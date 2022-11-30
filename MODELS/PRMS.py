import numpy as np
import os
import pandas as pd
import torch

class PRMS_pytorch(torch.nn.Module):
    def __init__(self):
        super(PRMS_pytorch, self).__init__()

    def precip_form(self, Precip, Tmaxf, Tminf, Tmax_allsnow_f,
                    Tmax_allrain_offset, args, Adjmix_rain = 1.0,
                    NEARZERO = 0.00001, rain_adj=1.0, snow_adj=1.0):
        """

        :param Precip: HRU Precipitation
        :param Tmaxf: max air temp
        :param Tminf: min air temp
        :param Tmax_allsnow_f: Monthly (January to December) maximum air temperature when precipitation is assumed to
                            be snow; if HRU air temperature is less than or equal to this value, precipitation is snow
        :param Tmax_allrain_offset: Monthly (January to December) maximum air temperature when precipitation is assumed to
                        be rain; if HRU air temperature is greater than or equal to Tmax_allsnow_f plus this value,
                        precipitation is rain
        :param Adjmix_rain: Monthly (January to December) factor to adjust rain proportion in
                        a mixed rain/snow event  (0.6 - 1.4)
        :param NEARZERO:
        :param rain_adj: Monthly (January to December) factor to adjust measured precipitation on each HRU to
                        account for differences in elevation, and so forth (0.5 - 2.0)
        :param snow_adj: Monthly (January to December) factor to adjust measured precipitation on each HRU to
                        account for differences in elevation, and so forth (0.5 - 2.0)
        :return: Basin area-weighted average rainfall,
                Basin area-weighted average snowfall
        """
        # if max temp is below or equal to the base temp for snow then
        # precipitation is all snow
        snow_hru = snow_adj * torch.where(Tmaxf <= Tmax_allsnow_f, Precip, Precip * 0.0)

        mask_snow_hru = snow_hru.le(0.0) .float()   # shows when it is not snowing
        # if min temp is above base temp for snow or
        # max temp is above all_rain temp then the precipitation is all rain

        rain_hru = rain_adj * torch.where((Tminf > Tmax_allsnow_f) | (Tmaxf >= Tmax_allsnow_f + Tmax_allrain_offset),
                                          Precip * mask_snow_hru, Precip * mask_snow_hru * 0.0)
        mask_rain_hru = rain_hru.le(0.0).float()
        # otherwise precipitation is a mixture of rain and snow
        tdiff = Tmaxf - Tminf
        if tdiff.le(-NEARZERO).int().sum() > 0.0:   # Tmax < Tmin
            print("ERROR, tmax < tmin ")

        tdiff_min = torch.zeros(tdiff.shape, device=args["device"]) + 0.0001
        tdiff = torch.where(tdiff < NEARZERO, tdiff_min, tdiff)
        Prmx = ((Tmaxf - Tmax_allsnow_f) / tdiff) * Adjmix_rain
        Prmx = torch.clamp(Prmx, min=0.0, max=1.0)
        hru_ppt = Precip * snow_adj * mask_rain_hru * mask_snow_hru
        rain_hru_mix = hru_ppt * Prmx
        snow_hru_mix = hru_ppt - rain_hru_mix

        Basin_rain = rain_hru + rain_hru_mix
        Basin_snow = snow_hru + snow_hru_mix

        return Basin_rain, Basin_snow

    def PRMS_transp_month_ON_OFF(self, months, args, transp_beg=None, transp_end=None):

        # here if we don't define the ON OFF time for transp, it is assumed to be started at month3 and ended at month 9
        if transp_beg == None:
            transp_beg = torch.ones(months.shape, dtype=torch.float32, device=args["device"]) + 2.0
        if transp_end == None:
            transp_end = torch.ones(months.shape, dtype=torch.float32, device=args["device"]) + 8.0

        # this mask tells us when the transpiration is ON
        mask_transp_ON = torch.where((months >= transp_beg) & (months < transp_end),
                                     torch.ones(months.shape, device=args["device"], dtype=torch.float32),
                                     torch.zeros(months.shape, device=args["device"], dtype=torch.float32))
        return mask_transp_ON

    def PRMS_transp_tindex(self, tmaxf, args, months, transp_beg=None, transp_end=None, transp_tmax=None):

        # getting the time that transp starts (0 or 1)
        mask_transp_month_ON = self.PRMS_transp_month_ON_OFF(months, args,
                                            transp_beg=transp_beg,
                                            transp_end=transp_end)

        # transp does not occurs in freezing temperature
        tmaxf_pos_mask = torch.where(tmaxf > 0.0, torch.ones(tmaxf.shape, device=args["device"], dtype=torch.float32),
                                     torch.zeros(tmaxf.shape, device=args["device"], dtype=torch.float32))
        tmax_mod = tmaxf * tmaxf_pos_mask * mask_transp_month_ON
        Tmax_sum = torch.cumsum(tmax_mod, axis=1)    #tmax_mod.shape[basin, rho + bufftime, nmul]

        if transp_tmax == None:
            transp_tmax = torch.ones(Tmax_sum.shape, device=args["device"], dtype=torch.float32) * 149.0  # we assume it is 300 in F and 149 in C

        mask_transp_tmax = torch.where(Tmax_sum >= transp_tmax,
                                       torch.ones(Tmax_sum.shape, device=args["device"], dtype=torch.float32),
                                       torch.zeros(Tmax_sum.shape, device=args["device"], dtype=torch.float32))

        # A mask that shows when transp is ON (it should be between transp_beg and transp_end
        # and also reaches the transp_tmax as well)
        Basin_transp_on = mask_transp_tmax * mask_transp_month_ON
        return Basin_transp_on

    def intercept(self, Precip, Stor_max, Cov, intcp_stor):
        Net_Precip = Precip * (1 - Cov)
        intcp_stor = intcp_stor + Precip
        Net_Precip = torch.where(intcp_stor > Stor_max,
                                 Net_Precip + (intcp_stor - Stor_max) * Cov,
                                 Net_Precip)
        intcp_stor = torch.where(intcp_stor > Stor_max,
                                 Stor_max,
                                 intcp_stor)
        return Net_Precip, intcp_stor

    def intcp(self, args, c_PRMS, Hru_rain, Hru_snow,  Basin_transp_on, intcp_stor):
        # why do we have these in the fortran code? intcp.f90 lines 310 - 314
        # IF (Transp_on(i) == 1)
        # THEN
        # Canopy_covden(i) = Covden_sum(i)
        # ELSE
        # Canopy_covden(i) = Covden_win(i)
        # ENDIF
        # if transp_On == 1 --> canopy_covden = covden_sum
        nmul = args["nmul"]
        covden_sum = c_PRMS[:, (args['optData']['varC_PRMS']).index('covden_sum')].unsqueeze(-1).repeat(1,nmul)
        covden_win = c_PRMS[:, (args['optData']['varC_PRMS']).index('covden_win')].unsqueeze(-1).repeat(1,nmul)
        srain_intcp = c_PRMS[:, (args['optData']['varC_PRMS']).index('srain_intcp(mm)')].unsqueeze(-1).repeat(1,nmul)
        snow_intcp = c_PRMS[:, (args['optData']['varC_PRMS']).index('snow_intcp(mm)')].unsqueeze(-1).repeat(1,nmul)
        wrain_intcp = c_PRMS[:, (args['optData']['varC_PRMS']).index('wrain_intcp(mm)')].unsqueeze(-1).repeat(1,nmul)
        cov_type = c_PRMS[:, (args['optData']['varC_PRMS']).index('cov_type')].unsqueeze(-1).repeat(1,nmul)

        #  translation of lines 313-340 of intcp.f90, adjusment interception amounts in summer and winter
        cov = torch.where(Basin_transp_on == 1.0, covden_sum, covden_win)
        intcp_form = torch.where(Hru_snow > 0.0,
                                 torch.ones(Hru_snow.shape, device=args["device"], dtype=torch.float32),
                                 torch.zeros(Hru_snow.shape, device=args["device"], dtype=torch.float32))
        # cov_type=0 --> bare ground --> no intcp storage
        # it should be done for lake area/hrus too, now I don't have any lake
        extrawater = torch.where((cov_type == 0.0) & (intcp_stor > 0.0),
                                 intcp_stor,
                                 torch.zeros(cov_type.shape, device=args["device"], dtype=torch.float32))
        # adjusting intcp storage for bare grounds.
        intcp_stor = torch.where((cov_type == 0.0) & (intcp_stor > 0.0),
                                 torch.zeros(intcp_stor.shape, device=args["device"], dtype=torch.float32),
                                 intcp_stor)

        # ###Determine the amount of interception from rain
        # go from summer to winter.   translation of  lines 344 - 362 intcp.f90
        #  Farshid: what happens if cov > covden_sum? Answer: It doesn't happen n current dataset
        diff = torch.where((Basin_transp_on==0.0) & (intcp_stor > 0.0),
                           covden_sum - cov,
                           torch.zeros(Basin_transp_on.shape, device=args["device"], dtype=torch.float32))
        changeover = torch.where((Basin_transp_on == 0.0) & (intcp_stor > 0.0),
                           diff * intcp_stor,
                           torch.zeros(Basin_transp_on.shape, device=args["device"], dtype=torch.float32))
        intcp_stor = torch.where((cov > 0.0) & (changeover < 0.0) & (Basin_transp_on==0.0) & (intcp_stor > 0.0),
                                 intcp_stor * covden_sum / cov,
                                 intcp_stor)
        changeover = torch.where((cov > 0.0) & (changeover < 0.0) & (Basin_transp_on==0.0) & (intcp_stor > 0.0),
                                 torch.zeros(changeover.shape, device=args["device"], dtype=torch.float32),
                                 changeover)
        intcp_stor = torch.where((Basin_transp_on==0.0) & (intcp_stor > 0.0) & (cov <= 0.0),
                                 torch.zeros(intcp_stor.shape, device=args["device"], dtype=torch.float32),
                                 intcp_stor)

        ##  go from winter to summer.   translation of  lines 365 - 382 intcp.f9
        diff = torch.where((Basin_transp_on == 1.0) & (intcp_stor > 0.0),
                           covden_win - cov,
                           diff)
        changeover = torch.where((Basin_transp_on == 1.0) & (intcp_stor > 0.0),
                                 diff * intcp_stor,
                                 changeover)
        intcp_stor = torch.where((cov > 0.0) & (changeover < 0.0) & (Basin_transp_on == 1.0) & (intcp_stor > 0.0),
                                 intcp_stor * covden_win / cov,
                                 intcp_stor)
        changeover = torch.where((cov > 0.0) & (changeover < 0.0) & (Basin_transp_on == 1.0) & (intcp_stor > 0.0),
                                 torch.zeros(changeover.shape, device=args["device"], dtype=torch.float32),
                                 changeover)
        intcp_stor = torch.where((Basin_transp_on == 1.0) & (intcp_stor > 0.0) & (cov <= 0.0),
                                 torch.zeros(intcp_stor.shape, device=args["device"], dtype=torch.float32),
                                 intcp_stor)

        ## determine the amount of interception from rain translation lines 386 - 410 of intcp.f90
        # stor = stor_Max
        stor = torch.where((cov_type != 0.0) & (Basin_transp_on == 1.0),
                           srain_intcp,
                           torch.zeros(srain_intcp.shape, device=args["device"], dtype=torch.float32))
        stor = torch.where((cov_type != 0.0) & (Basin_transp_on == 0.0),
                           wrain_intcp,
                           stor)

        # it should be done for cov_type>1
        net_rain_temp, intcp_stor_temp = self.intercept(Hru_rain, stor, cov, intcp_stor)
        net_rain = torch.where(cov_type > 1,
                               net_rain_temp,
                               Hru_rain)
        intcp_stor = torch.where(cov_type > 1,
                                 intcp_stor_temp,
                                 intcp_stor)

        # I didnot code the irrigation part  , lines 415-435 intcp.f90

        #  determine the amount of interception from snow


        return net_rain, net_snow, intcp_stor

        #intcp_form==1 if Hru_snow>0



    def forward(self, x, c_PRMS, params, args, warm_up=0):
        NEARZERO = args["NEARZERO"]
        nmul = args["nmul"]
        vars = args['optData']['varT_PRMS']
        if warm_up > 0:
            with torch.no_grad():
                xinit = x[0:warm_up, :, :]
                warm_up_model = PRMS_pytorch()
                intcpstor, B , C  = warm_up_model(xinit, c_PRMS, params, args, warm_up=0)
        else:
            # zero for initializiation
            intcpstor = torch.zeros([x.shape[0], nmul], dtype=torch.float32, device=args["device"])
            B = torch.zeros([x.shape[0], nmul], dtype=torch.float32, device=args["device"])
            C = torch.zeros([x.shape[0], nmul], dtype=torch.float32, device=args["device"])

        Ngrid, Ndays = x.shape[0], x.shape[1]
        Precip = x[:, warm_up:, vars.index('prcp(mm/day)')].unsqueeze(-1).repeat(1, 1, nmul)
        Tmaxf = x[:, warm_up:, vars.index("tmax(C)")].unsqueeze(-1).repeat(1, 1, nmul)
        Tminf = x[:, warm_up:, vars.index("tmin(C)")].unsqueeze(-1).repeat(1, 1, nmul)
        month = x[:, warm_up:, vars.index("month")].unsqueeze(-1).repeat(1, 1, nmul)

        mean_air_temp = (Tmaxf + Tminf) / 2
        Tmax_allsnow_f = torch.zeros(Tmaxf.shape, dtype=torch.float32, device=args["device"])
        Tmax_allrain_offset = torch.ones(Tmaxf.shape, dtype=torch.float32, device=args["device"]) + 0.5
        Basin_rain, Basin_snow = self.precip_form(Precip, Tmaxf, Tminf, Tmax_allsnow_f, Tmax_allrain_offset, args,
                                                  Adjmix_rain=1.0, NEARZERO=NEARZERO, rain_adj=1.0, snow_adj=1.0)
        Basin_transp_on = self.PRMS_transp_tindex(Tmaxf, args, months=month, transp_beg=None,
                                             transp_end=None, transp_tmax=None)

        for t in range(Ndays):
            Intcp_transp_on = Basin_transp_on[:, t, :]
            netrain = Basin_rain[:,t,:]
            netsnow = Basin_snow[:, t, :]


            intcp_rain, intcp_snow, intcpstor = self.intcp(args, c_PRMS, netrain, netsnow, Intcp_transp_on, intcpstor)





















# Precip = torch.rand((5, 10, 15))
# Tmax_allrain_f = torch.rand((5, 10, 15)) + 5.0
# Tmax_allsnow_f = torch.rand((5, 10, 15))
# Tmaxf = torch.rand((5, 10, 15)) + 10.0
# Tmaxf[0,0, :] = - 5.0
# Tminf = torch.rand((5, 10, 15))
# Tminf[0, 0, :] = - 15.0
# Tminf[0,2, :] = Tmax_allsnow_f[0,2, :] - 3.0
# Tmaxf[0,2, :] = Tmax_allrain_f[0,2, :] - 1.0
# basin_rain, Basin_snow = precip_form(self, Precip, Tmaxf, Tminf, Tmax_allsnow_f, Tmax_allrain_f)
#
#
#
# months = torch.randint(1, 12, Tminf.shape)
# Basin_transp_on = PRMS_transp_tindex(Tmaxf, months, transp_beg=None, transp_end=None, transp_tmax=None)


