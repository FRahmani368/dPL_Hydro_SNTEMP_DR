"""code to read the config file"""
import os

try:
    from ruamel.yaml import YAML
except ModuleNotFoundError:
    print("YAML Module not found.")

"""Local terminal path"""
"""pycharm path"""
config_path_SNTEMP = "config/config_SNTEMP_only.yaml"
config_path_PRMS = "config/config_marrmot_PRMS_only.yaml"
config_path_PRMS_SNTEMP = "config/config_marrmotPRMS_SNTEMP.yaml"
yaml = YAML(typ="safe")
path_SNTEMP = os.path.realpath(config_path_SNTEMP)
path_PRMS = os.path.realpath(config_path_PRMS)
path_PRMS_SNTEMP = os.path.realpath(config_path_PRMS_SNTEMP)
stream_SNTEMP = open(path_SNTEMP, "r")
stream_PRMS = open(path_PRMS, "r")
stream_PRMS_SNTEMP = open(path_PRMS_SNTEMP, "r")
config_SNTEMP = yaml.load(stream_SNTEMP)
config_PRMS = yaml.load(stream_PRMS)
config_PRMS_SNTEMP = yaml.load(stream_PRMS_SNTEMP)

