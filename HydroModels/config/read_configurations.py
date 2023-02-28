"""code to read the config file"""
import os

try:
    from ruamel.yaml import YAML
except ModuleNotFoundError:
    print("YAML Module not found.")

"""Local terminal path"""
"""pycharm path"""
config_path = "HydroModels/config/config.yaml"
yaml = YAML(typ="safe")
path = os.path.realpath(config_path)
stream = open(path, "r")
config = yaml.load(stream)
