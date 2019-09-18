import argparse

from utils import ConfigDict, create_directory

parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("root_gpu", type=int)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

config_loc = args.config
config = ConfigDict(config_loc)

config["name"] = config_loc.split("/")[-1][:-5]  # a/b/xxx.json => xxx
config["resume"] = args.resume


def log_config(cfg):
    log_folder = "output/" + cfg["env"]["name"] + "/" + \
        cfg["name"] + "/" + cfg["log_path"]
    hps_path = create_directory(log_folder) + "/hps.json"
    cfg_str = str(cfg)
    print(cfg_str)
    with open(hps_path, 'w') as f:
        f.write(cfg_str)
