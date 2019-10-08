import argparse

from utils import ConfigDict, create_directory

parser = argparse.ArgumentParser()
parser.add_argument("master")
parser.add_argument("config")
parser.add_argument("root_gpu", type=int)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

config_loc = args.config
config = ConfigDict(config_loc)

config["name"] = config_loc.split("/")[-1][:-5]  # a/b/xxx.json => xxx
config["resume"] = args.resume

# Extra config that original tf-STEVE config doesn't have
config["default_actor_batch"] = 4  # size of vector env for remote actors
config["policy_config"]["decay"] = 0.001  # weight decay when sync Q to old-Q
config["policy_config"]["actor_lr"] = 3e-4
config["policy_config"]["critic_lr"] = 3e-4
config["model_config"]["lr"] = 3e-4
config["max_sample_steps"] = 5e6


def log_config(cfg):
    log_folder = "output/" + cfg["env"]["name"] + "/" + \
        cfg["name"] + "/" + cfg["log_path"]
    hps_path = create_directory(log_folder) + "/hps.json"
    cfg_str = str(cfg)
    print(cfg_str)
    with open(hps_path, 'w') as f:
        f.write(cfg_str)
