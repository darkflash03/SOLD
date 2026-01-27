import numpy as np
import random
import os
import torch
import yaml

def seeding(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("seeding done !")

def load_config(config_path):
    with open(config_path, 'r', encoding="utf-8") as f:
        running_config = yaml.load(f, Loader=yaml.FullLoader)
    return running_config

def save_config(config_path, config):
    with open(config_path, 'w', encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)

def get_loss_fn(loss_dict, loss_config):
    loss = 0
    calc_loss_dict = {}
    for item in loss_dict.keys():
        item_loss = loss_dict[item].mean()
        loss += item_loss * loss_config[item]['weight']
        calc_loss_dict[item] = item_loss
    calc_loss_dict["total_loss"] = loss
    return calc_loss_dict
