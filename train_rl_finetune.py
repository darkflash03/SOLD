import warnings
warnings.filterwarnings("ignore")

from rl_trainers.register import *
from utils import *
import torch

def train_rl_finetune(config):
    torch.multiprocessing.set_start_method("spawn")
    init_rhofold_model()
    rl_trainer_dict = register_rl_trainer()
    trainer_name = config['train_config']['rl_config']['name']
    if trainer_name in rl_trainer_dict:
        trainer = rl_trainer_dict[trainer_name](config)
    else:
        raise ValueError('Trainer {} not registered'.format(trainer_name))

    print("---------- Start RL Finetuning----------")
    trainer.train()
    print("RL Finetuning Complete ......")