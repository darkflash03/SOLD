from rl_trainers import *
def register_rl_trainer():
    rl_trainer_dict = {}
    rl_trainer_dict['ddpo'] = DDPOTrainer
    rl_trainer_dict['sold'] = SOLDTrainer
    rl_trainer_dict['dpok'] = DPOKTrainer
    return rl_trainer_dict