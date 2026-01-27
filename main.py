import argparse
from utils import load_config
from train_encoder_decoder import train_encoder_decoder
from test_encoder_decoder import test_encoder_decoder
from train_latent_diffusion import train_latent_diffusion
from test_latent_diffusion import test_latent_diffusion
from sample_latent_diffusion import sample_latent_diffusion
from train_rl_finetune import train_rl_finetune
from test_rl_finetune import test_rl_finetune

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/latent_diffusion.yaml')
    args = parser.parse_args()
    config = load_config(args.config)

    pipe = config['pipeline']

    if pipe == "train_encoder_decoder":
        train_encoder_decoder(config)
    elif pipe == "test_encoder_decoder":
        test_encoder_decoder(config)
    elif pipe == "train_latent_diffusion":
        train_latent_diffusion(config)
    elif pipe == "test_latent_diffusion":
        test_latent_diffusion(config)
    elif pipe == "sample_latent_diffusion":
        sample_latent_diffusion(config)
    elif pipe == "train_rl_finetune":
        train_rl_finetune(config)
    elif pipe == "test_rl_finetune":
        test_rl_finetune(config)
    else:
        raise ValueError(f'pipeline {pipe} not supported')