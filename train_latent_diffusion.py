import warnings
warnings.filterwarnings("ignore")

from utils import *
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from dataset import *
from torch.utils.data import DataLoader
from models import *
from ema_pytorch import EMA
from torch.optim import AdamW
from tqdm import tqdm
import collections
import torch.nn.functional as F
import time

def train_latent_diffusion(config):
    print("Training Latent Diffusion model...")
    torch.multiprocessing.set_start_method('spawn')

    data_config = config['data_config']
    train_config = config['train_config']
    seed = config['seed']
    seeding(seed)

    encoder_decoder_config = config['encoder_decoder_config']
    llm_config = encoder_decoder_config['llm_config']
    llm_config['device'] = config['device']
    data_config['llm_config'] = llm_config

    llm_name = llm_config['name']
    compress_dim = encoder_decoder_config['decoder']['config']['encoder_hidden_dim']
    result_folder = os.path.join(config['train_config']['output_dir'], f'{llm_name}_compress_dim_{compress_dim}')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    save_config(os.path.join(result_folder, 'config.yaml'), config)

    # Initialize TensorBoard
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(result_folder, "runs", current_time)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    train_dataset = GraphDataset(data_path=data_config['train_data_path'],
                                 data_config=data_config)

    valid_dataset = GraphDataset(data_path=data_config['valid_data_path'],
                                 data_config=data_config)

    if train_config.get("sampler"):
        train_sampler = TrainSampler(
            train_dataset,
            batch_size=train_config['batch_size'],
            sample_mode=train_config["sample_mode"],
            max_squared_res=train_config["max_squared_res"],
        )
    else:
        train_sampler = None

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=train_config["num_workers"],
        collate_fn=train_dataset.collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=valid_dataset.collate_fn,
    )

    if encoder_decoder_config['decoder']['name'] == 'mlp':
        decoder = MLPDecoder(**encoder_decoder_config['decoder']['config'])
        encoder_decoder = EncoderMlpDecoder(decoder=decoder)
    else:
        raise NotImplementedError(encoder_decoder_config['decoder_config']['name'])

    encoder_decoder_ckpt_path = encoder_decoder_config['ckpt_path']
    print(f"loading encoder_decoder model from {encoder_decoder_config['ckpt_path']}")
    state_dict = torch.load(encoder_decoder_ckpt_path, map_location='cpu')
    if "encoder_decoder" in state_dict:
        state_dict = state_dict["encoder_decoder"]
    encoder_decoder.load_state_dict(state_dict, strict=False)
    encoder_decoder.to(config['device'])
    encoder_decoder.eval()

    model = GVPNet(config['model_config'])
    diffusion_model = DiffusionModel(model, train_config['latent_transition_config']).to(config['device'])
    if train_config.get("ckpt_path"):
        print("load pretrained model: {}...".format(train_config["ckpt_path"]))
        diffusion_model.load_state_dict(torch.load(train_config["ckpt_path"])["model"])
    ema = EMA(diffusion_model, beta=0.995, update_every=train_config["gradient_accumulate_every"]).to(config['device'])
    optimizer = AdamW(model.parameters(), lr=train_config["lr"], betas=(0.9, 0.99), weight_decay=train_config['wd'])

    print("---------- Start Training Latent Diffusion Model ----------")
    global_steps = 0
    gradient_step = 0
    accum_step = 0
    accum_loss_dict = {"loss_seq": 0, "loss_latent": 0, "total_loss": 0}
    train_elapsed_time = 0
    best_valid_seq_recovery = 0
    for epoch in range(train_config['max_train_epochs']):
        print(f"---------- Epoch {epoch} ----------")
        ema.ema_model.eval()
        if train_config.get("ddim_num_steps"):
            num_steps = train_config["ddim_num_steps"]
            val_recovery_seq_ddim, val_recovery_aa_ddim = validate_latent_diffusion(ema.ema_model, encoder_decoder,
                                                                                    config['device'], valid_dataloader,
                                                                                    'validate', epoch, num_steps=num_steps)
            print(f"epoch: {epoch}, ddim sampling with steps {num_steps} ..., val seq recovery: {val_recovery_seq_ddim}, val recovery aa: {val_recovery_aa_ddim}")
            writer.add_scalar(f"Validation_ddim_{num_steps}_steps/recovery_seq", val_recovery_seq_ddim, epoch)
            writer.add_scalar(f"Validation_ddim_{num_steps}_steps/recovery_aa", val_recovery_aa_ddim, epoch)
        val_recovery_seq, val_recovery_aa = validate_latent_diffusion(ema.ema_model, encoder_decoder, config['device'], valid_dataloader, 'validate', epoch)
        print(f"epoch: {epoch}, ddpm sampling..., val seq recovery: {val_recovery_seq}, val recovery aa: {val_recovery_aa}")
        writer.add_scalar(f"Validation_ddpm/recovery_seq", val_recovery_seq, epoch)
        writer.add_scalar(f"Validation_ddpm/recovery_aa", val_recovery_aa, epoch)

        if val_recovery_seq > best_valid_seq_recovery - train_config['early_stopping']['min_boost_recovery']:
            best_valid_seq_recovery = val_recovery_seq
            counter = 0
            data = {'model': diffusion_model.state_dict(), 'opt': optimizer.state_dict(), 'ema': ema.state_dict()}
            torch.save(data, os.path.join(result_folder, 'model_epoch_{}.pt'.format(epoch)))
        else:
            counter += 1
            if counter >= train_config['early_stopping']['patience']:
                print("Early stopping triggered ......")
                break

        diffusion_model.train()
        ema.ema_model.train()
        epoch_iterator = tqdm(train_dataloader)
        loss_log_dict = collections.defaultdict(list)
        epoch_start_time = time.time()
        for data in epoch_iterator:
            data = data.to(config['device'])
            pred_latent, t = diffusion_model(data)
            data['pred_latent'] = pred_latent
            gt_latent = data['x'].reshape(pred_latent.shape)
            loss_latent = torch.mean(F.mse_loss(pred_latent, gt_latent, reduction='none'), dim=(-2, -1))
            pred_logit = encoder_decoder(data, 'pred_latent')
            loss_seq = torch.mean(
                F.cross_entropy(
                    pred_logit.transpose(1, 2),
                    data['seq'].reshape(pred_logit.shape).transpose(1, 2),
                    reduction='none',
                ),
                dim=-1,
            )
            loss_dict = {"loss_seq": loss_seq, "loss_latent": loss_latent}
            calc_loss_dict = get_loss_fn(loss_dict, train_config['loss_config'])
            for k, v in calc_loss_dict.items():
                loss_log_dict[k].append(v.item())
                accum_loss_dict[k] += v.item()

            loss = calc_loss_dict["total_loss"]
            global_steps += 1
            accum_step += 1
            loss.backward()

            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), 1)

            if accum_step % train_config['gradient_accumulate_every'] == 0:
                for k, v in accum_loss_dict.items():
                    writer.add_scalar(f"Train-Step/{k}", v / train_config['gradient_accumulate_every'], gradient_step)
                    accum_loss_dict[k] = 0

                gradient_step += 1
                optimizer.step()
                optimizer.zero_grad()

            epoch_iterator.set_postfix(loss=loss.item())
        epoch_end_time = time.time()

        for k, v in loss_log_dict.items():
            print("train epoch_{}: {}".format(k, np.mean(v)), end=" ")
            train_elapsed_time += (epoch_end_time - epoch_start_time)
            writer.add_scalar(f"Train-Epoch/{k}", np.mean(v), epoch)
        writer.add_scalar(f"Train-Epoch/elapsed_time", train_elapsed_time, epoch)
        ema.update()

    writer.close()
    print("Training Latent Diffusion Complete ......")