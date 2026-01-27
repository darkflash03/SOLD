import torch
from tqdm import tqdm
from .metrics import seq_recovery

def validate_encoder_decoder(encoder_decoder, device, dataloader, loss_fn, name, epoch):
    total_val_loss = 0
    with torch.no_grad():
        ind_all = torch.tensor([])
        r_all = 0
        for batch in tqdm(dataloader, desc=f"after {epoch} epoch, validating on {name} dataset"):
            logits = encoder_decoder(batch)
            gt = torch.argmax(batch['seq'], dim=-1).to(device)
            loss = loss_fn(logits, gt)
            total_val_loss += loss.item()
            r, ind = seq_recovery(gt, logits)
            r_all += r.item()
            ind_all = torch.cat([ind_all, ind])

        val_loss = total_val_loss / len(dataloader)
        r = r_all / len(dataloader)
        rr = (ind_all.sum() / ind_all.shape[0]).item()
    return val_loss, r, rr

def validate_latent_diffusion(diffusion_model, encoder_decoder, device, dataloader, name, epoch, num_steps=0):
    with torch.no_grad():
        ind_all = torch.tensor([])
        r_all = 0
        if num_steps == 0:
            iter = tqdm(dataloader, desc=f"after {epoch} epoch, ddpm sampling, validating on {name} dataset")
        else:
            iter = tqdm(dataloader, desc=f"after {epoch} epoch, ddim sampling with step {num_steps}, validating on {name} dataset")

        for data in iter:
            data = data.to(device)
            if num_steps == 0:
                sample_latent = diffusion_model.ddpm_sample(data)
            else:
                sample_latent = diffusion_model.ddim_sample(data, num_steps)
            data['sample_latent'] = sample_latent.float()
            sample_logit = encoder_decoder(data, 'sample_latent')
            gt_logit = torch.argmax(data['seq'], dim=-1)
            r, ind = seq_recovery(gt_logit, sample_logit)
            r_all += r.item()
            ind_all = torch.cat([ind_all, ind])

        r = r_all / len(dataloader)
        rr = (ind_all.sum() / ind_all.shape[0]).item()
    return r, rr