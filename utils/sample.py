import torch
from tqdm import tqdm
from utils.metrics import *
from utils.constants import *
import pandas as pd

def latent_diffusion_sample(diffusion_model, encoder_decoder, device, dataloader, pdb_data_dir, predict_pdb_dir, num_steps=0):
    with torch.no_grad():
        result_dict = {
            "pdb_id": [],
            "gt_seq": [],
            "pred_seq": []
        }
        if num_steps == 0:
            iter = tqdm(dataloader, desc=f"ddpm sampling...")
        else:
            iter = tqdm(dataloader, desc=f"ddim sampling with step {num_steps}...")

        for data in iter:
            data = data.to(device)
            if num_steps == 0:
                traj, _ = diffusion_model.sample_with_logprob(data)
            else:
                traj, _ = diffusion_model.ddim_sample_with_logprob(data, num_steps)

            data['sample_latent'] = traj[0].float()
            sample_logit = encoder_decoder(data, 'sample_latent')
            gt_logit = torch.argmax(data['seq'].view(sample_logit.shape), dim=-1).to(device)
            batch_size = data['sample_latent'].shape[0]
            pdb_name = data['name'][0]

            for i in range(batch_size):
                r, _ = seq_recovery(gt_logit[i], sample_logit[i])
                result_dict['pdb_id'].append(pdb_name)

            pred_seqs = torch.argmax(sample_logit, dim=-1)

            pred_seq_strs = []
            for i in range(batch_size):
                pred_seq_str = "".join([nucleotides[i] for i in pred_seqs[i]])
                pred_seq_strs.append(pred_seq_str)
                result_dict['pred_seq'].append(pred_seq_str)
                gt_seq_str = "".join([nucleotides[i] for i in gt_logit[i]])
                result_dict['gt_seq'].append(gt_seq_str)

        result = pd.DataFrame(result_dict)
    return result