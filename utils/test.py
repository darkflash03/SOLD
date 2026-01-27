import torch
from tqdm import tqdm
from utils.constants import nucleotides
from utils.metrics import seq_recovery, secondary_structure_metric, three_dimension_metric
import collections

def latent_diffusion_test(diffusion_model, encoder_decoder, device, dataloader, name, epoch, pdb_data_dir, predict_pdb_dir, num_steps=0):
    with torch.no_grad():
        ind_all = torch.tensor([])
        r_all = 0
        ss_all = 0
        mfe_all = 0
        rmsd_all = 0
        lddt_all = 0
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

            ##calc ss，mfe metrics
            sample_logit = sample_logit.argmax(dim=-1)
            pred_seq_str = "".join([nucleotides[i] for i in sample_logit])
            gt_seq_str = "".join([nucleotides[i] for i in gt_logit])
            ss, mfe = secondary_structure_metric(gt_seq_str, pred_seq_str)
            ss_all += ss
            mfe_all += mfe

            ##calc rmsd, lddt metrics
            pdb_name = data['name'][0]
            rmsd_list, lddt_list = three_dimension_metric([pred_seq_str], pdb_name, pdb_data_dir, predict_pdb_dir, n_jobs=1)
            rmsd_all += rmsd_list[0]
            lddt_all += lddt_list[0]

        r = r_all / len(dataloader)
        rr = (ind_all.sum() / ind_all.shape[0]).item()
        ss = ss_all / len(dataloader)
        mfe = mfe_all / len(dataloader)
        rmsd = rmsd_all / len(dataloader)
        lddt = lddt_all / len(dataloader)
    return r, rr, ss / 100, mfe, rmsd, lddt

def rl_finetune_test(diffusion_model, encoder_decoder, device, dataloader, name, pdb_data_dir, predict_pdb_dir, config):
    eval_single_step_metric = config['eval_single_step_metric']
    eval_single_step_flag = False
    for metric in eval_single_step_metric:
        eval_single_step_flag |= eval_single_step_metric[metric]
    rewards_dict = collections.defaultdict(list)
    with torch.no_grad():
        ind_all = torch.tensor([])
        r_all = 0
        ss_all = 0
        mfe_all = 0
        rmsd_all = 0
        lddt_all = 0

        iter = tqdm(dataloader, desc=f"testing on {name} dataset")
        for data in iter:
            data = data.to(device)
            gt_logit = torch.argmax(data['seq'], dim=-1)
            sample_traj, sample_logprob_traj = diffusion_model.sample_with_logprob(data)
            if eval_single_step_flag:
                if eval_single_step_metric['recovery']:
                    single_recovery_rewards = []
                if eval_single_step_metric['ss']:
                    single_ss_rewards = []
                if eval_single_step_metric['mfe']:
                    single_mfe_rewards = []
                if eval_single_step_metric['rmsd']:
                    single_rmsd_rewards = []
                if eval_single_step_metric['lddt']:
                    single_lddt_rewards = []
                for step in range(len(sample_traj)):
                    sample_latent = sample_traj[step].squeeze()
                    data['sample_latent'] = sample_latent.float()
                    sample_logit = encoder_decoder(data, 'sample_latent')
                    if eval_single_step_metric['recovery']:
                        r, _ = seq_recovery(gt_logit, sample_logit)
                        single_recovery_rewards.append(r.item())

                    sample_logit = sample_logit.argmax(dim=-1)
                    pred_seq_str = "".join([nucleotides[i] for i in sample_logit])
                    gt_seq_str = "".join([nucleotides[i] for i in gt_logit])

                    if eval_single_step_metric['ss'] or eval_single_step_metric['mfe']:
                        ss, mfe = secondary_structure_metric(gt_seq_str, pred_seq_str)
                        single_ss_rewards.append(ss)
                        single_mfe_rewards.append(mfe)
                    if eval_single_step_metric['rmsd'] or eval_single_step_metric['lddt']:
                        pdb_name = data['name'][0]
                        rmsd_list, lddt_list = three_dimension_metric([pred_seq_str], pdb_name, pdb_data_dir,
                                                                      predict_pdb_dir, n_jobs=1)
                        single_rmsd_rewards.append(rmsd_list[0])
                        single_lddt_rewards.append(lddt_list[0])

                if eval_single_step_metric['recovery']:
                    rewards_dict['recovery'].append(single_recovery_rewards)
                if eval_single_step_metric['ss'] or eval_single_step_metric['mfe']:
                    rewards_dict['ss'].append(single_ss_rewards)
                    rewards_dict['mfe'].append(single_mfe_rewards)
                if eval_single_step_metric['rmsd'] or eval_single_step_metric['lddt']:
                    rewards_dict['rmsd'].append(single_rmsd_rewards)
                    rewards_dict['lddt'].append(single_lddt_rewards)

            sample_latent = sample_traj[0].squeeze()
            data['sample_latent'] = sample_latent.float()
            sample_logit = encoder_decoder(data, 'sample_latent')
            gt_logit = torch.argmax(data['seq'], dim=-1)
            r, ind = seq_recovery(gt_logit, sample_logit)
            r_all += r.item()
            ind_all = torch.cat([ind_all, ind])

            ##calc ss，mfe metrics
            sample_logit = sample_logit.argmax(dim=-1)
            pred_seq_str = "".join([nucleotides[i] for i in sample_logit])
            gt_seq_str = "".join([nucleotides[i] for i in gt_logit])
            ss, mfe = secondary_structure_metric(gt_seq_str, pred_seq_str)
            ss_all += ss
            mfe_all += mfe

            ##calc rmsd, lddt metrics
            pdb_name = data['name'][0]
            rmsd_list, lddt_list = three_dimension_metric([pred_seq_str], pdb_name, pdb_data_dir, predict_pdb_dir,
                                                          n_jobs=1)
            rmsd_all += rmsd_list[0]
            lddt_all += lddt_list[0]

        r = r_all / len(dataloader)
        rr = (ind_all.sum() / ind_all.shape[0]).item()
        ss = ss_all / len(dataloader)
        mfe = mfe_all / len(dataloader)
        rmsd = rmsd_all / len(dataloader)
        lddt = lddt_all / len(dataloader)
    return r, rr, ss / 100, mfe, rmsd, lddt, rewards_dict