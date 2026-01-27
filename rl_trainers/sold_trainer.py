from .base_rl_trainer import BaseRLTrainer
import numpy as np
import torch
import collections
from tqdm import tqdm
import copy
import os

class SOLDTrainer(BaseRLTrainer):
    """SOLD 训练器，使用网络预测 t=0 的 latent 计算奖励"""
    def __init__(self, config):
        super().__init__(config)
        self.pretrain_diffusion_model = self.setup_diffusion_model()
        self.pretrain_diffusion_model.eval()

    def setup_result_folder(self):
        reward_pattern = self.train_config['rl_config']['reward_pattern']
        pretrain_kl_weight = self.train_config['rl_config']['pretrain_kl_weight']
        val_step = self.train_config['rl_config']['val_step']
        self.result_folder = os.path.join(self.train_config['output_dir'],
                                          f'{self.rl_trainer_name}_{reward_pattern}_val_step_{val_step}_recovery_{self.recovery_weight}_ss_{self.ss_weight}_mfe_{self.mfe_weight}_lddt_{self.lddt_weight}')
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
    def sample_timesteps(self, batch_size):
        """采样单一时间步，支持均匀、指数或伽马分布"""
        rl_config = self.train_config["rl_config"]
        t_dist = rl_config.get("t_distribution", "uniform")
        if t_dist == "uniform":
            #return torch.randint(1, self.num_steps + 1, (batch_size,), device=self.device)
            return torch.randint(1, rl_config.get("val_step") + 1, (batch_size,), device=self.device)
        elif t_dist == "uniform_cut":
            long_reward_steps = rl_config["long_reward_steps"]
            short_reward_steps = rl_config["short_reward_steps"]
            t = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            for i in range(batch_size):
                while True:
                    sample = torch.randint(1, self.num_steps + 1, (1,), device=self.device)
                    if sample <= long_reward_steps or sample > self.num_steps - short_reward_steps:
                        t[i] = sample
                        break
            return t

        elif t_dist == "exponential":
            lambda_t = rl_config.get("lambda_t", 0.1)
            t_probs = torch.exp(-lambda_t * torch.arange(1, self.num_steps + 1, device=self.device))
            t_probs /= t_probs.sum()
            return torch.multinomial(t_probs, batch_size, replacement=True) + 1
        elif t_dist == "gamma":
            gamma = rl_config.get("gamma", 0.9)
            t_probs = torch.tensor([gamma ** t for t in range(self.num_steps)], device=self.device)
            t_probs /= t_probs.sum()
            return torch.multinomial(t_probs, batch_size, replacement=True) + 1

    def train_step(self, data):
        """单步训练，基于 t=0 的 latent 奖励"""
        rl_config = self.train_config["rl_config"]
        batch_size = data.batch[-1] + 1
        data = data.to(self.device)
        samples_collate = {}
        pdb_name = data['name'][0]
        length = data.x.shape[0] // batch_size
        latent_dim = data.x.shape[-1]

        # 采样时间步
        t = self.sample_timesteps(batch_size)
        t_tensor = t.unsqueeze(-1)

        # 前向加噪
        noisy_data = self.diffusion_model.add_noise(data, t_tensor)
        x_t = noisy_data.x.view(batch_size, length, latent_dim)

        # 使用参考模型生成 x_{t-1}
        with torch.no_grad():
            pred_latent_ref = self.ref_diffusion_model.model(noisy_data, t_tensor)
            ref_prev_latent, ref_logprob = self.ref_diffusion_model.latent_transition.step_with_logprob(
                x_t, pred_latent_ref, t
            )

        samples_collate['latents'] = x_t
        samples_collate['next_latents'] = ref_prev_latent
        samples_collate["log_probs"] = ref_logprob
        samples_collate["timesteps"] = t_tensor

        # 计算 t=0 的 latent 奖励，使用t=0预测的latent，还是用t-1时刻reverse sampling计算得到的latent，或者有其他模式
        if rl_config.get("reward_pattern") == 'predict':
            data["pred_latent"] = pred_latent_ref.float()
            pred_logit = self.encoder_decoder(data, "pred_latent")
            gt_seqs = torch.argmax(data["seq"].view(pred_logit.shape), dim=-1).to(self.device)
            pred_seqs = torch.argmax(pred_logit, dim=-1)
            rewards, _ = self.compute_rewards(pred_seqs, gt_seqs, pdb_name, self.data_config['pdb_data_dir'],
                                              out_dir='sold_train_tmp', n_jobs=rl_config['n_jobs'])
        elif rl_config.get("reward_pattern") == 'sampling':
            data["pred_latent"] = ref_prev_latent.float()
            pred_logit = self.encoder_decoder(data, "pred_latent")
            gt_seqs = torch.argmax(data["seq"].view(pred_logit.shape), dim=-1).to(self.device)
            pred_seqs = torch.argmax(pred_logit, dim=-1)
            rewards, _ = self.compute_rewards(pred_seqs, gt_seqs, pdb_name, self.data_config['pdb_data_dir'],
                                              out_dir='sold_train_tmp', n_jobs=rl_config['n_jobs'])
        elif rl_config.get("reward_pattern") == 'mix':
            data["pred_latent"] = pred_latent_ref.float()
            pred_logit = self.encoder_decoder(data, "pred_latent")
            data["sample_latent"] = ref_prev_latent.float()
            sample_logit = self.encoder_decoder(data, "sample_latent")
            gt_seqs = torch.argmax(data["seq"].view(pred_logit.shape), dim=-1).to(self.device)
            pred_seqs = torch.argmax(pred_logit, dim=-1)
            pred_rewards, _ = self.compute_rewards(pred_seqs, gt_seqs, pdb_name, self.data_config['pdb_data_dir'],
                                              out_dir='sold_train_pred_tmp', n_jobs=rl_config['n_jobs'])
            sample_seqs = torch.argmax(sample_logit, dim=-1)
            sample_rewards, _ = self.compute_rewards(sample_seqs, gt_seqs, pdb_name, self.data_config['pdb_data_dir'],
                                                   out_dir='sold_train_sample_tmp', n_jobs=rl_config['n_jobs'])
            # print("pred_rewards: ", pred_rewards)
            # print("sample_rewards: ", sample_rewards)
            rewards = (1 - t.cpu() / self.num_steps) * pred_rewards + t.cpu() / self.num_steps * sample_rewards
        elif rl_config.get("reward_pattern") == 'mix_cut':
            rewards = torch.zeros(batch_size)
            mask_long_reward = t.cpu() <= rl_config['long_reward_steps']
            # print("mask_long_reward", mask_long_reward)
            if mask_long_reward.any():
                data["pred_latent"] = pred_latent_ref[mask_long_reward].float()
                pred_logit_long_reward = self.encoder_decoder(data, "pred_latent")
                num_classes = pred_logit_long_reward.shape[-1]
                seq_reshaped = data["seq"].view(batch_size, -1, num_classes)
                seq_masked = seq_reshaped[mask_long_reward]
                gt_seqs_long_reward = torch.argmax(seq_masked, dim=-1).to(self.device)
                pred_seqs_long_reward = torch.argmax(pred_logit_long_reward, dim=-1)
                rewards_small_t, _ = self.compute_rewards(
                    pred_seqs_long_reward, gt_seqs_long_reward, pdb_name, self.data_config['pdb_data_dir'],
                    out_dir='sold_train_mix_cut_tmp', n_jobs=rl_config['n_jobs']
                )
                rewards[mask_long_reward] = rewards_small_t
            mask_short_reward = ~mask_long_reward
            if mask_short_reward.any():
                data["sample_latent"] = ref_prev_latent[mask_short_reward].float()
                pred_logit_short_reward = self.encoder_decoder(data, "sample_latent")
                num_classes = pred_logit_short_reward.shape[-1]
                seq_reshaped = data["seq"].view(batch_size, -1, num_classes)
                seq_masked = seq_reshaped[mask_short_reward]
                gt_seqs_short_reward = torch.argmax(seq_masked, dim=-1).to(self.device)
                pred_seqs_short_reward = torch.argmax(pred_logit_short_reward, dim=-1)
                rewards_large_t, _ = self.compute_rewards(
                    pred_seqs_short_reward, gt_seqs_short_reward, pdb_name, self.data_config['pdb_data_dir'],
                    out_dir='sold_train_mix_cut_tmp', n_jobs=rl_config['n_jobs']
                )
                rewards[mask_short_reward] = rewards_large_t
        else:
            raise ValueError("reward_pattern must be 'predict' or 'sampling'")

        samples_collate["rewards"] = rewards.to(self.device)

        # 计算优势
        samples_collate["advantages"] = (samples_collate["rewards"] - samples_collate["rewards"].mean(dim=0)) / (samples_collate["rewards"].std(dim=0) + 1e-8)

        # 重塑批次
        samples_batched = {k: v for k, v in samples_collate.items()}
        samples_batched = [samples_batched]  # 单步批次

        # 训练
        self.diffusion_model.train()
        step_log_dict = collections.defaultdict(list)
        step_log_dict["rewards"].append(rewards.mean().item())

        for sample in samples_batched:
            _, sample_logprob = self.diffusion_model.step_with_logprob(
                data, prev_latent_t=sample["latents"], t=sample["timesteps"], latent_t=sample["next_latents"]
            )
            # print("sample_logprob: ", sample_logprob.size(), sample_logprob)

            # 预训练模型的 logprob
            with torch.no_grad():
                _, pretrain_logprob = self.pretrain_diffusion_model.step_with_logprob(
                    data, prev_latent_t=sample["latents"], t=sample["timesteps"],
                    latent_t=sample["next_latents"]
                )

            ratio = torch.exp(sample_logprob - sample["log_probs"].detach())
            advantages = sample["advantages"]
            unclipped_loss = -advantages * ratio
            clipped_loss = -advantages * torch.clamp(
                ratio,
                1.0 - rl_config["clip_range"],
                1.0 + rl_config["clip_range"],
            )
            policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
            log_ratio = sample_logprob - sample["log_probs"].detach()
            kl_loss = 0.5 * torch.mean(log_ratio ** 2)
            step_log_dict["kl_loss"].append(kl_loss.item())
            kl_weight = rl_config.get("kl_weight", 1.0)

            ##和 pretrain model之间的kl散度
            pretrain_log_ratio = sample_logprob - pretrain_logprob.detach()
            pretrain_kl_loss = 0.5 * torch.mean(pretrain_log_ratio ** 2)
            step_log_dict["pretrain_kl_loss"].append(pretrain_kl_loss.item())
            pretrain_kl_weight = rl_config.get("pretrain_kl_weight", 0.0)

            # Compute total loss with KL regularization
            total_loss = policy_loss + kl_weight * kl_loss + pretrain_kl_weight * pretrain_kl_loss
            # total_loss = policy_loss

            scaled_loss = total_loss / self.train_config["gradient_accumulate_every"]
            scaled_loss.backward()

            # Log metrics
            step_log_dict["policy_loss"].append(policy_loss.item())
            step_log_dict["total_loss"].append(total_loss.item())
        return step_log_dict

    def train(self):
        """主训练循环，管理梯度累积"""
        print("---------- Start SOLD Finetuning ----------")
        best_valid_reward = 0

        for epoch in range(self.train_config["max_train_epochs"]):
            print(f"---------- Epoch {epoch} ----------")
            update_ref_model = False

            ##测试验证集
            metrics_info = self.validate(self.valid_dataloader, out_dir='sold_valid_tmp')
            for k, v in metrics_info.items():
                print(f"val epoch {k}: {np.round(v, 4)}", end=" ")
                self.writer.add_scalar(f"Validation/{k}", v, epoch)
            print("\n")

            if metrics_info["rewards"] > best_valid_reward:
                best_valid_reward = metrics_info["rewards"]
                update_ref_model = True
                data = {"model": self.diffusion_model.state_dict(), "opt": self.optimizer.state_dict()}
                save_path = os.path.join(self.result_folder, f"model_epoch_{epoch}.pt")
                torch.save(data, save_path)
                print("ckpt saving to: ", save_path)

            ##测试测试集
            metrics_info = self.validate(self.test_dataloader, out_dir='sold_test_tmp')
            for k, v in metrics_info.items():
                print(f"test epoch {k}: {np.round(v, 4)}", end=" ")
                self.writer.add_scalar(f"Test/{k}", v, epoch)
            print("\n")

            # Training
            log_dict = collections.defaultdict(list)
            epoch_iterator = tqdm(self.train_dataloader, desc=f"Training epoch {epoch}")
            if update_ref_model:
                print("update ref diffusion model...")
                self.ref_diffusion_model = copy.deepcopy(self.diffusion_model)
                self.ref_diffusion_model.eval()

            for _, data in enumerate(epoch_iterator):
                step_log_dict = self.train_step(data)
                self.accum_step += 1
                if self.accum_step % self.train_config["gradient_accumulate_every"] == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.diffusion_model.parameters(), self.train_config["rl_config"].get("max_grad_norm", 1.0)
                    )
                    #step_log_dict["grad_norm"].append(grad_norm.item())
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_gradient_step += 1

                epoch_iterator.set_postfix(
                    policy_loss=np.mean(step_log_dict["policy_loss"]),
                    kl_loss=np.mean(step_log_dict["kl_loss"]),
                    pretrain_kl_loss=np.mean(step_log_dict["pretrain_kl_loss"]),
                    reward=np.mean(step_log_dict["rewards"])
                )

                self.writer.add_scalar("Train-Step/policy_loss", np.mean(step_log_dict["policy_loss"]),
                                       self.global_gradient_step)
                self.writer.add_scalar("Train-Step/kl_loss", np.mean(step_log_dict["kl_loss"]),
                                       self.global_gradient_step)
                self.writer.add_scalar("Train-Step/pretrain_kl_loss", np.mean(step_log_dict["pretrain_kl_loss"]), self.global_gradient_step)
                self.writer.add_scalar("Train-Step/total_loss", np.mean(step_log_dict["total_loss"]),
                                       self.global_gradient_step)
                self.writer.add_scalar("Train-Step/rewards", np.mean(step_log_dict["rewards"]),
                                       self.global_gradient_step)

                for k, v in step_log_dict.items():
                    #print(k, len(v))
                    log_dict[k].append(np.mean(v))

            for k, v in log_dict.items():
                print(f"epoch_{k}: {np.mean(v)}", end=" ")
                self.writer.add_scalar(f'Train-Epoch/{k}', np.mean(v), epoch)

        self.writer.close()
        print("finetune sold complete ......")