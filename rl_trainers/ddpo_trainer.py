from .base_rl_trainer import BaseRLTrainer
import numpy as np
import torch
import collections
from tqdm import tqdm
import copy
import os

class DDPOTrainer(BaseRLTrainer):
    def sample_timesteps(self, batch_size):
        """DDPO 使用所有时间步"""
        return torch.from_numpy(
            np.arange(self.num_steps + 1)[::-1].copy()
        ).repeat(batch_size, 1).to(self.device).unsqueeze(-1)

    def train_step(self, data):
        rl_config = self.train_config["rl_config"]
        batch_size = data.batch[-1] + 1
        data = data.to(self.device)
        samples_collate = {}
        pdb_name = data['name'][0]

        # Sample trajectories
        sample_traj, sample_logprob_traj = self.ref_diffusion_model.sample_with_logprob(data)
        samples_collate["latents"] = torch.stack([sample_traj[t] for t in range(self.num_steps + 1)][::-1],
                                                 dim=1)  # (batch_size, num_steps+1, seq_len, latent_dim)
        samples_collate["next_latents"] = samples_collate["latents"][:,
                                          1:]  # (batch_size, num_steps, seq_len, latent_dim)
        samples_collate["log_probs"] = torch.stack([sample_logprob_traj[t] for t in range(self.num_steps)][::-1],
                                                   dim=1).unsqueeze(-1)  # (batch_size, num_steps+1, 1)
        samples_collate["timesteps"] = torch.from_numpy(np.arange(self.num_steps + 1)[::-1].copy()).repeat(batch_size,
                                                                                                      1).to(
            self.device).unsqueeze(-1)  # （batch_size, num_steps+1, 1）


        data["pred_latent"] = sample_traj[0].float()
        pred_logit = self.encoder_decoder(data, "pred_latent")
        gt_seqs = torch.argmax(data["seq"].view(pred_logit.shape), dim=-1).to(self.device)
        pred_seqs = torch.argmax(pred_logit, dim=-1)
        rewards, _ = self.compute_rewards(pred_seqs, gt_seqs, pdb_name, self.data_config['pdb_data_dir'], out_dir='ddpo_train_tmp_multi4', n_jobs=rl_config['n_jobs'])

        samples_collate["rewards"] = rewards.to(self.device)
        # shuffle samples along batch dimension
        perm = torch.randperm(batch_size, device=self.device)
        samples_collate = {k: v[perm] for k, v in samples_collate.items()}
        # shuffle along time dimension independently for each sample
        perms = torch.stack(
            [
                torch.randperm(self.num_steps, device=self.device)
                for _ in range(batch_size)
            ]
        )

        for key in ["timesteps", "latents", "next_latents", "log_probs"]:
            samples_collate[key] = samples_collate[key][
                torch.arange(batch_size, device=self.device)[:, None],
                perms,
            ]

        # Compute advantages
        samples_collate["advantages"] = (samples_collate["rewards"] - samples_collate["rewards"].mean(dim=0)) / (samples_collate["rewards"].std(dim=0) + 1e-8)

        # rebatch for training
        samples_batched = {
            k: v.reshape(-1, batch_size, *v.shape[1:])
            for k, v in samples_collate.items()
        }
        # dict of lists -> list of dicts for easier iteration
        samples_batched = [
            dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
        ]

        step_log_dict = collections.defaultdict(list)
        self.diffusion_model.train()
        for i, sample in enumerate(samples_batched):
            for j in range(self.num_steps):
                _, sample_logprob = self.diffusion_model.step_with_logprob(data, prev_latent_t=sample["latents"][:, j],
                                                                      t=sample["timesteps"][:, j],
                                                                      latent_t=sample["next_latents"][:, j])
                ratio = torch.exp(sample_logprob - samples_collate["log_probs"][:, j].squeeze(-1).detach())

                # Compute policy loss with clipping
                # advantages = torch.clamp(
                #     sample["advantages"],
                #     -config["train_config"]["rl_config"]["adv_clip_max"],
                #     config["train_config"]["rl_config"]["adv_clip_max"],
                # )
                advantages = sample["advantages"]
                unclipped_loss = -advantages * ratio
                clipped_loss = -advantages * torch.clamp(
                    ratio,
                    1.0 - rl_config["clip_range"],
                    1.0 + rl_config["clip_range"],
                )
                policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                # Compute KL divergence with gradient tracking
                log_ratio = sample_logprob - sample["log_probs"][:, j].squeeze(-1).detach()  # Detach old logprob
                # kl_loss = torch.mean(torch.exp(sample_logprob) * log_ratio)
                kl_loss = 0.5 * torch.mean(log_ratio ** 2)  # http://joschu.net/blog/kl-approx.html
                step_log_dict["kl_loss"].append(kl_loss.item())

                # Add KL penalty with adaptive weight
                kl_weight = rl_config.get("kl_weight", 1.0)
                # Compute total loss with KL regularization
                total_loss = policy_loss + kl_weight * kl_loss
                # total_loss = policy_loss

                scaled_loss = total_loss / self.train_config["gradient_accumulate_every"]
                scaled_loss.backward()

                step_log_dict["policy_loss"].append(policy_loss.item())
                step_log_dict["total_loss"].append(total_loss.item())
                step_log_dict["rewards"].append(rewards.mean().item())

                self.accum_step += 1

                if self.accum_step % self.train_config["gradient_accumulate_every"] == 0:
                    # Clip gradients by global norm
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.diffusion_model.parameters(),
                                                               rl_config.get("max_grad_norm", 1.0))
                    #step_log_dict["grad_norm"].append(grad_norm.item())
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_gradient_step += 1

                    if rl_config['log_gradient_step'] and self.global_gradient_step % rl_config['log_gradient_step'] == 0:
                        gradient_step_metrics_info = self.validate(self.valid_dataloader,
                                                                   out_dir='ddpo_valid_gradient_step')
                        for k, v in gradient_step_metrics_info.items():
                            print(f"train_gradient_step {self.global_gradient_step}, metric {k}: {np.round(v, 4)}",
                                  end=" ")
                            self.writer.add_scalar(f"train_gradient_step/{k}", v, self.global_gradient_step)
                        print("\n")

        return step_log_dict

    def train(self):
        """主训练循环，管理梯度累积"""
        print("---------- Start DDPO Finetuning ----------")
        best_valid_reward = 0

        for epoch in range(self.train_config["max_train_epochs"]):
            print(f"---------- Epoch {epoch} ----------")
            update_ref_model = False

            ##测试验证集
            metrics_info = self.validate(self.valid_dataloader, out_dir='ddpo_valid_tmp_multi4')
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
            metrics_info = self.validate(self.test_dataloader, out_dir='ddpo_test_tmp_multi4')
            for k, v in metrics_info.items():
                print(f"test epoch {k}: {np.round(v, 4)}", end=" ")
                self.writer.add_scalar(f"Test/{k}", v, epoch)
            print("\n")

            ##开始finetune diffusion model
            log_dict = collections.defaultdict(list)
            # Training
            epoch_iterator = tqdm(self.train_dataloader, desc=f"Training epoch {epoch}")
            if update_ref_model:
                print("update ref diffusion model...")
                self.ref_diffusion_model = copy.deepcopy(self.diffusion_model)
                self.ref_diffusion_model.eval()

            for _, data in enumerate(epoch_iterator):
                step_log_dict = self.train_step(data)
                epoch_iterator.set_postfix(
                    policy_loss=np.mean(step_log_dict["policy_loss"]),
                    kl_loss=np.mean(step_log_dict["kl_loss"]),
                    reward=np.mean(step_log_dict["rewards"])
                )

                self.writer.add_scalar("Train-Step/policy_loss", np.mean(step_log_dict["policy_loss"]), self.global_gradient_step)
                self.writer.add_scalar("Train-Step/kl_loss", np.mean(step_log_dict["kl_loss"]), self.global_gradient_step)
                self.writer.add_scalar("Train-Step/total_loss", np.mean(step_log_dict["total_loss"]), self.global_gradient_step)
                self.writer.add_scalar("Train-Step/rewards", np.mean(step_log_dict["rewards"]), self.global_gradient_step)

                for k, v in step_log_dict.items():
                    # print(k, len(v), np.mean(v))
                    log_dict[k].append(np.mean(v))

            for k, v in log_dict.items():
                print(f"epoch_{k}: {np.mean(v)}", end=" ")
                self.writer.add_scalar(f'Train-Epoch/{k}', np.mean(v), epoch)

        self.writer.close()
        print("finetune ddpo complete ......")