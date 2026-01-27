import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import collections
import abc
from dataset import GraphDataset, TrainSampler
from utils.common import seeding
from utils.constants import *
from utils.metrics import *
from models import *

class BaseRLTrainer(abc.ABC):
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        self.train_config = config["train_config"]
        self.data_config = config["data_config"]
        self.batch_size = self.train_config["batch_size"]
        self.num_steps = self.train_config["latent_transition_config"]["num_steps"]
        self.gradient_accumulate_every = self.train_config["gradient_accumulate_every"]
        seeding(config["seed"])

        self.rl_trainer_name = self.train_config['rl_config']['name']
        self.recovery_weight = self.train_config['rl_config']['recovery_weight']
        self.ss_weight = self.train_config['rl_config']['ss_weight']
        self.mfe_weight = self.train_config['rl_config']['mfe_weight']
        self.lddt_weight = self.train_config['rl_config']['lddt_weight']

        self.accum_step = 0
        self.global_gradient_step = 0

        self.setup_result_folder()
        self.setup_logging()
        self.setup_data()
        self.setup_models()
        self.setup_optimizer()

    def setup_result_folder(self):
        self.result_folder = os.path.join(self.train_config['output_dir'],
                                          f'{self.rl_trainer_name}_recovery_{self.recovery_weight}_ss_{self.ss_weight}_mfe_{self.mfe_weight}_lddt_{self.lddt_weight}')
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)

    def multi_reward(self):
        object_cnt = 0
        if self.recovery_weight > 0:
            object_cnt += 1
        if self.ss_weight > 0:
            object_cnt += 1
        if self.mfe_weight > 0:
            object_cnt += 1
        if self.lddt_weight > 0:
            object_cnt += 1
        return object_cnt >= 2

    def compute_rewards(self, pred_logits, gt_logits, pdb_name, pdb_data_dir, out_dir, n_jobs=1):
        rewards_info = collections.defaultdict(list)

        if pred_logits.dim() == 1:
            pred_logits = pred_logits.unsqueeze(0)
            gt_logits = gt_logits.unsqueeze(0)

        pred_seq_strs = []
        gt_seq_strs = []
        for gt_logit, pred_logit in zip(gt_logits, pred_logits):
            pred_seq_str = "".join([nucleotides[i] for i in pred_logit])
            gt_seq_str = "".join([nucleotides[i] for i in gt_logit])
            pred_seq_strs.append(pred_seq_str)
            gt_seq_strs.append(gt_seq_str)
            recovery = compute_sequence_similarity(pred_seq_str, gt_seq_str)
            rewards_info["recovery"].append(recovery)
            ss, mfe = secondary_structure_metric(gt_seq_str, pred_seq_str)
            rewards_info["ss"].append(ss / 100)
            if self.multi_reward():
                reward_mfe = np.exp(1 / (mfe - 1 / 4))
            else:
                reward_mfe = -mfe
            rewards_info["mfe"].append(reward_mfe)
            rewards_info["unscale_mfe"].append(-mfe)

        rewards = self.recovery_weight * torch.tensor(rewards_info["recovery"]) + self.ss_weight * torch.tensor(rewards_info["ss"]) + self.mfe_weight * torch.tensor(rewards_info["mfe"])

        if self.lddt_weight > 0:
            rmsd_list, lddt_list = three_dimension_metric(pred_seq_strs, pdb_name, pdb_data_dir, out_dir, n_jobs=n_jobs)
            rewards_info["rmsd"].extend(rmsd_list)
            rewards_info["lddt"].extend(lddt_list)
            rewards += self.lddt_weight * torch.tensor(rewards_info["lddt"])
        #print("rewards_info: ", rewards_info)
        return rewards, rewards_info

    def setup_logging(self):
        """初始化 TensorBoard"""
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(self.result_folder, "runs", current_time)
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs saved to: {log_dir}")

    def setup_data(self):
        """初始化数据集和 DataLoader"""
        llm_config = self.config["encoder_decoder_config"]["llm_config"]
        llm_config["device"] = self.device
        self.data_config["llm_config"] = llm_config

        train_dataset = GraphDataset(
            data_path=self.data_config["train_data_path"],
            data_config=self.data_config
        )
        valid_dataset = GraphDataset(
            data_path=self.data_config["valid_data_path"],
            data_config=self.data_config
        )
        test_dataset = GraphDataset(
            data_path=self.data_config["test_data_path"],
            data_config=self.data_config
        )

        train_sampler = TrainSampler(
            train_dataset,
            batch_size=self.batch_size,
            sample_mode=self.train_config["sample_mode"],
            max_squared_res=self.train_config["max_squared_res"]
        ) if self.train_config.get("sampler") else None

        self.train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.train_config["num_workers"],
            collate_fn=train_dataset.collate_fn
        )
        self.valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.train_config["num_workers"],
            collate_fn=valid_dataset.collate_fn
        )
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.train_config["num_workers"],
            collate_fn=test_dataset.collate_fn
        )

    def setup_encoder_decoder(self):
        """Setup and initialize encoder-decoder model."""
        encoder_decoder_config = self.config["encoder_decoder_config"]

        if encoder_decoder_config["decoder"]["name"] == "mlp":
            decoder = MLPDecoder(**encoder_decoder_config["decoder"]["config"])
            encoder_decoder = EncoderMlpDecoder(decoder=decoder)
        else:
            raise NotImplementedError(f"Unsupported decoder: {encoder_decoder_config['decoder']['name']}")

        # Load checkpoint
        ckpt_path = encoder_decoder_config["ckpt_path"]
        print(f"Loading encoder-decoder from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "encoder_decoder" in state_dict:
            state_dict = state_dict["encoder_decoder"]

        return_keys = encoder_decoder.load_state_dict(state_dict, strict=False)
        print(f"Loaded encoder-decoder keys: {return_keys}")

        return encoder_decoder.to(self.device).eval()

    #load ckpt from diffusion_model
    def setup_diffusion_model(self):
        """Setup and initialize diffusion model."""
        model = GVPNet(self.config["model_config"])
        diffusion_model = DiffusionModel(model, self.train_config["latent_transition_config"])

        if self.config["train_config"].get("ckpt_path"):
            ckpt_path = self.config["train_config"]["ckpt_path"]
            ckpt = torch.load(ckpt_path)
            print("load from diffusion model ckpt ...", ckpt_path)
            diffusion_model.load_state_dict(ckpt["model"])

        return diffusion_model.to(self.device)

    def setup_models(self):
        """初始化模型"""
        self.encoder_decoder = self.setup_encoder_decoder()
        self.diffusion_model = self.setup_diffusion_model()
        self.ref_diffusion_model = self.setup_diffusion_model()
        self.ref_diffusion_model.eval()

    def setup_optimizer(self):
        """初始化优化器"""
        self.optimizer = AdamW(
            self.diffusion_model.parameters(),
            lr=self.train_config["lr"],
            betas=(0.9, 0.99),
            weight_decay=self.train_config["wd"]
        )

    def validate(self, dataloader, out_dir):
        self.diffusion_model.eval()
        with torch.no_grad():
            rewards_collate = collections.defaultdict(list)
            rewards_all = 0
            ind_all = torch.tensor([])
            for data in tqdm(dataloader, desc="Validating"):
                pdb_name = data["name"][0]
                data = data.to(self.device)
                sample_latent_traj, _ = self.diffusion_model.sample_with_logprob(data)
                sample_latent = sample_latent_traj[0].squeeze(0)
                data["sample_latent"] = sample_latent.float()
                sample_logit = self.encoder_decoder(data, "sample_latent")
                gt_logit = torch.argmax(data["seq"], dim=-1)
                pred_seqs = torch.argmax(sample_logit, dim=-1)
                gt_seqs = torch.argmax(data["seq"].view(sample_logit.shape), dim=-1).to(self.device)

                _, ind = seq_recovery(gt_logit, sample_logit)
                ind_all = torch.cat([ind_all, ind])

                rewards, rewards_info = self.compute_rewards(pred_seqs, gt_seqs, pdb_name, self.data_config["pdb_data_dir"], out_dir, n_jobs=1)

                rewards_all += rewards.item()
                for k, v in rewards_info.items():
                    rewards_collate[k].append(v)

            aa_recovery = (ind_all.sum() / ind_all.shape[0]).item()
            metrics_info = {}
            rewards = rewards_all / len(dataloader)
            for k, v in rewards_collate.items():
                metrics_info[k] = np.mean(v)
            metrics_info["aa_recovery"] = aa_recovery
            metrics_info["rewards"] = rewards
        return metrics_info

    @abc.abstractmethod
    def sample_timesteps(self, batch_size):
        """采样时间步（子类实现）"""
        pass

    @abc.abstractmethod
    def train_step(self, data):
        """单步训练（子类实现）"""
        pass

    @abc.abstractmethod
    def train(self):
        """完整训练（子类实现）"""
        pass
