import warnings
warnings.filterwarnings("ignore")

from utils import *
from dataset import *
from torch.utils.data import DataLoader
from models import *
from dataset import TrainSampler
import time

def sample_latent_diffusion(config):
    print("Sampling Latent Diffusion model...")
    torch.multiprocessing.set_start_method('spawn')

    init_rhofold_model()

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
    sample_file_name = data_config['sample_data_path'].split('/')[-1][:-4]
    result_folder = os.path.join(config['train_config']['output_dir'],
                                 f'{llm_name}_compress_dim_{compress_dim}/SAMPLE_{sample_file_name}')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    save_config(os.path.join(result_folder, 'config.yaml'), config)


    sample_dataset = GraphDataset(data_path=data_config['sample_data_path'],
                                data_config=data_config)

    sampler = TrainSampler(
            sample_dataset,
            batch_size=train_config['sample_num'],
            sample_mode=train_config["sample_mode"],
            max_squared_res=train_config["max_squared_res"],
        )

    sample_dataloader = DataLoader(
        sample_dataset,
        sampler=sampler,
        batch_size=train_config['sample_num'],
        shuffle=False,
        num_workers=train_config["num_workers"],
        collate_fn=sample_dataset.collate_fn,
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

    diffusion_model.load_state_dict(torch.load(train_config["ckpt_path"])["model"])
    print("load from diffusion ckpt ...", config['train_config']['ckpt_path'])
    diffusion_model.eval()

    start_time = time.time()

    if train_config.get("ddim_num_steps"):
        num_steps = train_config["ddim_num_steps"]
        result = latent_diffusion_sample(diffusion_model, encoder_decoder,
                                config['device'], sample_dataloader,
                                data_config['pdb_data_dir'], os.path.join(result_folder, f'sample_predict_pdbs_ddim_{num_steps}_steps'),
                                num_steps=num_steps)
        result.to_csv(os.path.join(result_folder, f'sample_predict_ddim_{num_steps}_steps.csv'), index=False)
    else:
        result = latent_diffusion_sample(diffusion_model, encoder_decoder,
                                config['device'], sample_dataloader,
                                data_config['pdb_data_dir'],
                                os.path.join(result_folder, f'test_predict_pdbs_ddpm'))
        result.to_csv(os.path.join(result_folder, f'sample_predict_ddpm.csv'), index=False)

    end_time = time.time()
    print(f"time taken: {end_time - start_time}")