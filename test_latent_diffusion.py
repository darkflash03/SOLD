import warnings
warnings.filterwarnings("ignore")

from utils import *
from dataset import *
from torch.utils.data import DataLoader
from models import *

def test_latent_diffusion(config):
    print("Testing Latent Diffusion model...")
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
    test_file_name = data_config['test_data_path'].split('/')[-1][:-4]
    result_folder = os.path.join(config['train_config']['output_dir'], f'{llm_name}_compress_dim_{compress_dim}/TEST_{test_file_name}')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    save_config(os.path.join(result_folder, 'config.yaml'), config)

    test_dataset = GraphDataset(data_path=data_config['test_data_path'],
                                 data_config=data_config)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=test_dataset.collate_fn,
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

    if train_config.get("ddim_num_steps"):
        num_steps = train_config["ddim_num_steps"]
        seq_recovery, aa_recovery, ss, mfe, rmsd, lddt = latent_diffusion_test(diffusion_model, encoder_decoder,
                                                                                config['device'], test_dataloader,
                                                                                'validate', 0,
                                                                                      data_config['pdb_data_dir'], os.path.join(result_folder, f'test_predict_pdbs_ddim_{num_steps}_steps'),
                                                                                      num_steps=num_steps)
        print(f"ema ddim sampling with steps {num_steps} ..., "
              f"test seq recovery: {seq_recovery}, "
              f"test aa recovery: {aa_recovery},"
              f"test ss recovery: {ss}, "
              f"test mfe: {mfe}, "
              f"test rmsd: {rmsd}, "
              f"test lddt: {lddt}")

    else:
        seq_recovery, aa_recovery, ss, mfe, rmsd, lddt = latent_diffusion_test(diffusion_model, encoder_decoder,
                                                                               config['device'], test_dataloader,
                                                                               'validate', 0,
                                                                               data_config['pdb_data_dir'],
                                                                               os.path.join(result_folder, 'test_predict_pdbs_ddpm'),
                                                                              )

        print(f"ema ddpm sampling..., "
              f"test seq recovery: {seq_recovery}, "
              f"test aa recovery: {aa_recovery},"
              f"test ss recovery: {ss}, "
              f"test mfe: {mfe}, "
              f"test rmsd: {rmsd}, "
              f"test lddt: {lddt}")


    print("Testing Latent Diffusion Complete ......")