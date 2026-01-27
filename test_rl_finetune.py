import warnings
warnings.filterwarnings("ignore")

from utils import *
from dataset import *
from torch.utils.data import DataLoader
from models import *

def test_rl_finetune(config):
    print("Testing RL-Finetune model...")
    torch.multiprocessing.set_start_method('spawn')

    init_rhofold_model()

    data_config = config['data_config']
    train_config = config['train_config']
    test_config = config['test_config']
    seed = config['seed']
    seeding(seed)

    encoder_decoder_config = config['encoder_decoder_config']
    llm_config = encoder_decoder_config['llm_config']
    llm_config['device'] = config['device']
    data_config['llm_config'] = llm_config

    ckpt_path = test_config['ckpt_path']
    ckpt_dir = ''
    for path in ckpt_path.split('/')[:-1]:
        ckpt_dir = os.path.join(ckpt_dir, path)

    test_file_name = data_config['test_data_path'].split('/')[-1][:-4]
    result_folder = os.path.join(ckpt_dir, f'TEST_{test_file_name}')
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

    diffusion_model.load_state_dict(torch.load(test_config["ckpt_path"])["model"])
    print("load from diffusion ckpt ...", test_config['ckpt_path'])
    diffusion_model.eval()

    seq_recovery, aa_recovery, ss, mfe, rmsd, lddt, rewards_dict = rl_finetune_test(diffusion_model, encoder_decoder,
                                                                           config['device'], test_dataloader,
                                                                           'test', data_config['pdb_data_dir'],
                                                                           os.path.join(result_folder, 'predict_pdbs_tmp'), test_config
                                                                           )

    print(f"rl-finetune testing ..., "
          f"test seq recovery: {seq_recovery}, "
          f"test aa recovery: {aa_recovery},"
          f"test ss recovery: {ss}, "
          f"test mfe: {mfe}, "
          f"test rmsd: {rmsd}, "
          f"test lddt: {lddt}")

    for metric in rewards_dict.keys():
        value = np.array(rewards_dict[metric])
        print("metric", metric, "value", value)
        np.save(os.path.join(result_folder, metric + '.npy'), value)

    print("Testing RL-Finetune Complete ......")
