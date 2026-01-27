import warnings
warnings.filterwarnings("ignore")

from utils import *
from dataset import *
from torch.utils.data import DataLoader
from models import *
import torch.nn as nn

def test_encoder_decoder(config):
    print("Testing Encoder-Decoder model...")
    data_config = config['data_config']
    train_config = config['train_config']
    model_config = config['model_config']
    seed = config['seed']
    seeding(seed)

    llm_config = model_config['llm_config']
    llm_config['device'] = config['device']
    data_config['llm_config'] = llm_config

    test_dataset = GraphDataset(
        data_path=data_config['test_data_path'],
        data_config=data_config
    )

    loss_fn = nn.CrossEntropyLoss()

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=train_config['batch_size'],
                                 shuffle=False,
                                 collate_fn=test_dataset.collate_fn)

    if model_config['decoder']['name'] == 'mlp':
        decoder = MLPDecoder(**model_config['decoder']['config'])
        if model_config.get("encoder"):
            if model_config["encoder"]["name"] == "mlp":
                encoder = MLPEncoder(**model_config["encoder"]["config"])
            else:
                raise NotImplementedError(model_config["encoder"]["name"])
            encoder_decoder = EncoderMlpDecoder(decoder=decoder, encoder=encoder).to(config['device'])
        else:
            encoder_decoder = EncoderMlpDecoder(decoder=decoder).to(config['device'])
    else:
        raise NotImplementedError(model_config['decoder']['name'])

    state_dict = torch.load(train_config['ckpt_path'], map_location='cpu')['encoder_decoder']

    encoder_decoder.load_state_dict(state_dict, strict=True)
    encoder_decoder.eval()

    test_loss, test_recovery_seq, test_recovery_aa = validate_encoder_decoder(encoder_decoder,
                                                                           config['device'],
                                                                           test_dataloader,
                                                                           loss_fn,
                                                                           'testing',
                                                                           0)
    print("test loss: {}, seq recovery {}, aa recovery {}".format(test_loss, test_recovery_seq, test_recovery_aa))
    print("testing complete ......")