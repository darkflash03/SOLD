import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from utils import *
from dataset.graph_dataset import GraphDataset
from torch.utils.data import DataLoader
from models import *
import torch.nn as nn
from tqdm import tqdm

def train_encoder_decoder(config):
    print("Training Encoder-Decoder model...")
    data_config = config['data_config']
    train_config = config['train_config']
    model_config = config['model_config']
    llm_config = model_config['llm_config']
    llm_config['device'] = config['device']
    data_config['llm_config'] = llm_config

    encoder_config = model_config['encoder']
    decoder_config = model_config['decoder']

    seed = config['seed']
    seeding(seed)

    llm_name = llm_config['name']
    encoder_name = encoder_config['name']
    decoder_name = decoder_config['name']
    compress_dim = decoder_config['config']['encoder_hidden_dim']

    result_folder = os.path.join(config['train_config']['output_dir'], f'{llm_name}_encoder_{encoder_name}_decoder_{decoder_name}_compress_dim_{compress_dim}')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    save_config(os.path.join(result_folder, 'config.yaml'), config)

    # Initialize TensorBoard
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(result_folder, "runs", current_time)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    train_dataset = GraphDataset(data_path=data_config['train_data_path'],
                                 data_config=data_config)

    valid_dataset = GraphDataset(data_path=data_config['valid_data_path'],
                                 data_config=data_config)

    test_dataset = GraphDataset(data_path=data_config['test_data_path'],
                                data_config=data_config)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=train_config['batch_size'],
                                  shuffle=True,
                                  collate_fn=train_dataset.collate_fn)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=train_config['batch_size'],
                                  shuffle=False,
                                  collate_fn=valid_dataset.collate_fn)
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

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=train_config['lr'])
    loss_fn = nn.CrossEntropyLoss()

    print("---------- Start Training Encoder-Decoder ----------")
    gradient_step = 0
    accum_step = 0
    best_valid_loss = np.inf
    patient = 0
    for epoch in range(train_config['max_train_epochs']):
        print(f"---------- Epoch {epoch} ----------")

        encoder_decoder.eval()
        val_loss, val_recovery_seq, val_recovery_aa = validate_encoder_decoder(encoder_decoder, config['device'], valid_dataloader, loss_fn, 'validate', epoch)
        print(f"epoch: {epoch}, val seq recovery: {val_recovery_seq}, val recovery aa: {val_recovery_aa}")
        test_loss, test_recovery_seq, test_recovery_aa = validate_encoder_decoder(encoder_decoder, config['device'], test_dataloader, loss_fn, 'test', epoch)
        print(f"epoch: {epoch}, test seq recovery: {test_recovery_seq}, test recovery aa: {test_recovery_aa}")
        writer.add_scalar(f"Validation/loss", val_loss, epoch)
        writer.add_scalar(f"Validation/recovery_seq", val_recovery_seq, epoch)
        writer.add_scalar(f"Validation/recovery_aa", val_recovery_aa, epoch)
        writer.add_scalar(f"Test/loss", test_loss, epoch)
        writer.add_scalar(f"Test/recovery_seq", test_recovery_seq, epoch)
        writer.add_scalar(f"Test/recovery_aa", test_recovery_aa, epoch)
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            patient = 0
            state_dict = {
                "encoder": encoder_decoder.encoder.state_dict(),
                "decoder": encoder_decoder.decoder.state_dict(),
                "encoder_decoder": encoder_decoder.state_dict(),
            }
            print("now saving model on epoch {} ......".format(epoch))
            torch.save(state_dict, os.path.join(result_folder, 'model_separate_epoch_{}.pt'.format(epoch)))
        else:
            patient += 1

        if patient > train_config['patience']:
            print("Early Stopping .......")
            break

        encoder_decoder.train()
        total_loss = 0
        accum_loss = 0
        epoch_iterator = tqdm(train_dataloader)
        for batch in epoch_iterator:
            logits = encoder_decoder(batch)
            gt = torch.argmax(batch['seq'], dim=-1).to(config['device'])
            loss = loss_fn(logits, gt)
            total_loss += loss.item()
            accum_loss += loss.item()
            accum_step += 1
            loss.backward()
            epoch_iterator.set_postfix(loss=loss.item())

            if accum_step % train_config['gradient_accumulate_every'] == 0:
                writer.add_scalar("Train-Step/loss", accum_loss / train_config['gradient_accumulate_every'], gradient_step)
                gradient_step += 1
                accum_loss = 0
                optimizer.step()
                optimizer.zero_grad()

        train_epoch_loss = total_loss / len(train_dataloader)
        writer.add_scalar(f'Train-Epoch/loss', train_epoch_loss, epoch)

    writer.close()
    print("training complete ......")