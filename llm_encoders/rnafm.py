import torch
from utils import nucleotides
from multimolecule import RnaTokenizer, RnaFmModel

class RnaFMEmbeddingExtractor:
    def __init__(self, model_path, config):
        self.tokenizer = RnaTokenizer.from_pretrained(model_path)
        self.model = RnaFmModel.from_pretrained(model_path)
        self.config = config
        self.device = config["device"]
        self.model.to(self.device)
        self.idx2res = {}
        for ix in range(len(nucleotides)):
            self.idx2res[ix] = nucleotides[ix]

    def __call__(self, label_seqs, device=None):
        return self.extract(label_seqs)

    def extract(self, label_seqs):
        batch_seq = []
        label_seqs = label_seqs.cpu().numpy()
        res_seq = [self.idx2res[ix] for ix in label_seqs]
        res_seq = ''.join(res_seq)
        batch_seq.append(res_seq)
        max_length = max(len(seq) for seq in batch_seq) + 2
        inputs = self.tokenizer(batch_seq, return_tensors="pt", padding="max_length", max_length=max_length)
        with torch.no_grad():
            output = self.model(
                inputs['input_ids'].to(self.device),
                attention_mask=inputs['attention_mask'].to(self.device)
            )
            embeddings = output['last_hidden_state'][:, 1:-1, :]
        return embeddings.squeeze(0)