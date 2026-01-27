import torch.nn as nn

class EncoderMlpDecoder(nn.Module):

    def __init__(self, decoder, encoder=None):
        super().__init__()
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, graph_batch, key='x'):
        graph_embedding = self.get_embeddings(graph_batch=graph_batch, key=key)

        if self.encoder is not None:
            graph_embedding = self.encoder(graph_embedding)

        logits = self.decoder(graph_embedding)

        return logits

    def get_embeddings(self, graph_batch, key='x'):
        graph_embedding = graph_batch[key]
        return graph_embedding