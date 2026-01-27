import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dit import (SinusoidalPositionalEmbedding, TransformerEncoderCondLayer)
from models.gvp_pytorch import GVP, DihedralFeatures, GVPConvLayer
from models.layers import LearnedSinusoidalPosEmb
from models.utils import LayerNorm

class GVPNet(nn.Module):

    def __init__(self, config):
        super(GVPNet, self).__init__()
        self.node_in_dim = config["node_in_dim"]
        self.node_h_dim = config["node_h_dim"]
        self.edge_in_dim = config["edge_in_dim"]
        self.edge_h_dim = config["edge_h_dim"]
        self.num_layers = config["num_layers"]
        self.latent_out_dim = config["latent_out_dim"]
        self.dihedral_angle = config["dihedral_angle"]
        drop_rate = config["drop_rate"]
        activations = (F.relu, None)

        # Node input embedding
        self.W_v = torch.nn.Sequential(
            LayerNorm(self.node_in_dim),
            GVP(self.node_in_dim, self.node_h_dim, activations=(None, None), vector_gate=True))

        # Edge input embedding
        self.W_e = torch.nn.Sequential(
            LayerNorm(self.edge_in_dim),
            GVP(self.edge_in_dim, self.edge_h_dim, activations=(None, None), vector_gate=True))

        # Encoder layers (supports multiple conformations)
        self.encoder_layers = nn.ModuleList(
            GVPConvLayer(
                self.node_h_dim, self.edge_h_dim, activations=activations, vector_gate=True, drop_rate=drop_rate)
            for _ in range(self.num_layers))

        # Output
        self.W_out = GVP(self.node_h_dim, (self.node_h_dim[0], 0), activations=(None, None))

        # Transformer Layers
        self.seq_res = nn.Linear(self.node_in_dim[0], self.node_h_dim[0])
        self.mix_lin = nn.Linear(self.node_h_dim[0] * 2, self.node_h_dim[0])
        self.num_trans_layer = config["num_trans_layer"]
        self.embed_positions = SinusoidalPositionalEmbedding(
            self.node_h_dim[0],
            -1,
        )
        self.trans_layers = nn.ModuleList(
            TransformerEncoderCondLayer(config["trans"]) for _ in range(self.num_trans_layer))
        self.MLP_seq_out = nn.Sequential(nn.Linear(self.node_h_dim[0], self.node_h_dim[0]), nn.ReLU(),
                                         nn.Linear(self.node_h_dim[0], self.latent_out_dim))

        learned_sinu_pos_emb_dim = 16
        time_cond_dim = config["node_h_dim"][0] * 2
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
        sinu_pos_emb_input_dim = learned_sinu_pos_emb_dim + 1
        self.to_time_hiddens = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(sinu_pos_emb_input_dim, time_cond_dim),
            nn.SiLU(),
            nn.Linear(time_cond_dim, config["node_h_dim"][0]),
        )

        # Dihedral angle
        if self.dihedral_angle:
            self.embed_dihedral = DihedralFeatures(config["node_h_dim"][0])

    def struct_forward(self, batch, init_seq):
        h_V = (init_seq, batch.node_v)
        h_E = (batch.edge_attr, batch.edge_v)

        edge_index = batch.edge_index

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        if self.dihedral_angle:
            dihedral_feats = self.embed_dihedral(batch.coords).reshape_as(h_V[0])
            h_V = (h_V[0] + dihedral_feats, h_V[1])

        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)

        gvp_output = self.W_out(h_V)
        return gvp_output

    def forward(self, data, t):
        x = data.x
        init_seq = torch.cat([x, data.extra_x], dim=1).float()

        batch_size = t.shape[0]
        length = data.x.shape[0] // batch_size

        gvp_output = self.struct_forward(data, init_seq).reshape(batch_size, length, -1)

        trans_x = torch.cat([gvp_output, self.seq_res(init_seq.reshape(batch_size, length, -1))], dim=-1)
        trans_x = self.mix_lin(trans_x)

        noise_level = t.squeeze(1)
        time_cond = self.to_time_hiddens(noise_level)
        time_cond = time_cond.unsqueeze(1).repeat(1, length, 1)

        # add position embedding
        seq_mask = torch.ones((batch_size, length), device=data.x.device)
        pos_emb = self.embed_positions(seq_mask)

        trans_x = trans_x + pos_emb
        trans_x = trans_x.transpose(0, 1)

        # transformer layers
        for layer in self.trans_layers:
            trans_x = layer(trans_x, None, cond=time_cond.transpose(0, 1))

        pred_latent = self.MLP_seq_out(trans_x.transpose(0, 1)).squeeze(0)

        return pred_latent