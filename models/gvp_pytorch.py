import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import _norm_no_nan
from models.utils import tuple_cat, tuple_index, tuple_sum, _merge, _split, LayerNorm, Dropout
from torch_scatter import scatter_add
import functools

from torch_geometric.nn import MessagePassing

class GVP(nn.Module):
    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.relu, torch.sigmoid), vector_gate=True):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate: self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)

        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo:
                v = self.wv(vh)
                v = torch.transpose(v, -1, -2)
                if self.vector_gate:
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(
                        _norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3, device=x.device)
        if self.scalar_act:
            s = self.scalar_act(s)

        return (s, v) if self.vo else s

class GVPConv(MessagePassing):
    def __init__(self, in_dims, out_dims, edge_dims,
                 n_layers=3, module_list=None, aggr="mean",
                 activations=(F.relu, torch.sigmoid), vector_gate=True):
        super(GVPConv, self).__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims

        GVP_ = functools.partial(GVP,
                                 activations=activations,
                                 vector_gate=vector_gate)

        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP_((2 * self.si + self.se, 2 * self.vi + self.ve),
                         (self.so, self.vo), activations=(None, None)))
            else:
                module_list.append(
                    GVP_((2 * self.si + self.se, 2 * self.vi + self.ve), out_dims)
                )
                for i in range(n_layers - 2):
                    module_list.append(GVP_(out_dims, out_dims))
                module_list.append(GVP_(out_dims, out_dims,
                                        activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        x_s, x_v = x
        message = self.propagate(edge_index,
                                 s=x_s, v=x_v.contiguous().view(x_v.shape[0], x_v.shape[1] * 3),
                                 edge_attr=edge_attr)
        return _split(message, self.vo)

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1] // 3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1] // 3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge(*message)

class GVPConvLayer(nn.Module):
    def __init__(
            self,
            node_dims,
            edge_dims,
            n_message=3,
            n_feedforward=2,
            drop_rate=.1,
            autoregressive=False,
            activations=(F.relu, torch.sigmoid),
            vector_gate=True,
            residual=True
    ):
        super(GVPConvLayer, self).__init__()
        self.conv = GVPConv(node_dims, node_dims, edge_dims, n_message,
                            aggr="add" if autoregressive else "mean",
                            activations=activations, vector_gate=vector_gate)
        GVP_ = functools.partial(GVP,
                                 activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims, activations=(None, None)))
        else:
            hid_dims = 4 * node_dims[0], 2 * node_dims[1]
            ff_func.append(GVP_(node_dims, hid_dims))
            for i in range(n_feedforward - 2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)
        self.residual = residual

    def forward(self, x, edge_index, edge_attr,
                autoregressive_x=None, node_mask=None):
        if autoregressive_x is not None:
            src, dst = edge_index
            mask = src < dst
            edge_index_forward = edge_index[:, mask]
            edge_index_backward = edge_index[:, ~mask]
            edge_attr_forward = tuple_index(edge_attr, mask)
            edge_attr_backward = tuple_index(edge_attr, ~mask)

            dh = tuple_sum(
                self.conv(x, edge_index_forward, edge_attr_forward),
                self.conv(autoregressive_x, edge_index_backward, edge_attr_backward)
            )

            count = scatter_add(torch.ones_like(dst), dst,
                                dim_size=dh[0].size(0)).clamp(min=1).unsqueeze(-1)

            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)

        else:
            dh = self.conv(x, edge_index, edge_attr)

        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)

        x = self.norm[0](tuple_sum(x, self.dropout[0](dh))) if self.residual else dh

        dh = self.ff_func(x)
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh))) if self.residual else dh

        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_
        return x


class Normalize(nn.Module):
    def __init__(self, features, epsilon=1e-6):
        super(Normalize, self).__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x, dim=-1):
        # x = torch.nan_to_num(x, nan=0.0, posinf=1e5, neginf=-1e5)
        mu = x.mean(dim, keepdim=True)
        sigma = torch.sqrt(x.var(dim, keepdim=True) + self.epsilon)
        gain = self.gain
        bias = self.bias
        # Reshape
        if dim != -1:
            shape = [1] * len(mu.size())
            shape[dim] = self.gain.size()[0]
            gain = gain.view(shape)
            bias = bias.view(shape)
        return gain * (x - mu) / (sigma + self.epsilon) + bias

class DihedralFeatures(nn.Module):
    def __init__(self, node_embed_dim):
        """ Embed dihedral angle features. """
        super(DihedralFeatures, self).__init__()
        # 3 dihedral angles; sin and cos of each angle
        node_in = 6
        # Normalization and embedding
        self.node_embedding = nn.Linear(node_in,  node_embed_dim, bias=True)
        self.norm_nodes = Normalize(node_embed_dim)
    def forward(self, X):
        """ Featurize coordinates as an attributed graph """
        with torch.no_grad():
            V = self._dihedrals(X)
        V = self.node_embedding(V)
        V = self.norm_nodes(V)
        return V

    @staticmethod
    def _dihedrals(X, eps=1e-7, return_angles=False):
        # First 3 coordinates are [N, CA, C] / [C4', C1', N1/N9]
        if len(X.shape) == 4:
            X = X[..., :3, :].reshape(X.shape[0], 3*X.shape[1], 3)
        else:
            X = X[:, :3, :]

        # Shifted slices of unit vectors
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, (1,2), 'constant', 0)
        D = D.view((D.size(0), int(D.size(1)/3), 3))

        D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
        return D_features