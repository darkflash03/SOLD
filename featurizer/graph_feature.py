import torch
import numpy as np
import torch.nn.functional as F
import torch_geometric
import torch_cluster

def lengths(atom_i, atom_j, distance_eps=0.001):
    dX = atom_j - atom_i
    L = torch.sqrt((dX ** 2).sum(dim=-1) + distance_eps)
    return L

def normed_vec(V, distance_eps=0.001):
    mag_sq = (V ** 2).sum(dim=-1, keepdim=True)
    mag = torch.sqrt(mag_sq + distance_eps)
    U = V / mag
    return U

def angles(atom_i, atom_j, atom_k, distance_eps=0.001, degrees=False):
    U_ji = normed_vec(atom_i - atom_j, distance_eps=distance_eps)
    U_jk = normed_vec(atom_k - atom_j, distance_eps=distance_eps)
    inner_prod = torch.einsum("ix,ix->i", U_ji, U_jk)
    inner_prod = torch.clamp(inner_prod, -1, 1)
    A = torch.acos(inner_prod)
    if degrees:
        A = A * 180.0 / np.pi
    return A

def normed_cross(V1, V2, distance_eps=0.001):
    C = normed_vec(torch.cross(V1, V2, dim=-1), distance_eps=distance_eps)
    return C

def dihedrals(atom_i, atom_j, atom_k, atom_l, distance_eps=0.001, degrees=False):
    U_ij = normed_vec(atom_j - atom_i, distance_eps=distance_eps)
    U_jk = normed_vec(atom_k - atom_j, distance_eps=distance_eps)
    U_kl = normed_vec(atom_l - atom_k, distance_eps=distance_eps)
    normal_ijk = normed_cross(U_ij, U_jk, distance_eps=distance_eps)
    normal_jkl = normed_cross(U_jk, U_kl, distance_eps=distance_eps)
    _inner_product = lambda a, b: (a * b).sum(-1)
    cos_dihedrals = _inner_product(normal_ijk, normal_jkl)
    angle_sign = _inner_product(U_ij, normal_jkl)
    cos_dihedrals = torch.clamp(cos_dihedrals, -1, 1)
    D = torch.sign(angle_sign) * torch.acos(cos_dihedrals)
    if degrees:
        D = D * 180.0 / np.pi
    return D

def internal_coords(coords, C, distance_eps=0.001, return_masks=False):
    mask = (C > 0).float()
    X_chain = coords[:, :2, :]
    num_residues, _, _ = X_chain.shape
    X_chain = X_chain.reshape(2 * num_residues, 3)

    _lengths = lambda Xi, Xj: lengths(Xi, Xj, distance_eps=distance_eps)
    _angles = lambda Xi, Xj, Xk: np.pi - angles(
        Xi, Xj, Xk, distance_eps=distance_eps
    )
    _dihedrals = lambda Xi, Xj, Xk, Xl: dihedrals(
        Xi, Xj, Xk, Xl, distance_eps=distance_eps
    )

    PC4p_L = _lengths(X_chain[1:, :], X_chain[:-1, :])
    PC4p_A = _angles(X_chain[:-2, :], X_chain[1:-1, :], X_chain[2:, :])
    PC4p_D = _dihedrals(
        X_chain[:-3, :],
        X_chain[1:-2, :],
        X_chain[2:-1, :],
        X_chain[3:, :],
    )

    X_P, X_C4p, X_N = coords.unbind(dim=1)
    X_P_next = coords[1:, 0, :]
    N_L = _lengths(X_C4p, X_N)
    N_A = _angles(X_P, X_C4p, X_N)
    N_D = _dihedrals(X_P_next, X_N[:-1, :], X_C4p[:-1, :], X_P[:-1, :])

    if C is None:
        C = torch.zeros_like(mask)

    C = C * (mask.type(torch.long))
    ii = torch.stack(2 * [C], dim=-1).view([-1])
    L0, L1 = ii[:-1], ii[1:]
    A0, A1, A2 = ii[:-2], ii[1:-1], ii[2:]
    D0, D1, D2, D3 = ii[:-3], ii[1:-2], ii[2:-1], ii[3:]

    # Mask for linear backbone
    mask_L = torch.eq(L0, L1)
    mask_A = torch.eq(A0, A1) * torch.eq(A0, A2)
    mask_D = torch.eq(D0, D1) * torch.eq(D0, D2) * torch.eq(D0, D3)
    mask_L = mask_L.type(torch.float32)
    mask_A = mask_A.type(torch.float32)
    mask_D = mask_D.type(torch.float32)

    # Masks for branched nitrogen
    mask_N_D = torch.eq(C[:-1], C[1:])
    mask_N_D = mask_N_D.type(torch.float32)
    mask_N_A = mask
    mask_N_L = mask

    def _pad_pack(D, A, L, N_D, N_A, N_L):
        # Pad and pack together the components
        D = F.pad(D, (1, 2))
        A = F.pad(A, (1, 1))
        L = F.pad(L, (0, 1))
        N_D = F.pad(N_D, (0, 1))
        D, A, L = [x.reshape(num_residues, 2) for x in [D, A, L]]
        _pack = lambda a, b: torch.cat([a, b.unsqueeze(-1)], dim=-1)
        L = _pack(L, N_L)
        A = _pack(A, N_A)
        D = _pack(D, N_D)
        return D, A, L

    D, A, L = _pad_pack(PC4p_D, PC4p_A, PC4p_L, N_D, N_A, N_L)
    mask_D, mask_A, mask_L = _pad_pack(
        mask_D, mask_A, mask_L, mask_N_D, mask_N_A, mask_N_L
    )
    mask_expand = mask.unsqueeze(-1)
    mask_D = mask_expand * mask_D
    mask_A = mask_expand * mask_A
    mask_L = mask_expand * mask_L

    D = mask_D * D
    A = mask_A * A
    L = mask_L * L

    if not return_masks:
        return D, A, L
    else:
        return D, A, L, mask_D, mask_A, mask_L

def internal_vecs(X):
    p, c4p, n = X[:, 0], X[:, 1], X[:, 2]
    n, p = n - c4p, p - c4p
    forward = F.pad(c4p[1:] - c4p[:-1], [0, 0, 0, 1])
    backward = F.pad(c4p[:-1] - c4p[1:], [0, 0, 1, 0])
    return torch.cat([
        normed_vec(p).unsqueeze_(-2),
        normed_vec(n).unsqueeze_(-2),
        normed_vec(forward).unsqueeze_(-2),
        normed_vec(backward).unsqueeze_(-2),
    ], dim=-2)

def rbf(D, D_min=0., D_max=20., D_count=16):
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def get_posenc(edge_index, num_posenc=16):
    num_posenc = num_posenc
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(0, num_posenc, 2, dtype=torch.float32, device=d.device)
        * -(np.log(10000.0) / num_posenc)
    )

    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E

def normalize(tensor, dim=-1):
    return torch.nan_to_num(torch.div(tensor, torch.linalg.norm(tensor, dim=dim, keepdim=True)))

def construct_graph_data_single(coords, seq=None, mask=None, num_posenc=16, num_rbf=16, knn_num=10):
    seq = torch.as_tensor(seq, dtype=torch.long)
    coords = torch.as_tensor(coords, dtype=torch.float32)
    if mask is None:
        mask = coords.sum(dim=(2, 3)) == 0.
    mask = torch.tensor(mask)
    dihedrals, angles, lengths = internal_coords(coords, mask)
    angle_stack = torch.cat([dihedrals, angles], dim=-1)
    lengths = torch.log(lengths + 0.001)
    internal_coords_feat = torch.cat([torch.cos(angle_stack), torch.sin(angle_stack), lengths], dim=-1)
    internal_vecs_feat = internal_vecs(coords)

    coord_C = coords[:, 1].clone()
    edge_index = torch_cluster.knn_graph(coord_C, k=knn_num)
    edge_index = torch_geometric.utils.coalesce(edge_index)
    edge_vectors = coord_C[edge_index[0]] - coord_C[edge_index[1]]
    edge_rbf = rbf(edge_vectors.norm(dim=-1), D_count=num_rbf)
    edge_posenc = get_posenc(edge_index, num_posenc)

    node_s = (seq.unsqueeze(-1) == torch.arange(4).unsqueeze(0)).float()
    node_v = internal_vecs_feat

    edge_s = torch.cat([edge_rbf, edge_posenc], dim=-1)
    edge_v = normalize(edge_vectors).unsqueeze(-2)

    node_s, node_v, edge_s, edge_v = map(
        torch.nan_to_num,
        (node_s, node_v, edge_s, edge_v)
    )

    return {
        "seq": node_s,
        "extra_x": internal_coords_feat,
        "pos": coords[:, 1],
        "edge_index": edge_index,
        "edge_attr": edge_s,
        "node_v": node_v,
        "edge_v": edge_v,
        "coords": coords
    }