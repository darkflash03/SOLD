from featurizer.utils import parse_pdb
import numpy as np
import torch
from utils import nucleotides, nt_atom_constant

def make_feature(pdb_path):
    struc = parse_pdb(pdb_path)
    N = 0
    res_list = []
    for chain in struc.get_chains():
        for x in chain:
            if x.get_id()[0] != ' ' or x.get_id()[2] == ' ':
                N += 1
                res_list.append(x)

    coords = np.zeros((N, 3, 3), dtype=np.float32)
    coord_mask = np.zeros((N, 3), dtype=bool)

    for i in range(N):
        seq_idx = i
        res = res_list[i]
        res_id = res.resname.replace(" ", "")
        for atom in res.get_atoms():
            if atom.id not in nt_atom_constant:
                continue
            atom14idx = nt_atom_constant.index(atom.id)
            if res_id not in ['U', 'C'] and atom.id == "N9":
                atom14idx -= 1
            if res_id not in ['U', 'C'] and atom.id == "N1":
                continue
            coords[seq_idx, atom14idx] = atom.get_coord()
            coord_mask[seq_idx, atom14idx] = True

    gt_seq = [x.resname.replace(" ", "") for x in res_list]

    ##对于一些不合法的氨基酸，做一个映射到 'A'
    for i in range(len(gt_seq)):
        if gt_seq[i] not in nucleotides:
            gt_seq[i] = 'A'

    gt_seq = "".join(gt_seq)

    feature = dict(
        seq=gt_seq,
        coords=coords,
        coord_mask=coord_mask
    )
    return feature

def get_structure(data_config, pdb_id):
    pdb_data_dir = data_config['pdb_data_dir']
    PDB_feature = make_feature(f"{pdb_data_dir}/{pdb_id}.pdb")
    coords = torch.tensor(PDB_feature['coords'])
    coord_mask = torch.tensor(PDB_feature['coord_mask'])
    str_seq = PDB_feature['seq']
    assert len(str_seq) == coords.shape[0] and len(str_seq) == coord_mask.shape[0] and len(str_seq) > 0
    letter_to_num = dict(zip(
        nucleotides,
        list(range(len(nucleotides)))
    ))
    seq2num = [letter_to_num[residue] for residue in str_seq]
    seq = torch.tensor(seq2num, dtype=torch.int64)
    mask = coord_mask[:, 1]

    ret = dict(
        name=pdb_id,
        str_seq=str_seq,
        seq=seq,
        mask=mask,
        coords=coords,
        coord_mask=coord_mask
    )
    return ret