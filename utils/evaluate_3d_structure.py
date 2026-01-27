from Bio.PDB.PDBParser import PDBParser
import numpy as np
from itertools import permutations
from Bio.PDB import MMCIFParser

# It's a very simple letter correspondence table, but it works in most cases here.
PROTEIN_DNA_COMMON_N_TO_ONE = {
    'ALA': 'A',
    'ARG': 'R',
    'ASN': 'N',
    'ASP': 'D',
    'CYS': 'C',
    'GLN': 'Q',
    'GLU': 'E',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LEU': 'L',
    'LYS': 'K',
    'MET': 'M',
    'PHE': 'F',
    'PRO': 'P',
    'SER': 'S',
    'THR': 'T',
    'TRP': 'W',
    'TYR': 'Y',
    'VAL': 'V',
    'DA': 'A',
    'DG': 'G',
    'DC': 'C',
    'DT': 'T',
    'UNK': 'X'
}


def kabsch_rotation(P, Q):
    C = P.transpose(-1, -2) @ Q
    V, _, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        V[:, -1] = -V[:, -1]
    U = V @ W
    return U


def get_optimal_transform(src_atoms, tgt_atoms):
    src_center = src_atoms.mean(-2)[None, :]
    tgt_center = tgt_atoms.mean(-2)[None, :]
    r = kabsch_rotation(src_atoms - src_center, tgt_atoms - tgt_center)
    x = tgt_center - src_center @ r
    return r, x


def get_pdb(filename):
    parser = PDBParser()
    data = parser.get_structure("tmp", filename)
    chains = []

    for chain in data.get_chains():
        residues = {}
        for res in chain.get_residues():
            if res.get_id()[0] != " ":
                continue
            res_id = res.get_id()[1]
            d = {}
            d["resname"] = res.resname
            if len(d["resname"]) > 1 and d["resname"] in PROTEIN_DNA_COMMON_N_TO_ONE:
                d["resname"] = PROTEIN_DNA_COMMON_N_TO_ONE[d["resname"]]
            atoms = {}
            for a in res.get_atoms():
                atoms[a.name] = a.get_coord()
            d["atoms"] = atoms
            residues[res_id] = d
        chains.append(residues)
    return chains


def get_cif(filename):
    parser = MMCIFParser()
    structure = parser.get_structure("tmp", filename)
    chains = []
    for model in structure:
        for chain in model:
            residues = {}
            for res in chain:
                res_id = res.get_id()[1]
                d = {}
                d["resname"] = res.get_resname()
                if len(d["resname"]) > 1 and d["resname"] in PROTEIN_DNA_COMMON_N_TO_ONE:
                    d["resname"] = PROTEIN_DNA_COMMON_N_TO_ONE[d["resname"]]
                atoms = {}
                for atom in res:
                    atoms[atom.get_name()] = atom.get_coord()
                d["atoms"] = atoms
                residues[res_id] = d
            chains.append(residues)
    return chains


def recursive_perm(intervals, cur_idx=0):
    if cur_idx >= len(intervals):
        return [()]
    ret = []
    for cur_perm in permutations(intervals[cur_idx]):
        for right in recursive_perm(intervals, cur_idx + 1):
            ret.append(cur_perm + right)
    return ret


def generate_perm(entity):
    intervals = []
    pre_eid = -1
    for i, eid in enumerate(entity):
        if eid != pre_eid:
            intervals.append([])
        intervals[-1].append(i)
        pre_eid = eid
    return recursive_perm(intervals)


def get_coords(gt, pred):
    gt_coords = []
    pred_coords = []
    for i in range(len(gt)):
        for r in gt[i]:
            if gt[i][r]["resname"] == "UNK":
                continue
            assert r in pred[i]
            if "C4'" in gt[i][r]["atoms"] and "C4'" in pred[i][r]["atoms"]:
                gt_coords.append(gt[i][r]["atoms"]["C4'"])
                pred_coords.append(pred[i][r]["atoms"]["C4'"])
            # if "CA" in gt[i][r]["atoms"] and "CA" in pred[i][r]["atoms"]:
            #     gt_coords.append(gt[i][r]["atoms"]["CA"])
            #     pred_coords.append(pred[i][r]["atoms"]["CA"])
    if gt_coords and pred_coords:
        gt_coords = np.stack(gt_coords)
        pred_coords = np.stack(pred_coords)
        return gt_coords, pred_coords
    else:
        return [], []


def compute_rmsd(true_atom_pos, pred_atom_pos, eps: float = 1e-6):
    sd = np.square(true_atom_pos - pred_atom_pos).sum(axis=-1)
    msd = np.mean(sd)
    return np.sqrt(msd + eps)


def compute_tm(true_atom_pos, pred_atom_pos, eps: float = 1e-6):
    sd = np.square(true_atom_pos - pred_atom_pos).sum(axis=-1)
    num_res = true_atom_pos.shape[0]
    d0 = 1.24 * (max(num_res, 19) - 15) ** (1.0 / 3) - 1.8
    nsd = 1.0 / (1.0 + (sd) / (d0 ** 2.0))
    return nsd.mean()


def compute_gdt(true_atom_pos, pred_atom_pos, eps: float = 1e-6):
    d = np.sqrt(np.square(true_atom_pos - pred_atom_pos).sum(axis=-1))

    def p(d, k):
        return (d <= k).astype(np.float32).sum() / d.size

    p0_5 = p(d, 0.5)
    p1 = p(d, 1)
    p2 = p(d, 2)
    p4 = p(d, 4)
    p8 = p(d, 8)
    return 0.25 * (p1 + p2 + p4 + p8), 0.25 * (p0_5 + p1 + p2 + p4)


def compute_lddt(
        true_atom_pos,
        pred_atom_pos,
        cutoff: float = 15.0,
        eps: float = 1e-10,
):
    n = true_atom_pos.shape[-2]
    dmat_true = np.sqrt(
        eps
        + np.sum(
            (true_atom_pos[..., None, :] - true_atom_pos[..., None, :, :]) ** 2,
            axis=-1,
        )
    )

    dmat_pred = np.sqrt(
        eps
        + np.sum(
            (pred_atom_pos[..., None, :] - pred_atom_pos[..., None, :, :]) ** 2,
            axis=-1,
        )
    )
    dists_to_score = (dmat_true < cutoff).astype(np.float32) * (1.0 - np.eye(n))

    dist_l1 = np.abs(dmat_true - dmat_pred)

    score = (
            (dist_l1 < 0.5).astype(np.float32)
            + (dist_l1 < 1.0).astype(np.float32)
            + (dist_l1 < 2.0).astype(np.float32)
            + (dist_l1 < 4.0).astype(np.float32)
    )
    score = score * 0.25

    norm = 1.0 / (eps + np.sum(dists_to_score, axis=-1))
    score = norm * (eps + np.sum(dists_to_score * score, axis=-1))
    return score.mean()

def align_residues(gt, pred):
    """
    Align residues between two structures (gt and pred) based on residue names.
    Returns filtered gt and pred with aligned residues.
    """
    from Bio import pairwise2

    # Extract residue sequences (residue names) for both structures
    def get_residue_sequence(structure):
        sequence = []
        for chain in structure:
            for res_id in sorted(chain.keys()):  # Sort by residue ID
                res_name = chain[res_id]["resname"]
                sequence.append(res_name)
        return sequence

    gt_sequence = get_residue_sequence(gt)
    pred_sequence = get_residue_sequence(pred)

    # Perform sequence alignment to find the longest common subsequence (LCS)
    alignments = pairwise2.align.globalms(
        ''.join(gt_sequence),
        ''.join(pred_sequence),
        2, -1, -99, -99  # Match score, mismatch penalty, open/extend gap penalties
    )
    best_alignment = alignments[0]

    aligned_gt_seq = best_alignment.seqA
    aligned_pred_seq = best_alignment.seqB
    # Find the indices of aligned residues
    gt_indices = []
    pred_indices = []
    gt_idx = 0
    pred_idx = 0
    for gt_char, pred_char in zip(aligned_gt_seq, aligned_pred_seq):
        if gt_char != '-' and pred_char != '-':
            gt_indices.append(gt_idx)
            pred_indices.append(pred_idx)
        if gt_char != '-':
            gt_idx += 1
        if pred_char != '-':
            pred_idx += 1

    # 确保索引数量一致
    assert len(gt_indices) == len(pred_indices), "Alignment indices mismatch"

    # 使用全局索引过滤残基
    def filter_structure(structure, indices):
        filtered = {}
        counter = 0
        global_idx = 0
        for chain in structure:
            for res_id in sorted(chain.keys()):
                if global_idx in indices:
                    filtered[counter] = chain[res_id]
                    counter += 1
                global_idx += 1
        return filtered

    filtered_gt = filter_structure(gt, gt_indices)
    filtered_pred = filter_structure(pred, pred_indices)

    return filtered_gt, filtered_pred


def extract_c4_coords(filtered_gt, filtered_pred):
    """
    Extract C4' coordinates from filtered_gt and filtered_pred where the residue indices match.
    Returns two (N, 3) numpy arrays for gt and pred.
    """
    gt_coords = []
    pred_coords = []

    # Iterate through the common keys (residue indices)
    for key in filtered_gt:
        if key in filtered_pred:  # Ensure the residue exists in both structures
            gt_res = filtered_gt[key]
            pred_res = filtered_pred[key]
            if "C4'" in gt_res["atoms"] and "C4'" in pred_res["atoms"]:
                gt_coords.append(gt_res["atoms"]["C4'"])
                pred_coords.append(pred_res["atoms"]["C4'"])
            # if "CA" in gt_res["atoms"] and "CA" in pred_res["atoms"]:
            #     gt_coords.append(gt_res["atoms"]["CA"])
            #     pred_coords.append(pred_res["atoms"]["CA"])
    # Convert to numpy arrays
    if gt_coords and pred_coords:
        gt_coords = np.array(gt_coords)
        pred_coords = np.array(pred_coords)
        return gt_coords, pred_coords
    else:
        raise ValueError("Empty Coords")


def compute_monomer(gt_pdb, pred_pdb):
    """
    Compute monomer metrics
    : param gt_pdb: ground truth pdb file
    : param pred_pdb: predicted pdb file
    """
    gt = get_pdb(gt_pdb)
    if pred_pdb.endswith('.cif'):
        pred = get_cif(pred_pdb)
    else:
        pred = get_pdb(pred_pdb)
    filtered_gt, filtered_pred = align_residues(gt, pred)
    # return filtered_gt, filtered_pred
    gt_coords, pred_coords = extract_c4_coords(filtered_gt, filtered_pred)
    r, x = get_optimal_transform(pred_coords, gt_coords)
    pred_coords = pred_coords @ r + x
    best_rmsd = compute_rmsd(gt_coords, pred_coords)
    best_lddt = compute_lddt(gt_coords, pred_coords)
    metrics = {
        "rmsd": float(best_rmsd),
        "lddt": float(best_lddt)
    }
    return metrics