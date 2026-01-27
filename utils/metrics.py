import os
import RNA
from utils.predict_3d_structure import batch_rhofold
from utils.evaluate_3d_structure import compute_monomer
def seq_recovery(gt_seq, pred_seq):
    ind = (gt_seq == pred_seq.argmax(dim=-1))
    recovery = ind.sum() / ind.shape[0]
    return recovery, ind.cpu()
def secondary_structure_metric(gt_seq, pred_seq):
    pred_structure, pred_mfe = RNA.fold(pred_seq)
    gt_structure, gt_mfe = RNA.fold(gt_seq)

    similarity_count = 0
    min_len = min(len(pred_structure), len(gt_structure))
    for i in range(min_len):
        if pred_structure[i] == gt_structure[i]:
            similarity_count += 1

    secondary_structure_similarity = (similarity_count / min_len) * 100 if min_len > 0 else 0.0
    return secondary_structure_similarity, pred_mfe

def three_dimension_metric(pred_seq, pdb_name, pdb_data_dir, out_dir, n_jobs=1):
    output_files = batch_rhofold(pred_seq, pdb_name, out_dir, n_jobs=n_jobs)
    rmsd_list = []
    lddt_list = []
    for output_file in output_files:
        gt_pdb_file = os.path.join(pdb_data_dir, pdb_name + ".pdb")
        metrics = compute_monomer(gt_pdb_file, output_file)
        rmsd = metrics['rmsd']
        lddt = metrics['lddt']
        rmsd_list.append(rmsd)
        lddt_list.append(lddt)
    return rmsd_list, lddt_list

def compute_sequence_similarity(pred_seq, gt_seq):
    assert len(pred_seq) == len(gt_seq)
    eq_cnt = 0
    for i in range(len(pred_seq)):
        if pred_seq[i] == gt_seq[i]:
            eq_cnt += 1
    recovery = eq_cnt / len(pred_seq)
    return recovery