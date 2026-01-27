import sys
import torch
import logging
import os
import joblib

logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RHOFOLD_ROOT = '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/cfff_siqi/kaggle/RhoFold'  # please pre-install rhofold from source code, and ref to source code path
if RHOFOLD_ROOT not in sys.path:
    sys.path.append(RHOFOLD_ROOT)

from rhofold.rhofold import RhoFold
from rhofold.config import rhofold_config
from rhofold.utils.alphabet import get_features

# 全局Rhofold模型单例
RHOFOLD_MODEL = None
RHOFOLD_CONFIG = {
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'ckpt': '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/cfff_siqi/kaggle/RhoFold/pretrained/rhofold_pretrained.pt',
    'single_seq_pred': True
}

def init_rhofold_model():
    """初始化全局Rhofold模型（延迟加载）。"""
    global RHOFOLD_MODEL
    if RHOFOLD_MODEL is None:
        logger.debug("加载Rhofold模型")
        RHOFOLD_MODEL = RhoFold(rhofold_config)
        RHOFOLD_MODEL.load_state_dict(torch.load(RHOFOLD_CONFIG['ckpt'], map_location='cpu')['model'])
        RHOFOLD_MODEL.eval()
        RHOFOLD_MODEL = RHOFOLD_MODEL.to(RHOFOLD_CONFIG['device'])

def cleanup_rhofold_model():
    global RHOFOLD_MODEL
    if RHOFOLD_MODEL is not None:
        RHOFOLD_MODEL = RHOFOLD_MODEL.to('cpu')
        del RHOFOLD_MODEL
        RHOFOLD_MODEL = None
        torch.cuda.empty_cache()
        logger.debug("Rhofold模型已清理")

def run_rhofold_sync(sequence: str, fasta_file: str, output_dir: str, seq_index: int) -> tuple:
    try:
        if RHOFOLD_MODEL is None:
            raise RuntimeError("Rhofold模型未初始化")

        with open(fasta_file, "w") as f:
            f.write(f">100500\n{sequence}\n")

        # config output dir
        pdb_file = os.path.join(output_dir, 'unrelaxed_model.pdb')

        # prepare input
        data_dict = get_features(fasta_file, fasta_file)  # single sequence predict，input_a3m=input_fas

        # inference
        with torch.no_grad():
            outputs = RHOFOLD_MODEL(
                tokens=data_dict['tokens'].to(RHOFOLD_CONFIG['device']),
                rna_fm_tokens=data_dict['rna_fm_tokens'].to(RHOFOLD_CONFIG['device']),
                seq=data_dict['seq'],
            )
            output = outputs[-1]  # invalid in my environment
            node_cords_pred = output['cord_tns_pred'][-1].squeeze(0).data.cpu().numpy()

        RHOFOLD_MODEL.structure_module.converter.export_pdb_file(
            data_dict['seq'],
            node_cords_pred,
            path=pdb_file,
            chain_id=None,
            confidence=output['plddt'][0].data.cpu().numpy(),
            logger=logger
        )
        os.remove(fasta_file)
        return seq_index, pdb_file
    except Exception as e:
        logger.error(f"Rhofold失败，序列 ({sequence}): {str(e)}")
        return seq_index, None

def batch_rhofold(sequences: list, pdb_name: str, out_dir: str = 'tmp_output', n_jobs: int = 1) -> list:
    os.makedirs(f'./{out_dir}/pdb', exist_ok=True)
    os.makedirs(f'./{out_dir}/fasta', exist_ok=True)

    output_dirs = [f"./{out_dir}/pdb/{pdb_name}_{i}" for i in range(len(sequences))]
    fasta_files = [f"./{out_dir}/fasta/{pdb_name}_{i}.fasta" for i in range(len(sequences))]

    results = joblib.Parallel(n_jobs=n_jobs, require="sharedmem")(
        joblib.delayed(run_rhofold_sync)(seq, fasta_file, output_dir, seq_index)
        for seq, fasta_file, output_dir, seq_index in zip(sequences, fasta_files, output_dirs, range(len(sequences)))
    )
    sorted_results = sorted(results, key=lambda x: x[0])
    output_files = [result[1] for result in sorted_results]
    torch.cuda.empty_cache()
    return output_files