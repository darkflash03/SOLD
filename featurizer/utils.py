from Bio.PDB.PDBParser import PDBParser
from os.path import basename, splitext

def get_pdb_id(pdb_file_path):
    return splitext(basename(pdb_file_path))[0]

def parse_pdb(pdb_file, model=0):
    parser = PDBParser()
    structure = parser.get_structure(get_pdb_id(pdb_file), pdb_file)
    return structure[model]