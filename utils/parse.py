import numpy as np


alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
states = len(alpha_1)
alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
            'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']

atoms_codebook = ['N','CA','C', 'O', 'CB', 'CG', 'CG1', 'CG2', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CZ', 'CZ2', 'CZ3', 'CH2', 'OH', 'SD', 'SG', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NZ', 'NH1', 'NH2', 'OG', 'OG1', 'OD1', 'OD2', 'OE1', 'OE2', 'SD', 'SG']
residue_codebook = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C']

aa_1_N = {a:n for n,a in enumerate(alpha_1)}
aa_3_N = {a:n for n,a in enumerate(alpha_3)}
aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}

def AA_to_N(x):
    # ["ARND"] -> [[0,1,2,3]]
    x = np.array(x)
    if x.ndim == 0: x = x[None]
    return [[aa_1_N.get(a, states-1) for a in y] for y in x]
    

def N_to_AA(x):
    # [[0,1,2,3]] -> ["ARND"]
    x = np.array(x)
    if x.ndim == 1: x = x[None]
    return ["".join([aa_N_1.get(a,"-") for a in y]) for y in x]
    
    
def parse_PDB_biounits(x, atoms=None, chain=None):
    '''
    input:  x = PDB filename
            atoms = atoms to extract (optional)
    output: (length, atoms, coords=(x,y,z)), sequence
    '''
    if atoms is None:
        atoms = atoms_codebook
    xyz,seq,min_resn,max_resn = {},{},1e6,-1e6
    for line in open(x,"rb"):
        line = line.decode("utf-8","ignore").rstrip()

        if line[:6] == "HETATM" and line[17:17+3] == "MSE":
            line = line.replace("HETATM","ATOM  ")
            line = line.replace("MSE","MET")

        if line[:4] == "ATOM":
            ch = line[21:22]
            if ch == chain or chain is None:
                atom = line[12:12+4].strip()
                resi = line[17:17+3]
                resn = line[22:22+5].strip()
                x,y,z = [float(line[i:(i+8)]) for i in [30,38,46]]

                if resn[-1].isalpha(): 
                    resa,resn = resn[-1],int(resn[:-1])-1
                else: 
                    resa,resn = "",int(resn)-1
        #         resn = int(resn)
                if resn < min_resn: 
                    min_resn = resn
                if resn > max_resn: 
                    max_resn = resn
                if resn not in xyz: 
                    xyz[resn] = {}
                if resa not in xyz[resn]: 
                    xyz[resn][resa] = {}
                if resn not in seq: 
                    seq[resn] = {}
                if resa not in seq[resn]: 
                    seq[resn][resa] = resi

                if atom not in xyz[resn][resa]:
                    xyz[resn][resa][atom] = np.array([x,y,z])

    # convert to numpy arrays, fill in missing values
    seq_,xyz_ = [],[]
    try:
        for resn in range(min_resn,max_resn+1):
            if resn in seq:
                for k in sorted(seq[resn]): seq_.append(aa_3_N.get(seq[resn][k],20))
            else: seq_.append(20)
            if resn in xyz:
                for k in sorted(xyz[resn]):
                    for atom in atoms:
                        if atom in xyz[resn][k]: xyz_.append(xyz[resn][k][atom])
                        else: xyz_.append(np.full(3,np.nan))
            else:
                for atom in atoms: xyz_.append(np.full(3,np.nan))
        return np.array(xyz_).reshape(-1,len(atoms),3), N_to_AA(np.array(seq_))
    except TypeError:
        return 'no_chain', 'no_chain'

from Bio.PDB.PDBParser import PDBParser
def parse_PDB_Biopython(x, atoms=None, chain=None):
    if atoms is None:
        atoms = atoms_codebook
    pdb_parser = PDBParser(PERMISSIVE=1)
    structure_id = "2O2B"
    filename = "data/2O2B.pdb"
    structure = pdb_parser.get_structure(structure_id, filename)
    seq_, xyz_ = [], []
    for model in structure:
        for chain_ in model:
            if chain is None or chain == chain_.id:
                for residue in chain_:
                    seq_.append(aa_3_N.get(residue.get_resname(),20))
                    for atom in atoms:
                        if atom in residue:
                            xyz_.append(residue[atom].get_coord())
                        else:
                            xyz_.append(np.full(3,np.nan))
    return np.array(xyz_).reshape(-1,len(atoms),3), N_to_AA(np.array(seq_))
    