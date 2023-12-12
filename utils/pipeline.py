from pathlib import Path
import pandas as pd

from .mutate import mutate_all, mutate_one, mutate_two, strategy_libary
from .parse import parse_PDB_Biopython, parse_PDB_biounits, atoms_codebook, residue_codebook
from .model import model_name_library

from typing import overload, Union, List, Dict, Tuple

from Bio import SeqIO

def read_sequence(sequence_file):
    """load from sequence file, only load the first sequence

    Args:
        sequence_file (str or pathlib.Path): sequence file

    Returns:
        str: sequence
    """
    seq = str(list(SeqIO.parse(sequence_file, "fasta"))[0].seq)
    return seq


def read_sites(site_file, one_based=False):
    """load from site file, each line is a (site, residue) pair, means the site is mutated from the residue

    Args:
        site_file (str or pathlib.Path): site file
        one_based (bool, optional): whether the site is one-based. Defaults to False.

    Returns:
        list: each element is a (site, residue) pair, note that the site is a string and 0-based, and the residue is 1-letter code
    """
    sites = []
    for line in Path(site_file).read_text().splitlines():
        if line == '':
            continue
        if line.startswith('#'):
            continue
        site = line.split()
        if one_based:
            site = (str(int(site[0]) - 1), site[1])
        sites.append(site)
    return sites

def read_pdb(pdb_file, atoms=atoms_codebook):
    """load from pdb file, only load the atoms in `atoms`, using `parse_PDB_biounits` from ProteinMPNN

    Args:
        pdb_file (str or pathlib.Path): pdb file
        atoms (list, optional): atoms list for loading. Defaults to atoms_codebook.

    Returns:
        tuple(np.array, list): pdb array (sequence_length, len(atoms), 3) and sequence list
    """
    if pdb_file is None:
        return None, None
    pdb_str = Path(pdb_file)
    pdb, sequence = parse_PDB_biounits(pdb_str, atoms=atoms)
    return pdb, sequence

def read_pdb_Bio(pdb_file, atoms=atoms_codebook):
    """load from pdb file, only load the atoms in `atoms`, using `parse_PDB_Biopython` from biopython

    Args:
        pdb_file (str or pathlib.Path): pdb file
        atoms (list, optional): atoms list for loading. Defaults to atoms_codebook.

    Returns:
        tuple(np.ndarray, list): pdb array (sequence_length, len(atoms), 3) and sequence list
    """
    pdb_str = Path(pdb_file)
    pdb, sequence = parse_PDB_Biopython(pdb_str, atoms=atoms)
    return pdb, sequence
    

def check_sequence_and_sites(sequence, sites):
    for idx, site in enumerate(sites):
        assert sequence[int(site[0])] == site[1], f"the site#{idx} {site[1]} in site file is not the same as the site#{idx} {sequence[int(site[0])]} in sequence file" # check if the site in site file is the same as the site in sequence file
        
def check_fasta_sequence_and_pdb_sequence(fasta_sequence, pdb_sequence):
    if pdb_sequence is None:
        print("pdb_sequence is None, skip checking")
        return
    if not fasta_sequence == pdb_sequence[0]: 
        print("the sequence in pdb file is not the same as the sequence in sequence file") # check if the sequence in pdb file is the same as the sequence in sequence file
        return

def get_mutations(sequence, sites, pdb, model_name="esm2_t6_8M_UR50D", strategy="mutate_one"):
    """get mutations profiles for each site, potentially combining pdb and using different model and strategy.

    Args:
        sequence (str): sequence
        sites (list): sites from site file
        pdb (np.ndarray): pdb array (sequence_length, len(atoms), 3)
        model_name (str, optional): model name of esm. Defaults to "esm2_t6_8M_UR50D".
        strategy (str, optional): strategy to mutate. Defaults to "mutate_one".

    Returns:
        dict: mutations profiles for each site
    """
    assert strategy in strategy_libary, f"strategy {strategy} is not in strategy_libary {strategy_libary}"
    if isinstance(model_name, str):
        assert model_name in model_name_library, f"model_name {model_name} is not in model_name_libary {model_name_library}"
    elif isinstance(model_name, Path):
        assert model_name.name[:-3] in model_name_library, f"model_name {model_name} is not in model_name_libary {model_name_library}"
    else:
        assert False, f"model_name {model_name} is not a string or pathlib.Path"
        
    if strategy == "mutate_all":
        mutations = mutate_all(sequence, sites, pdb, model_name)
    elif strategy == "mutate_one":
        mutations = mutate_one(sequence, sites, pdb, model_name)
    elif strategy == "mutate_two":
        mutations = mutate_two(sequence, sites, pdb, model_name)
    
    return mutations

def get_sequence_library(mutations_profiles, sequence_number=5):
    """get sequence library from mutations profiles

    Args:
        mutations_profiles (torch.tensor): mutations profiles of selected sites (n, 20)
        sequence_number (int, optional): selected sequence number. Defaults to 5.

    Returns:
        List((selected amino acid, probability)): sequence library
    """
    sequences = [[list(), 1.0]]
    
    for row_idx in range(mutations_profiles.shape[0]):
        row = mutations_profiles[row_idx]
        all_candidates = list()

        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * float(row[j])]
                all_candidates.append(candidate)

        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)  # 按score排序
        sequences = ordered[:sequence_number]  # choose the top sequence_number sequences
    return sequences

def save_sequence_library_for_sites(sequence_library: List[Tuple[list, float]], sequence: str, sites: list, output_file: str, one_based: bool = False) -> None:
    output_file = Path(output_file)
    output_list = []
    return_list = []
    saved_wt = False
    for item in sequence_library:
        comment_string_list = []
        mutated_sites = item[0]
        mutated_probability = item[1]
        random_choice_probability = 0.05**len(sites)
        
        comment_string_list.append("mutations:")
        sequence_list = list(sequence)
        mutant_list = []
        for m_s, s in zip(mutated_sites, sites):
            sequence_list[int(s[0])] = residue_codebook[m_s]
            if s[1] != residue_codebook[m_s]:
                mut = f"{s[1]}{int(s[0])+1}{residue_codebook[m_s]}"
                mutant_list.append(mut)
                comment_string_list.append(mut)
        else:
            # if in the last run, the site is not mutated, save once, then remove this sequence
            if len(mutant_list) == 0:
                if saved_wt:
                    continue
                else:
                    saved_wt = True
                    mutant_list.append("WT")
                    comment_string_list.append("WT")
        
        comment_string_list.append(f"probability: {mutated_probability:.4f}")
        # comment_string_list.append(f"better than random choice: {mutated_probability/random_choice_probability:.2f} times")
        comment_string = " ".join(comment_string_list)
        sequence_string = "".join(sequence_list)
        mutant_string = ",".join(mutant_list)
        output_list.append((comment_string, sequence_string))
        return_list.append((comment_string, mutant_string, sequence_string, mutated_probability))
        # return_list.append((mutated_sites, mutated_probability))
    
    
    if output_file.suffix == ".csv":
        return_pd = pd.DataFrame(return_list, columns=["comment", "mut", "sequence", "fitness"])
        return_pd.to_csv(output_file, index=False)
    elif output_file.suffix == ".fasta" or output_file.suffix == ".fa":
        with open(output_file, "w") as f:
            for comment_string, sequence_string in output_list:
                f.write(f">{comment_string}\n{sequence_string}\n")

def save_sequence_library_for_sequence(sequence_library: List[Tuple[str, float]], sequence: str, sites: list, output_file: str):
    output_file = Path(output_file)
    output_list = []
    return_list = []
    for item in sequence_library:
        comment_string_list = []
        mutated_sequence = item[0]
        mutated_probability = item[1]
        
        sequence_list = list(mutated_sequence)
        comment_string_list.append("mutations:")
        for s in sites:
            m_s_token = sequence_list[int(s[0])]
            comment_string_list.append(f"{s[1]}{int(s[0])+1}{m_s_token}")
        
        comment_string_list.append(f"probability: {mutated_probability:.4f}")
        comment_string = " ".join(comment_string_list)
        sequence_string = "".join(sequence_list)
        output_list.append((comment_string, sequence_string))
        return_list.append((sequence_string, mutated_probability))
    
    
    if output_file.suffix == ".csv":
        return_pd = pd.DataFrame(return_list, columns=["mutant", "fitness"])
        return_pd.to_csv(output_file, index=False, header=None)
    elif output_file.suffix == ".fasta" or output_file.suffix == ".fa":
        with open(output_file, "w") as f:
            for comment_string, sequence_string in output_list:
                f.write(f">{comment_string}\n{sequence_string}\n")
