import argparse
import os
from utils.model import model_name_library
from utils.rank import rank_strategy_dictionary
from transformers import AutoTokenizer, EsmForMaskedLM
import warnings
warnings.filterwarnings("ignore")

def parse_args_rank_all_dms(argument_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dms-dir', type=str, required=True, help="Directory containing DMS files")
    parser.add_argument('--model-name', type=str, default="esm2_t6_8M_UR50D", choices=model_name_library, help="model name")
    parser.add_argument('--checkpoint', type=str, default=None, help="checkpoint")
    parser.add_argument('--rank-strategy', type=str, default="esm1v_1", choices=rank_strategy_dictionary, help="rank strategy")
    parser.add_argument('--reverse', action='store_true', help="whether to reverse the sequence")
    parser.add_argument("-f", "--force-overwrite", action="store_true", default=False, help="force overwrite")
    if argument_list is None:
        return parser.parse_args()
    else:
        return parser.parse_args(argument_list)

from utils.rank import rank_for_all_strategies

def check_exist(output_file, force_overwrite, verbose=True):
    if Path(output_file).exists() and force_overwrite is False:
        if verbose:
            print(f"output file {output_file} exists, skip", flush=True)
        return True
    else:
        return False

from utils.pipeline import read_sequence, check_sequence_and_sites, save_sequence_library_for_sites
from pathlib import Path
import torch
import esm
from tqdm import tqdm
def main(args):
    dms_home = Path(args.dms_dir)
    results_dir = dms_home / "results"
    fasta_dir = dms_home / "fasta"
    VERBOSE = False

    # check if the output file exists, if exists then stop this run
    for fasta_file in fasta_dir.glob("*.fa"):
        if args.checkpoint is not None:
            file_name = (fasta_file.stem + f"_{args.checkpoint}_{args.rank_strategy}_.csv")
        else:
            file_name = (fasta_file.stem + f"_{args.model_name}_{args.rank_strategy}_.csv")
        output_file = results_dir / file_name
        if check_exist(output_file, args.force_overwrite, verbose=VERBOSE):
            # if you want to stop this run, uncomment the following line
            # return
            # if you want to skip this file, uncomment the following line
            continue
        else:
            stop_file_name = file_name
            break
    else:
        print("all files exist, skip", flush=True)
        return
    
    print("loading model", flush=True)
    if args.checkpoint is None and args.model_name != "regression_transformer":
        model_name = Path("/path/to/esm/model/"+args.model_name+".pt")
        model, alphabet = esm.pretrained.load_model_and_alphabet(str(model_name))
    elif args.model_name != "regression_transformer":
        model_name = Path("/path/to/huggingface/models/"+args.model_name)
        model = EsmForMaskedLM.from_pretrained(model_name)
        alphabet = AutoTokenizer.from_pretrained(model_name)
        checkpoint_path = "/path/to/checkpoint/"+args.checkpoint+".pt"
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict["model"], strict=False)
    else:
        from gt4sd.algorithms.conditional_generation.regression_transformer.implementation import (
            ConditionalGenerator
        )
        from gt4sd.algorithms.conditional_generation.regression_transformer import (
            RegressionTransformerProteins
        )
        from terminator.search import SEARCH_FACTORY

        config = RegressionTransformerProteins(
            algorithm_version='stability', search='greedy'
        )

        model = ConditionalGenerator(
                    resources_path=config.ensure_artifacts(),
                    device="cuda:0",
                    tolerance=config.tolerance,
                )
        model.search = SEARCH_FACTORY[config.search](temperature=config.temperature)
        model.max_samples = 1
        alphabet = "<stab>[MASK][MASK][MASK][MASK][MASK]|{}"
        
    if torch.cuda.is_available() and args.model_name != "regression_transformer":
        model = model.cuda()
    
    for fasta_file in fasta_dir.glob("*.fa"):
        if args.checkpoint is not None:
            file_name = (fasta_file.stem + f"_{args.checkpoint}_{args.rank_strategy}_.csv")
        else:
            file_name = (fasta_file.stem + f"_{args.model_name}_{args.rank_strategy}_.csv")
        if stop_file_name is not None and file_name != stop_file_name:
            continue
        else:
            stop_file_name = None
            
        output_file = results_dir / file_name
        if check_exist(output_file, args.force_overwrite, verbose=VERBOSE):
            # if you want to stop this run, uncomment the following line
            return
            # if you want to skip this file, uncomment the following line
            # continue
        sequence = read_sequence(fasta_file)
        # since every 1st site tends to be M, we add M to the first token
        sequence = "M" + sequence
        sites = list(zip(range(len(sequence)), list(sequence)))
        single_site = True
        
        check_sequence_and_sites(sequence, sites)
        
        print("ranking", flush=True)
        if check_exist(output_file, args.force_overwrite, verbose=VERBOSE):
            # if you want to stop this run, uncomment the following line
            return
            # if you want to skip this file, uncomment the following line
            continue
        sequence_library = rank_for_all_strategies(model, alphabet, sequence, sites, strategy=args.rank_strategy, sequence_number=-1, reverse=args.reverse, single_site=single_site)
        if check_exist(output_file, args.force_overwrite, verbose=VERBOSE):
            # if you want to stop this run, uncomment the following line
            return
            # if you want to skip this file, uncomment the following line
            continue
        save_sequence_library_for_sites(sequence_library, sequence, sites, output_file)

if __name__ == "__main__":
    args = parse_args_rank_all_dms()
    print(args, flush=True)
    main(args)
