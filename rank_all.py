import argparse
from utils.model import model_name_library
from utils.rank import rank_strategy_dictionary

def parse_args_rank_all(argument_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-file', type=str, required=True, help="fasta file")
    parser.add_argument('--site-file', type=str, default=None, help="site file, if not provided, will use all sites")
    parser.add_argument('--one-based', action='store_true', help="whether site file is one-based or not")
    parser.add_argument('--pdb-file', type=str, help="pdb file")
    parser.add_argument('--model-name', type=str, default="esm2_t6_8M_UR50D", choices=model_name_library, help="model name")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint")
    parser.add_argument('--rank-strategy', type=str, default="esm1v_1", choices=rank_strategy_dictionary, help="rank strategy")
    parser.add_argument('--sequence-number', type=int, default=-1, help="number of sequences to be ranked")
    parser.add_argument('--output-file', type=str, required=True, help="output file")
    parser.add_argument('--reverse', action='store_true', help="whether to reverse the sequence")
    parser.add_argument('--gt-file', type=str, default=None, help="ground truth file")
    parser.add_argument("--single-site", action="store_true", default=False, help="only rank for single site")
    parser.add_argument("-f", "--force-overwrite", action="store_true", default=False, help="force overwrite")
    if argument_list is None:
        return parser.parse_args()
    else:
        return parser.parse_args(argument_list)

from utils import residue_codebook
import itertools
from utils.rank import rank_for_all_strategies


from utils.pipeline import read_sequence, read_sites, check_sequence_and_sites, save_sequence_library_for_sites
from pathlib import Path
import torch
from transformers import AutoTokenizer, EsmForMaskedLM
import esm
from tqdm import tqdm
def main(args):
    # print(f"model name: {args.model_name}\tstrategy: {args.rank_strategy}", flush=True)
    if Path(args.output_file).exists() and args.force_overwrite is False:
        print(f"output file {args.output_file} exists, skip", flush=True)
        return
    sequence = read_sequence(args.sequence_file)
    # if site_file is None, we will use all sites
    if args.site_file is None:
        sites = list(zip(range(len(sequence)), list(sequence)))
    else:
        sites = read_sites(args.site_file, one_based=args.one_based)

    check_sequence_and_sites(sequence, sites)
    
    print("loading model", flush=True)
    if args.checkpoint is None:
        try:
            print("traditional path", flush=True)
            model_name = Path("/path/to/esm/models/"+args.model_name+".pt")
            model, alphabet = esm.pretrained.load_model_and_alphabet(str(model_name))
        except:
            print("huggingface path", flush=True)
            model_name = Path("/path/to/huggingface/models/"+args.model_name)
            model = EsmForMaskedLM.from_pretrained(model_name)
            alphabet = AutoTokenizer.from_pretrained(model_name)
    else:
        model_name = Path("/path/to/huggingface/models/"+args.model_name)
        model = EsmForMaskedLM.from_pretrained(model_name)
        alphabet = AutoTokenizer.from_pretrained(model_name)
        checkpoint_path = "/path/to/checkpoint/"+args.checkpoint+".pt"
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict["model"], strict=False)
    
    if torch.cuda.is_available():
        model = model.cuda()

    print("ranking", flush=True)
    sequence_library = rank_for_all_strategies(model, alphabet, sequence, sites, strategy=args.rank_strategy, sequence_number=args.sequence_number, reverse=args.reverse, single_site=args.single_site)
    # print(sequence_library[:5], flush=True)
    save_sequence_library_for_sites(sequence_library, sequence, sites, args.output_file)

if __name__ == "__main__":
    args = parse_args_rank_all()
    print(args, flush=True)
    main(args)
