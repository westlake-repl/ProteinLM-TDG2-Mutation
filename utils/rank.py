import torch
from tqdm import tqdm
from utils import residue_codebook
import itertools
from transformers import EsmTokenizer, EsmForMaskedLM
import esm

rank_strategy_dictionary = {
    "esm1v_1": "MaskMP_A", # esm1v Masked Marginal Probability Strategy A
    "esm1v_2": "MaskMP_B", # esm1v Masked Marginal Probability Strategy B
    "esm1v_3": "MaskMP_C", # esm1v Masked Marginal Probability Strategy C
    "esm1v_4": "MutantMP", # esm1v Mutant Marginal Probability
    "esm1v_5": "WildtypeMP", # esm1v Wildtype Marginal Probability
    "AR_1": "AutoRegressive_A", # AutoRegressive Strategy A
    "AR_2": "AutoRegressive_B", # AutoRegressive Strategy B
    "AR_3": "AutoRegressive_C", # AutoRegressive Strategy C
    "AR_4": "AutoRegressive_D", # AutoRegressive Strategy D
}

def rank_for_all_strategies(model, alphabet, seq, sites, strategy, sequence_number, reverse, single_site):
    assert strategy in rank_strategy_dictionary, f"strategy {strategy} not found in rank_strategy_dictionary"
    if not (isinstance(model, esm.model.esm2.ESM2) or isinstance(model, EsmForMaskedLM) or isinstance(model, esm.model.esm1.ProteinBertModel)):
        print(type(model))
        import gt4sd
        return rank_for_regression_transformer(model, alphabet, seq, sites, strategy, sequence_number, reverse, single_site)
    if rank_strategy_dictionary[strategy] == "MaskMP_A":
        return rank_for_MaskMP_A(model, alphabet, seq, sites, sequence_number, reverse=reverse, single_site=single_site)
    elif rank_strategy_dictionary[strategy] == "MaskMP_B":
        return rank_for_MaskMP_B(model, alphabet, seq, sites, sequence_number, reverse=reverse, single_site=single_site)
    elif rank_strategy_dictionary[strategy] == "MaskMP_C":
        return rank_for_MaskMP_C(model, alphabet, seq, sites, sequence_number, reverse=reverse, single_site=single_site)
    elif rank_strategy_dictionary[strategy] == "MutantMP":
        return rank_for_MutantMP(model, alphabet, seq, sites, sequence_number, reverse=reverse, single_site=single_site)
    elif rank_strategy_dictionary[strategy] == "WildtypeMP":
        return rank_for_WildtypeMP(model, alphabet, seq, sites, sequence_number, reverse=reverse, single_site=single_site)
    elif rank_strategy_dictionary[strategy] == "AutoRegressive_A":
        return rank_for_AutoRegressive_A(model, alphabet, seq, sites, sequence_number, reverse=reverse, single_site=single_site)
    elif rank_strategy_dictionary[strategy] == "AutoRegressive_B":
        return rank_for_AutoRegressive_B(model, alphabet, seq, sites, sequence_number, reverse=reverse, single_site=single_site)
    elif rank_strategy_dictionary[strategy] == "AutoRegressive_C":
        return rank_for_AutoRegressive_C(model, alphabet, seq, sites, sequence_number, reverse=reverse, single_site=single_site)
    elif rank_strategy_dictionary[strategy] == "AutoRegressive_D":
        return rank_for_AutoRegressive_D(model, alphabet, seq, sites, sequence_number, reverse=reverse, single_site=single_site)
    elif rank_strategy_dictionary[strategy] == "AutoRegressive_E":
        return rank_for_AutoRegressive_E(model, alphabet, seq, sites, sequence_number, reverse=reverse, single_site=single_site)

def make_mutant_library(sites, single_site):
    """make mutant library for single site or multiple sites
    Args:
    sites: [(site1, res1), (site2, res2), ...]
    single_site: bool
        single_site = True if only one site is mutated at a time
    Returns:
    mutant_library: [[(site1, res1), (site2, res2), ...], ...]
        len(mutant_library) = 20 * len(sites) if single_site else 20 ** len(sites)
        len(mutant_library[0]) = len(sites)
        mutant_library[0][0] = (site1, res1)
    """
    if single_site:
        mutant_library = []
        for site in sites: # 所有的该突变的地方
            for res in residue_codebook: # 突变成的蛋白质
                _mutant = []
                for _site in sites: # 遍历全长序列，构建突变
                    if _site[0] == site[0]:
                        _mutant.append((site[0], res)) # mutate
                    else:
                        _mutant.append(_site) # 维持原状
                mutant_library.append(_mutant)
    else:
        mutant_library = list(itertools.product(*[[(site[0], res) for res in residue_codebook] for site in sites]))
    return mutant_library

def rank_for_regression_transformer(model, alphabet, seq, sites, strategy, sequence_number, reverse, single_site):
    sequence_library = []
    mutant_library = make_mutant_library(sites, single_site)
    for mutant in tqdm(mutant_library):
        mutant_sequence_list = list(seq)
        for m in mutant:
            mutant_sequence_list[int(m[0])] = m[1]
        mutant_sequence = "".join(mutant_sequence_list)
        result = model.generate_batch_regression(alphabet.format(mutant_sequence))
        result = float(result[0].split("<stab>")[1])
        sequence_library.append((tuple([residue_codebook.index(m[1]) for m in mutant]), result))
    sequence_library = sorted(sequence_library, key=lambda x: x[1], reverse=True)
    
    if sequence_number > 0:
        return sequence_library[:sequence_number]
    else:
        return sequence_library


def rank_for_MaskMP_A(model, alphabet, seq, sites, sequence_number, reverse, single_site):
    sequence_library = []
    if isinstance(alphabet, esm.data.Alphabet):
        mask_token = alphabet.get_tok(alphabet.mask_idx)
        converter = alphabet.get_batch_converter()
    elif isinstance(alphabet, EsmTokenizer):
        mask_token = alphabet.mask_token
        converter = alphabet
    mutant_library = make_mutant_library(sites, single_site)
    
    # For MaskMP_A, mask every sites in seq
    sequence_list = list(seq)
    for site in sites:
        sequence_list[int(site[0])] = mask_token
    masked_sequence = "".join(sequence_list)
    
    if isinstance(converter, EsmTokenizer):
        mask_tokens = converter([("", masked_sequence)])["input_ids"]
        mask_tokens = torch.Tensor(mask_tokens).long()
        mask_tokens = torch.cat([mask_tokens[:, 0:1], mask_tokens[:, 2:]], dim=1)
    else:
        _, _, mask_tokens = converter([("", masked_sequence)])
    if torch.cuda.is_available() and not isinstance(mask_tokens, str):
        mask_tokens = mask_tokens.cuda()
    with torch.no_grad():
        if isinstance(model, EsmForMaskedLM):
            result = model(mask_tokens)
        else:
            result = model(mask_tokens, return_contacts=False)
    # probabilties = result["logits"][0, [int(site[0])+1 for site in sites], 4:24].softmax(-1)
    probabilties = result["logits"][0, [int(site[0])+1 for site in sites], :].softmax(-1)
    
    sequence_library = []
    for mutant in tqdm(mutant_library):
        mutant_score = 0
        for idx, (mut, site) in enumerate(zip(mutant, sites)):
            assert mut[0] == site[0]
            # mutant_score += torch.log(probabilties[idx, residue_codebook.index(mut[1])]) - torch.log(probabilties[idx, residue_codebook.index(site[1])])
            mutant_score += torch.log(probabilties[idx, residue_codebook.index(mut[1])]) - torch.log(probabilties[idx, residue_codebook.index(site[1])])
            # mutant_score += torch.log(probabilties[idx, residue_codebook.index(mut[1])])
        ################ try reverse ####################
        if reverse:
            mutant_score = -mutant_score
        ##################################################
        
        mutant_save = tuple([residue_codebook.index(m[1]) for m in mutant])
        mutant_score = float(mutant_score)
        sequence_library.append((mutant_save, mutant_score))
        
    sequence_library = sorted(sequence_library, key=lambda x: x[1], reverse=True)
        
    if sequence_number > 0:
        return sequence_library[:sequence_number]
    else:
        return sequence_library


def rank_for_MaskMP_B(model, alphabet, seq, sites, sequence_number, reverse, single_site):
    sequence_library = []
    if not isinstance(alphabet, EsmTokenizer):
        mask_token = alphabet.get_tok(alphabet.mask_idx)
        converter = alphabet.get_batch_converter()
    else:
        mask_token = alphabet.mask_token
        converter = alphabet
    mutant_library = make_mutant_library(sites, single_site)
    
    # For MaskMP_B, mask each site, then predict the probability of the wildtype residue
    masked_probabilities = []
    for site in sites:
        sequence_list = list(seq)
        sequence_list[int(site[0])] = mask_token
        masked_sequence = "".join(sequence_list)
        if isinstance(converter, EsmTokenizer):
            mask_tokens = converter([("", masked_sequence)])["input_ids"]
            mask_tokens = torch.Tensor(mask_tokens).long()
            mask_tokens = torch.cat([mask_tokens[:, 0:1], mask_tokens[:, 2:]], dim=1)
        else:
            _, _, mask_tokens = converter([("", masked_sequence)])
        print(masked_sequence)
        if torch.cuda.is_available():
            mask_tokens = mask_tokens.cuda()
        with torch.no_grad():
            if isinstance(model, EsmForMaskedLM):
                result = model(mask_tokens)
            else:
                result = model(mask_tokens, return_contacts=False)
        probabilties = result["logits"][0, int(site[0])+1, 4:24].softmax(-1)
        masked_probabilities.append(torch.log(probabilties[residue_codebook.index(site[1])]))
    
    sequence_library = []
    for mutant in tqdm(mutant_library):
        mutant_score = 0
        # substitute the wildtype residue with the mutant residue
        mutant_sequence_list = list(seq)
        for m in mutant:
            mutant_sequence_list[int(m[0])] = m[1]
        mutant_sequence = "".join(mutant_sequence_list)
        print(mutant_sequence)
        # mask each site, then predict the probability of the mutant residue
        mutant_masked_sequence_list = []
        for m in mutant:
            masked_sequence_list = list(mutant_sequence)
            masked_sequence_list[int(m[0])] = mask_token
            mutant_masked_sequence_list.append("".join(masked_sequence_list))
        print(mutant_masked_sequence_list)
        if isinstance(converter, EsmTokenizer):
            mask_tokens = converter([("", mutant_masked_sequence) for mutant_masked_sequence in mutant_masked_sequence_list])["input_ids"]
            mask_tokens = torch.Tensor(mask_tokens).long()
            mask_tokens = torch.cat([mask_tokens[:, 0:1], mask_tokens[:, 2:]], dim=1)
        else:
            _, _, mask_tokens = converter([("", mutant_masked_sequence) for mutant_masked_sequence in mutant_masked_sequence_list])
        if torch.cuda.is_available():
            mask_tokens = mask_tokens.cuda()
        with torch.no_grad():
            if isinstance(model, EsmForMaskedLM):
                result = model(mask_tokens)
            else:
                result = model(mask_tokens, return_contacts=False)
        for idx, (m, m_p) in enumerate(zip(mutant, masked_probabilities)):
            # print(mask_tokens[idx, int(m[0])+1])
            probabilties = result["logits"][idx, int(m[0])+1, 4:24].softmax(-1)
            mutant_score += torch.log(probabilties[residue_codebook.index(m[1])]) - m_p
        ################ try reverse ####################
        if reverse:
            mutant_score = -mutant_score
        ##################################################
        
        mutant_save = tuple([residue_codebook.index(m[1]) for m in mutant])
        mutant_score = float(mutant_score)
        sequence_library.append((mutant_save, mutant_score))
        # break
    
    sequence_library = sorted(sequence_library, key=lambda x: x[1], reverse=True)
    
    if sequence_number > 0:
        return sequence_library[:sequence_number]
    else:
        return sequence_library 

def rank_for_MaskMP_C(model, alphabet, seq, sites, sequence_number, reverse, single_site):
    sequence_library = []
    if not isinstance(alphabet, EsmTokenizer):
        mask_token = alphabet.get_tok(alphabet.mask_idx)
        converter = alphabet.get_batch_converter()
    else:
        mask_token = alphabet.mask_token
        converter = alphabet
    mutant_library = make_mutant_library(sites, single_site)
    
    sequence_library = []
    for mutant in tqdm(mutant_library):
        mutant_score = 0
        mutant_sequence_list = list(seq)
        for m in mutant:
            mutant_sequence_list[int(m[0])] = m[1]
        mutant_sequence = "".join(mutant_sequence_list)
    
        mutant_masked_sequence_list = []
        for m in mutant:
            masked_sequence_list = list(mutant_sequence)
            masked_sequence_list[int(m[0])] = mask_token
            mutant_masked_sequence_list.append("".join(masked_sequence_list))
        print(mutant_masked_sequence_list)
        if isinstance(converter, EsmTokenizer):
            mask_tokens = converter([("", mutant_masked_sequence) for mutant_masked_sequence in mutant_masked_sequence_list])["input_ids"]
            mask_tokens = torch.Tensor(mask_tokens).long()
            mask_tokens = torch.cat([mask_tokens[:, 0:1], mask_tokens[:, 2:]], dim=1)
        else:
            _, _, mask_tokens = converter([("", mutant_masked_sequence) for mutant_masked_sequence in mutant_masked_sequence_list])
        
        if torch.cuda.is_available():
            mask_tokens = mask_tokens.cuda()
        with torch.no_grad():
            if isinstance(model, EsmForMaskedLM):
                result = model(mask_tokens)
            else:
                result = model(mask_tokens, return_contacts=False)
        for idx, (m, s) in enumerate(zip(mutant, sites)):
            # print(mask_tokens[idx, int(m[0])+1], s)
            probabilties = result["logits"][idx, int(m[0])+1, 4:24].softmax(-1)
            mutant_score += torch.log(probabilties[residue_codebook.index(m[1])]) - torch.log(probabilties[residue_codebook.index(s[1])])
        ################ try reverse ####################
        if reverse:
            mutant_score = -mutant_score
        ##################################################
        
        mutant_save = tuple([residue_codebook.index(m[1]) for m in mutant])
        mutant_score = float(mutant_score)
        sequence_library.append((mutant_save, mutant_score))
        # break
    
    sequence_library = sorted(sequence_library, key=lambda x: x[1], reverse=True)
    
    if sequence_number > 0:
        return sequence_library[:sequence_number]
    else:
        return sequence_library 

def rank_for_MutantMP(model, alphabet, seq, sites, sequence_number, reverse, single_site):
    sequence_library = []
    if not isinstance(alphabet, EsmTokenizer):
        mask_token = alphabet.get_tok(alphabet.mask_idx)
        converter = alphabet.get_batch_converter()
    else:
        mask_token = alphabet.mask_token
        converter = alphabet
    mutant_library = make_mutant_library(sites, single_site)

    sequence_library = []
    for mutant in tqdm(mutant_library):
        sequence_list = list(seq)
        for m in mutant:
            sequence_list[int(m[0])] = m[1]
        mutant_sequence = "".join(sequence_list)
            
        if isinstance(converter, EsmTokenizer):
            mutant_tokens = converter([("", mutant_sequence)])["input_ids"]
            mutant_tokens = torch.Tensor(mutant_tokens).long()
            mutant_tokens = torch.cat([mutant_tokens[:, 0:1], mutant_tokens[:, 2:]], dim=1)
        else:
            _, _, mutant_tokens = converter([("", mutant_sequence)])
        if torch.cuda.is_available():
            mutant_tokens = mutant_tokens.cuda()
        with torch.no_grad():
            if isinstance(model, EsmForMaskedLM):
                result = model(mutant_tokens)
            else:
                result = model(mutant_tokens, return_contacts=False)
        probabilties = result["logits"][0, [int(m[0])+1 for m in mutant], 4:24].softmax(-1)

        mutant_score = 0
        for idx, (mut, site) in enumerate(zip(mutant, sites)):
            assert mut[0] == site[0]
            mutant_score += torch.log(probabilties[idx, residue_codebook.index(mut[1])]) - torch.log(probabilties[idx, residue_codebook.index(site[1])])
        ################ try reverse ####################
        if reverse:
            mutant_score = -mutant_score
        ##################################################
        mutant_save = tuple([residue_codebook.index(m[1]) for m in mutant])
        mutant_score = float(mutant_score)
        sequence_library.append((mutant_save, mutant_score))
        
    sequence_library = sorted(sequence_library, key=lambda x: x[1], reverse=True)
        
    if sequence_number > 0:
        return sequence_library[:sequence_number]
    else:
        return sequence_library

def rank_for_WildtypeMP(model, alphabet, seq, sites, sequence_number, reverse, single_site):
    sequence_library = []
    if not isinstance(alphabet, EsmTokenizer):
        mask_token = alphabet.get_tok(alphabet.mask_idx)
        converter = alphabet.get_batch_converter()
    else:
        mask_token = alphabet.mask_token
        converter = alphabet
    mutant_library = make_mutant_library(sites, single_site)
    
    if isinstance(converter, EsmTokenizer):
        tokens = converter([("", seq)])["input_ids"]
        tokens = torch.Tensor(tokens).long()
        tokens = torch.cat([tokens[:, 0:1], tokens[:, 2:]], dim=1)
    else:
        _, _, tokens = converter([("", seq)])
    if torch.cuda.is_available():
        tokens = tokens.cuda()
    with torch.no_grad():
        if isinstance(model, EsmForMaskedLM):
            result = model(tokens)
        else:
            result = model(tokens, return_contacts=False)
    probabilties = result["logits"][0, [int(site[0])+1 for site in sites], 4:24].softmax(-1)
    
    sequence_library = []
    for mutant in tqdm(mutant_library):
        mutant_score = 0
        for idx, (mut, site) in enumerate(zip(mutant, sites)):
            assert mut[0] == site[0]
            mutant_score += torch.log(probabilties[idx, residue_codebook.index(mut[1])]) - torch.log(probabilties[idx, residue_codebook.index(site[1])])
        ################ try reverse ####################
        if reverse:
            mutant_score = -mutant_score
        ##################################################
        
        mutant_save = tuple([residue_codebook.index(m[1]) for m in mutant])
        mutant_score = float(mutant_score)
        sequence_library.append((mutant_save, mutant_score))
        
    sequence_library = sorted(sequence_library, key=lambda x: x[1], reverse=True)
    
    if sequence_number > 0:
        return sequence_library[:sequence_number]
    else:
        return sequence_library
    # print("last")
    # return sequence_library[-sequence_number:]

def rank_for_AutoRegressive_A(model, alphabet, seq, sites, sequence_number, reverse, single_site):
    sequence_library = []
    if not isinstance(alphabet, EsmTokenizer):
        mask_token = alphabet.get_tok(alphabet.mask_idx)
        converter = alphabet.get_batch_converter()
    else:
        mask_token = alphabet.mask_token
        converter = alphabet
    mutant_library = make_mutant_library(sites, single_site)
    
    sequence_library = []
    for mutant in tqdm(mutant_library):
        mutant_score = 0
        mutant_masked_sequence_list = []
        for m_idx in range(len(mutant)):
            mutant_sequence_list = list(seq)
            for m in mutant[:m_idx]:
                mutant_sequence_list[int(m[0])] = m[1]
            mutant_sequence_list[int(mutant[m_idx][0])] = mask_token
            mutant_masked_sequence_list.append("".join(mutant_sequence_list))
            
        if isinstance(converter, EsmTokenizer):
            mask_tokens = converter([("", mutant_masked_sequence) for mutant_masked_sequence in mutant_masked_sequence_list])["input_ids"]
            mask_tokens = torch.Tensor(mask_tokens).long()
            mask_tokens = torch.cat([mask_tokens[:, 0:1], mask_tokens[:, 2:]], dim=1)
        else:
            _, _, mask_tokens = converter([("", mutant_masked_sequence) for mutant_masked_sequence in mutant_masked_sequence_list])
        if torch.cuda.is_available():
            mask_tokens = mask_tokens.cuda()
        with torch.no_grad():
            if isinstance(model, EsmForMaskedLM):
                result = model(mask_tokens)
            else:
                result = model(mask_tokens, return_contacts=False)
        for idx, (m, s) in enumerate(zip(mutant, sites)):
            # print(mask_tokens[idx, int(m[0])+1], s)
            probabilties = result["logits"][idx, int(m[0])+1, 4:24].softmax(-1)
            mutant_score += torch.log(probabilties[residue_codebook.index(m[1])]) - torch.log(probabilties[residue_codebook.index(s[1])])
        ################ try reverse ####################
        if reverse:
            mutant_score = -mutant_score
        ##################################################
        
        mutant_save = tuple([residue_codebook.index(m[1]) for m in mutant])
        mutant_score = float(mutant_score)
        sequence_library.append((mutant_save, mutant_score))
        # break
    
    sequence_library = sorted(sequence_library, key=lambda x: x[1], reverse=True)
    
    if sequence_number > 0:
        return sequence_library[:sequence_number]
    else:
        return sequence_library 

def rank_for_AutoRegressive_B(model, alphabet, seq, sites, sequence_number, reverse, single_site):
    sequence_library = []
    if not isinstance(alphabet, EsmTokenizer):
        mask_token = alphabet.get_tok(alphabet.mask_idx)
        converter = alphabet.get_batch_converter()
    else:
        mask_token = alphabet.mask_token
        converter = alphabet
    mutant_library = make_mutant_library(sites, single_site)
    
    masked_probabilities = []
    for site in sites:
        sequence_list = list(seq)
        sequence_list[int(site[0])] = mask_token
        masked_sequence = "".join(sequence_list)
        if isinstance(converter, EsmTokenizer):
            mask_tokens = converter([("", masked_sequence)])["input_ids"]
            mask_tokens = torch.Tensor(mask_tokens).long()
            mask_tokens = torch.cat([mask_tokens[:, 0:1], mask_tokens[:, 2:]], dim=1)
        else:
            _, _, mask_tokens = converter([("", masked_sequence)])
        
        if torch.cuda.is_available():
            mask_tokens = mask_tokens.cuda()
        with torch.no_grad():
            if isinstance(model, EsmForMaskedLM):
                result = model(mask_tokens)
            else:
                result = model(mask_tokens, return_contacts=False)
        probabilties = result["logits"][0, int(site[0])+1, 4:24].softmax(-1)
        masked_probabilities.append(torch.log(probabilties[residue_codebook.index(site[1])]))
    
    sequence_library = []
    for mutant in tqdm(mutant_library):
        mutant_score = 0
        mutant_masked_sequence_list = []
        for m_idx in range(len(mutant)):
            mutant_sequence_list = list(seq)
            for m in mutant[:m_idx]:
                mutant_sequence_list[int(m[0])] = m[1]
            mutant_sequence_list[int(mutant[m_idx][0])] = mask_token
            mutant_masked_sequence_list.append("".join(mutant_sequence_list))
            
        if isinstance(converter, EsmTokenizer):
            mask_tokens = converter([("", mutant_masked_sequence) for mutant_masked_sequence in mutant_masked_sequence_list])["input_ids"]
            mask_tokens = torch.Tensor(mask_tokens).long()
            mask_tokens = torch.cat([mask_tokens[:, 0:1], mask_tokens[:, 2:]], dim=1)
        else:
            _, _, mask_tokens = converter([("", mutant_masked_sequence) for mutant_masked_sequence in mutant_masked_sequence_list])
        
        if torch.cuda.is_available():
            mask_tokens = mask_tokens.cuda()
        with torch.no_grad():
            if isinstance(model, EsmForMaskedLM):
                result = model(mask_tokens)
            else:
                result = model(mask_tokens, return_contacts=False)
        for idx, (m, m_p) in enumerate(zip(mutant, masked_probabilities)):
            # print(mask_tokens[idx, int(m[0])+1])
            probabilties = result["logits"][idx, int(m[0])+1, 4:24].softmax(-1)
            mutant_score += torch.log(probabilties[residue_codebook.index(m[1])]) - m_p
        ################ try reverse ####################
        if reverse:
            mutant_score = -mutant_score
        ##################################################
        
        mutant_save = tuple([residue_codebook.index(m[1]) for m in mutant])
        mutant_score = float(mutant_score)
        sequence_library.append((mutant_save, mutant_score))
        # break
    
    sequence_library = sorted(sequence_library, key=lambda x: x[1], reverse=True)
    
    if sequence_number > 0:
        return sequence_library[:sequence_number]
    else:
        return sequence_library 

def rank_for_AutoRegressive_C(model, alphabet, seq, sites, sequence_number, reverse, single_site):
    sequence_library = []
    if not isinstance(alphabet, EsmTokenizer):
        mask_token = alphabet.get_tok(alphabet.mask_idx)
        converter = alphabet.get_batch_converter()
    else:
        mask_token = alphabet.mask_token
        converter = alphabet
    mutant_library = make_mutant_library(sites, single_site)
    
    sequence_library = []
    for mutant in tqdm(mutant_library):
        mutant_score = 0
        mutant_masked_sequence_list = []
        for m_idx in range(len(mutant)):
            mutant_sequence_list = list(seq)
            for m in mutant[:m_idx]:
                mutant_sequence_list[int(m[0])] = m[1]
            for m in mutant[m_idx:]:
                mutant_sequence_list[int(m[0])] = mask_token
            mutant_masked_sequence_list.append("".join(mutant_sequence_list))
            
        if isinstance(converter, EsmTokenizer):
            mask_tokens = converter([("", mutant_masked_sequence) for mutant_masked_sequence in mutant_masked_sequence_list])["input_ids"]
            mask_tokens = torch.Tensor(mask_tokens).long()
            mask_tokens = torch.cat([mask_tokens[:, 0:1], mask_tokens[:, 2:]], dim=1)
        else:
            _, _, mask_tokens = converter([("", mutant_masked_sequence) for mutant_masked_sequence in mutant_masked_sequence_list])
        
        
        if torch.cuda.is_available():
            mask_tokens = mask_tokens.cuda()
        with torch.no_grad():
            if isinstance(model, EsmForMaskedLM):
                result = model(mask_tokens)
            else:
                result = model(mask_tokens, return_contacts=False)
        for idx, (m, s) in enumerate(zip(mutant, sites)):
            # print(mask_tokens[idx, int(m[0])+1], s)
            probabilties = result["logits"][idx, int(m[0])+1, 4:24].softmax(-1)
            mutant_score += torch.log(probabilties[residue_codebook.index(m[1])]) - torch.log(probabilties[residue_codebook.index(s[1])])
        ################ try reverse ####################
        if reverse:
            mutant_score = -mutant_score
        ##################################################
        
        mutant_save = tuple([residue_codebook.index(m[1]) for m in mutant])
        mutant_score = float(mutant_score)
        sequence_library.append((mutant_save, mutant_score))
        # break
    
    sequence_library = sorted(sequence_library, key=lambda x: x[1], reverse=True)
    
    if sequence_number > 0:
        return sequence_library[:sequence_number]
    else:
        return sequence_library 


def rank_for_AutoRegressive_D(model, alphabet, seq, sites, sequence_number, reverse, single_site):
    sequence_library = []
    if not isinstance(alphabet, EsmTokenizer):
        mask_token = alphabet.get_tok(alphabet.mask_idx)
        converter = alphabet.get_batch_converter()
    else:
        mask_token = alphabet.mask_token
        converter = alphabet
    mutant_library = make_mutant_library(sites, single_site)
    
    masked_probabilities = []
    for site_idx in range(len(sites)-1, -1, -1):
        sequence_list = list(seq)
        for site in sites[site_idx:]:
            sequence_list[int(site[0])] = mask_token
        masked_sequence = "".join(sequence_list)
        if isinstance(converter, EsmTokenizer):
            mask_tokens = converter([("", masked_sequence)])["input_ids"]
            mask_tokens = torch.Tensor(mask_tokens).long()
            mask_tokens = torch.cat([mask_tokens[:, 0:1], mask_tokens[:, 2:]], dim=1)
        else:
            _, _, mask_tokens = converter([("", masked_sequence)])
        
        
        if torch.cuda.is_available():
            mask_tokens = mask_tokens.cuda()
        with torch.no_grad():
            if isinstance(model, EsmForMaskedLM):
                result = model(mask_tokens)
            else:
                result = model(mask_tokens, return_contacts=False)
        probabilties = result["logits"][0, int(sites[site_idx][0])+1, 4:24].softmax(-1)
        masked_probabilities.append(torch.log(probabilties[residue_codebook.index(site[1])]))
    masked_probabilities = masked_probabilities[::-1]
    
    sequence_library = []
    for mutant in tqdm(mutant_library):
        mutant_score = 0
        mutant_masked_sequence_list = []
        for m_idx in range(len(mutant)):
            mutant_sequence_list = list(seq)
            for m in mutant[:m_idx]:
                mutant_sequence_list[int(m[0])] = m[1]
            mutant_sequence_list[int(mutant[m_idx][0])] = mask_token
            mutant_masked_sequence_list.append("".join(mutant_sequence_list))
        if isinstance(converter, EsmTokenizer):
            mask_tokens = converter([("", mutant_masked_sequence) for mutant_masked_sequence in mutant_masked_sequence_list])["input_ids"]
            mask_tokens = torch.Tensor(mask_tokens).long()
            mask_tokens = torch.cat([mask_tokens[:, 0:1], mask_tokens[:, 2:]], dim=1)
        else:
            _, _, mask_tokens = converter([("", mutant_masked_sequence) for mutant_masked_sequence in mutant_masked_sequence_list])
        
        if torch.cuda.is_available():
            mask_tokens = mask_tokens.cuda()
        with torch.no_grad():
            if isinstance(model, EsmForMaskedLM):
                result = model(mask_tokens)
            else:
                result = model(mask_tokens, return_contacts=False)
        for idx, (m, m_p) in enumerate(zip(mutant, masked_probabilities)):
            # print(mask_tokens[idx, int(m[0])+1])
            probabilties = result["logits"][idx, int(m[0])+1, 4:24].softmax(-1)
            mutant_score += torch.log(probabilties[residue_codebook.index(m[1])]) - m_p
        ################ try reverse ####################
        if reverse:
            mutant_score = -mutant_score
        ##################################################
        
        mutant_save = tuple([residue_codebook.index(m[1]) for m in mutant])
        mutant_score = float(mutant_score)
        sequence_library.append((mutant_save, mutant_score))
        # break
    
    sequence_library = sorted(sequence_library, key=lambda x: x[1], reverse=True)
    
    if sequence_number > 0:
        return sequence_library[:sequence_number]
    else:
        return sequence_library 


def combine_seq_sites(seq, sites):
    seq_list = list(seq)
    for site in sites:
        seq_list[int(site[0])] = site[1]
    return "".join(seq_list)

def extract_probability(model, converter, mutant_sequences, mutants_list):
    assert isinstance(mutant_sequences, list)
    if len(mutant_sequences) == 0 or len(mutants_list) == 0:
        return None
    _, _, tokens = converter([("", mutant_sequence) for mutant_sequence in mutant_sequences])
    if torch.cuda.is_available():
        tokens = tokens.cuda()
    with torch.no_grad():
        result = model(tokens, return_contacts=False)
        
    probabilities = torch.zeros(len(mutant_sequences), len(mutants_list[0]), 20)
    for idx, (seq, mutants) in enumerate(zip(mutant_sequences, mutants_list)):
        probabilities[idx] = result["logits"][idx, [int(m[0])+1 for m in mutants], 4:24].softmax(-1).cpu()

    """
tensor([[[0.0307, 0.0160, 0.0141, 0.7843, 0.0091, 0.0141, 0.0085, 0.0090,
          0.0199, 0.0123, 0.0117, 0.0091, 0.0058, 0.0051, 0.0187, 0.0131,
          0.0042, 0.0043, 0.0063, 0.0037],
         [0.0206, 0.0259, 0.0135, 0.0273, 0.0294, 0.0289, 0.0204, 0.0342,
          0.0128, 0.6782, 0.0245, 0.0263, 0.0147, 0.0128, 0.0069, 0.0066,
          0.0038, 0.0082, 0.0025, 0.0023],
         [0.0307, 0.0263, 0.7447, 0.0481, 0.0076, 0.0084, 0.0091, 0.0091,
          0.0194, 0.0070, 0.0092, 0.0077, 0.0038, 0.0030, 0.0222, 0.0170,
          0.0064, 0.0054, 0.0067, 0.0081],
         [0.0267, 0.0357, 0.0173, 0.6147, 0.0170, 0.0311, 0.0516, 0.0165,
          0.0072, 0.0219, 0.0368, 0.0459, 0.0283, 0.0087, 0.0113, 0.0086,
          0.0061, 0.0075, 0.0042, 0.0031]],

        [[0.8327, 0.0136, 0.0115, 0.0228, 0.0069, 0.0100, 0.0066, 0.0064,
          0.0169, 0.0081, 0.0083, 0.0064, 0.0045, 0.0034, 0.0160, 0.0112,
          0.0029, 0.0037, 0.0047, 0.0034],
         [0.0221, 0.0275, 0.0142, 0.0265, 0.0304, 0.0278, 0.0228, 0.0323,
          0.0125, 0.6690, 0.0283, 0.0255, 0.0168, 0.0134, 0.0068, 0.0064,
          0.0037, 0.0090, 0.0023, 0.0026],
         [0.0233, 0.0245, 0.7845, 0.0352, 0.0079, 0.0087, 0.0091, 0.0085,
          0.0143, 0.0073, 0.0099, 0.0075, 0.0040, 0.0030, 0.0165, 0.0135,
          0.0050, 0.0050, 0.0053, 0.0068],
         [0.0274, 0.0350, 0.0176, 0.6153, 0.0166, 0.0310, 0.0528, 0.0162,
          0.0074, 0.0214, 0.0367, 0.0444, 0.0283, 0.0086, 0.0116, 0.0089,
          0.0059, 0.0077, 0.0042, 0.0030]],

        [[0.8290, 0.0136, 0.0102, 0.0201, 0.0089, 0.0110, 0.0118, 0.0102,
          0.0119, 0.0076, 0.0065, 0.0089, 0.0070, 0.0056, 0.0120, 0.0097,
          0.0031, 0.0056, 0.0038, 0.0036],
         [0.7002, 0.0261, 0.0155, 0.0274, 0.0237, 0.0233, 0.0206, 0.0243,
          0.0130, 0.0192, 0.0323, 0.0205, 0.0124, 0.0116, 0.0079, 0.0068,
          0.0032, 0.0066, 0.0024, 0.0027],
         [0.0162, 0.0207, 0.7872, 0.0165, 0.0135, 0.0131, 0.0152, 0.0108,
          0.0067, 0.0116, 0.0203, 0.0152, 0.0090, 0.0055, 0.0081, 0.0079,
          0.0029, 0.0130, 0.0025, 0.0041],
         [0.0279, 0.0332, 0.0184, 0.6478, 0.0158, 0.0276, 0.0455, 0.0149,
          0.0079, 0.0183, 0.0333, 0.0356, 0.0251, 0.0079, 0.0123, 0.0086,
          0.0055, 0.0073, 0.0037, 0.0033]],

        [[0.7402, 0.0199, 0.0202, 0.0217, 0.0150, 0.0234, 0.0201, 0.0145,
          0.0113, 0.0175, 0.0108, 0.0151, 0.0140, 0.0102, 0.0117, 0.0118,
          0.0037, 0.0091, 0.0041, 0.0057],
         [0.6078, 0.0273, 0.0183, 0.0278, 0.0281, 0.0324, 0.0336, 0.0362,
          0.0092, 0.0253, 0.0605, 0.0270, 0.0201, 0.0137, 0.0063, 0.0063,
          0.0034, 0.0112, 0.0023, 0.0032],
         [0.7434, 0.0227, 0.0211, 0.0224, 0.0147, 0.0178, 0.0190, 0.0144,
          0.0088, 0.0139, 0.0245, 0.0146, 0.0116, 0.0066, 0.0111, 0.0117,
          0.0035, 0.0102, 0.0035, 0.0046],
         [0.0279, 0.0350, 0.0180, 0.6281, 0.0180, 0.0347, 0.0469, 0.0166,
          0.0077, 0.0228, 0.0369, 0.0367, 0.0269, 0.0089, 0.0092, 0.0066,
          0.0053, 0.0076, 0.0032, 0.0032]]])"""
    return probabilities

def rank_for_AutoRegressive_E(model, alphabet, seq, sites, sequence_number, reverse, single_site):
    sequence_library = []
    mask_token = alphabet.get_tok(alphabet.mask_idx)
    mutant_library = make_mutant_library(sites, single_site)
    converter = alphabet.get_batch_converter()
    
    # print(combine_seq_sites("ABCDEFG", [("0", "D"), ("1", "E"), ("2", "F")]))
    print(sites)
    new_mutant_dict = {}
    for mutant in tqdm(mutant_library):
        mutant = list(mutant)
        # print(mutant)
        mutant_sequences = []
        new_mutants = []
        for idx in range(len(mutant)):
            new_mutant = tuple(mutant[:idx] + sites[idx:])
            if new_mutant in new_mutant_dict:
                continue
            mutant_sequence = combine_seq_sites(seq, new_mutant)
            new_mutants.append(new_mutant)
            mutant_sequences.append(mutant_sequence)
            
        probabilities = extract_probability(model, converter, mutant_sequences, new_mutants)
        for idx, n_m in enumerate(new_mutants):
            if probabilities is None:
                break
            new_mutant_dict[tuple(n_m)] = probabilities[idx]
        # print(new_mutant_dict)
        # break
    
    sequence_library = []
    for mutant in tqdm(mutant_library):
        mutant = list(mutant)
        mutant_score = 0
        new_mutants = []
        for idx in range(len(mutant)):
            new_mutant = tuple(mutant[:idx] + sites[idx:])
            new_mutant_prob = new_mutant_dict[new_mutant]
            # mutant_score += torch.log(new_mutant_prob[idx, residue_codebook.index(mutant[idx][1])]) - torch.log(new_mutant_prob[idx, residue_codebook.index(sites[idx][1])])
            mutant_score += torch.log(new_mutant_prob[idx, residue_codebook.index(mutant[idx][1])])
        sequence_library.append(([residue_codebook.index(m[1]) for m in mutant], float(mutant_score)))
        # break
        
    sequence_library = sorted(sequence_library, key=lambda x: x[1], reverse=True)
    
    if sequence_number > 0:
        return sequence_library[:sequence_number]
    else:
        return sequence_library