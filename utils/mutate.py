import itertools

import esm
import torch
import torch.nn.functional as F

strategy_libary = [
    "mutate_all",
    "mutate_one",
    "mutate_two",
    "ga",
]

def mask_sequence(sequence, sites, site_num, mask_token):
    assert site_num <= len(sites), f"site_num {site_num} is larger than the number of sites {len(sites)}"
    masked_sites = [tuple([int(site[0]) for site in sites]) for sites in itertools.combinations(sites, site_num)]
    masked_sequences = []
    for masked_site in masked_sites:
        sequence_list = list(sequence)
        for site in masked_site:
            sequence_list[site] = mask_token
        masked_sequences.append("".join(sequence_list))
    return masked_sequences, masked_sites

def mutate(sequence, sites, site_num, model_name):
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name if isinstance(model_name, str) else str(model_name))
    converter = alphabet.get_batch_converter()
    
    masked_token = alphabet.get_tok(alphabet.mask_idx)
    masked_sequences, masked_sites = mask_sequence(sequence, sites, site_num, masked_token)
    
    mutations = {}
    for masked_sequence, site in zip(masked_sequences, masked_sites):
        _, _, token = converter([(str(site), masked_sequence)])
        with torch.no_grad():
            results = model(token)
        mutations[site] = F.softmax(results["logits"][0, [s+1 for s in site], 4:24], dim=-1)
    return mutations
    
def split_mutation_dicts(mutation_dicts, sites=None):
    mutation_site_tensors = {}
    mutation_sites = {}
    if sites is None:
        sites_keys = set([site for k in mutation_dicts.keys() for site in k])
    else:
        sites_keys = set([int(site[0]) for site in sites])
    
    sites_keys = sorted(list(sites_keys))
    
    for key in sites_keys:
        mutation_site_tensors[key] = []
        mutation_sites[key] = []
    
    for k, v in mutation_dicts.items():
        for idx, site in enumerate(k):
            mutation_site_tensors[site].append(v[idx])
            mutation_sites[site].append(k)
            
    for k, v in mutation_site_tensors.items():
        mutation_site_tensors[k] = torch.stack(v)
        
    return mutation_site_tensors, mutation_sites

def calculate_weights(mutation_sites, pdb):
    def _calculate_weight(site, pdb):
        if pdb is None:
            return torch.tensor(1.0)
        # use only backbone information
        site_tensor = torch.index_select(input=torch.tensor(pdb), dim=0, index=torch.tensor(site))[:, :4, :]
        # use backbone mean as residue coordinate
        site_tensor = site_tensor.mean(1)
        # use mean of distance to center
        site_mean = site_tensor.mean(0)
        site_weight = (site_tensor - site_mean).pow(2).sum(-1).sqrt().mean()
        return site_weight
    mutation_site_weights = {}
    for site, sites in mutation_sites.items():
        mutation_site_weights[site] = []
        for s in sites:
            weight = _calculate_weight(s, pdb)
            mutation_site_weights[site].append(weight)
        mutation_site_weights[site] = F.softmax(torch.tensor(mutation_site_weights[site]), dim=-1)
        
    return mutation_site_weights

def collect_mutation_profiles(mutation_site_tensors, mutation_site_weights):
    # print(mutation_site_tensors)
    mutation_profiles = []
    for site in mutation_site_tensors.keys():
        mutation_profiles.append((torch.mm(mutation_site_weights[site].unsqueeze(0).float(), mutation_site_tensors[site].float())).squeeze(0))
    return torch.stack(mutation_profiles)

def mutate_one(sequence, sites, pdb, model_name):
    mutation_dicts = mutate(sequence, sites, 1, model_name)
    mutation_site_tensors, mutation_sites = split_mutation_dicts(mutation_dicts)
    mutation_site_weights = calculate_weights(mutation_sites, pdb)
    mutation_profiles = collect_mutation_profiles(mutation_site_tensors, mutation_site_weights)
    return mutation_profiles
    


def mutate_two(sequence, sites, pdb, model_name):
    mutation_dicts = mutate(sequence, sites, 2, model_name)
    mutation_site_tensors, mutation_sites = split_mutation_dicts(mutation_dicts)
    mutation_site_weights = calculate_weights(mutation_sites, pdb)
    mutation_profiles = collect_mutation_profiles(mutation_site_tensors, mutation_site_weights)
    return mutation_profiles
    


def mutate_all(sequence, sites, pdb, model_name):
    mutation_dicts = mutate(sequence, sites, len(sites), model_name)
    mutation_site_tensors, mutation_sites = split_mutation_dicts(mutation_dicts)
    mutation_site_weights = calculate_weights(mutation_sites, pdb)
    mutation_profiles = collect_mutation_profiles(mutation_site_tensors, mutation_site_weights)
    return mutation_profiles