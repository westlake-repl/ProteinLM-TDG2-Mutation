import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import seaborn as sns

aa_list = list("ACDEFGHIKLMNPQRSTVWY")
def plot_landscape(mutate_one_sites, sites=None, figsize=None, prefix=None):
    if figsize is None:
        fig, ax = plt.subplots(figsize=(20, 10))
    else:
        fig, ax = plt.subplots(figsize=figsize)

    if prefix is None:
        prefix = ""
    # plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    assert "mut" in mutate_one_sites.columns and "fitness" in mutate_one_sites.columns, f"mutate_one_sites should have columns 'mut' and 'fitness', but got {mutate_one_sites.columns}"
    records = []
    for mut, fitness in tqdm(mutate_one_sites[["mut", "fitness"]].values):
        if mut == "WT":
            continue
        if sites is not None:
            if int(mut[1:-1]) not in sites:
                continue
        records.append({
            "site": mut[:-1],
            "mut": mut[-1],
            "fitness": fitness
        })

    site_dict = {}
    for record in records:
        site_dict.setdefault(record["site"], {})
        site_dict[record["site"]][record["mut"]] = record["fitness"]
    
    # check if all sites have all mutations
    for site, mut_dict in site_dict.items():
        for mut in aa_list:
            if mut not in mut_dict:
                mut_dict[mut] = None
    
    # sort by site
    sorted_site_dict = {k: v for k, v in sorted(list(site_dict.items()), key=lambda item: int(item[0][1:]))}

    # plot heatmap
    df = pd.DataFrame(sorted_site_dict)
    df = df.loc[aa_list]
    df = df.sort_index()
    sns.heatmap(df, ax=ax, cmap="RdBu_r", center=0, cbar_kws={"label": "Fitness"}, annot=True, fmt=".2f", annot_kws={"size": 8})
    ax.set_xlabel("Mutation")
    ax.set_ylabel("Site")
    if sites is None:
        ax.set_title(f"{prefix} full landscape")
    else:
        ax.set_title(f"{prefix} landscape for\n {','.join([str(s) for s in sites])}")
    # compact the figure
    fig.tight_layout()
    return fig, ax


import argparse
def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Generate landscape for a given protein")
    parser.add_argument("--mutate_one_sites", type=str, required=True, help="Path to the csv file containing the fitness of each mutation")
    parser.add_argument("--sites", type=str, default=None, help="Sites to plot, separated by comma")
    parser.add_argument("--figsize", type=str, default=None, help="Figure size, separated by comma, for example: (20,10)")
    parser.add_argument("--prefix", type=str, default=None, help="Prefix of the plot")
    parser.add_argument("--output", type=str, default=None, help="Path to save the plot")
    return parser.parse_args(args)

if __name__ == "__main__":
    args = parse_args()
    mutate_one_sites_path = Path(args.mutate_one_sites)
    mutate_one_sites = pd.read_csv(mutate_one_sites_path, sep=",")

    figsize = eval(args.figsize) if args.figsize is not None else None
    assert figsize is None or (isinstance(figsize, tuple) and len(figsize) == 2), f"figsize should be a tuple of length 2, but got {figsize}"

    # 打印全长的序列
    fig, ax = plot_landscape(mutate_one_sites, sites=args.sites, figsize=figsize, prefix=args.prefix)
    fig.savefig(args.output)