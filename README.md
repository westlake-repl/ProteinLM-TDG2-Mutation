# ProteinLM-TDG2-Mutation

## Install enviroment

Install the conda environment with the following command:

```bash
bash env.sh
```

## Create single mutations

First, rank your own data by feeding it into the `rank_all.py` script. 

Note that you should change the `--sequence-file` and `--output-file` arguments to your own data.

```bash
python rank_all.py --sequence-file data/datadir/filename.fa --rank-strategy esm1v_5 --output-file data/datadir/filename_mutate_one_allsites.csv --model-name esm2_t33_650M_UR50D --single-site -f
```

Then, you can generate the landscape of your data by feeding it into the `generate_landscape.py` script.

```bash
python generate_landscape.py --mutate_one_sites data/datadir/filename_mutate_one_allsites.csv --figsize "(250,10)" --prefix filename --output data/datadir/filename_full_landscape.png
```

## Benchmark in dms dataset

If you want to benchmark your model in the dms dataset, you can use the `rank_all_dms.py` script.

```bash
python rank_all_dms.py --dms-dir data/dms --model-name model_name --rank-strategy rank_strategy
```

For your convenience, we provide the `rank_all_dms.sh` script to run the benchmark in all models and rank strategies. 

```bash
bash rank_all_dms.sh
```