# ProteinLM-TDG2-Mutation

Code for paper: 📖**Protein language models-assisted optimization of a uracil-N-glycosylase variant enables programmable T-to-G and T-to-C base editing**

DOI: 🔗[https://doi.org/10.1016/j.molcel.2024.01.021](https://doi.org/10.1016/j.molcel.2024.01.021)

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

# Citation

If our work has been of assistance to your work, please cite our paper as :  

```
@article{he2024protein,
  title={Protein language models-assisted optimization of a uracil-N-glycosylase variant enables programmable T-to-G and T-to-C base editing},
  author={He, Yan and Zhou, Xibin and Chang, Chong and Chen, Ge and Liu, Weikuan and Li, Geng and Fan, Xiaoqi and Sun, Mingsun and Miao, Chensi and Huang, Qianyue and others},
  journal={Molecular Cell},
  year={2024},
  publisher={Elsevier}
}
```
