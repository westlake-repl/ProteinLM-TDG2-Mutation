# ProteinLM-TDG2-Mutation

Code for paper: 📖**Protein language models-assisted optimization of a uracil-N-glycosylase variant enables programmable T-to-G and T-to-C base editing**

DOI: 🔗[https://doi.org/10.1016/j.molcel.2024.01.021](https://doi.org/10.1016/j.molcel.2024.01.021)

## Install enviroment

Install the conda environment refering to the `env.sh` script. 

**Note that**, *DONNOT* run the `env.sh` script directly. Instead, copy the commands in the `env.sh` script and run them in your terminal.

## Download ESM series models

You can download the ESM series models from the [huggingface model hub](https://huggingface.co) by searching `facebook/model_name` in the search bar.

Available models include:

| Model name | link |
| --- | --- |
| esm1b_t33_650M_UR50S | [link](https://huggingface.co/facebook/esm1b_t33_650M_UR50S) |
| esm1v_t33_650M_UR90S_1 | [link](https://huggingface.co/facebook/esm1v_t33_650M_UR90S_1) |
| esm1v_t33_650M_UR90S_2 | [link](https://huggingface.co/facebook/esm1v_t33_650M_UR90S_2) |
| esm1v_t33_650M_UR90S_3 | [link](https://huggingface.co/facebook/esm1v_t33_650M_UR90S_3) |
| esm1v_t33_650M_UR90S_4 | [link](https://huggingface.co/facebook/esm1v_t33_650M_UR90S_4) |
| esm1v_t33_650M_UR90S_5 | [link](https://huggingface.co/facebook/esm1v_t33_650M_UR90S_5) |
| esm2_t6_8M_UR50D | [link](https://huggingface.co/facebook/esm2_t6_8M_UR50D) |
| esm2_t12_35M_UR50D | [link](https://huggingface.co/facebook/esm2_t12_35M_UR50D) |
| esm2_t30_150M_UR50D | [link](https://huggingface.co/facebook/esm2_t30_150M_UR50D) |
| esm2_t33_650M_UR50D | [link](https://huggingface.co/facebook/esm2_t33_650M_UR50D) |
| esm2_t36_3B_UR50D | [link](https://huggingface.co/facebook/esm2_t36_3B_UR50D) |
| esm2_t48_15B_UR50D | [link](https://huggingface.co/facebook/esm2_t48_15B_UR50D) |

**Note that**: For those who cannot access the huggingface model hub, you can download the models from the [facebookresearch/esm](https://github.com/facebookresearch/esm).

We recommend you put the downloaded models in the `Model` directory, but this is not compulsory. No matter where you put the models, you should change the specified model path in the `rank_all.py`.

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
