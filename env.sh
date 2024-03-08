# DONNOT run the file directly. Instead, run the following command line by line in the terminal.
conda create -n TDG2 python=3.9 -y
conda activate TDG2
conda install numpy biopython tqdm pandas matplotlib seaborn -y
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia -y
pip install fair-esm
pip install transformers
