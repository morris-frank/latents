#! /bin/sh

set -e

python_version="3.8"
env_name=latents
conda_bin=$HOME/conda/bin/conda

eval "$(${conda_bin} shell.bash hook)"

# Remove the hbdx enviroment if already installed
if conda env list | grep -q $env_name; then
    conda deactivate
    conda env remove --name $env_name
fi

conda create -n $env_name python="${python_version}" -y
conda activate $env_name

conda install -y pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

conda install mamba -c conda-forge -y

pip install tensorflow h5py"==2.10.0"
pip install torch torchvision torchaudio
mamba install -c conda-forge pytorch-lightning kornia ftfy rich tqdm pandas scipy ipython omegaconf regex ipdb -y

# mkdir lib
git clone https://github.com/openai/CLIP.git lib/clip
git clone https://github.com/CompVis/taming-transformers  lib/taming-transformers
git clone https://github.com/idealo/image-super-resolution lib/image-super-resolution
