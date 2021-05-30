#! /bin/sh

mamba install -c conda-forge pytorch-lightning kornia ftfy

mkdir lib
mkdir -p models/vqgan

git clone https://github.com/openai/CLIP.git lib/clip
git clone https://github.com/CompVis/taming-transformers  lib/taming-transformers

wget 'https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1' -O 'models/vqgan/last.ckpt' 
wget 'https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1' -O 'models/vqgan/model.yaml' 
