# Slice-Specific Phase Selection for Fast MR Scanning with Transformer

This repository contains scripts to replicate the experiments performed in:



[Yiming Liu, Yanwei Pang, Ruiqi Jin, Zhenchang Wang, 
"Active Phase-Encode Selection for Slice-Specific Fast MR Scanning Using a Transformer-Based Deep Reinforcement Learning Framework".](https://arxiv.org/abs/2203.05756.pdf)

# Data & Code Resources

The code is mainly modified on
[active-mri-acquisition](https://github.com/facebookresearch/active-mri-acquisition)
and
[fastMRI repository](https://github.com/facebookresearch/fastMRI).

Data is available at
[fastMRI dataset](http://fastmri.med.nyu.edu/), and data of size [640, 368] can be selected according to the files in ./Phase_selection/activemri/data/splits/knee_singlecoil

# Getting started

## Installation

We have tested the Python environment setting below, newer versions of packages may also work well.
```bash
conda create --name title python=3.7
```

```bash
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
```

```bash
cd title
pip install -e .
```

## Configuration

Setting your path of dataset, reconstruction model, and the checkpoint.
```bash
"path/to/TITLE/Phase_selection/configs/.activemri/defaults.json".
```

### Train TITLE

```bash
cd ./Phase_selection
./examples/train_ddqn.sh ./checkpoint/TITLE/ miccai 4 ss_maskfusedtransformer 0
```
### Evaluation for TITLE

```bash
python ./examples/run_evaluation.py --baseline ss-ddqn --env miccai
```

#### Note
Inference of TITLE does not require the computation of reward metrics. The 9ms performance in the paper is obtained without calculating reward metrics. 

### Train Unet
Unet reconstruction related to sampling pattern should provide the sampling pattern for each slice.
```bash
cd ../Reconstruction
python ./fastmri_examples/unet/train_unet_demo.py --trajectory_dir "../Phase_selection/trajectories4IFTRecon/TIPS/TIPS_trajectories.pkl"
```

# Results Reproduction

The data used in the results will be obtained from the experiments described above, and the code for the data analysis is provided in ./Phase_selection/Data_analysis. 
