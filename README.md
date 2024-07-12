# On the Expressive Power of Sparse Geometric MPNNs

## Table of Contents

- [About](#about)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)
- [Authors](#authors)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yonatansverdlov/E-GenNet/blob/master/k_chains_baselines.ipynb)

## About

Motivated by applications in chemistry and other sciences, we study the expressive power of message-passing neural networks for geometric graphs. 

This reposetory illustrates our findings via sparse separation, power graph experiments, and prediction of chemical properties.

## Installation

# Conda
```bash
conda create --name egenet -c conda-forge python=3.11

conda activate egenet

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install torch_geometric

pip install pytorch-lightning

pip install easydict

pip install pandas

pip install rdkit
```

## Usage
- Sanity Check
For a sanity check of our model, please run 
```bash
python sanity_check.py
```
## Synthetic experiments
In this section, we illustrate our ability to separate the sparsest graphs and show that our model can separate pairs that no I-GGNN can separate and a pair that no I-GGNN can separate even when considering cross edges.
We illustrate our results here and the baselines via Google Colab.
Our experiments show we succeeded in separating the k-chain pairs for ten different seeds.
Next, we show that our model can separate pair A and B, but I-GGNN struggles to separate them.
To run our results, please run the following:
```bash
python k_chain_exps.py
```
For baselines, we add a link to the Colab notebook illustrating our tasks, results and baselines.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yonatansverdlov/E-GenNet/blob/master/k_chains_baselines.ipynb)
## Chemical properties 
Here, we present our results on predicting chemical properties and our train/test/val accuracy.
- Data:
Please download the Kraken, Drugs, and BDE datasets from https://github.com/SXKDZ/MARCEL and put them in /data.

- Training:
To run, use python chemical_property_exps.py --dataset_name name --task task.

- options:
The options are Drugs, Kraken, BDE.
The chemical properties of drugs are ip, ea, and chi for Kraken B5, L,burB5, burL, and BDE Binding Energy.

Example of run:
```bash
python chemical_property_exps.py --dataset_name Kraken --task B5
```

## Acknowledgements

We thank Idan Tankel for his great help and discussion during the project.

## Contact

You can email yonatans@campus.technion.ac.il

## Authors

Yonatan Sverdlov: yonatans@campus.technion.ac.il

Nadav Dym: nadavdym@technion.ac.il
