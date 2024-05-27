# On the Expressive Power of Sparse Geometric MPNNs

Short description of your project.

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
In this section, we illustrate our ability to separate the sparsest graphs, and then we show our model can separate pairs. No I-GGNN can separate, and a pair no I-GGNN can separate even when considering cross edges.
We illustrate our results here and the baselines via Google-Colab.
In order to run our results, please run the following:
```bash
python k_chain_exps.py
```
For baselines, we add a link to the Colab notebook illustrating our tasks, results and baselines.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yonatansverdlov/E-GenNet/blob/master/k_chains_baselines.ipynb)
## Chemical properties 
- Data:
For the Kraken, Drugs, and BDE datasets, please download them from https://github.com/SXKDZ/MARCEL, and should be put in /data.

- Training:
In order to run, use python chemical_property_exps.py --dataset_name name --task task.

- options:
The options are Drugs, Kraken, BDE.
Drugs' chemical properties are ip, ea, chi, for Kraken B5, L,burB5, burL, and for BDE BindingEnergy.
Example of run:
```bash
cd script
python chemical_property_exps.py --dataset_name Kraken --task B5
```

## Acknowledgements

We thank Idan Tankel for his great help and discussion during the project.

## Contact

You can email yonatans@campus.technion.ac.il

## Authors

Yonatan Sverdlov: yonatans@campus.technion.ac.il

Nadav Dym: nadavdum@technion.ac.il
