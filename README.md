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

Driven by applications in chemistry and various other scientific fields, we investigate the expressive capabilities of message-passing neural networks (MPNNs) for geometric graphs. We demonstrate which graphs invariant and equivariant MPNNs can separate. We found that this separation generically doesn't depend on the point cloud but rather on the graph structure. This repository showcases our results through sparse separation, power graph experiments, and the prediction of chemical properties.

## Installation

# Conda
```bash
git clone git@github.com:yonatansverdlov/E-GenNet.git
conda env create -f egenet_environment.yml
conda activate egenet
```

## Usage
- Sanity Check
For a sanity check of our model, please run 
```bash
python sanity_check.py
```
## Hard Example
In order to show an exmaple of a pair, no I-GWL can seperate, we train our model on seperation such a pair.
Run: 
```bash
python hard_example.py
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

Example of run for A10 users using one GPU and 24GB:
```bash
python chemical_property_exps.py --dataset_name Kraken --task B5 --batch_size 5
```
Example of run for A40 users using one GPU and 48GB:
```bash
python chemical_property_exps.py --dataset_name Kraken --task B5
```
## Acknowledgements

We thank Idan Tankel for his great help and discussion during the project.

## Contact

You are more than welcome to send an email yonatans@campus.technion.ac.il


## Authors

Yonatan Sverdlov: yonatans@campus.technion.ac.il

Nadav Dym: nadavdym@technion.ac.il


