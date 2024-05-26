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

We showed the generic expressiveness power of I-GGNNs and E-GGNNs and illustrate it via the Google-Colab notebook.

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
## Sanity Check
For a sanity check of our model, please run 

python sanity_check.py

## Experiments
The repository consists of two main parts: synthetic experiments and chemical properties prediction.
## Synthetic experiments.
We illustrate our results via terminal and the baselines via Google-Colab.
## Our results.
In order to run our results, please run the following:
```bash
cd script
python k_chain_exps.py
```
## Baselines
For baselines, we add a link to the Colab notebook illustrating our results and baselines.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yonatansverdlov/E-GenNet/blob/master/k_chains_baselines.ipynb)
## Chemical properties 
## Data 
For the Kraken, Drugs, and BDE datasets, please download it from https://github.com/SXKDZ/MARCEL.

And put it in /data.

## Training
In order to run, use python chemical_property_exps.py --dataset-name __task
The options are Drugs, Kraken, BDE.
Drugs' chemical properties are ip, ea, chi, Kraken B5, L,burB5, burL, and BDE BindingEnergy.
Example of run:
```bash
cd script

python chemical_property_exps.py --Kraken --B5
```

## License

Include information about the license under which your project is distributed. You can also include a link to the full text of the license.

## Acknowledgements

We thank Idan Tankel for his great help and discussion during the project.

## Contact

You can email yonatans@campus.technion.ac.il

## Authors

Yonatan Sverdlov: yonatans@campus.technion.ac.il

Nadav Dym: nadavdum@technion.ac.il
