# On the Expressive Power of Sparse Geometric MPNNs

## Table of Contents

- [About](#about)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
  - [Sanity Check](#sanity-check)
  - [Hard Example](#hard-example)
  - [Power-Graph & k-Chain Experiments](#power-graph--k-chain-experiments)
  - [Chemical Property Prediction](#chemical-property-prediction)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)
- [Authors](#authors)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yonatansverdlov/E-GenNet/blob/master/k_chains_baselines.ipynb)

## About

Driven by applications in chemistry and various scientific fields, we investigate the expressive capabilities of message-passing neural networks (MPNNs) for geometric graphs. We demonstrate which graphs invariant and equivariant MPNNs can separate. Our findings indicate that this separation is primarily influenced by graph structure rather than the underlying point cloud. This repository provides implementations for sparse separation, power graph experiments, and chemical property prediction.

## Installation

### **Using Conda**
```bash
git clone git@github.com:yonatansverdlov/E-GenNet.git
conda env create -f egenet_environment.yml
conda activate egenet
```

## Usage

### **Sanity Check**
To verify the correctness of our model, run:
```bash
python sanity_check.py
```

## Experiments

### **Hard Example**
To demonstrate a pair that no I-GWL can separate, we train our model on separating such a pair:
```bash
python hard_example.py
```

### **Power-Graph & k-Chain Experiments**
In this section, we showcase our ability to separate sparsest graphs, demonstrating that our model can outperform I-GGNN. Our experiments validate that the k-chain pairs can be separated across ten different seeds. To run:
```bash
python k_chain.py
```
For baselines, refer to the Colab notebook illustrating our tasks, results, and baselines:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yonatansverdlov/E-GenNet/blob/master/k_chains_baselines.ipynb)

### **Chemical Property Prediction**
We present our results on predicting chemical properties along with train/test/val accuracy.

#### **Data**
Please download the **Kraken, Drugs, and BDE** datasets from [MARCEL GitHub](https://github.com/SXKDZ/MARCEL) and place them in `/data`.

#### **Options**
- `dataset_name` options: `Drugs`, `Kraken`, `BDE`
- `task` options:
  - **Drugs**: `ip`, `ea`, `chi`
  - **Kraken**: `B5`, `L`, `burB5`, `burL`
  - **BDE**: `BindingEnergy`

#### **Training**
To start training:
```bash
python chemical_property_exps.py --dataset_name <dataset_name> --task <task>
```

## Acknowledgements

We thank **Idan Tankel** for his valuable contributions and discussions throughout the project.

## Contact

For inquiries, feel free to reach out via email:

📧 **Yonatan Sverdlov** - yonatans@campus.technion.ac.il

## Authors

- **Yonatan Sverdlov** - yonatans@campus.technion.ac.il
- **Nadav Dym** - nadavdym@technion.ac.il


