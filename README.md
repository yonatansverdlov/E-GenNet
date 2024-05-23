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

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)

## About

Motivated by applications in chemistry and other sciences, we study the expressive power of message-passing neural networks for geometric graphs, whose node features correspond to 3-dimensional positions. 

## Installation

# Pip
conda create --name egenet -c conda-forge python=3.11

conda activate egenet

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install torch_geometric

pip install pytorch-lightning

pip install easydict

pip install pandas

pip install rdkit



## Usage

The repository consists of two main parts: synthetic experiments and chemical properties prediction.
## Syntetic experiments.

We add a link to the Colab notebook illustrating our results and baselines.
The link: 
## Chemical properties

cd scripts

python chemical_property_exps.py 
## Contributing

Explain how others can contribute to your project. Include guidelines for submitting bug reports, feature requests, and code contributions. You can also include information about your coding style, code of conduct, and how to set up a development environment.

## License

Include information about the license under which your project is distributed. You can also include a link to the full text of the license.

## Acknowledgements

If your project uses third-party libraries, services, or resources, you can acknowledge them here.

## Contact

Provide contact information for users to reach out to you with questions, feedback, or other inquiries.

## Authors

List the authors or contributors of the project.