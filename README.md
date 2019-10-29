# Simple implementation of Conditional VAE

This my simple implementation of [Conditional Variational Autoencoder for Neural Machine Translation](https://arxiv.org/pdf/1812.04405.pdf) paper by Artidoro Pagnoni, Kevin Liu and Shangyan Li

## Requirements

The project is built on top of Pytorch 	`1.3.0`.
To install all requirement packages, run the command:
```bash
pip install -r requirements.txt
```

## How to use
The **Conditional Variational Autoencoder** (CVAE) is a an algorithm to generate certain image (e.g. generation of the digit image of given label) from the latent space

### Datasets

The model was taught on MNIST dataset, which can be obtained as:
```bash
torchvision.datasets.MNIST
```

## Results

For demonstration 10 random digits were generated
<p align="center"><img src="10_random_digits.png"></p>