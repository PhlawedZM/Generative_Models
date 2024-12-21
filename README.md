# Genrative Models
A bunch of generative image models designed to see if dynamic training was a good idea. It failed but documentation is nice. Will likely update and store all attempts.

## Installation Instructions

Follow these steps to get the project up and running on your local machine:

### Prerequisites
Make sure you have the following installed:

- [Python](https://www.python.org/)
- [Pytorch CUDA](https://pytorch.org/get-started/locally/)

### Cloning the Repository
First, clone this repository to your local machine using Git:

```bash
git clone https://github.com/PhlawedZM/Generative_Models.git
cd Generative_Models
```

### Style Transfer Gan (Img2Img)
Completely dynamic style transfer gan. Completely inefficient compared to other models but does slightly more through interpolation. Needs about 10k+ images for both input and stylized and required a while to get out of instability (took me 3.1k epochs). Falls into equilibrium easily, pain to train.


### Dynamic Img2Img VAE
Completely dynamic vae. Tested on MNINT and worked fine, but because of dynamic tensors, the model performs rough.
