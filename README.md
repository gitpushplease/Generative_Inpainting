# Generative Image and Video Inpainting using AOT and WGAN-GP

This repository provides a PyTorch-based implementation of a generative inpainting model that leverages:

- AOT (Aggregation of Transformations) blocks for context-aware generation
- WGAN-GP (Wasserstein GAN with Gradient Penalty) loss for stable adversarial training
- An optional L1 reconstruction loss for improved visual fidelity

The model is designed for filling in missing regions in images and videos, using either randomly generated masks or actual damaged/masked input.
