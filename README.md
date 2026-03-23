1. Tests on Two Moons
A comparative study of three normalizing flow architectures RealNVP, MAF, and TARFlow 
applied to the Two Moons synthetic dataset.

2. TARFlow Reimplementation
An independent reimplementation of TARFlow (Zhai et al., 2024), trained and evaluated on MNIST.
The experiments explore the effect of different sampling methods on generation quality, including
Gaussian noise augmentation

3. Pretrained Modules
A collection of loading utilities for the pretrained components of STARFlow (Gu et al., 2025),
covering the variational autoencoder, text encoders, llm.

To Do
Test on Fashion-MNIST. Evaluate the TARFlow reimplementation on Fashion-MNIST to assess performance on a more challenging dataset at the same resolution as MNIST.
Deep-shallow architecture on TARFlow. Investigate the deep-shallow design — introduced in STARFlow — applied directly to TARFlow (i.e., in pixel space, not latent space). This involves allocating greater Transformer depth to the first flow block and fewer layers to subsequent blocks, to test whether the architectural benefit observed at scale transfers to smaller settings.
