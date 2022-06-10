# Making a linear Autoencoder learn PCA

As shown in the Appendix of ..., it can be shown that a **linear autoencoder** trained with weight-decay learns **PCA**.

This is a library which trains a lineat autoencoder with and without weight-decay and compares the results. See for example the following image:

![comparison_good](https://user-images.githubusercontent.com/41115254/173034245-2b45b685-d8ca-4c65-916d-e477a40c5716.png)

From left to right one can see the embedding of MNIST into 2D latent space as done by a linear autoencoder trained only with MSE loss, by PCA and by a linear Autoencoder regularized via weight-decay.
