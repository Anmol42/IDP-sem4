# IDP-sem4

This project aims to specialise in colouring black and white images of people.
Using a VAE based generator for a GAN to colourise the image.

## Inputs

The input data is 128x128 images of famous celebrities. 
We use Lab colour scheme for producing colour in the image.
Training dataset consists of 10000 images split into 8000,1000,1000 for training, validation, testing respectively.

## Training
The input is normalised to range from -1 to 1 before feeding it into the network.
For the generator Batch Normalisation is applied after every layer along with a ReLU activation function.
In the Forward pass of the generator the 'L' channel of the image is fed into the network where a 512 channeled 4x4 latent space is generated,
The Generator uses BCEwithLogitsLoss() utility from torch.nn as the 'classification' part of generator loss function and CosineSimilarity()
utility from torch.nn as the 'generation' part of the loss function.  

We use BCEwithLogitsLoss() from torch.nn for calculating loss for the discriminator.  
  
The optimiser used is Adam for this project with momentum coefficient beta=0.5  
The results obtained so far have been after training for 50 epochs with learning rate in the order of -4

