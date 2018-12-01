# Membership Inference of Generative Models

This is a toy example (on CIFAR-10) of how to run attacks presented in https://arxiv.org/abs/1705.07663

## To run

1. Train a GAN on CIFAR-10 by `python dcgan.py --outf ./models --dataroot ./data --cuda --niter 50`
2. Run attack by `python attack.py --outf ./models --dataroot ./data --niter 10000 --cuda --netBBG ./models/netG_epoch_49.pth --netBBD ./models/netD_epoch_49.pth`

## Results:

Running the above gives approx:

baseline (random guess) accuracy: 0.167

white-box attack accuracy: 0.260

black-box attack accuracy: 0.317


## Notes:

This is a toy example of how the attack should run. 
- Hyperparameter optimization is needed for decent results (and run for a larger number of epochs and steps, as the above results show the black-box attack fails when training for only 1000 steps).
- To change the size of the training set one must edit the DataLoader method.
- I swapped around the train set and test set for CIFAR-10, so the models are trained on the test set.
