# Membership Inference of Generative Models

This is a toy example (on CIFAR-10) of how to run attacks presented in https://arxiv.org/abs/1705.07663

## To run

1. Train a GAN on CIFAR-10 by `python dcgan.py --outf ./models --dataroot ./data --cuda --niter 10`
2. Run attack by `python attack.py --outf ./models --dataroot ./data --niter 1000 --cuda --netBBG ./models/netG_epoch_9.pth --netBBD ./models/netD_epoch_9.pth`

## Results:

Running the above gives 



## Notes:

This is a toy example of how the attack should run. Hyperparameter optimization is needed for decent results. To change
the size of the training set one must edit the DataLoader method.
