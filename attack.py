from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np

from models import weights_init, Generator, Discriminator

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=100, help='number of steps to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netBBG', type=str, required=True, help="path to netBBG (to attack)")
parser.add_argument('--netWBG', default='', help="path to netWBG (to continue training)")
parser.add_argument('--netBBD', type=str, required=True, help="path to netBBD (to attack)")
parser.add_argument('--netWBD', default='', help="path to netWBD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Data
print('==> Preparing data..')
data_transform = transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

])

trainset = dset.CIFAR10(root=opt.dataroot, train=True, download=True, transform=data_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

testset = dset.CIFAR10(root=opt.dataroot, train=False, download=True, transform=data_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


netBBG = Generator(ngpu, nz, ngf, nc).to(device)
netBBG.load_state_dict(torch.load(opt.netBBG))

netBBD = Discriminator(ngpu, nc, ndf).to(device)
netBBD.load_state_dict(torch.load(opt.netBBD))

netWBG = Generator(ngpu, nz, ngf, nc).to(device)
netWBG.apply(weights_init)
if opt.netWBG != '':
    netWBG.load_state_dict(torch.load(opt.netWBG))

netWBD = Discriminator(ngpu, nc, ndf).to(device)
netWBD.apply(weights_init)
if opt.netWBD != '':
    netWBD.load_state_dict(torch.load(opt.netWBD))

netBBG.eval()
netBBD.eval()

##### White-box attack ####
# Assumes we have direct access to BBD
wb_predictions = []

# loop over training data
for i, data in enumerate(trainloader, 0):
    real_cpu = data[0].to(device)
    output = netBBD(real_cpu)
    output = [x for x in output.detach().cpu().numpy()]
    output = list(zip(output, ['train' for _ in range(len(output))]))
    wb_predictions.extend(output)

# loop over test data
for i, data in enumerate(testloader, 0):
    real_cpu = data[0].to(device)
    output = netBBD(real_cpu)
    output = output.detach().cpu().numpy()
    output = list(zip(output, ['test' for _ in range(len(output))]))
    wb_predictions.extend(output)

wb_predictions = wb_predictions
wb_predictions = [x[1] for x in sorted(wb_predictions, reverse=True)[:len(trainset)]]
wb_accuracy = wb_predictions.count('train')/float(len(trainset))

##### Black-box attack ####
# Trains another GAN on the output of the black-box
# Then launches whitebox attack with trained Discriminator 


criterion = nn.BCELoss()

fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netWBD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netWBG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for step in range(opt.niter):
    ############################
    # (1) update d network: maximize log(d(x)) + log(1 - d(g(z)))
    ###########################

    # generate "real"
    real_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
    real_cpu = netBBG(real_noise)

    # train with "real"
    netWBD.zero_grad()
    batch_size = real_cpu.size(0)
    label = torch.full((batch_size,), real_label, device=device)

    output = netWBD(real_cpu)
    errD_real = criterion(output, label)
    errD_real.backward()
    D_x = output.mean().item()

    # train with fake
    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    fake = netWBG(noise)
    label.fill_(fake_label)
    output = netWBD(fake.detach())
    errD_fake = criterion(output, label)
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    errD = errD_real + errD_fake
    optimizerD.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netWBG.zero_grad()
    label.fill_(real_label)  # fake labels are real for generator cost
    output = netWBD(fake)
    errG = criterion(output, label)
    errG.backward()
    D_G_z2 = output.mean().item()
    optimizerG.step()

    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
          % (step, opt.niter,
             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
    if i % 100 == 0:
        vutils.save_image(real_cpu,
                '%s/real_samples.png' % opt.outf,
                normalize=True)
        fake = netG(fixed_noise)
        vutils.save_image(fake.detach(),
                '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                normalize=True)

# do checkpointing
torch.save(netWBG.state_dict(), '%s/netWBG_step_%d.pth' % (opt.outf, step))
torch.save(netWBD.state_dict(), '%s/netWBD_step_%d.pth' % (opt.outf, step))

netWBD.eval()

##### Black-box attack ####
# Assumes we have direct access to WBD
bb_predictions = []

# loop over training data
for i, data in enumerate(trainloader, 0):
    real_cpu = data[0].to(device)
    output = netBBD(real_cpu)
    output = [x for x in output.detach().cpu().numpy()]
    output = list(zip(output, ['train' for _ in range(len(output))]))
    bb_predictions.extend(output)

# loop over test data
for i, data in enumerate(testloader, 0):
    real_cpu = data[0].to(device)
    output = netWBD(real_cpu)
    output = output.detach().cpu().numpy()
    output = list(zip(output, ['test' for _ in range(len(output))]))
    bb_predictions.extend(output)

bb_predictions = bb_predictions
bb_predictions = [x[1] for x in sorted(bb_predictions, reverse=True)[:len(trainset)]]
bb_accuracy = bb_predictions.count('train')/float(len(trainset))

print("baseline (random guess) accuracy: {:.3f}".format(len(trainset)/float(len(trainset)+len(testset))))
print("white-box attack accuracy: {:.3f}".format(wb_accuracy))
print("black-box attack accuracy: {:.3f}".format(bb_accuracy))


