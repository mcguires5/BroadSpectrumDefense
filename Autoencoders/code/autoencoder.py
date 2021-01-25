import os
import torch
import torchvision
import numpy as np
from torch import nn

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, ImageNet
from torchvision.utils import save_image
from torch.nn import functional as F
from sklearn.preprocessing import normalize
from torch.optim.lr_scheduler import StepLR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from Datasets.occupancy_dataset import occupancyDataset
from datasets_objects import generate_struct
from model.defense_models import autoencoder2, autoencoder
import torchvision.datasets as datasets

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
'''
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
'''

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), z, mu, logvar



class CIFARSINGLEVAE(nn.Module):
    def __init__(self):
        super(CIFARSINGLEVAE, self).__init__()

        self.fc1 = nn.Linear(1024, 512)
        self.fc21 = nn.Linear(512, 256)
        self.fc22 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), z, mu, logvar

class newConvAutoencoder(nn.Module):
    def __init__(self):
        super(newConvAutoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 15, 8, stride=1, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(15, 30, 6, stride=1, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(30, 60, 4, stride=1, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
# 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
#             nn.ReLU(),
        )
        self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
#             nn.ReLU(),
            nn.ConvTranspose2d(60, 30, 4, stride=1, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(30, 15, 6, stride=1, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(15, 3, 8, stride=1, padding=1),   # [batch, 3, 32, 32]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

class newConvAutoencoderTanh(nn.Module):
    def __init__(self):
        super(newConvAutoencoderTanh, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 15, 8, stride=1, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(15, 30, 6, stride=1, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(30, 60, 4, stride=1, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
# 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
#             nn.ReLU(),
        )
        self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
#             nn.ReLU(),
            nn.ConvTranspose2d(60, 30, 4, stride=1, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(30, 15, 6, stride=1, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(15, 3, 8, stride=1, padding=1),   # [batch, 3, 32, 32]
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class ConvAutoencoder256(nn.Module):
    def __init__(self):
        super(ConvAutoencoder256, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 256, 3),            # [batch, 12, 16, 16]
            nn.ReLu(),
            nn.AvgPool2d(2),
            nn.Conv2d(128, 256, 3),           # [batch, 24, 8, 8]
            nn.ReLu(),
# 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
#             nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3),  # [batch, 24, 8, 8]
            nn.ReLu(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(256, 256, 3),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(15, 3, 8, stride=1, padding=1),   # [batch, 3, 32, 32]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

class defendedConvAutoencoder(nn.Module):
    def __init__(self):
        super(defendedConvAutoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 15, 8, stride=1, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(15, 30, 6, stride=1, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(30, 60, 4, stride=1, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
# 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
#             nn.ReLU(),
        )
        self.decoder1 = nn.Sequential(
#             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
#             nn.ReLU(),
            nn.ConvTranspose2d(60, 30, 4, stride=1, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(30, 15, 6, stride=1, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(15, 3, 8, stride=1, padding=1),   # [batch, 3, 32, 32]
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(3, 15, 8, stride=1, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(15, 30, 6, stride=1, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(30, 60, 4, stride=1, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            # 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
            #             nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            #             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            #             nn.ReLU(),
            nn.ConvTranspose2d(60, 30, 4, stride=1, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(30, 15, 6, stride=1, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(15, 3, 8, stride=1, padding=1),  # [batch, 3, 32, 32]
        )

    def forward(self, x):
        encoded1 = self.encoder1(x)
        decoded1 = self.decoder1(encoded1)
        encoded2 = self.encoder2(x)
        decoded2 = self.decoder2(encoded2)
        output = (decoded1 + decoded2)/2
        return output

class defendedVAE(nn.Module):
    def __init__(self):
        super(defendedVAE, self).__init__()

        self.fc1_1 = nn.Linear(784, 400)
        self.fc21_1 = nn.Linear(400, 20)
        self.fc22_1 = nn.Linear(400, 20)
        self.fc3_1 = nn.Linear(20, 400)
        self.fc4_1 = nn.Linear(400, 784)

        self.fc1_2 = nn.Linear(784, 400)
        self.fc21_2 = nn.Linear(400, 20)
        self.fc22_2 = nn.Linear(400, 20)
        self.fc3_2 = nn.Linear(20, 400)
        self.fc4_2 = nn.Linear(400, 784)

    def encode1(self, x):
        h1 = F.relu(self.fc1_1(x))
        return self.fc21_1(h1), self.fc22_1(h1)

    def reparametrize1(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode1(self, z):
        h3 = F.relu(self.fc3_1(z))
        return torch.sigmoid(self.fc4_1(h3))

    def encode2(self, x):
        h1 = F.relu(self.fc1_2(x))
        return self.fc21_2(h1), self.fc22_2(h1)

    def reparametrize2(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode2(self, z):
        h3 = F.relu(self.fc3_2(z))
        return torch.sigmoid(self.fc4_2(h3))

    def forward(self, x):
        x = x.view(-1, 784)
        mu, logvar = self.encode1(x)
        z1 = self.reparametrize1(mu, logvar)
        out1 = self.decode1(z1)
        mu, logvar = self.encode2(x)
        z2 = self.reparametrize2(mu, logvar)
        out2 = self.decode2(z2)
        output = (out1 + out2) / 2
        output = output.view(-1, 1, 28, 28)
        return output


class MagNetAE256(nn.Module):
    def __init__(self):
        super(MagNetAE256, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 256, 3, stride=1, padding=1),  # [batch, 12, 16, 16]
            nn.Sigmoid(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),  # [batch, 48, 4, 4]
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 3, 1, stride=1, padding=0),  # [batch, 48, 4, 4]
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class TinyImageNetConvAE(nn.Module):
    def __init__(self):
        super(TinyImageNetConvAE, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 15, 3, stride=1, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(15, 15, 3, stride=1, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(15, 30, 3, stride=1, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
        )
        #self.pool1 = nn.AvgPool2d(2, return_indices=True)
        self.encoder2 = nn.Sequential(
            nn.Conv2d(30, 60, 3, stride=1, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
        )
        #self.pool2 = nn.AvgPool2d(2, return_indices=True)

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(60, 30, 3, stride=1, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
        )
        #self.unpool1 = nn.MaxUnpool2d(2)
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(30, 15, 3, stride=1, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
        )
        #self.unpool2 = nn.MaxUnpool2d(2)
        self.decoder3 = nn.Sequential(
            nn.Conv2d(15, 15, 3, stride=1, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(15, 3, 3, stride=1, padding=1),   # [batch, 3, 32, 32]
            nn.Tanh(),
        )
    def forward(self, x):
        encoded1 = self.encoder1(x)
        #out1, indicies1 = self.pool1(encoded1)
        encoded2 = self.encoder2(encoded1)
        #out2, indicies2 = self.pool2(encoded2)

        decoded1 = self.decoder1(encoded2)
        #up1 = self.unpool1(decoded1, indicies2)
        decoded2 = self.decoder2(decoded1)
        #up2 = self.unpool2(decoded2, indicies1)
        output = self.decoder3(decoded2)
        return output


class occupancy_ae(nn.Module):
    def __init__(self):
        super(occupancy_ae, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(True),
            nn.Linear(32, 5))
        self.decoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(True),
            nn.Linear(32, 5))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def main():
    if torch.cuda.device_count() == 1:
        cuda_name = "0"
    else:
        cuda_name = "5"
        #torch.cuda.set_device(4)
        device = torch.device('cuda:5,6,7' if torch.cuda.is_available() else 'cpu')
        print('AE Device ' + str(torch.cuda.current_device()))
    num_epochs = 1001
    batch_size = 128
    learning_rate = .001
    #learning_rate = 1e-5
    # This is based on dimensionality of data 1 channel or 3 channel, color or greyscale
    # I hope we can leave all of this just 1 dim so it works on all data
    dataset = 'CIFAR100'




    if dataset == 'MNIST':
        img_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # FOR MNIST
        dataset = MNIST('./Datasets/', transform=img_transform, download=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        test_dataset = MNIST('./Datasets/', train=False, transform=img_transform, download=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        model = VAE().cuda()
    elif dataset == 'CIFAR10':
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        '''
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ])
        '''

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        trainset = CIFAR10(root='./Datasets/', train=True,
                                                download=True, transform=transform)
        dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=False, num_workers=1)

        testset = CIFAR10(root='./Datasets/', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=1)
        # Hyperparameters

        # Architecture
        num_classes = 10
        num_features = 3072
        num_latent = 200
        #model = newConvAutoencoderTanh().cuda()
        #model = MagNetAE256().cuda()
        model = autoencoder2(3).cuda()
        data_obj, x_test, y_test, x_train, y_train, dataloader, testloader = generate_struct(1, dataset,
                                                                                             0.01, cuda_name)
    elif dataset == 'CIFAR100':
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        '''
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ])
        '''

        testnorm = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             ])
        transform2 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        trainset = torchvision.datasets.CIFAR100(root='./Datasets/', train=True,
                                                download=False, transform=transform2)
        dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=False, num_workers=0)

        testset = torchvision.datasets.CIFAR100(root='./Datasets/', train=False,
                                               download=False, transform=testnorm)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)

        # Hyperparameters

        # Architecture
        num_classes = 10
        num_features = 3072
        num_latent = 200
        #model = newConvAutoencoderTanh().cuda()
        #model = MagNetAE256().cuda()
        #model = autoencoder(3).cuda()
        model = TinyImageNetConvAE().cuda()
        data_obj, x_test, y_test, x_train, y_train, dataloader, testloader = generate_struct(1, dataset,
                                                                                             0.01, cuda_name)
    elif dataset == 'TinyImageNet':

        data_obj, x_test, y_test, x_train, y_train, dataloader, testloader = generate_struct(1, dataset,
                                                                                             0.01, cuda_name)
        model = TinyImageNetConvAE().cuda()

    elif dataset == 'ImageNet':

        data_obj, x_test, y_test, x_train, y_train, dataloader, testloader = generate_struct(1, dataset,
                                                                                             0.01, cuda_name)
        model = TinyImageNetConvAE()
        model = torch.nn.DataParallel(model, device_ids=[3, 4, 5, 6, 7])
        model.to('cuda:3')
        print(torch.cuda.device_count())
    elif dataset == 'occupancy':
        '''
        # Set min pixel and max pixel size input size
        data_obj = {'name': 'occupancy', 'percent': .05, 'CUDA_name': "1",
                    'classifier': 'MNIST_standard_classifier', 'min_pixel_value': 0,
                    'max_pixel_value': 1, 'input_size': (1, 28, 28), 'output_size': 10,
                    'dims': 1, 'L1_dir': 'Autoencoders/MNIST/1000ep_MAE_loss_1-e5_LR.pth',
                    'L2_dir': 'Autoencoders/MNIST/1000ep_MSE_loss_1-e5_LR.pth'}
        batch_size = 128
        '''
        data_obj, x_test, y_test, x_train, y_train, dataloader, testloader = generate_struct(1, dataset,
                                                                                             0.01, cuda_name)
        model = occupancy_ae().cuda()

    random_seed = 0
    learning_rate = 0.001
    num_epochs = 50
    batch_size = 128

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    loss_values = []
    channel = 0
    # Hyperparameters
    num_epochs = 500
    batch_size = 128

    # Architecture
    num_classes = 10
    num_features = 784
    num_latent = 50
    #v_noise = 0.1
    v_noise = 0.05
    for epoch in range(num_epochs):

        model.train()
        # scheduler.step()
        train_loss = 0
        #criterion = nn.L1Loss()
        criterion = nn.MSELoss()
        #criterion = nn.L1Loss()
        outputs = None
        img = None
        # criterion = nn.BCELoss()
        for batch_idx, data in enumerate(dataloader):
            img, _ = data

            noise = v_noise * np.random.normal(size=np.shape(img))
            noise = torch.from_numpy(noise)

            noisy_train_data = img.double() + noise

            # noisy_train_data.clamp(-1.0, 1.0)
            noisy_train_data = noisy_train_data.clamp(data_obj['min_pixel_value'], data_obj['max_pixel_value'])

            noisy_train_data = noisy_train_data.cuda().float()

            img = img.cuda().float()

            optimizer.zero_grad()
            noisy_train_data = noisy_train_data.to('cuda:0')
            output = model(noisy_train_data)
            img = img.to('cuda:0')
            loss = criterion(output, img)

            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % 10 == 0:
                '''
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                    epoch,
                    batch_idx * len(img),
                    len(dataloader.dataset), 100. * batch_idx / len(dataloader),
                    loss.item() / len(img), scheduler.get_lr()))
                '''
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {:.6f}'.format(epoch,
                    batch_idx * len(img),
                    len(dataloader.dataset), batch_idx * len(img) / len(dataloader),
                    loss.item() / len(img), scheduler.get_lr()[0]))
        print('====> Epoch: {} Average loss: {:.6}'.format(
            epoch, train_loss / len(dataloader.dataset)))
        loss_values.append(train_loss / len(dataloader.dataset))
        scheduler.step()
        # if epoch % 10 == 0:
            # save = to_img(recon_batch.cpu().data)
            # save_image(save, './vae_img/image_{}.png'.format(epoch))

        if epoch % 50 == 0:
            idx = np.random.randint(len(output)-1)
            unorm = UnNormalize(mean=(.5, .5, .5), std=(.5, .5, .5))
            show_output = unorm(output[idx]).data.cpu().detach().numpy() * 255
            og_out = unorm(img[idx]).data.cpu().detach().numpy() * 255
            #show_output = np.reshape(show_output, (32, 32))
            #og_out = np.reshape(og_out, (32, 32))
            plt.figure(1)
            plt.imshow(np.moveaxis(show_output, 0, -1).astype('uint8'), interpolation='nearest')
            #plt.imshow(show_output.astype('uint8'), interpolation='nearest')
            plt.savefig('./Results/' + str(dataset) + '/' + str(epoch) + 'reconstruct.png')
            plt.figure(2)
            plt.imshow(np.moveaxis(og_out, 0, -1).astype('uint8'), interpolation='nearest')
            #plt.imshow(og_out.astype('uint8'), interpolation='nearest')
            plt.savefig('./Results/'+ str(dataset) + '/' + str(epoch) + 'input.png')
        if epoch % 50 == 0 and epoch != 0:
            plt.figure(3)
            plt.plot(np.array(np.asarray(loss_values)[5:]))
            plt.savefig('./Results/' + str(epoch) + '0.05Noise_MSE.png')

        pwd = os.getcwd()
        save_dir = os.path.join(pwd, 'Autoencoders/' + str(dataset) + '/' + str(epoch) + "ep_TinyImageNetAE_0.05_noise_-1to1MSE.pth")
        if epoch == num_epochs or epoch == num_epochs/20:
            torch.save(model.state_dict(), save_dir)

    torch.save(model.state_dict(), save_dir)
    '''
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_dataloader):
            data = data.view(data.size(0), -1)
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(batch_size, 1, 28*28)[:n]])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_dataloader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    '''
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()