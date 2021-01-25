import os
import torch
import torchvision
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from torch.nn import functional as F
from sklearn.preprocessing import normalize
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


def to_onehot(labels, num_classes, device):
    labels_onehot = torch.zeros(labels.size()[0], num_classes).to(device)
    labels_onehot.scatter_(1, labels.view(-1, 1), 1)

    return labels_onehot


class ConditionalVariationalAutoencoder(torch.nn.Module):

    def __init__(self, num_features, num_latent):
        super(ConditionalVariationalAutoencoder, self).__init__()


        ###############
        # ENCODER
        ##############

        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2

        self.enc_conv_1 = torch.nn.Conv2d(in_channels=3,
                                          out_channels=16,
                                          kernel_size=(6, 6),
                                          stride=(2, 2),
                                          padding=0)

        self.enc_conv_2 = torch.nn.Conv2d(in_channels=16,
                                          out_channels=32,
                                          kernel_size=(4, 4),
                                          stride=(2, 2),
                                          padding=0)

        self.enc_conv_3 = torch.nn.Conv2d(in_channels=32,
                                          out_channels=64,
                                          kernel_size=(4, 4),
                                          stride=(2, 2),
                                          padding=0)

        self.z_mean = torch.nn.Linear(64 * 4 * 4, num_latent)
        # in the original paper (Kingma & Welling 2015, we use
        # have a z_mean and z_var, but the problem is that
        # the z_var can be negative, which would cause issues
        # in the log later. Hence we assume that latent vector
        # has a z_mean and z_log_var component, and when we need
        # the regular variance or std_dev, we simply use
        # an exponential function
        self.z_log_var = torch.nn.Linear(64 * 4 * 4, num_latent)

        ###############
        # DECODER
        ##############

        self.dec_linear_1 = torch.nn.Linear(num_latent, 64 * 4 * 4)

        self.dec_deconv_1 = torch.nn.ConvTranspose2d(in_channels=64,
                                                     out_channels=32,
                                                     kernel_size=(4, 4),
                                                     stride=(2, 2),
                                                     padding=0)

        self.dec_deconv_2 = torch.nn.ConvTranspose2d(in_channels=32,
                                                     out_channels=16,
                                                     kernel_size=(4, 4),
                                                     stride=(2, 2),
                                                     padding=0)

        self.dec_deconv_3 = torch.nn.ConvTranspose2d(in_channels=16,
                                                     out_channels=3,
                                                     kernel_size=(6, 6),
                                                     stride=(2, 2),
                                                     padding=0)

    def reparameterize(self, z_mu, z_log_var):
        # Sample epsilon from standard normal distribution
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(device)
        # note that log(x^2) = 2*log(x); hence divide by 2 to get std_dev
        # i.e., std_dev = exp(log(std_dev^2)/2) = exp(log(var)/2)
        z = z_mu + eps * torch.exp(z_log_var / 2.)
        return z

    def encoder(self, features):
        ### Add condition


        x = features

        x = self.enc_conv_1(x)
        x = F.leaky_relu(x)
        # print('conv1 out:', x.size())

        x = self.enc_conv_2(x)
        x = F.leaky_relu(x)
        # print('conv2 out:', x.size())

        x = self.enc_conv_3(x)
        x = F.leaky_relu(x)
        # print('conv3 out:', x.size())

        z_mean = self.z_mean(x.view(-1, 64 * 4 * 4))
        z_log_var = self.z_log_var(x.view(-1, 64 * 4 * 4))
        encoded = self.reparameterize(z_mean, z_log_var)
        return z_mean, z_log_var, encoded

    def decoder(self, encoded):
        ### Add condition


        x = self.dec_linear_1(encoded)
        x = x.view(-1, 64, 2, 2)

        x = self.dec_deconv_1(x)
        x = F.leaky_relu(x)
        # print('deconv1 out:', x.size())

        x = self.dec_deconv_2(x)
        x = F.leaky_relu(x)
        # print('deconv2 out:', x.size())

        x = self.dec_deconv_3(x)
        x = F.leaky_relu(x)
        # print('deconv1 out:', x.size())

        decoded = torch.sigmoid(x)
        return decoded

    def forward(self, features):
        z_mean, z_log_var, encoded = self.encoder(features)
        decoded = self.decoder(encoded)

        return decoded, z_mean, z_log_var, encoded,


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = 1001
    batch_size = 128
    learning_rate = .001
    #learning_rate = 1e-5
    # This is based on dimensionality of data 1 channel or 3 channel, color or greyscale
    # I hope we can leave all of this just 1 dim so it works on all data
    dataset = 'CIFAR10'
    if dataset == 'MNIST':
        img_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # FOR MNIST
        dataset = MNIST('./Datasets/', transform=img_transform, download=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        test_dataset = MNIST('./Datasets/', train=False, transform=img_transform, download=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        num_classes = 10
        num_features = 784
        num_latent = 50
        model = ConditionalVariationalAutoencoder(num_features,
                                          num_latent,
                                          num_classes)
    elif dataset == 'CIFAR10':
        normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
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
            normalize,
        ])
        trainset = torchvision.datasets.CIFAR10(root='./Datasets/', train=True,
                                                download=True, transform=transform)
        dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./Datasets/', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)
        # Hyperparameters

        # Architecture
        num_classes = 10
        num_features = 3072
        num_latent = 200
        model = ConditionalVariationalAutoencoder(num_features,
                                          num_latent)
        model.to(device)
    random_seed = 0
    learning_rate = 0.001
    num_epochs = 50
    batch_size = 128

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    loss_values = [];
    channel = 0;
    # Hyperparameters
    random_seed = 0
    learning_rate = 0.001
    num_epochs = 50
    batch_size = 128

    # Architecture
    num_classes = 10
    num_features = 784
    num_latent = 50
    print("Channel:" + str(channel))

    start_time = time.time()
    loss_values = []
    for epoch in range(num_epochs):

        model.train()
        # scheduler.step()
        train_loss = 0
        # criterion = nn.L1Loss()
        criterion = nn.MSELoss()
        outputs = None
        img = None
        # criterion = nn.BCELoss()
        for batch_idx, (features, targets) in enumerate(dataloader):

            features = features.to(device)

            ### FORWARD AND BACK PROP
            decoded, z_mean, z_log_var, encoded = model(features)

            # cost = reconstruction loss + Kullback-Leibler divergence
            kl_divergence = (0.5 * (z_mean ** 2 +
                                    torch.exp(z_log_var) - z_log_var - 1)).sum()
            pixelwise_bce = F.binary_cross_entropy_with_logits(decoded, features, reduction='sum')

            #loss = criterion(decoded, features)
            cost = kl_divergence + pixelwise_bce
            ### UPDATE MODEL PARAMETERS
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            train_loss += cost.item()
            ### LOGGING
            if not batch_idx % 50:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f'
                      % (epoch + 1, num_epochs, batch_idx,
                         len(dataloader), cost))
        loss_values.append(train_loss / len(dataloader.dataset))
        print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))
        print('====> Epoch: {} Average loss: {:.6}'.format(
            epoch, train_loss / len(dataloader.dataset)))
        loss_values.append(train_loss / len(dataloader.dataset))
        scheduler.step()
        # if epoch % 10 == 0:
            # save = to_img(recon_batch.cpu().data)
            # save_image(save, './vae_img/image_{}.png'.format(epoch))
        if epoch % 5 == 0:
            unorm = UnNormalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
            show_output = unorm(decoded[0]).data.cpu().detach().numpy() * 255
            og_out = unorm(features[0]).data.cpu().detach().numpy() * 255
            #show_output = np.reshape(show_output, (32, 32))
            #og_out = np.reshape(og_out, (32, 32))
            plt.figure(1)
            plt.imshow(np.moveaxis(show_output, 0, -1).astype('uint8'), interpolation='nearest')
            #plt.imshow(show_output.astype('uint8'), interpolation='nearest')
            plt.savefig('./Results/ConvVAE/' + str(epoch) + 'reconstruct.png')
            plt.figure(2)
            plt.imshow(np.moveaxis(og_out, 0, -1).astype('uint8'), interpolation='nearest')
            #plt.imshow(og_out.astype('uint8'), interpolation='nearest')
            plt.savefig('./Results/ConvVAE/' + str(epoch) + 'input.png')
        if epoch % 10 == 0 and epoch != 0:
            plt.figure(3)
            plt.plot(np.array(np.asarray(loss_values)[3:]))
            plt.savefig('./Results/' + str(epoch) + 'ConvVAEloss.png')
        pwd = os.getcwd()
        save_dir = os.path.join(pwd, 'Autoencoders/' + str(dataset) + '/' + str(epoch) + "ep_MSE_loss_ConvVae.pth")
        if epoch == 1000 or epoch == 500:
            torch.save(model.state_dict(), save_dir)
    print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))
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