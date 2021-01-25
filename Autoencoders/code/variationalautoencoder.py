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

if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


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
        return self.decode(z), mu, logvar


def add_noise(img):
    noise = abs(torch.randn(img.size()) * 0.1)
    torch.FloatTensor(noise).uniform_(0, 1)
    noisy_img = img + noise
    noisy_img = noisy_img/torch.max(noisy_img).item()
    # return torch.from_numpy(noisy_img)
    return noisy_img

def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    reconstruction_function = nn.MSELoss(size_average=False)
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = 1000
    batch_size = 128
    #learning_rate = 1e-3
    learning_rate = 1e-5
    # This is based on dimensionality of data 1 channel or 3 channel, color or greyscale
    # I hope we can leave all of this just 1 dim so it works on all data
    img_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = MNIST('./Datasets/', transform=img_transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataset = MNIST('./Datasets/', train=False, transform=img_transform, download=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(dataloader):
            img, _ = data
            img = img.view(img.size(0), -1)
            noisy_img = add_noise(img)
            noisy_img = Variable(noisy_img.type(torch.float32))
            if torch.cuda.is_available():
                noisy_img = noisy_img.cuda()
                img = img.cuda()
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(noisy_img)
            loss = loss_function(recon_batch, img, mu, logvar)
            #recon_batch, mu, logvar = model(img)
            #loss = loss_function(recon_batch, img, mu, logvar)
            loss.backward()
            train_loss += loss.data.item()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(img),
                    len(dataloader.dataset), 100. * batch_idx / len(dataloader),
                    loss.item() / len(img)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(dataloader.dataset)))
        if epoch % 10 == 0:
            save = to_img(recon_batch.cpu().data)
            save_image(save, './vae_img/image_{}.png'.format(epoch))
        if epoch == 500:
            torch.save(model.state_dict(), '.\\Autoencoders\\' + str(epoch) + "ep_VAE_loss_newnoise_1-e5_LR.pth")
    torch.save(model.state_dict(), '.\\Autoencoders\\' + str(num_epochs) + "ep_VAE_loss_newnoise_1-e5_LR.pth")

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
                #comparison = torch.cat([data[:n],
                #                        recon_batch.view(batch_size, 1, 28*28)[:n]])
                #save_image(comparison.cpu(),
                #           'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_dataloader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()