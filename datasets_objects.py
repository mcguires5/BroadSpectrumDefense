
import torchvision

from torchvision import transforms
import torch
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
import os
import torchvision.datasets as datasets
import torch.utils.data as dataL
from Datasets.occupancy_dataset import occupancyDataset


def generate_struct(scale, set, percent, cuda_name):
    torch.cuda.empty_cache()
    #master_seed(123456789)
    # Step 0: Define the neural network model, return logits instead of activation in forward method
    # Step 1: Load the MNIST dataset
    dataset = set
    print(dataset)
    data_obj = None
    if dataset == 'MNIST':
        '''
        data_obj = {'name': 'MNIST', 'percent': percent,'CUDA_name': cuda_name,'classifier': 'MNIST_standard_classifier', 'min_pixel_value': 0,
                    'max_pixel_value': 1, 'input_size': (1,28,28), 'output_size': 10,
                    'dims': 1,'L1_dir':'Autoencoders/MNIST/1000ep_MAE_loss_1-e5_LR.pth','L2_dir':'Autoencoders/MNIST/1000ep_MSE_loss_1-e5_LR.pth'}
        '''
        data_obj = {'name': 'MNIST', 'percent': percent,'CUDA_name': cuda_name,'classifier': 'MNIST_standard_classifier', 'min_pixel_value': 0,
                    'max_pixel_value': 1, 'input_size': (1,28,28), 'output_size': 10,
                    'dims': (1, 2, 3),'L1_dir':'checkpoint/MNIST/autoencoder1L1.pth','L2_dir':'checkpoint/MNIST/autoencoder2.pth'}
        batch_size = 128
        transform = transforms.Compose([
            transforms.ToTensor()])
        trainset = torchvision.datasets.MNIST(root='./Datasets/', train=True,
                                                download=False, transform=transform)
        dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                 shuffle=True, num_workers=0)

        testset = torchvision.datasets.MNIST(root='./Datasets/', train=False,
                                               download=False,transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)

        x_train = []
        y_train = []
        for batch_idx, data in enumerate(dataloader):
            img, label = data

            img = Variable(img.type(torch.float32))
            x_train.append(img)
            y_train.append(label)
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        x_test = []
        y_test = []
        for batch_idx, data in enumerate(testloader):
            img, label = data
            img = Variable(img.type(torch.float32))
            y_test.append(label)
            x_test.append(img)
        x_test = np.concatenate(x_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)
    elif dataset == 'CIFAR10':
        batch_size = 128
        #normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

        #testnorm = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
        testnorm = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             ])
        transform2 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./Datasets/', train=True,
                                                download=False, transform=transform2)
        dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=False, num_workers=0)

        testset = torchvision.datasets.CIFAR10(root='./Datasets/', train=False,
                                               download=False, transform=testnorm)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)
        x_train = []
        y_train = []
        for batch_idx, data in enumerate(dataloader):
            img, label = data
            img = Variable(img.type(torch.float32))
            x_train.append(img)
            y_train.append(label)
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        x_test = []
        y_test = []
        for batch_idx, data in enumerate(testloader):
            img, label = data
            img = Variable(img.type(torch.float32))
            y_test.append(label)
            x_test.append(img)
        x_test = np.concatenate(x_test, axis=0)
        np.save('500_x_test.npy', np.reshape(x_test, (10000, 32, 32, 3))[0:500]*255)
        y_test = np.concatenate(y_test, axis=0)
        np.save('500_y_test.npy', y_test[0:500])
        # TODO: FIX min and Max pixel values
        #data_obj = {'name':'CIFAR10', 'percent': percent,'CUDA_name': cuda_name, 'classifier': 'CIFAR10_Standard-1to1','min_pixel_value':np.float(np.min(x_train)),'max_pixel_value':np.float(np.max(x_train)),'input_size':(3,32,32), 'output_size':10, 'dims':(1,2,3),'L1_dir':'Autoencoders/CIFAR10/999ep_MAE_loss_-1to1.pth','L2_dir':'Autoencoders/CIFAR10/999ep_MSE_loss_-1to1.pth'}
        '''
        data_obj = {'name': 'CIFAR10', 'percent': percent, 'CUDA_name': cuda_name,
                    'classifier': 'CIFAR10_Standard-1to1', 'min_pixel_value': np.float(np.min(x_train)),
                    'max_pixel_value': np.float(np.max(x_train)), 'input_size': (3, 32, 32), 'output_size': 10,
                    'dims': (1, 2, 3), 'L1_dir': 'Autoencoders/CIFAR10/999ep_MAE_loss_-1to1.pth',
                    'L2_dir': 'Autoencoders/CIFAR10/999ep_MSE_loss_-1to1.pth'}
        '''
        data_obj = {'name': 'CIFAR10', 'percent': percent, 'CUDA_name': cuda_name,
                    'classifier': 'CIFAR10_Standard-1to1', 'min_pixel_value': np.float(np.min(x_train)),
                    'max_pixel_value': np.float(np.max(x_train)), 'input_size': (3, 32, 32), 'output_size': 10,
                    'dims': (1, 2, 3), 'L1_dir': 'Autoencoders/CIFAR10/autoencoder_L1_2.pth',
                    'L2_dir': 'Autoencoders/CIFAR10/autoencoder2.pth'}

    elif dataset == 'CIFAR100':
        batch_size = 128
        #normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

        #testnorm = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
        testnorm = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             ])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        trainset = torchvision.datasets.CIFAR100(root='./Datasets/', train=True,
                                                download=True, transform=transform_train)
        dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=False, num_workers=0)

        testset = torchvision.datasets.CIFAR100(root='./Datasets/', train=False,
                                               download=True, transform=testnorm)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)
        x_train = []
        y_train = []
        for batch_idx, data in enumerate(dataloader):
            img, label = data
            img = Variable(img.type(torch.float32))
            x_train.append(img)
            y_train.append(label)
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        x_test = []
        y_test = []
        for batch_idx, data in enumerate(testloader):
            img, label = data
            img = Variable(img.type(torch.float32))
            y_test.append(label)
            x_test.append(img)
        x_test = np.concatenate(x_test, axis=0)
        np.save('500_x_test.npy', np.reshape(x_test, (10000, 32, 32, 3))[0:500]*255)
        y_test = np.concatenate(y_test, axis=0)
        np.save('500_y_test.npy', y_test[0:500])
        # TODO: FIX min and Max pixel values

        data_obj = {'name': 'CIFAR100', 'percent': percent, 'CUDA_name': cuda_name,
                    'classifier': 'CIFAR100_Standard-1to1', 'min_pixel_value': np.float(np.min(x_train)),
                    'max_pixel_value': np.float(np.max(x_train)), 'input_size': (3, 32, 32), 'output_size': 100,
                    'dims': (1, 2, 3), 'L1_dir': 'Autoencoders/CIFAR100/499ep_AE_TinyImageNetAEStruct_0.05_noise_-1to1MSE.pth',
                    'L2_dir': 'Autoencoders/CIFAR100/autoencoder2.pth'}
    elif dataset == 'TinyImageNet':
        batch_size = 128
        #normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

        #testnorm = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
        data_dir = 'tiny-imagenet-200/'
        num_workers = {'train': 0, 'val': 0, 'test': 0}
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        }

        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                          for x in ['train', 'val', 'test']}

        dataloader = dataL.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers['train'])

        testloader = dataL.DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers['test'])
        x_train = []
        y_train = []
        for batch_idx, data in enumerate(dataloader):
            img, label = data
            img = Variable(img.type(torch.float32))
            x_train.append(img)
            y_train.append(label)
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        x_test = []
        y_test = []
        for batch_idx, data in enumerate(testloader):
            img, label = data
            img = Variable(img.type(torch.float32))
            y_test.append(label)
            x_test.append(img)
        x_test = np.concatenate(x_test, axis=0)

        y_test = np.concatenate(y_test, axis=0)


        data_obj = {'name': 'TinyImageNet', 'percent': percent, 'CUDA_name': cuda_name,
                    'classifier': 'TinyImageNetFix', 'min_pixel_value': np.float(np.min(x_train)),
                    'max_pixel_value': np.float(np.max(x_train)), 'input_size': (3, 64, 64), 'output_size': 200,
                    'dims': (1, 2, 3), 'L1_dir': 'Autoencoders/TinyImageNet/250ep_AE_0.05_noise_-1to1MSE.pth',
                    'L2_dir': 'Autoencoders/CIFAR100/autoencoder2.pth'}
    elif dataset == 'occupancy':

        batch_size = 128

        normalize = transforms.Normalize([1.6790455e+05, 2.0953647e+05, 9.7324562e+05, 4.9391105e+06, 3.1452608e+01], [1.0168638e+00, 5.5308995e+00, 1.9474380e+02, 3.1430164e+02, 8.5227750e-04])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        params = {'batch_size': batch_size,
                  'shuffle': False,
                  'num_workers': 1}
        trainset = occupancyDataset('Datasets/datatraining.txt', batch_size=batch_size, transform=transforms.ToTensor())
        dataloader = torch.utils.data.DataLoader(trainset, **params)
        testset = occupancyDataset('Datasets/datatest.txt', batch_size=batch_size,
                                 transform=transforms.ToTensor())
        testloader = torch.utils.data.DataLoader(testset, **params)
        x_train = []
        y_train = []
        nimages = 0
        mean = 0.
        std = 0.
        for batch_idx, data in enumerate(dataloader):
            img, label = data
            img = Variable(img.type(torch.float32))
            x_train.append(np.asarray(img))
            y_train.append(label)

        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        x_test = []
        y_test = []
        for batch_idx, data in enumerate(testloader):
            img, label = data
            img = Variable(img.type(torch.float32))
            y_test.append(label)
            x_test.append(np.asarray(img))
        x_test = np.concatenate(x_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)

        # Set min pixel and max pixel size input size
        data_obj = {'name': 'occupancy', 'percent': percent, 'CUDA_name': cuda_name,
                    'classifier': 'occupancy_classifier', 'min_pixel_value': np.float(np.min(x_train)),
                    'max_pixel_value': np.float(np.max(x_train)), 'input_size': 5, 'output_size': 10,
                    'dims': 1, 'L1_dir': 'Autoencoders/occupancy/999ep_MAE_noisy.pth',
                    'L2_dir': 'Autoencoders/occupancy/999ep_MSE_noisy.pth'}
    elif dataset == 'ImageNet':
        # TO TRAIN
        batch_size = 512
        batch_size = 64
        #normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

        #testnorm = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
        data_dir = '/data/ImageNet/'
        num_workers = {'train': 0, 'val': 0}
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        }

        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                          for x in ['train', 'val']}

        dataloader = dataL.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers['train'])

        testloader = dataL.DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers['val'])

        testloader = torch.utils.data.DataLoader(
            datasets.ImageFolder('/data/ImageNet/val/', transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True)

        x_train = []
        y_train = []
        '''
        for batch_idx, data in enumerate(dataloader):
            img, label = data
            img = Variable(img.type(torch.float32))
            x_train.append(img)
            y_train.append(label)
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        '''
        x_test = []
        y_test = []

        for batch_idx, data in enumerate(testloader):
            img, label = data
            img = Variable(img.type(torch.float32))
            y_test.append(label)
            x_test.append(img)
        x_test = np.concatenate(x_test, axis=0)

        y_test = np.concatenate(y_test, axis=0)

        #TODO actually check max and min pixel values
        data_obj = {'name': 'ImageNet', 'percent': percent, 'CUDA_name': cuda_name,
                    'classifier': 'ResNet18ImageNet', 'min_pixel_value': -1,
                    'max_pixel_value': 1, 'input_size': (3, 256, 256), 'output_size': 1000,
                    'dims': (1, 2, 3), 'L1_dir': 'Autoencoders/ImageNet/1ep_AE_0.05_noise_-1to1MSE.pth',
                    'L2_dir': 'Autoencoders/CIFAR100/autoencoder2.pth'}
        '''
        data_obj = {'name': 'TinyImageNet', 'percent': percent, 'CUDA_name': cuda_name,
                    'classifier': 'ResNet18ImageNet', 'min_pixel_value': np.float(np.min(x_train)),
                    'max_pixel_value': np.float(np.max(x_train)), 'input_size': (3, 256, 256), 'output_size': 1000,
                    'dims': (1, 2, 3), 'L1_dir': 'Autoencoders/TinyImageNet/250ep_AE_0.05_noise_-1to1MSE.pth',
                    'L2_dir': 'Autoencoders/CIFAR100/autoencoder2.pth'}
        '''
    return data_obj, x_test, y_test, x_train, y_train, dataloader, testloader