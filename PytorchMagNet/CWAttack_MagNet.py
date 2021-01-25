## test_defense.py -- test defense
##
## Copyright (C) 2017, Dongyu Meng <zbshfmmm@gmail.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import numpy as np
import os
import torch
if torch.cuda.device_count() == 1:
    cuda_name = "0"
else:
    cuda_name = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_name
    torch.cuda.set_device(2)
    print(torch.cuda.current_device())
print("using GPU " + str(cuda_name))
import torch.nn as nn
import os
import sys
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary
import copy
import pickle
from torch.autograd import Variable
from model.StandardCIFAR10 import StandardCIFAR10
from optparse import OptionParser
from torchvision.datasets import MNIST, CIFAR10, ImageNet
from util import make_one_hot
from setup_cifar import CIFAR
from PytorchMagNet.defense_models import autoencoder, autoencoder2
from loss import dice_score
from PytorchMagNet.worker import AEDetector, SimpleReformer, Classifier, DBDetector, IdReformer, Operator
dataset = "CIFAR"

data = CIFAR()

attacking = False


def get_args():
    parser = OptionParser()
    parser.add_option('--data_path', dest='data_path', type='string',
                      default='data/samples', help='data path')
    parser.add_option('--model_path', dest='model_path', type='string',
                      default='models/CIFAR10_Standard-1to1.pth', help='model_path')
    parser.add_option('--reformer_path', dest='reformer_path', type='string',
                      default='checkpoints/', help='reformer_path')
    parser.add_option('--detector_path', dest='detector_path', type='string',
                      default='checkpoints/', help='detector_path')
    parser.add_option('--classes', dest='classes', default=10, type='int',
                      help='number of classes')
    parser.add_option('--channels', dest='channels', default=3, type='int',
                      help='number of channels')
    parser.add_option('--width', dest='width', default=32, type='int',
                      help='image width')
    parser.add_option('--height', dest='height', default=32, type='int',
                      help='image height')
    parser.add_option('--model', dest='model', type='string',
                      help='model name(UNet, SegNet, DenseNet)', default="standardCIFAR")
    parser.add_option('--reformer', dest='reformer', type='string',
                      help='reformer name(autoencoder1 or autoencoder2)', default="autoencoder1")
    parser.add_option('--detector', dest='detector', type='string',
                      help='detector name(autoencoder1 or autoencoder2)', default="autoencoder1")
    parser.add_option('--gpu', dest='gpu', type='string',
                      default='gpu', help='gpu or cpu')
    parser.add_option('--model_device1', dest='model_device1', default=0, type='int',
                      help='device1 index number')
    parser.add_option('--model_device2', dest='model_device2', default=-1, type='int',
                      help='device2 index number')
    parser.add_option('--model_device3', dest='model_device3', default=-1, type='int',
                      help='device3 index number')
    parser.add_option('--model_device4', dest='model_device4', default=-1, type='int',
                      help='device4 index number')
    parser.add_option('--defense_model_device', dest='defense_model_device', default=0, type='int',
                      help='defense_model_device gpu index number')

    (options, args) = parser.parse_args()
    return options


def test(model, args):
    data_path = args.data_path
    n_channels = args.channels
    n_classes = args.classes
    data_width = args.width
    data_height = args.height
    gpu = args.gpu

    # Hyper paremter for MagNet
    thresholds = [0.1, 0.01, 0.05, 0.001, 0.005]

    reformer_model = None

    if args.reformer == 'autoencoder1':

        reformer_model = autoencoder(n_channels)

    elif args.reformer == 'autoencoder2':

        reformer_model = autoencoder2(n_channels)

    else:
        print("wrong reformer model : must be autoencoder1 or autoencoder2")
        raise SystemExit

    print('reformer model')
    reformer_model.cuda()
    summary(reformer_model, input_size=(n_channels, data_height, data_width), device='cuda')

    detector_model = None


    detector_model1 = autoencoder(n_channels)


    detector_model2 = autoencoder2(n_channels)


    print('detector model')

    #summary(detector_model, input_size=(n_channels, data_height, data_width), device='cuda')


    # set device configuration
    device_ids = []

    if gpu == 'gpu':

        if not torch.cuda.is_available():
            print("No cuda available")
            raise SystemExit

        device = torch.device(args.model_device1)
        device_defense = torch.device(args.defense_model_device)

        device_ids.append(args.model_device1)

        if args.model_device2 != -1:
            device_ids.append(args.model_device2)

        if args.model_device3 != -1:
            device_ids.append(args.model_device3)

        if args.model_device4 != -1:
            device_ids.append(args.model_device4)

    else:
        device = torch.device("cuda")
        device_defense = torch.device("cuda")

    detector = [AEDetector(detector_model1, device_defense, args.detector_path + 'autoencoder1' + '.pth', p=2),AEDetector(detector_model2, device_defense, args.detector_path + 'autoencoder2' + '.pth', p=2) ]
    #detector = [AEDetector(detector_model, device_defense, args.detector_path + args.detector + '.pth', p=2)]
    reformer = [SimpleReformer(reformer_model, device_defense, args.reformer_path + args.reformer + '.pth')]
    classifier = Classifier(model, device, args.model_path, device_ids)

    id_reformer = IdReformer()
    if dataset == "MNIST":
        detector_JSD = []
    else:
        detector_JSD = [DBDetector(id_reformer, ref, classifier, T=10) for i, ref in enumerate(reformer)]
        detector_JSD += [DBDetector(id_reformer, ref, classifier, T=40) for i, ref in enumerate(reformer)]
    # set testdataset

    detector_dict = dict()

    for i, det in enumerate(detector):
        detector_dict["II" + str(i)] = det
    for i, det in enumerate(detector_JSD):
        detector_dict["JSD" + str(i)] = det
    dr = dict([("II" + str(i), .005) for i in range(len(detector))] + [("JSD" + str(i), .01) for i in range(len(detector_JSD))])

    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    batch_size = 64
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    trainset = CIFAR10(root='./Datasets/', train=True,
                       download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
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

    testset = CIFAR10(root='./Datasets/', train=False,
                      download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    print('test_dataset : {}, test_loader : {}'.format(len(testloader.dataset), len(testloader.dataset)))

    # Defense with MagNet
    print('test start')




    counter = 0
    avg_score = 0.0
    correct = 0
    removed = 0
    operator = Operator(data, classifier, detector_dict, reformer[0])

    thresh = operator.get_thrs(dataloader, detector_dict, dr)
    print(thresh)

    if attacking:
        thrs = operator.get_thrs(dict((k, v * 4) for k, v in dr.items()))

        attack = CarliniL2(sess, [Pred2(x) for x in reformer], detector_dict, thrs, batch_size=100,
                           binary_search_steps=4, learning_rate=1e-2,
                           max_iterations=10000, targeted=True,
                           initial_const=1, confidence=1,
                           boxmin=0, boxmax=1)

        adv = attack.attack(dat, lab)
        np.save("/tmp/" + dataset + ".npy", adv)
    else:
        adv = np.load("/tmp/" + dataset + ".npy")

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(testloader):
            inputs = inputs.cuda().float()
            labels = labels.cuda().long()

            # Y prime values
            all_pass, _ = operator.filters(inputs, thresh)

            if len(all_pass) == 0:
                continue
            operate_results = operator.operate(inputs)
            filtered_results = operate_results[all_pass]

            pred = filtered_results.cuda().float()

            removed += len(inputs) - len(all_pass)
            correct += torch.sum(labels[all_pass] == torch.argmax(pred, axis=1))
            # statistics
            counter += 1

            del inputs, labels, pred


    print('avg_score : {:.4f}'.format(correct.item()/len(testloader.dataset)))
    print('removed ' + str(removed))



if __name__ == "__main__":

    args = get_args()

    n_channels = args.channels
    n_classes = args.classes

    model = None

    if args.model == 'standardCIFAR':
        model = StandardCIFAR10()
        tmp_dict = torch.load('models/CIFAR10_Standard-1to1.pth')
        model.load_state_dict(tmp_dict)
    else:
        print("wrong model : must be UNet, SegNet, or DenseNet")
        raise SystemExit

    print('segmentation model')
    model.cuda()
    summary(model, input_size=(n_channels, args.height, args.width), device='cuda')

    test(model, args)
