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
    cuda_name = "4"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_name
    print(torch.cuda.current_device())
print("using GPU " + str(cuda_name))
import torch.nn as nn
import os
import sys
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision

from torchsummary import summary
import copy
import pickle
from torch.autograd import Variable
from optparse import OptionParser
from torchvision.datasets import MNIST, CIFAR10, ImageNet
from Networks.CIFAR10 import StandardCIFAR10
from PytorchMagNet.defense_models import autoencoder, autoencoder2
from PytorchMagNet.worker import AEDetector, SimpleReformer, Classifier, Operator, IdReformer, DBDetector
import collections
import itertools
import wandb
dataset = "CIFAR"


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



    reformer_model1 = autoencoder(n_channels)



    reformer_model2 = autoencoder2(n_channels)




    #print('reformer model')
    #reformer_model.cuda()
    #summary(reformer_model, input_size=(n_channels, data_height, data_width))

    detector_model = None

    detector_model1 = autoencoder(n_channels)
    detector_model2 = autoencoder2(n_channels)

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

    # TODO: Check difference calculation and make it take into account the p norm
    #detector = [AEDetector(detector_model1, device_defense, args.detector_path + 'autoencoder1' + '.pth', p=2),
    #            AEDetector(detector_model2, device_defense, args.detector_path + 'autoencoder2' + '.pth', p=2)]
    detector = [AEDetector(detector_model2, device_defense, args.detector_path + 'autoencoder2' + '.pth', p=1)]
    reformer = [SimpleReformer(reformer_model2, device_defense, args.reformer_path + 'autoencoder2' + '.pth')]
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

    transform2 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    testnorm = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ])
    batch_size = 64
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
    np.save('500_x_test.npy', np.reshape(x_test, (10000, 32, 32, 3))[0:500] * 255)
    y_test = np.concatenate(y_test, axis=0)
    np.save('500_y_test.npy', y_test[0:500])



    # set testdataset
    # Default values for hyper-parameters we're going to sweep over
    #run = wandb.init(project='BroadSpectrumDefense',
    #                 group="CIFAR10")
    # wandb.config.update(allow_val_change=True)
    # wandb.config.update({'s':s}, allow_val_change=True)
    # print(wandb.config)
    avg = 0
    all_data = []
    d = []
    c = 0
    pwd = os.getcwd()

    path = os.listdir(os.path.join(pwd, 'attack_data/'))
    operator = Operator(data, classifier, detector_dict, reformer[0])
    thresh = operator.get_thrs(dataloader, detector_dict, dr)
    print(thresh)
    path = sorted(path)
    for attack_type in path:
        print(attack_type)
        isfile = False
        p = os.path.join(pwd, 'attack_data/') + attack_type
        if os.path.isfile(p):
            isfile = True
            x_test_adv = np.load(p)

        if isfile:
            # print(attack_type)

            # evaluate_attack_impact(x_test_adv)

            NaN_idx = np.unique(np.where(np.isnan(x_test_adv))[0])
            x_test_adv = np.delete(x_test_adv, NaN_idx, axis=0)
            x_test_prune = np.delete(x_test, NaN_idx, axis=0)
            x_test_prune = x_test_prune[:len(x_test_adv)]
            y_test_adv = np.delete(y_test, NaN_idx, axis=0)
            y_test_adv = y_test_adv[:len(x_test_adv)]

            eval_def(operator, x_test_adv, y_test_adv, thresh)

            '''
            d.append(
                [ret['No_defense_accuracy'], ret['Total-detected-acc'], attack_type, ret['Comb_Raw-Accuracy'],
                 ret['Comb-detected'], ret['Class_diff_samples'], ret['Average-L2-Dist'], ret['Total-detected-diff'],
                 ret['Number-Attack-Samples'], ret['Total-JSD-detected-acc'], ret['newJSD']])
            if data_obj['name'] == 'occupancy':
                wandb.log(
                    {"Table": wandb.Table(data=d, columns=["Original Accuracy", "Defended Accuracy", "Attack Type",
                                                           "Autoencoded Accuracy", "Distance Based Detection",
                                                           "Class Difference Detected", "Average L2 Distance",
                                                           'Total Removed Samples', 'Attack Samples Generated',
                                                           'JSD Samples Accuracy', 'Additional JSD Removed']),
                     'Original Accuracy': ret['No_defense_accuracy'],
                     'Autoencoded Accuracy': ret['Comb_Raw-Accuracy'],
                     'Defended Accuracy': ret['Total-detected-acc'],
                     'Average L2 Distance': ret['Average-L2-Dist'],
                     'Total Removed Samples': ret['Total-detected-diff'],
                     'Distance Based Detection': ret['Comb-detected'],
                     'Attack Samples Generated': ret['Number-Attack-Samples'],
                     'JSD Samples Removed': ret['JSD_rejected_samples'],
                     'Total Acc With JSD': ret['Total-JSD-detected-acc'],
                     'New JSD Detections': ret['newJSD']})
            '''
    print('test_dataset : {}, test_loader : {}'.format(len(testloader.dataset), len(testloader.dataset)))

    # Defense with MagNet
    print('test start')

    for thrs in thresholds:

        counter = 0
        avg_score = 0.0
        correct = 0
        removed = 0
        dr = torch.tensor(thrs)


        print(thresh)


def eval_def(operator, x_test_adv, y_test_adv, thresh, device):
    with torch.no_grad():
        removed= 0
        counter = 0
        correct = torch.tensor(0)
        collection = {}
        all_reformed = torch.tensor(0).to(device)
        for i in range(0, len(x_test_adv), 64):
            inputs = torch.as_tensor(x_test_adv[i:i+64]).to(device).float()
            labels = torch.as_tensor(y_test_adv[i:i+64]).to(device).long()

            # Y prime values
            operate_results = operator.operate(inputs, thresh)
            all_reformed += torch.sum(labels.to(device) == torch.argmax(operate_results.to(device), axis=1))
            all_pass, collector_tmp = operator.filters(inputs, thresh)
            for k in collector_tmp.keys():
                if k in collection:
                    collection[k] += len(inputs) - collector_tmp[k]
                else:
                    collection[k] = len(inputs) - collector_tmp[k]

            filtered_results = operate_results[all_pass]

            pred = filtered_results.to(device).float()

            removed += len(inputs) - len(all_pass)
            if len(all_pass) == 0:
                continue
            # statistics
            counter += 1

            correct = correct +torch.sum(labels[all_pass] == torch.argmax(pred, axis=1))
            del inputs, labels, pred

    print(collection)
    #print('removed ' + str(removed))
    print('number of adv' + str(len(x_test_adv)))
    print('total ' + str((correct.item() + removed) / len(x_test_adv)))
    print('Clean' + str(correct.item() / len(x_test_adv)))
    return (correct.item() + removed) / len(x_test_adv), collection, correct.item() / len(x_test_adv), removed / len(x_test_adv), all_reformed.item() / len(x_test_adv)

if __name__ == "__main__":

    args = get_args()

    n_channels = args.channels
    n_classes = args.classes

    model = None

    if args.model == 'standardCIFAR':
        model = StandardCIFAR10()
        tmp_dict = torch.load('Classifiers/Evasion/NN/CIFAR10_Standard-1to1.pth')
        model.load_state_dict(tmp_dict)
        model.cuda()
    else:
        print("wrong model : must be UNet, SegNet, or DenseNet")
        raise SystemExit

    print('segmentation model')
    summary(model, input_size=(n_channels, args.height, args.width))

    test(model, args)
