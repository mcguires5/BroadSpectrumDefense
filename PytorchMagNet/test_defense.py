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
from model.defense_models import autoencoder, autoencoder2

from PytorchMagNet.worker import AEDetector, SimpleReformer, Classifier, DBDetector, IdReformer, Operator
from PytorchMagNet.pytorch_cw2_MagNet import AttackCarliniWagnerL2
from PytorchMagNet.pytorch_cw2_MagNet_TwoSearch import AttackCarliniWagnerL2TwoSearch
from datasets_objects import generate_struct
from helpers import *
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


def test():
    dataset = "CIFAR100"
    percent = 0.01
    data_obj, x_test, y_test, x_train, y_train, dataloader, testloader = generate_struct(1, dataset,
                                                                                         percent, cuda_name)

    device_defense = torch.device("cuda")
    device = torch.device("cuda")
    if data_obj['name'] == 'CIFAR10':
        n_channels = 3
        detector_model1 = autoencoder(n_channels)
        detector_model2 = autoencoder2(n_channels)
        detector_model2.load_state_dict(torch.load('checkpoint/CIFAR10/49ep_autoencoder2_-1to1MSE.pth'))
        detector = [AEDetector(detector_model2, device_defense, p=1)]
        reformer = [SimpleReformer(detector_model2, device_defense, data_obj['min_pixel_value'], data_obj['max_pixel_value'])]
        model = StandardCIFAR10()
        tmp_dict = torch.load('Classifiers/Evasion/NN/CIFAR10_Standard-1to1.pth')
        model.load_state_dict(tmp_dict)
        model.cuda()
        classifier = Classifier(model, device, 'Classifiers/Evasion/NN/CIFAR10_Standard-1to1.pth', 1)
        id_reformer = IdReformer()

        detector_JSD = [DBDetector(id_reformer, ref, classifier, T=10) for i, ref in enumerate(reformer)]
        detector_JSD += [DBDetector(id_reformer, ref, classifier, T=40) for i, ref in enumerate(reformer)]
        detector_dict = dict()

    elif data_obj['name'] == 'MNIST':
        n_channels = 1
        detector_model1 = autoencoder(n_channels)
        detector_model2 = autoencoder2(n_channels)
        detector_model1.load_state_dict(torch.load('checkpoint/MNIST/autoencoder1.pth'))
        detector_model2.load_state_dict(torch.load('checkpoint/MNIST/autoencoder2.pth'))

        detector = [AEDetector(detector_model1, device_defense, p=2, dims=data_obj['dims']),
                    AEDetector(detector_model2, device_defense, p=2, dims=data_obj['dims'])]

        reformer = [SimpleReformer(detector_model1, device_defense,
                                    data_obj['min_pixel_value'], data_obj['max_pixel_value'])]
        '''
        detector = [AEDetector(detector_model1, device_defense, p=2, dims=data_obj['dims']),
                    AEDetector(detector_model2, device_defense, p=2, dims=data_obj['dims']),
                    AEDetectorC([detector_model1,detector_model2], device_defense, p=2)]

        reformer = [SimpleReformerC([detector_model1, detector_model2], device_defense,
                                   data_obj['min_pixel_value'], data_obj['max_pixel_value'])]
        '''
        model = get_model(data_obj)

        id_reformer = IdReformer()
        classifier = Classifier(model, device, 1)
        #detector_JSD = [DBDetector(id_reformer, ref, classifier, T=10) for i, ref in enumerate(reformer)]
        #detector_JSD += [DBDetector(id_reformer, ref, classifier, T=40) for i, ref in enumerate(reformer)]
        detector_JSD = []
        detector_dict = dict()
    elif data_obj['name'] == 'CIFAR100':
        n_channels = 3
        detector_model1 = autoencoder(n_channels)
        detector_model2 = autoencoder2(n_channels)
        detector_model1.load_state_dict(torch.load('Autoencoders/CIFAR100/999ep_autoencoder_-1to1MSE.pth'))
        detector = [AEDetector(detector_model1, device_defense,  p=2)]
        reformer = [SimpleReformer(detector_model1, device_defense, data_obj['min_pixel_value'], data_obj['max_pixel_value'])]

        model = get_model(data_obj)
        classifier = Classifier(model, device, 1)
        id_reformer = IdReformer()

        detector_JSD = [DBDetector(id_reformer, ref, classifier, T=10) for i, ref in enumerate(reformer)]
        detector_JSD += [DBDetector(id_reformer, ref, classifier, T=40) for i, ref in enumerate(reformer)]

        detector_dict = dict()
    elif data_obj['name'] == 'occupancy':
        aeMSE = get_autoencoder(data_obj)
        aeMSE.load_state_dict(torch.load(data_obj['L2_dir']))
        aeMAE = get_autoencoder(data_obj)
        aeMAE.load_state_dict(torch.load(data_obj['L1_dir']))
        detector = [AEDetector(aeMAE, device_defense, data_obj['L1_dir'], p=1, dims=data_obj['dims']),
                    AEDetector(aeMSE, device_defense, data_obj['L2_dir'], p=2, dims=data_obj['dims'])]
        reformer = [SimpleReformer(aeMAE, device_defense, data_obj['L1_dir'], data_obj['min_pixel_value'], data_obj['max_pixel_value'])]
        model = get_model(data_obj)
        model.cuda()

        id_reformer = IdReformer()
        classifier = Classifier(model, device, 'Classifiers/Evasion/NN/occupancy_classifier.pth', 1)
        detector_JSD = [DBDetector(id_reformer, ref, classifier, T=10) for i, ref in enumerate(reformer)]
        detector_JSD += [DBDetector(id_reformer, ref, classifier, T=40) for i, ref in enumerate(reformer)]
        detector_dict = dict()

    for i, det in enumerate(detector):
        detector_dict["II" + str(i)] = det
    for i, det in enumerate(detector_JSD):
        detector_dict["JSD" + str(i)] = det
    dr = dict([("II" + str(i), percent) for i in range(len(detector))] + [("JSD" + str(i), percent) for i in range(len(detector_JSD))])
    operator = Operator(x_train, classifier, detector_dict, reformer[0], data_obj['dims'])
    thresh = operator.get_thrs(dataloader, detector_dict, dr)

    print('test_dataset : {}, test_loader : {}'.format(len(testloader.dataset), len(testloader.dataset)))

    # Defense with MagNet
    print('test start')
    attacking = True

    if attacking:
        c = 20
        attack = AttackCarliniWagnerL2TwoSearch(classifier.model, reformer, detector_dict, thresh, debug=False,
                           search_steps=6, learning_rate=1e-2,
                           max_steps=10000, targeted=False,
                           confidence=c,
                           clip_min=data_obj['min_pixel_value'], clip_max=data_obj['max_pixel_value'])
        adv = []
        for batch_idx, data in enumerate(dataloader):
            if batch_idx <= 16:
                img, label = data
                img = img.cuda()
                label = label.cuda()
                adv.append(attack.run(img, label, batch_idx))
                print(batch_idx)
        adv = np.concatenate(adv, axis=0)
        #np.save("AttackedData/Evasion/CIFAR10/CIFAR10_Standard-1to1/FullKnowledgeCWL2_BS6_Const2Search_1k_c_" + str(c).zfill(3) + ".npy", adv)
        pwd = os.getcwd()
        np.save(pwd + '/Results/' + data_obj['name'] + '/' + data_obj['classifier'] + '/' + "/FullKnowledgeCWL2_BS6_Const2Search_1k_c_" + str(
            c).zfill(3) + ".npy", adv)
    else:
        adv = np.load("/tmp/" + dataset + ".npy")


    counter = 0
    avg_score = 0.0
    correct = 0
    removed = 0

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
    test()
