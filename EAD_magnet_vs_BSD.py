import matplotlib
matplotlib.use('Agg')
from scipy.stats import entropy
from numpy.linalg import norm
from matplotlib.ticker import FuncFormatter
from keras.models import Sequential, load_model
from keras.activations import softmax
from keras.layers import Lambda
import numpy as np
import pylab
import os
import matplotlib.pyplot as plt
import pickle

import os
import torch

if torch.cuda.device_count() == 1 and torch.cuda.current_device() == 0:
    cuda_name = "0"
    print("WARNING: CHANGED TO CUDA DEVICE 0")
else:
    cuda_name = "4"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_name
print("using GPU " + str(cuda_name))
print(torch.cuda.current_device())
device = torch.device("cuda:" + cuda_name)
import torch.nn as nn
import torch.optim as optim
from Autoencoders.code.autoencoder import VAE, newConvAutoencoder, defendedConvAutoencoder, defendedVAE

import numpy as np
import torchvision
from art import attacks
from art.classifiers import PyTorchClassifier
from torchvision import transforms
from torch.autograd import Variable
import torchvision.models as models
import pickle
from collections import OrderedDict
import csv
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import eagerpy as ep
import cleverhans
# from Networks.MNIST import Net, DefendedClassifier
from Datasets.occupancy_dataset import occupancyDataset
from Networks.CIFAR10 import Net, CNN, StandardCIFAR10
from Networks.MNIST import Net, StandardMNIST
from Networks.occupancy import occupancy_classifier
from Networks.ResNet import ResNet18
from AttackedData.CWAttack.pytorch_cw2 import AttackCarliniWagnerL2
from AttackedData.CWAttack.pytorch_custom import AttackCarliniWagnerCustom
from torchsummary import summary
import foolbox
import advertorch
import wandb
from helpers import *
from evaluate_defense import evaluate_defense_impact
import matplotlib.pyplot as plt
from datasets_objects import generate_struct
from scipy.stats import wilcoxon
from PytorchMagNet.worker import AEDetector, SimpleReformer, DBDetector, Operator, Classifier, IdReformer, CDDetector, AEDetectorC, SimpleReformerC, CDDetectorC, CDDetectorNoisy, CDDetectorMedian
from PytorchMagNet.evaluate_adversarial import eval_def
from PytorchMagNet.defense_models import autoencoder, autoencoder2
from Networks.CIFAR100 import Wide_ResNet
from Autoencoders.code.autoencoder import TinyImageNetConvAE
from collections import *

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


flags.DEFINE_boolean('select', True, 'Select correctly classified examples for the experiement.')
flags.DEFINE_integer('nb_examples', 100, 'The number of examples selected for attacks.')
flags.DEFINE_boolean('balance_sampling', False, 'Select the same number of examples for each class.')
flags.DEFINE_boolean('test_mode', False, 'Only select one sample for each class.')

flags.DEFINE_string('attacks',
                    "FGSM?eps=0.1;BIM?eps=0.1&eps_iter=0.02;JSMA?targeted=next;CarliniL2?targeted=next&batch_size=100&max_iterations=1000;CarliniL2?targeted=next&batch_size=100&max_iterations=1000&confidence=2",
                    'Attack name and parameters in URL style, separated by semicolon.')
flags.DEFINE_float('clip', -1, 'L-infinity clip on the adversarial perturbations.')
flags.DEFINE_boolean('visualize', True, 'Output the image examples for each attack, enabled by default.')

# MNIST
#flags.DEFINE_string('robustness', 'FeatureSqueezing?squeezer=bit_depth_1;FeatureSqueezing?squeezer=median_filter_2_2;FeatureSqueezing?squeezer=median_filter_3_3;FeatureSqueezing?squeezers=bit_depth_1&distance_measure=l1&fpr=0.01;FeatureSqueezing?squeezers=bit_depth_2&distance_measure=l1&fpr=0.01;FeatureSqueezing?squeezers=bit_depth_1,median_filter_2_2&distance_measure=l1&fpr=0.01;', 'Supported: FeatureSqueezing.')
flags.DEFINE_string('robustness', 'FeatureSqueezing?squeezer=bit_depth_5;FeatureSqueezing?squeezer=bit_depth_4;FeatureSqueezing?squeezer=median_filter_2_2;FeatureSqueezing?squeezer=non_local_means_color_13_3_4;FeatureSqueezing?squeezers=bit_depth_1&distance_measure=l1&fpr=0.01;FeatureSqueezing?squeezers=bit_depth_2&distance_measure=l1&fpr=0.01;FeatureSqueezing?squeezers=bit_depth_3&distance_measure=l1&fpr=0.01;FeatureSqueezing?squeezers=bit_depth_4&distance_measure=l1&fpr=0.01;FeatureSqueezing?squeezers=bit_depth_5&distance_measure=l1&fpr=0.01;FeatureSqueezing?squeezers=median_filter_2_2&distance_measure=l1&fpr=0.01;FeatureSqueezing?squeezers=median_filter_3_3&distance_measure=l1&fpr=0.01;FeatureSqueezing?squeezers=non_local_means_color_11_3_2&distance_measure=l1&fpr=0.01;FeatureSqueezing?squeezers=non_local_means_color_11_3_4&distance_measure=l1&fpr=0.01;FeatureSqueezing?squeezers=non_local_means_color_13_3_2&distance_measure=l1&fpr=0.01;FeatureSqueezing?squeezers=non_local_means_color_13_3_4&distance_measure=l1&fpr=0.01;FeatureSqueezing?squeezers=bit_depth_5,median_filter_2_2,non_local_means_color_13_3_2&distance_measure=l1&fpr=0.01;', 'Supported: FeatureSqueezing.')
flags.DEFINE_string('detection', '', 'Supported: feature_squeezing.')
flags.DEFINE_boolean('detection_train_test_mode', True, 'Split into train/test datasets.')

flags.DEFINE_string('result_folder', "results", 'The output folder for results.')
flags.DEFINE_boolean('verbose', False, 'Stdout level. The hidden content will be saved to log files anyway.')


def prepare_data(dataset, idx):
    """
    Extract data from index.

    dataset: Full, working dataset. Such as MNIST().
    idx: Index of test examples that we care about.
    return: X, targets, Y
    """
    return dataset.test_data[idx], dataset.test_labels[idx], np.argmax(dataset.test_labels[idx], axis=1)


def save_obj(obj, name, directory='./attack_data/'):
    with open(os.path.join(directory, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, directory='./attack_data/'):
    if name.endswith(".pkl"): name = name[:-4]
    with open(os.path.join(directory, name + '.pkl'), 'rb') as f:
        return pickle.load(f)


def plot_sweep_conf(dataset, percent, class_type, attack):
    pylab.rcParams['figure.figsize'] = 6, 4
    fig = plt.figure(1, (6, 4))
    ax = fig.add_subplot(1, 1, 1)
    data_obj, x_test, y_test, x_train, y_train, dataloader, testloader = generate_struct(1, dataset,
                                                                                         percent, cuda_name)
    pwd = os.getcwd()
    path = sorted(os.listdir(os.path.join(pwd, 'AttackedData/Evasion/' + data_obj['name'] + '/' + class_type + '_classifier/')))

    confs = []
    JSD_acc = []
    Conf_acc = []
    No_JSD_CONF = []
    for attack_type in path:
        # attack_type = "SinglePixelFoolbox.npy"
        if os.path.isfile(
                os.path.join(os.path.join(pwd, 'AttackedData/Evasion/' + data_obj['name'] + '/' + class_type + '_classifier/'),
                             attack_type)) and ('White_' + attack + '_Foolbox_Logits_1k_c_' in attack_type):
            x_test_adv = np.load(
                'AttackedData/Evasion/' + data_obj['name'] + '/' + class_type + '_classifier/' + attack_type)
            # evaluate_attack_impact(x_test_adv)
            print(attack_type)
            confs.append(int(attack_type.split('_')[-1].split('.')[0]))

            NaN_idx = np.unique(np.where(np.isnan(x_test_adv))[0])
            x_test_adv = np.delete(x_test_adv, NaN_idx, axis=0)
            x_test_prune = np.delete(x_test, NaN_idx, axis=0)
            x_test_prune = x_test_prune[:len(x_test_adv)]
            y_test_adv = np.delete(y_test, NaN_idx, axis=0)
            y_test_adv = y_test_adv[:len(x_test_adv)]

            if x_test_adv.shape[1:] != (3, 32, 32):
                x_test_adv = np.moveaxis(x_test_adv, 3, -3)

            ret, ret2, ret3, det = evaluate_defense_impact('NN', data_obj, x_train, 1, x_test_adv,
                                                      y_test_adv, attack_type, dataloader, x_test_prune)
            print(ret)
            JSD_acc.append(float(ret['Total-JSD-detected-acc'])/100)
            Conf_acc.append(float(det['w/o_JSD']))
            No_JSD_CONF.append(float(ret['Total-detected-acc'])/100)


    print(JSD_acc)
    print(Conf_acc)
    print(No_JSD_CONF)
    print(confs)
    if data_obj['name'] == 'MNIST':
        MagNet = [.79, .36, .18, .17, .20, .24, .36, .41, .48, float('NaN')]
    #elif data_obj['name'] == 'CIFAR10':
        # EAD
        # MagNet = [.70, float('NaN'), .28, .3, .38, .45, .62, .71, .83, .9, .93, .95]
        # CW
        #MagNet = [.92, .8, .9, .95, 1, 1,1, 1,1,1,1]
        #MagNet = [.92, .8, .9, .95, 1, 1, 1, 1, 1, 1]
    size = 2.5
    plt.plot(confs, MagNet, c="green", label="MagNet", marker="x", markersize=size)
    plt.plot(confs, JSD_acc, c="orange", label="BSD with JSD", marker="o", markersize=size)
    plt.plot(confs, Conf_acc, c="blue", label="BSD with Confidence", marker="^", markersize=size)
    plt.plot(confs, No_JSD_CONF, c="gray", label="BSD w/o Confidence and JSD", marker="<", markersize=size)
    pylab.legend(loc='lower left', bbox_to_anchor=(0.02, 0.1), prop={'size': 8})
    plt.grid(linestyle='dotted')
    plt.xlabel(r"Confidence in " + attack )
    plt.ylabel("Classification accuracy")
    graph_name = attack + " against BSD at " + str(percent*100) + "% Detection Threshold " + data_obj['name']
    plt.title(graph_name)
    if data_obj['name'] == 'MNIST':
        plt.xlim(min(confs) - 1.0, 41)
    else:
        plt.xlim(min(confs) - 1.0, max(confs) + 1.0)
    plt.ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))

    plt.show()
    save_path = os.path.join(pwd + '/Results/' + data_obj['name'] + '/' + class_type + '_classifier/' + graph_name + ".pdf")
    plt.savefig(save_path)
    plt.clf()
    print("saved image")


def plot_clean(dataset, percent):
    pylab.rcParams['figure.figsize'] = 6, 4
    fig = plt.figure(1, (6, 4))
    ax = fig.add_subplot(1, 1, 1)
    data_obj, x_test, y_test, x_train, y_train, dataloader, testloader = generate_struct(1, dataset,
                                                                                         percent, cuda_name)
    pwd = os.getcwd()

    clean_adj_JSD = []
    clean_adj_conf = []
    conf_det = []
    class_div_det = []
    distance_det = []
    JSD_det = []
    JSD_detections = {}
    MagNet = []
    MagNetCDD = []
    no_def_acc = []
    BSD_removed = []
    MagNet_removed = []
    BSD_reformed = []
    MagNet_reformed = []
    BSD_AllReformed = []
    MagNet_AllReformed = []
    FS_Acc = []
    for perc in percent:
        data_obj['percent'] = perc
        # evaluate_attack_impact(x_test_adv)
        print(perc)

        operatorCDD, threshCDD = setup_MagNetCDD(dataloader, perc, data_obj)
        MagNetAccCDD, MagNetCollectionCDD, clean_accCDD, BSDRemoved, BSDallref = eval_def(operatorCDD, x_test, y_test, threshCDD, device)

        # MAGNET
        operator, thresh = setup_MagNet(dataloader, perc, data_obj)
        MagNetAcc, MagNetCollection, clean_acc, MagNetRemoved, MagNetallref = eval_def(operator, x_test, y_test, thresh, device)

        #FSAcc, legit_acc_FS = setup_FS(dataloader, percent, data_obj, x_test, y_test, "Clean")
        #print("FS ACC " + str(FSAcc) + " Legit Acc " + str(legit_acc_FS))
        correct_nodef = 0
        for i in range(0, len(x_test), 64):
            Y_logit = operator.no_def_acc(torch.as_tensor(x_test[i:i + 64]))
            correct_nodef += torch.sum(torch.argmax(Y_logit, dim=1) == torch.as_tensor(y_test[i:i + 64]))
        no_def_acc.append(correct_nodef.item() / len(x_test))

        #ret, ret2, ret3, det, JSD_detectors = evaluate_defense_impact('NN', data_obj, x_train, 1, x_test,
        #                                                              y_test, "Clean Data", dataloader, x_test)
        #FS_Acc.append(FSAcc)
        MagNet.append(clean_acc)
        MagNetCDD.append(clean_accCDD)
        BSD_removed.append(BSDRemoved)
        MagNet_removed.append(MagNetRemoved)
        BSD_reformed.append(clean_accCDD)
        MagNet_reformed.append(clean_acc)
        MagNet_AllReformed.append(MagNetallref)
        BSD_AllReformed.append(BSDallref)
        #if JSD_detections == {}:
            #for key in JSD_detectors.keys():
                #JSD_detections[key] = []
        #for key in JSD_detections.keys():
            #JSD_detections[key].append(sum(JSD_detectors[key]) / len(x_test))
        #conf_det.append(float(ret['Conf_rejected_samples']) / len(x_test))
        #class_div_det.append(float(ret['Class_diff_samples']) / len(x_test))
        #distance_det.append(float(ret['Comb-detected']) / len(x_test))
        #JSD_det.append(float(ret['JSD_rejected_samples']) / len(x_test))
        #clean_adj_JSD.append(float(ret['Clean_Adj_Acc_JSD']))
        #clean_adj_conf.append(float(ret['Clean_Adj_Acc_conf']))
    print(clean_adj_JSD)
    #print(FS_Acc)
    print(no_def_acc)
    print(percent)
    print(MagNetCDD)

    size = 2.5
    #for key in JSD_detections.keys():
        #plt.plot(percent, JSD_detections[key], linestyle='dashed', label="JSD " + str(key) + " Accuracy")
    #plt.plot(percent, clean_adj_JSD, linestyle='solid', c="green", label="Clean Accuarcy BSD")
    #plt.plot(percent, class_div_det, linestyle='dashdot', c="orange", label="Class Divergence Detector")
    #plt.plot(percent, JSD_det, linestyle='dashdot', c="gray", label="All JSD detectors")
    #plt.plot(percent, distance_det, linestyle='dashdot', c="red", label="Distance Based Detections")
    plt.plot(percent, no_def_acc, linestyle='solid', c="black", label="No Defense Accuracy")
    plt.plot(percent, MagNet, c="magenta", linestyle='solid', label="MagNet")
    plt.plot(percent, MagNetCDD, c="blue", linestyle='solid', label="BSD")
    #plt.plot(percent, FS_Acc, c="red", linestyle='solid', label="FS")
    #plt.plot(percent, MagNet_AllReformed, c="red", linestyle='solid', label="MagNet All Reformed")
    #plt.plot(percent, BSD_AllReformed, c="green", linestyle='solid', label="BSD All Reformed")
    pylab.legend(loc='lower left', bbox_to_anchor=(0.02, 0.1), prop={'size': 8})
    plt.grid(linestyle='dotted')
    plt.xlabel(r"Threshold")
    plt.ylabel("Classification accuracy")
    graph_name = "New Clean Accuracy " + data_obj['name']
    #plt.title(graph_name)
    plt.xlim(min(percent), max(percent))
    plt.ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))

    plt.show()
    save_path = os.path.join(pwd + '/Results/' + data_obj['name'] + '/' + data_obj['classifier'] + '/' + graph_name + ".pdf")
    plt.savefig(save_path)
    plt.clf()
    print("saved image")


def plot_contributions(dataset, percent, attack):
    pylab.rcParams['figure.figsize'] = 6, 4
    fig = plt.figure(1, (6, 4))
    ax = fig.add_subplot(1, 1, 1)
    data_obj, x_test, y_test, x_train, y_train, dataloader, testloader = generate_struct(1, dataset,
                                                                                         percent, cuda_name)
    pwd = os.getcwd()
    path = sorted(os.listdir(os.path.join(pwd, 'AttackedData/Evasion/' + data_obj['name'] + '/' + data_obj['classifier'] + '/')))


    # MagNet
    operator, thresh = setup_MagNet(dataloader, percent, data_obj)

    operatorCDD, threshCDD = setup_MagNetCDD(dataloader, percent, data_obj)
    print(thresh)
    print('\n')
    confs = []
    BSDacc = []
    JSD_det = []
    conf_acc = []
    conf_det = []
    class_div_det = []
    distance_det = []
    JSD_detections = {}
    no_def = []
    MagNet = []
    JSD_detections = {}
    attack_name = []
    no_def_acc = []
    MagNet = []
    BSD_removed = []
    MagNet_removed = []
    BSD_reformed = []
    MagNet_reformed = []
    MagNet_Allreformed = []
    BSD_Allreformed = []
    Avg_distance = []
    for attack_type in path:
        # attack_type = "SinglePixelFoolbox.npy"
        if data_obj['name'] == 'CIFAR10':
            middle_name =  '_Foolbox_Logits_1k_c_'
        else:
            middle_name = '_Foolbox_Logits_c_'
        if os.path.isfile(
                os.path.join(os.path.join(pwd, 'AttackedData/Evasion/' + data_obj['name'] + '/' + data_obj['classifier'] + '/'),
                             attack_type)) and ('White_' + attack + middle_name in attack_type):
            x_test_adv = np.load(
                'AttackedData/Evasion/' + data_obj['name'] + '/' + data_obj['classifier'] + '/' + attack_type)
            # evaluate_attack_impact(x_test_adv)
            print(attack_type)
            confs.append(int(attack_type.split('_')[-1].split('.')[0]))

            NaN_idx = np.unique(np.where(np.isnan(x_test_adv))[0])
            x_test_adv = np.delete(x_test_adv, NaN_idx, axis=0)
            x_test_prune = np.delete(x_test, NaN_idx, axis=0)
            x_test_prune = x_test_prune[:len(x_test_adv)]
            y_test_adv = np.delete(y_test, NaN_idx, axis=0)
            y_test_adv = y_test_adv[:len(x_test_adv)]

            # MAGNET
            correct_nodef = 0
            for i in range(0, len(x_test_adv), 64):
                Y_logit = operator.no_def_acc(torch.as_tensor(x_test_adv[i:i+64]))
                correct_nodef += torch.sum(torch.argmax(Y_logit, dim=1) == torch.as_tensor(y_test[i:i+64]))
            no_def.append(correct_nodef.item()/len(x_test_adv))
            # MAGNET
            MagNetAcc, MagNetCollection, clean_acc, MagNet_removed_percent, MagNet_allref = eval_def(operator, x_test_adv, y_test_adv, thresh, device)

            BSD_acc, CDDCollection, BSD_clean_acc, BSD_removed_percent, BSD_allref = eval_def(operatorCDD, x_test_adv, y_test_adv, threshCDD, device)
            if x_test_adv.shape[1:] != (3, 32, 32) and data_obj['name'] == 'CIFAR10':
                x_test_adv = np.moveaxis(x_test_adv, 3, -3)

            '''
            ret, ret2, ret3, det, JSD_detectors = evaluate_defense_impact('NN', data_obj, x_train, 1, x_test_adv,
                                                      y_test_adv, attack_type, dataloader, x_test_prune)
            if JSD_detections == {}:
                for key in JSD_detectors.keys():
                    JSD_detections[key] = []
            for key in JSD_detections.keys():
                JSD_detections[key].append(sum(JSD_detectors[key])/len(x_test_adv))
            '''

            Avg_distance.append(torch.sqrt(torch.sum((x_test_adv-x_test)^2)))
            MagNet.append(MagNetAcc)
            BSDacc.append(BSD_acc)
            #conf_det.append(float(ret['Conf_rejected_samples']) / len(x_test_adv))
            #conf_acc.append(float(det['w/o_JSD']))
            #class_div_det.append(float(ret['Class_diff_samples'])/len(x_test_adv))
            #distance_det.append(float(ret['Comb-detected'])/len(x_test_adv))
            #JSD_det.append(float(ret['JSD_rejected_samples'])/len(x_test_adv))
            #JSD_acc.append(float(ret['Total-JSD-detected-acc'])/100)

    print(BSDacc)
    print(class_div_det)
    print(distance_det)
    print(JSD_det)
    print(confs)
    print(MagNet)
    print("No Defense " + str(wilcoxon(BSDacc, no_def, alternative="greater")))
    '''
    if data_obj['name'] == 'MNIST':
        if attack == 'ElasticNetAttack':
            # EAD
            MagNet = [.79, .36, .18, .17, .20, .24, .36, .41, .48]
        # CWL2
        elif attack == 'CWAttack':
            MagNet = [.97, .92, .9, .88, .9, .92, .95, .96, .98]

    elif data_obj['name'] == 'CIFAR10':
        if 'EADAttackEN_B.1' == attack:
            # EAD
            # This is 256 AE
            MagNet = [.65, .12, .10, .15, .2, .26, .33, .41, .48, .57, .62]
        elif 'EADAttackEN_B.01' == attack:
            # This is default
            MagNet = [.70, .25, .26, .38, .45, .61, .71, .83, .9, .93, .95]
        elif 'EADAttackL1_B.1' == attack:
            # EAD
            # This is 256 AE
            MagNet = [.65, .12, .10, .15, .2, .26, .33, .41, .48, .57, .62]
        elif 'L2CarliniWagnerAttack' in attack:
            # CW
            MagNet = [.8, .5, .48, .62, .7, .81, .9,.95, .98,.99,1]
        '''
    if data_obj['name'] == 'occupancy':
        MagNet = []
    print("MagNet our favor" + str(wilcoxon(BSDacc, MagNet, alternative="greater")))
    print("MagNet their favor" + str(wilcoxon(BSDacc, MagNet, alternative="less")))
    size = 2.5
    #plt.plot(confs, conf_det, c="aqua", label="Confidence Detections", marker=">", markersize=size)
    #plt.plot(confs, conf_acc, c="gray", label="Total Conf Accuracy", marker="<", markersize=size)
    #for key in JSD_detections.keys():
    #    plt.plot(confs, JSD_detections[key], linestyle='dashed', label="JSD " + str(key) + " Accuracy")
    if MagNet != []:
        plt.plot(confs, MagNet, c="magenta", linestyle='solid', label="MagNet Accuracy")
    plt.plot(confs, BSDacc, c="green", linestyle='solid', label="BSD Accuracy")
    #plt.plot(confs, class_div_det, linestyle='dashdot', c="orange", label="Class Divergence Detector")
    #plt.plot(confs, JSD_det, linestyle='dashdot', c="blue", label="All JSD detectors")
    #plt.plot(confs, distance_det, linestyle='dashdot', c="red", label="Distance Based Detections")
    plt.plot(confs, no_def, linestyle='solid', c="black", label="No Defense Accuracy")

    pylab.legend(loc='lower left', bbox_to_anchor=(0.02, 0.1), prop={'size': 8})
    plt.grid(linestyle='dotted')
    plt.xlabel(r"Confidence in " + attack)
    plt.ylabel("Classification accuracy")
    graph_name = "Contributions of BSD at " + str(percent*100) + " Detection Threshold " + attack + " on " + data_obj['name']
    plt.title(graph_name)
    if data_obj['name'] == 'MNIST':
        plt.xlim(min(confs) - 1.0, 41)
    else:
        plt.xlim(min(confs) - 1.0, max(confs) + 1.0)
    plt.ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))

    plt.show()
    save_path = os.path.join(pwd + '/Results/' + data_obj['name'] + '/' + data_obj['classifier'] + '/' + graph_name + ".pdf")
    plt.savefig(save_path)
    plt.clf()
    print("saved image")

def plot_compare(imgs, attack_name, data_obj):
    from mpl_toolkits.axes_grid1 import ImageGrid
    fig = plt.figure(figsize=(10., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(int(np.sqrt(len(imgs)))+1, int(np.sqrt(len(imgs)))+1),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    tmp_attack_name = attack_name.copy()
    tmp_attack_name.insert(0, "Original")
    for ax, im, name in zip(grid, imgs, tmp_attack_name):
        # Iterating over the grid returns the Axes.
        if data_obj['name'] != "MNIST":
            im = np.moveaxis(im, 0, -1)
        ax.imshow(im)
        ax.set_title(name)

    plt.show()
    pwd = os.getcwd()
    save_path = os.path.join(pwd + '/Results/' + data_obj['name'] + '/' + data_obj['classifier'] + "/ComparisonOfImages.pdf")
    plt.savefig(save_path)

class OrderedCounter(Counter, OrderedDict):
    pass


def plot_all(dataset, percent):
    pylab.rcParams['figure.figsize'] = 6, 4
    fig = plt.figure(1, (6, 4))
    ax = fig.add_subplot(1, 1, 1)
    data_obj, x_test, y_test, x_train, y_train, dataloader, testloader = generate_struct(1, dataset,
                                                                                         percent, cuda_name)
    pwd = os.getcwd()
    path = sorted(os.listdir(os.path.join(pwd, 'AttackedData/Evasion/' + data_obj['name'] + '/' + data_obj['classifier'] + '/All/')))

    operator, thresh, detector_dict, reformer = setup_MagNet(dataloader, percent, data_obj)

    operatorCDD, threshCDD = setup_MagNetCDD(dataloader, percent, data_obj)
    JSD_acc = []
    JSD_det = []
    conf_acc = []
    conf_det = []
    class_div_det = []
    distance_det = []
    JSD_detections = {}
    attack_name = []
    no_def_acc = []
    MagNet = []
    BSD_removed = []
    MagNet_removed = []
    BSD_reformed = []
    MagNet_reformed = []
    MagNet_Allreformed = []
    BSD_Allreformed = []
    Avg_distance = []
    imgs = []
    labels = []
    FS = []
    print(x_test.shape)
    imgs.append(x_test[0])
    for attack_type in path:
        p = os.path.join(pwd, 'AttackedData/Evasion/' + data_obj['name'] + '/' + data_obj['classifier'] + '/All/' + attack_type)
        if os.path.isfile(p):
            x_test_adv = np.load(p)
            NaN_idx = np.unique(np.where(np.isnan(x_test_adv))[0])
            x_test_adv = np.delete(x_test_adv, NaN_idx, axis=0)
            print(attack_type)

            NaN_idx = np.unique(np.where(np.isnan(x_test_adv))[0])
            x_test_adv = np.delete(x_test_adv, NaN_idx, axis=0)
            x_test_prune = np.delete(x_test, NaN_idx, axis=0)
            x_test_prune = x_test_prune[:len(x_test_adv)]
            y_test_adv = np.delete(y_test, NaN_idx, axis=0)
            y_test_adv = y_test_adv[:len(x_test_adv)]
            labels.append(y_test_adv)
            print(x_test_adv.shape)
            if x_test_adv.shape[1:] != (3, 32, 32) and data_obj['name'] == 'CIFAR10':
                x_test_adv = np.moveaxis(x_test_adv, 3, -3)
                print(x_test_adv.shape)
            if x_test_adv.shape[1:] != (1, 28, 28) and data_obj['name'] == 'MNIST':
                x_test_adv = np.moveaxis(x_test_adv, 3, -3)
            correct_nodef = 0
            for i in range(0, len(x_test_adv), 64):
                Y_logit = operator.no_def_acc(torch.as_tensor(x_test_adv[i:i+64]))
                correct_nodef = correct_nodef + torch.sum(torch.argmax(Y_logit, dim=1) == torch.as_tensor(y_test[i:i+len(Y_logit)]))
            no_def_acc.append(correct_nodef.item()/len(x_test_adv))

            FSAcc, legit_acc_FS = setup_FS(dataloader, percent, data_obj, x_test_adv, y_test_adv, attack_type)
            # MAGNET
            MagNetAcc, MagNetCollection, clean_acc, MagNet_removed_percent, MagNet_allref = eval_def(operator, x_test_adv, y_test_adv, thresh, device)

            JSDacc, CDDCollection, BSD_clean_acc, BSD_removed_percent, BSD_allref = eval_def(operatorCDD, x_test_adv, y_test_adv, threshCDD, device)


            '''
            ret, ret2, ret3, det, JSD_detectors = evaluate_defense_impact('NN', data_obj, x_train, 1, x_test_adv,
                                                      y_test_adv, attack_type, dataloader, x_test_prune)
                                                      
            print('No def ' + str(float(ret['No_defense_accuracy'] / 100)))
            if JSD_detections == {}:
                for key in JSD_detectors.keys():
                    JSD_detections[key] = []
            for key in JSD_detections.keys():
                JSD_detections[key].append(sum(JSD_detectors[key])/len(x_test_adv))
            conf_det.append(float(ret['Conf_rejected_samples']) / len(x_test_adv))
            conf_acc.append(float(det['w/o_JSD']))
            class_div_det.append(float(ret['Class_diff_samples'])/len(x_test_adv))
            no_def.append(float(ret['No_defense_accuracy'] / 100))
            distance_det.append(float(ret['Comb-detected'])/len(x_test_adv))
            JSD_det.append(float(ret['JSD_rejected_samples'])/len(x_test_adv))
            JSD_acc.append(float(ret['Total-JSD-detected-acc'])/100)
            '''
            FS.append(FSAcc)
            imgs.append(x_test_adv[0])
            Avg_distance.append(torch.sqrt(torch.sum((torch.as_tensor(x_test_adv)-torch.as_tensor(x_test_prune))**2))/len(x_test_adv))
            JSD_acc.append(JSDacc)
            attack_name.append(attack_type)
            MagNet.append(MagNetAcc)
            MagNet_reformed.append(clean_acc)
            BSD_reformed.append(BSD_clean_acc)
            MagNet_removed.append(MagNet_removed_percent)
            BSD_removed.append(BSD_removed_percent)
            BSD_Allreformed.append(BSD_allref)
            MagNet_Allreformed.append(MagNet_allref)
    print(Avg_distance)
    print(FSAcc)
    print(JSD_acc)
    print(BSD_reformed)
    print(BSD_removed)
    print(MagNet_reformed)
    print(MagNet_removed)
    print(MagNet_Allreformed)
    print(BSD_Allreformed)
    #print(class_div_det)
    #print(distance_det)
    #print(JSD_det)
    #print(no_def)
    print(MagNet)
    #print("No Defense " + str(wilcoxon(JSD_acc, no_def)))

    size = 2.5
    #plt.plot(confs, conf_det, c="aqua", label="Confidence Detections", marker=">", markersize=size)
    #plt.plot(confs, conf_acc, c="gray", label="Total Conf Accuracy", marker="<", markersize=size)
    #for key in JSD_detections.keys():
        #plt.bar(attack_name, JSD_detections[key])

    # set width of bar
    barWidth = 0.25


    # Set position of bar on X axis
    r1 = np.arange(len(JSD_acc))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    r5 = [x + barWidth for x in r4]
    r6 = [x + barWidth for x in r5]

    # Make the plot
    plt.bar(r1, JSD_acc, width=barWidth, edgecolor='white', label='BSD Accuracy')
    #plt.bar(r2, class_div_det, width=barWidth, edgecolor='white', label='Class Diff Detector')
    #plt.bar(r3, distance_det, width=barWidth, edgecolor='white', label='Distance Detections')
    #plt.bar(r4, JSD_det, width=barWidth, edgecolor='white', label='JSD detections')
    plt.bar(r2, no_def_acc, width=barWidth, edgecolor='white', label="No Defense Accuracy")
    plt.bar(r3, MagNet, width=barWidth, edgecolor='white', label="MagNet")



    #plt.plot(confs, class_div_det, linestyle='dashdot', c="orange", label="Class Divergence Detector")
    #plt.plot(confs, JSD_det, linestyle='dashdot', c="blue", label="All JSD detectors")
    #plt.plot(confs, distance_det, linestyle='dashdot', c="red", label="Distance Based Detections")

    #pylab.legend(loc='lower left', bbox_to_anchor=(0.02, 0.1), prop={'size': 8})


    plt.xlabel("Attack", fontweight='bold')
    plt.ylabel("Classification accuracy")
    graph_name = "BSD at " + str(percent*100) + " Detection Threshold all attacks on " + data_obj['name']
    #plt.title(graph_name)

    plt.ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))
    plt.legend()
    plt.show()
    save_path = os.path.join(pwd + '/Results/' + data_obj['name'] + '/' + data_obj['classifier'] + '/' + graph_name + ".pdf")
    plt.savefig(save_path)
    plt.clf()
    print("saved image")
    print(labels[0])
    count_label = OrderedCounter(labels[0])
    print(count_label.keys())
    print(str(count_label.keys()))
    '''
    with open(os.path.join(pwd, 'Results/CountOfClasses' + data_obj['name'] + data_obj['classifier'] + str(percent) + '.csv'), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=count_label.keys())
        writer.writeheader()
        for data in count_label:
            writer.writerow(data)
    '''
    with open(os.path.join(pwd, 'Results/Table' + data_obj['name'] + data_obj['classifier'] + str(percent) + '.csv'), 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["Name", "BSD", "MagNet", "FS", "No Def", "L2dist", "BSD Removed", "BSD Reformed", "MagNet Removed", "MagNet Reformed", "All MagNet Reformed", "All BSD Reformed"])
        for idx in range(0,len(attack_name)):
            #wr.writerow([str(attack_name[idx]), str(JSD_acc[idx]), str(MagNet[idx]), str(no_def[idx])])
            wr.writerow([str(attack_name[idx]), str(JSD_acc[idx]), str(MagNet[idx]), str(FS[idx]), str(no_def_acc[idx]), str(Avg_distance[idx].data), str(BSD_removed[idx]), str(BSD_reformed[idx]), str(MagNet_removed[idx]), str(MagNet_reformed[idx]), str(MagNet_Allreformed[idx]), str(BSD_Allreformed[idx])])
    print("saved table")

    plot_compare(imgs, attack_name, data_obj)


def setup_FS(dataloader, percent, data_obj, x_test_adv, labels, attk_name):
    # NEXT TRY 5%
    if data_obj['name'] == 'MNIST':
        robutsness_str = 'FeatureSqueezing?squeezer=bit_depth_1;FeatureSqueezing?squeezer=median_filter_2_2;FeatureSqueezing?squeezer=median_filter_3_3;FeatureSqueezing?squeezers=bit_depth_1&distance_measure=l1&fpr='+str(percent)+';FeatureSqueezing?squeezers=bit_depth_2&distance_measure=l1&fpr='+str(percent)+';FeatureSqueezing?squeezers=bit_depth_1,median_filter_2_2&distance_measure=l1&fpr='+str(percent)+';'
        detection_str = 'FeatureSqueezing?squeezers=bit_depth_1&distance_measure=l1&fpr='+str(percent)+';FeatureSqueezing?squeezers=bit_depth_2&distance_measure=l1&fpr='+str(percent)+';FeatureSqueezing?squeezers=bit_depth_1,median_filter_2_2&distance_measure=l1&fpr='+str(percent)+';'
    if data_obj['name'] == 'CIFAR10' or data_obj['name'] == 'CIFAR100':
        robutsness_str ='FeatureSqueezing?squeezer=bit_depth_5;FeatureSqueezing?squeezer=bit_depth_4;FeatureSqueezing?squeezer=median_filter_2_2;FeatureSqueezing?squeezer=non_local_means_color_13_3_4;FeatureSqueezing?squeezers=bit_depth_1&distance_measure=l1&fpr='+str(percent)+';FeatureSqueezing?squeezers=bit_depth_2&distance_measure=l1&fpr='+str(percent)+';FeatureSqueezing?squeezers=bit_depth_3&distance_measure=l1&fpr='+str(percent)+';FeatureSqueezing?squeezers=bit_depth_4&distance_measure=l1&fpr='+str(percent)+';FeatureSqueezing?squeezers=bit_depth_5&distance_measure=l1&fpr='+str(percent)+';FeatureSqueezing?squeezers=median_filter_2_2&distance_measure=l1&fpr='+str(percent)+';FeatureSqueezing?squeezers=median_filter_3_3&distance_measure=l1&fpr='+str(percent)+';FeatureSqueezing?squeezers=non_local_means_color_11_3_2&distance_measure=l1&fpr='+str(percent)+';FeatureSqueezing?squeezers=non_local_means_color_11_3_4&distance_measure=l1&fpr='+str(percent)+';FeatureSqueezing?squeezers=non_local_means_color_13_3_2&distance_measure=l1&fpr='+str(percent)+';FeatureSqueezing?squeezers=non_local_means_color_13_3_4&distance_measure=l1&fpr='+str(percent)+';FeatureSqueezing?squeezers=bit_depth_5,median_filter_2_2,non_local_means_color_13_3_2&distance_measure=l1&fpr='+str(percent)+';'
        detection_str = 'FeatureSqueezing?squeezers=bit_depth_1&distance_measure=l1&fpr='+str(percent)+';FeatureSqueezing?squeezers=bit_depth_2&distance_measure=l1&fpr='+str(percent)+';FeatureSqueezing?squeezers=bit_depth_3&distance_measure=l1&fpr='+str(percent)+';FeatureSqueezing?squeezers=bit_depth_4&distance_measure=l1&fpr='+str(percent)+';FeatureSqueezing?squeezers=bit_depth_5&distance_measure=l1&fpr='+str(percent)+';FeatureSqueezing?squeezers=median_filter_2_2&distance_measure=l1&fpr='+str(percent)+';FeatureSqueezing?squeezers=median_filter_3_3&distance_measure=l1&fpr='+str(percent)+';FeatureSqueezing?squeezers=non_local_means_color_11_3_2&distance_measure=l1&fpr='+str(percent)+';FeatureSqueezing?squeezers=non_local_means_color_11_3_4&distance_measure=l1&fpr='+str(percent)+';FeatureSqueezing?squeezers=non_local_means_color_13_3_2&distance_measure=l1&fpr='+str(percent)+';FeatureSqueezing?squeezers=non_local_means_color_13_3_4&distance_measure=l1&fpr='+str(percent)+';FeatureSqueezing?squeezers=bit_depth_5,median_filter_2_2,non_local_means_color_13_3_2&distance_measure=l1&fpr='+str(percent)+';'
    if data_obj['name'] == 'ImageNet' or data_obj['name'] == 'TinyImageNet':
        robutsness_str = 'FeatureSqueezing?squeezers=bit_depth_5,median_filter_2_2,non_local_means_color_11_3_4&distance_measure=l1&fpr=0.01'
        detection_str = 'FeatureSqueezing?squeezers=bit_depth_5,median_filter_2_2,non_local_means_color_11_3_4&distance_measure=l1&fpr=' + str(percent)
    attk_name = attk_name.split(".")[0]
    #print(x_test_adv.shape)
    #x_test_adv = np.moveaxis(x_test_adv, 1, 3)
    #print(x_test_adv.shape)
    # 0. Select a dataset.
    #from datasets import MNISTDataset, CIFAR10Dataset, ImageNetDataset
    from Datasets import get_correct_prediction_idx, evaluate_adversarial_examples, calculate_mean_confidence, \
        calculate_accuracy
    '''
    if FLAGS.dataset_name == "MNIST":
        dataset = MNISTDataset()
    elif FLAGS.dataset_name == "CIFAR-10":
        dataset = CIFAR10Dataset()
    elif FLAGS.dataset_name == "ImageNet":
        dataset = ImageNetDataset()
    # elif FLAGS.dataset_name == "SVHN":
    # dataset = SVHNDataset()
    '''
    data_obj, X_test_all, Y_test, x_train, y_train, dataloader, testloader = generate_struct(1, data_obj['name'],
                                                                                         percent, cuda_name)

    model = get_model(data_obj)
    model = model.to(device)

    # 3. Evaluate the trained model.
    # TODO: add top-5 accuracy for ImageNet.
    print("Evaluating the pre-trained model...")
    pred_lst = []
    for batch_idx, data in enumerate(testloader):
        img, label = data

        img = Variable(img.type(torch.float32))
        tmp = model(img.to(device)).detach().cpu().numpy()
        pred_lst.append(tmp)

    Y_pred_all = np.vstack(pred_lst)
    Y_test_all = torch.zeros(len(Y_test), Y_test.max() + 1).scatter_(1, torch.as_tensor(Y_test).unsqueeze(1), 1.)
    Y_test_all = np.asarray(Y_test_all)
    '''
    mean_conf_all = calculate_mean_confidence(Y_pred_all, Y_test_all)
    accuracy_all = calculate_accuracy(Y_pred_all, Y_test_all)
    print('Test accuracy on raw legitimate examples %.4f' % (accuracy_all))
    print('Mean confidence on ground truth classes %.4f' % (mean_conf_all))
    '''
    # 4. Select some examples to attack.
    import hashlib
    from Datasets import get_first_n_examples_id_each_class

    if FLAGS.select:
        # Filter out the misclassified examples.
        correct_idx = get_correct_prediction_idx(Y_pred_all, Y_test_all)
        if FLAGS.test_mode:
            # Only select the first example of each class.
            correct_and_selected_idx = get_first_n_examples_id_each_class(Y_test_all[correct_idx])
            selected_idx = [correct_idx[i] for i in correct_and_selected_idx]
        else:
            if not FLAGS.balance_sampling:
                selected_idx = correct_idx[:FLAGS.nb_examples].astype('int32')
            else:
                # select the same number of examples for each class label.
                nb_examples_per_class = int(FLAGS.nb_examples / Y_test_all.shape[1])
                correct_and_selected_idx = get_first_n_examples_id_each_class(Y_test_all[correct_idx],
                                                                              n=nb_examples_per_class)
                selected_idx = [correct_idx[i] for i in correct_and_selected_idx]
    else:
        selected_idx = np.array(range(FLAGS.nb_examples))

    from utils.output import format_number_range
    X_test, Y_test, Y_pred = X_test_all[:len(x_test_adv)], Y_test_all[:len(x_test_adv)], Y_pred_all[:len(x_test_adv)]

    # The accuracy should be 100%.
    #accuracy_selected = calculate_accuracy(Y_pred, Y_test)
    #mean_conf_selected = calculate_mean_confidence(Y_pred, Y_test)
    #print('Test accuracy on selected legitimate examples %.4f' % (accuracy_selected))
    #print('Mean confidence on ground truth classes, selected %.4f\n' % (mean_conf_selected))

    task = {}



    task['test_set_selected_length'] = len(selected_idx)
    task['test_set_selected_idx_hash'] = hashlib.sha1(str(selected_idx).encode('utf-8')).hexdigest()



    task_id = "%s" % \
              (data_obj['name'])


    if not os.path.isdir(FLAGS.result_folder):
        os.makedirs(FLAGS.result_folder)

    from utils.output import save_task_descriptor
    save_task_descriptor(data_obj['name'], FLAGS.result_folder, [task])

    # 5. Generate adversarial examples.
    #from attacks import maybe_generate_adv_examples
    from utils.squeeze import reduce_precision_py
    from utils.parameter_parser import parse_params
    attack_string_hash = hashlib.sha1(FLAGS.attacks.encode('utf-8')).hexdigest()[:5]
    sample_string_hash = task['test_set_selected_idx_hash'][:5]

    from Datasets.datasets_utils import get_next_class, get_least_likely_class
    Y_test_target_next = get_next_class(Y_test)
    Y_test_target_ll = get_least_likely_class(Y_pred)

    X_test_adv_list = [x_test_adv]
    X_test_adv_discretized_list = []
    Y_test_adv_discretized_pred_list = []

    attack_string_list = filter(lambda x: len(x) > 0, FLAGS.attacks.lower().split(';'))
    to_csv = []

    X_adv_cache_folder = os.path.join(FLAGS.result_folder, 'adv_examples')
    adv_log_folder = os.path.join(FLAGS.result_folder, 'adv_logs')
    predictions_folder = os.path.join(FLAGS.result_folder, 'predictions')
    for folder in [X_adv_cache_folder, adv_log_folder, predictions_folder]:
        if not os.path.isdir(folder):
            os.makedirs(folder)

    predictions_fpath = os.path.join(predictions_folder, "legitimate.npy")
    np.save(predictions_fpath, Y_pred, allow_pickle=False)

    if FLAGS.clip >= 0:
        epsilon = FLAGS.clip
        print("Clip the adversarial perturbations by +-%f" % epsilon)
        max_clip = np.clip(X_test + epsilon, 0, 1)
        min_clip = np.clip(X_test - epsilon, 0, 1)

    attk_lst = []

    # 6. Evaluate robust classification techniques.
    # Example: --robustness \
    #           "Base;FeatureSqueezing?squeezer=bit_depth_1;FeatureSqueezing?squeezer=median_filter_2;"

    # 7. Detection experiment.
    # Example: --detection "FeatureSqueezing?distance_measure=l1&squeezers=median_smoothing_2,bit_depth_4,bilateral_filter_15_15_60;"

    X_test_adv_discret = reduce_precision_py(x_test_adv, 256)
    X_test_adv_discretized_list = [X_test_adv_discret]
    preds = []
    for i in range(0, len(X_test_adv_discret), 64):
        preds.append(model(torch.as_tensor(X_test_adv_discret[i:i + 64]).to(device)).detach().cpu().numpy())
    Y_test_adv_discret_pred = np.concatenate(preds, axis=0)
    Y_test_adv_discretized_pred_list = [Y_test_adv_discret_pred]
    attk_lst = [attk_name]
    if FLAGS.visualize is True:
        from Datasets.visualization import show_imgs_in_rows
        if FLAGS.test_mode or FLAGS.balance_sampling:
            selected_idx_vis = range(Y_test.shape[1])
        else:
            selected_idx_vis = get_first_n_examples_id_each_class(Y_test, 1)

        legitimate_examples = X_test[selected_idx_vis]

        rows = [legitimate_examples]
        rows += map(lambda x:x[selected_idx_vis], X_test_adv_list)

        img_fpath = os.path.join(FLAGS.result_folder, '%s_attacks_examples.png' % (attk_name) )
        show_imgs_in_rows(rows, img_fpath)
        print ('\n===Adversarial image examples are saved in ', img_fpath)

        # TODO: output the prediction and confidence for each example, both legitimate and adversarial.
    '''
    if robutsness_str != '':
        """
        Test the accuracy with robust classifiers.
        Evaluate the accuracy on all the legitimate examples.
        """
        from robustness.base import evaluate_robustness
        result_folder_robustness = os.path.join(FLAGS.result_folder, "robustness")
        print(result_folder_robustness)
        fname_prefix = "%s_robustness" % (attk_name)

        if X_test_all.shape[1] != 3 and data_obj['name'] == 'CIFAR10':
            X_test_all = np.moveaxis(X_test_all, 3, -3)
        best_acc, name, legit_acc = evaluate_robustness(robutsness_str, model, Y_test_all, X_test_all, Y_test, \
                attk_lst, X_test_adv_discretized_list,
                fname_prefix, selected_idx_vis, result_folder_robustness)
    '''
    if detection_str != '':
        from detections.base import DetectionEvaluator

        result_folder_detection = os.path.join(FLAGS.result_folder, "detection")
        csv_fname = "%s_attacks_%s_detection.csv" % (task_id, attack_string_hash)
        de = DetectionEvaluator(model, result_folder_detection, csv_fname, data_obj['name'], device)
        Y_test_all_pred = Y_pred_all
        de.build_detection_dataset(X_test_all, Y_test_all, Y_test_all_pred, selected_idx, X_test_adv_discretized_list,
                                   Y_test_adv_discretized_pred_list, attk_lst, attack_string_hash, FLAGS.clip,
                                   Y_test_target_next, Y_test_target_ll)
        best_acc = de.evaluate_detections(detection_str)
        legit_acc = 0

    return best_acc, legit_acc


def setup_MagNet(dataloader, percent, data_obj):

    # MAGNET SETUP
    if data_obj['name'] == 'CIFAR10':
        n_channels = 3
        detector_model1 = autoencoder(n_channels)
        detector_model2 = autoencoder2(n_channels)
        detector_model2.load_state_dict(torch.load('checkpoint/CIFAR10/49ep_autoencoder2_-1to1MSE.pth'))
        detector_model2 = detector_model2.to(device)
        detector = [AEDetector(detector_model2, device, p=1)]
        reformer = [SimpleReformer(detector_model2, device, data_obj['min_pixel_value'], data_obj['max_pixel_value'])]
        model = StandardCIFAR10()
        tmp_dict = torch.load('Classifiers/Evasion/NN/CIFAR10_Standard-1to1.pth')
        model.load_state_dict(tmp_dict)
        model.to(device)
        classifier = Classifier(model, device,  1)
        id_reformer = IdReformer()

        detector_JSD = [DBDetector(id_reformer, ref, classifier, T=10) for i, ref in enumerate(reformer)]
        detector_JSD += [DBDetector(id_reformer, ref, classifier, T=40) for i, ref in enumerate(reformer)]
        detector_dict = dict()
    elif data_obj['name'] == 'CIFAR100':
        n_channels = 3
        #detector_model1 = autoencoder(n_channels)
        #detector_model2 = autoencoder2(n_channels)
        detector_model1 = TinyImageNetConvAE()
        detector_model1.load_state_dict(torch.load(data_obj['L1_dir']))
        detector_model1 = detector_model1.to(device)
        detector = [AEDetector(detector_model1, device,  p=2)]
        reformer = [SimpleReformer(detector_model1, device, data_obj['min_pixel_value'], data_obj['max_pixel_value'])]

        model = get_model(data_obj)
        model = model.to(device)
        classifier = Classifier(model, device, 1)
        id_reformer = IdReformer()

        detector_JSD = [DBDetector(id_reformer, ref, classifier, T=10) for i, ref in enumerate(reformer)]
        detector_JSD += [DBDetector(id_reformer, ref, classifier, T=40) for i, ref in enumerate(reformer)]

        detector_dict = dict()
    elif data_obj['name'] == 'TinyImageNet':
        n_channels = 3
        detector_model1 = TinyImageNetConvAE()
        detector_model2 = autoencoder2(n_channels)
        detector_model1.load_state_dict(torch.load(data_obj['L1_dir']))
        detector = [AEDetector(detector_model1, device,  p=2)]
        reformer = [SimpleReformer(detector_model1, device, data_obj['min_pixel_value'], data_obj['max_pixel_value'])]

        model = get_model(data_obj)
        model = model.to(device)
        classifier = Classifier(model, device, 1)
        id_reformer = IdReformer()

        detector_JSD = [DBDetector(id_reformer, ref, classifier, T=10) for i, ref in enumerate(reformer)]
        detector_JSD += [DBDetector(id_reformer, ref, classifier, T=40) for i, ref in enumerate(reformer)]

        detector_dict = dict()
    elif data_obj['name'] == 'ImageNet':
        n_channels = 3
        detector_model1 = TinyImageNetConvAE()

        detector_model2 = autoencoder2(n_channels)
        state_dict = torch.load(data_obj['L1_dir'])
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        detector_model1.load_state_dict(new_state_dict)

        detector = [AEDetector(detector_model1, device,  p=2)]
        reformer = [SimpleReformer(detector_model1, device, data_obj['min_pixel_value'], data_obj['max_pixel_value'])]

        model = get_model(data_obj)
        model = model.to(device)
        classifier = Classifier(model, device, 1)
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

        detector = [AEDetector(detector_model1, device, p=2, dims=data_obj['dims']),
                    AEDetector(detector_model2, device, p=2, dims=data_obj['dims'])]

        reformer = [SimpleReformer(detector_model1, device,
                                    data_obj['min_pixel_value'], data_obj['max_pixel_value'])]
        '''
        detector = [AEDetector(detector_model1, device_defense, p=2, dims=data_obj['dims']),
                    AEDetector(detector_model2, device_defense, p=2, dims=data_obj['dims']),
                    AEDetectorC([detector_model1,detector_model2], device_defense, p=2)]

        reformer = [SimpleReformerC([detector_model1, detector_model2], device_defense,
                                   data_obj['min_pixel_value'], data_obj['max_pixel_value'])]
        '''
        model = get_model(data_obj)
        model.to(device)

        id_reformer = IdReformer()
        classifier = Classifier(model, device, 1)
        #detector_JSD = [DBDetector(id_reformer, ref, classifier, T=10) for i, ref in enumerate(reformer)]
        #detector_JSD += [DBDetector(id_reformer, ref, classifier, T=40) for i, ref in enumerate(reformer)]
        detector_JSD = []
        detector_dict = dict()

    elif data_obj['name'] == 'occupancy':
        aeMSE = get_autoencoder(data_obj)
        aeMSE.load_state_dict(torch.load(data_obj['L2_dir']))
        aeMAE = get_autoencoder(data_obj)
        aeMAE.load_state_dict(torch.load(data_obj['L1_dir']))
        detector = [AEDetector(aeMAE, device, data_obj['L1_dir'], p=1, dims=data_obj['dims']),
                    AEDetector(aeMSE, device, data_obj['L2_dir'], p=2, dims=data_obj['dims'])]
        reformer = [SimpleReformer(aeMAE, device, data_obj['L1_dir'], data_obj['min_pixel_value'], data_obj['max_pixel_value'])]
        model = get_model(data_obj)
        model.to(device)

        id_reformer = IdReformer()
        classifier = Classifier(model, device, 1)
        detector_JSD = [DBDetector(id_reformer, ref, classifier, T=10) for i, ref in enumerate(reformer)]
        detector_JSD += [DBDetector(id_reformer, ref, classifier, T=40) for i, ref in enumerate(reformer)]
        detector_dict = dict()

    for i, det in enumerate(detector):
        detector_dict["II" + str(i)] = det
    for i, det in enumerate(detector_JSD):
        detector_dict["JSD" + str(i)] = det
    dr = dict([("II" + str(i), percent) for i in range(len(detector))] + [("JSD" + str(i), percent) for i in range(len(detector_JSD))])
    operator = Operator(classifier, detector_dict, reformer[0], device, data_obj['dims'])
    thresh = operator.get_thrs(dataloader, detector_dict, dr)

    return operator, thresh, detector_dict, reformer


def setup_MagNetCDD(dataloader, percent, data_obj):
    # MAGNET SETUP
    if data_obj['name'] == 'CIFAR10':
        n_channels = 3
        detector_model1 = autoencoder(n_channels)
        detector_model2 = autoencoder2(n_channels).to(device)
        detector_model2.load_state_dict(torch.load('checkpoint/CIFAR10/49ep_autoencoder2_-1to1MSE.pth'))
        detector = [AEDetector(detector_model2, device,  p=1)]
        reformer = [SimpleReformer(detector_model2, device, data_obj['min_pixel_value'], data_obj['max_pixel_value'])]
        model = StandardCIFAR10()
        tmp_dict = torch.load('Classifiers/Evasion/NN/CIFAR10_Standard-1to1.pth')
        model.load_state_dict(tmp_dict)
        model.to(device)
        classifier = Classifier(model, device, 1)
        id_reformer = IdReformer()

        detector_JSD = [DBDetector(id_reformer, ref, classifier, T=10) for i, ref in enumerate(reformer)]
        detector_JSD += [DBDetector(id_reformer, ref, classifier, T=40) for i, ref in enumerate(reformer)]

        CDD = [CDDetector(detector_model2, classifier, device)]
        #CDDNoise = [CDDetectorNoisy(detector_model2, classifier, device, data_obj['min_pixel_value'], data_obj['max_pixel_value'])]
        CDDNoise = [CDDetectorMedian(detector_model2, classifier, device, data_obj['min_pixel_value'],
                                    data_obj['max_pixel_value'])]
        CDDNoise = []
        detector_dict = dict()
    elif data_obj['name'] == 'CIFAR100':
        n_channels = 3
        #detector_model1 = autoencoder(n_channels).to(device)
        #detector_model2 = autoencoder2(n_channels)
        detector_model1 = TinyImageNetConvAE()
        detector_model1.load_state_dict(torch.load(data_obj['L1_dir']))
        detector = [AEDetector(detector_model1, device,  p=2)]
        reformer = [SimpleReformer(detector_model1, device, data_obj['min_pixel_value'], data_obj['max_pixel_value'])]


        tmp_dict = torch.load('Classifiers/Evasion/NN/wide-resnet-28x20.t7')
        #model.load_state_dict(tmp_dict['net'])
        model = tmp_dict['net']
        model.to(device)
        classifier = Classifier(model, device, 1)
        id_reformer = IdReformer()

        detector_JSD = [DBDetector(id_reformer, ref, classifier, T=10) for i, ref in enumerate(reformer)]
        detector_JSD += [DBDetector(id_reformer, ref, classifier, T=40) for i, ref in enumerate(reformer)]

        CDD = [CDDetector(detector_model1, classifier, device)]
        #CDDNoise = [CDDetectorNoisy(detector_model2, classifier, device, data_obj['min_pixel_value'], data_obj['max_pixel_value'])]
        #CDDNoise = [CDDetectorMedian(detector_model1, classifier, device, data_obj['min_pixel_value'],
        #                            data_obj['max_pixel_value'])]
        CDDNoise = []
        del detector_model1, model
        detector_dict = dict()
    elif data_obj['name'] == 'TinyImageNet':
        n_channels = 3
        detector_model1 = TinyImageNetConvAE().to(device)
        detector_model2 = autoencoder2(n_channels)
        detector_model1.load_state_dict(torch.load(data_obj['L1_dir']))
        detector = [AEDetector(detector_model1, device,  p=2)]
        reformer = [SimpleReformer(detector_model1, device, data_obj['min_pixel_value'], data_obj['max_pixel_value'])]

        model = get_model(data_obj)
        model = model.to(device)
        classifier = Classifier(model, device, 1)
        id_reformer = IdReformer()

        detector_JSD = [DBDetector(id_reformer, ref, classifier, T=10) for i, ref in enumerate(reformer)]
        detector_JSD += [DBDetector(id_reformer, ref, classifier, T=40) for i, ref in enumerate(reformer)]

        detector_dict = dict()
        CDD = [CDDetector(detector_model1, classifier, device)]
        del detector_model1, model
        CDDNoise = []
    elif data_obj['name'] == 'ImageNet':
        n_channels = 3
        detector_model1 = TinyImageNetConvAE().to(device)

        detector_model2 = autoencoder2(n_channels)
        state_dict = torch.load(data_obj['L1_dir'])
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        detector_model1.load_state_dict(new_state_dict)
        detector = [AEDetector(detector_model1, device,  p=2)]
        reformer = [SimpleReformer(detector_model1, device, data_obj['min_pixel_value'], data_obj['max_pixel_value'])]

        model = get_model(data_obj)
        classifier = Classifier(model, device, 1)
        id_reformer = IdReformer()

        detector_JSD = [DBDetector(id_reformer, ref, classifier, T=10) for i, ref in enumerate(reformer)]
        detector_JSD += [DBDetector(id_reformer, ref, classifier, T=40) for i, ref in enumerate(reformer)]

        detector_dict = dict()
        CDD = [CDDetector(detector_model1, classifier, device)]
        del detector_model1, model
        CDDNoise = []
    elif data_obj['name'] == 'MNIST':
        n_channels = 1
        detector_model1 = autoencoder(n_channels).to(device)
        detector_model2 = autoencoder2(n_channels).to(device)
        # Add combinded model
        detector_model1.load_state_dict(torch.load('checkpoint/MNIST/autoencoder1.pth'))
        detector_model2.load_state_dict(torch.load('checkpoint/MNIST/autoencoder2.pth'))
        '''
        detector = [AEDetector(detector_model1, device_defense, p=2, dims=data_obj['dims']),
                    AEDetector(detector_model2, device_defense, p=2, dims=data_obj['dims']),
                    AEDetectorC([detector_model1,detector_model2], device_defense, p=2, minimum=data_obj['min_pixel_value'], maximum=data_obj['max_pixel_value']) ]
        '''
        detector = [AEDetector(detector_model1, device, p=2, dims=data_obj['dims']),
                    AEDetector(detector_model2, device, p=2, dims=data_obj['dims'])]
        reformer = [SimpleReformer(detector_model1, device,
                                    data_obj['min_pixel_value'], data_obj['max_pixel_value'])]
        #reformer = [SimpleReformerC([detector_model1, detector_model2], device_defense,
        #                           data_obj['min_pixel_value'], data_obj['max_pixel_value'])]

        '''
        detector = [AEDetector(detector_model1, device_defense, p=2, dims=data_obj['dims']),
                    AEDetector(detector_model2, device_defense, p=2, dims=data_obj['dims']),
                    AEDetectorC([detector_model1,detector_model2], device_defense, p=2)]

        reformer = [SimpleReformerC([detector_model1, detector_model2], device_defense,
                                   data_obj['min_pixel_value'], data_obj['max_pixel_value'])]
        '''
        model = get_model(data_obj)
        model = model.to(device)

        id_reformer = IdReformer()
        classifier = Classifier(model, device, 1)
        #detector_JSD = [DBDetector(id_reformer, ref, classifier, T=10) for i, ref in enumerate(reformer)]
        #detector_JSD += [DBDetector(id_reformer, ref, classifier, T=40) for i, ref in enumerate(reformer)]
        detector_JSD = []
        CDD = [CDDetector(detector_model1, classifier, device),
               CDDetector(detector_model2, classifier, device)]
        #CDDNoise = [CDDetectorNoisy(detector_model1, classifier, device, 0, 1),
        #       CDDetectorNoisy(detector_model2, classifier, device, 0, 1)]
        CDDNoise = []
        detector_dict = dict()

    elif data_obj['name'] == 'occupancy':
        aeMSE = get_autoencoder(data_obj)
        aeMSE.load_state_dict(torch.load(data_obj['L2_dir']))
        aeMAE = get_autoencoder(data_obj)
        aeMAE.load_state_dict(torch.load(data_obj['L1_dir']))
        detector = [AEDetector(aeMAE, device, data_obj['L1_dir'], p=1, dims=data_obj['dims']),
                    AEDetector(aeMSE, device, data_obj['L2_dir'], p=2, dims=data_obj['dims'])]
        reformer = [SimpleReformer(aeMAE, device, data_obj['L1_dir'], data_obj['min_pixel_value'], data_obj['max_pixel_value'])]
        model = get_model(data_obj)
        model.to(device)

        id_reformer = IdReformer()
        classifier = Classifier(model, device, 'Classifiers/Evasion/NN/occupancy_classifier.pth', 1)
        detector_JSD = [DBDetector(id_reformer, ref, classifier, T=10) for i, ref in enumerate(reformer)]
        detector_JSD += [DBDetector(id_reformer, ref, classifier, T=40) for i, ref in enumerate(reformer)]
        detector_dict = dict()

    for i, det in enumerate(detector):
        detector_dict["II" + str(i)] = det
    for i, det in enumerate(detector_JSD):
        detector_dict["JSD" + str(i)] = det
    for i, det in enumerate(CDD):
        detector_dict["CDD" + str(i)] = det
    for i, det in enumerate(CDDNoise):
        detector_dict["CDDNoise" + str(i)] = det
    dr = dict([("II" + str(i), percent) for i in range(len(detector))] + [("JSD" + str(i), percent) for i in range(len(detector_JSD))] + [("CDD" + str(i), -1) for i in range(len(CDD))] + [("CDDNoise" + str(i), -1) for i in range(len(CDDNoise))])
    operator = Operator(classifier, detector_dict, reformer[0], device, data_obj['dims'])
    thresh = operator.get_thrs(dataloader, detector_dict, dr)

    return operator, thresh



def main():
    data_obj = 0
    x_train = 0
    x_test = 0
    y_test = 0
    dataloader = 0
    #plot_static(data_obj, x_train, x_test, y_test, dataloader)
    '''
    per = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]
    for p in per:
        plot_sweep_conf('MNIST', p, 'standard', 'CWAttack')
    '''
    #plot_clean('MNIST', [0.001, 0.01, 0.02, 0.03, 0.05])
    #plot_clean('CIFAR10', [0.001, 0.01, 0.02, 0.03, 0.05])
    #plot_clean('CIFAR100', [0.001, 0.01, 0.02, 0.03, 0.05])

    per = [0.01]
    for p in per:
        print(p)
        #plot_contributions('CIFAR10', p, 'EADAttackEN_B.1')
        #plot_contributions('MNIST', p, 'CWAttack')
        plot_all('MNIST', p)




if __name__ == "__main__":
    main()