import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle
from art.classifiers import PyTorchClassifier
import numpy as np
from numpy.linalg import norm
from scipy.stats import entropy
import torchvision.models as models
from Networks.CIFAR10 import StandardCIFAR10
from Networks.CIFAR100 import SimpleCIFAR100
from Networks.ResNet import ResNet18
from Networks.MNIST import Net, StandardMNIST
from Networks.occupancy import occupancy_classifier
from Autoencoders.code.autoencoder import newConvAutoencoderTanh, occupancy_ae, MagNetAE256, TinyImageNetConvAE
from PytorchMagNet.defense_models import autoencoder, autoencoder2

def get_model(data_obj):
    if data_obj['name'] == 'MNIST':
        model = StandardMNIST()
        tmp_dict = torch.load('Classifiers/Evasion/NN/' + data_obj['classifier'] + '.pth')
        model.load_state_dict(tmp_dict)
    elif data_obj['name'] == 'CIFAR10':
        if data_obj['classifier'] == 'CIFAR10_deep_classifier':
            model = ResNet18()
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            tmp_dict = torch.load('Classifiers/Evasion/NN/' + data_obj['classifier'] + '.pth')
            for k, v in tmp_dict['net'].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model = StandardCIFAR10()
            tmp_dict = torch.load('Classifiers/Evasion/NN/' + data_obj['classifier'] + '.pth')
            model.load_state_dict(tmp_dict)
    elif data_obj['name'] == 'CIFAR100':
        if data_obj['classifier'] == 'CIFAR100_Standard-1to1':
            tmp_dict = torch.load('Classifiers/Evasion/NN/wide-resnet-28x20.t7')
            # model.load_state_dict(tmp_dict['net'])
            model = tmp_dict['net']
        elif data_obj['classifier'] == 'SimpleCIFAR100-1to1':
            model = SimpleCIFAR100()
            tmp_dict = torch.load('Classifiers/Evasion/NN/' + data_obj['classifier'] + '.pth')
            model.load_state_dict(tmp_dict)
    elif data_obj['name'] == 'TinyImageNet':
        model = models.resnet18(True)

        model.avgpool = nn.AdaptiveAvgPool2d(1)
        m_in = model.fc.in_features
        model.fc.out_features = 200
        model.fc = nn.Linear(m_in, 200)
        tmp_dict = torch.load('Classifiers/Evasion/NN/' + data_obj['classifier'] + '.pth')
        model.load_state_dict(tmp_dict)
    elif data_obj['name'] == 'ImageNet':
        model = models.resnet18(False)
        tmp_dict = torch.load('Classifiers/Evasion/NN/' + data_obj['classifier'] + '.pth')

        state_dict = tmp_dict['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)

    elif data_obj['name'] == 'occupancy':
        model = occupancy_classifier(5, 10)
        tmp_dict = torch.load('Classifiers/Evasion/NN/' + data_obj['classifier'] + '.pth')
        model.load_state_dict(tmp_dict)
    return model

def get_aemodel(data_obj):
    model = ResNet18()
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    tmp_dict = torch.load('Classifiers/Evasion/NN/AE_Class_Sig_CIFAR10.pth')

    for k, v in tmp_dict['net'].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    return model

def get_torch_classifier(data_obj, device):
    pwd = os.getcwd()
    model = get_model(data_obj)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    classifier = PyTorchClassifier(model=model, clip_values=(data_obj['min_pixel_value'], data_obj['max_pixel_value']), loss=criterion,
                                   optimizer=optimizer, input_shape=data_obj['input_size'], nb_classes=data_obj['output_size'])
    return classifier

def get_torch_ae_classifier(data_obj):
    model = get_aemodel(data_obj)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    classifier = PyTorchClassifier(model=model, clip_values=(data_obj['min_pixel_value'], data_obj['max_pixel_value']), loss=criterion,
                                   optimizer=optimizer, input_shape=data_obj['input_size'], nb_classes=data_obj['output_size'])
    return classifier

def get_classifier(classifier_name, type):
    pwd = os.getcwd()
    if type == 'NN':
        with open(os.path.join(pwd, 'Classifiers/Evasion/NN/' + classifier_name), 'rb') as input:
            classifier = pickle.load(input)
        return classifier
    if type == 'SVM':
        with open(os.path.join(pwd, 'Classifiers/Evasion/SVM/' + classifier_name), 'rb') as input:
            classifier = pickle.load(input)
        return classifier

def empty_temp(data_obj):
    pwd = os.getcwd()
    directory = os.path.join(pwd, 'AttackedData/Evasion/' + data_obj[
        'name'] + '/standard_classifier/temp/*')
    import glob
    files = glob.glob(directory)
    for f in files:
        os.remove(f)

def get_autoencoder(data_obj):
    if data_obj['name'] == 'MNIST':
        model = autoencoder(1)
        model.eval()

        model2 = autoencoder2(1)
        model2.eval()

    elif data_obj['name'] == 'CIFAR10':
        AE_type = data_obj['L1_dir'].split('_')[-1].split('.')[0]
        if AE_type == 'MagNet256AE':
            model = MagNetAE256()
        elif AE_type == '2':
            model = autoencoder2(3)
        else:
            model = newConvAutoencoderTanh()
        model.eval()

        model2 = None
    elif data_obj['name'] == 'CIFAR100':
        AE_type = data_obj['L1_dir'].split('_')[-1].split('.')[0]
        if AE_type == 'MagNet256AE':
            model = MagNetAE256()
        elif AE_type == '2':
            model = autoencoder2(3)
        else:
            model = newConvAutoencoderTanh()
        model.eval()
        model2 = None
    elif data_obj['name'] == 'TinyImageNet':
        AE_type = data_obj['L1_dir'].split('_')[-1].split('.')[0]
        if AE_type == 'MagNet256AE':
            model = TinyImageNetConvAE()
        model.eval()
        model2 = None
    elif data_obj['name'] == 'occupancy':
        model = occupancy_ae()
        model.eval()
        model2 = None
    return model, model2

def train_recon_error(x_train, data_obj, L1_ae_dir, L2_ae_dir, dataloader, percent, device):

    model, model2 = get_autoencoder(data_obj)
    model.load_state_dict(torch.load(L1_ae_dir))
    L1_train_recon_error = []
    L2_train_recon_error = []
    x_train = torch.from_numpy(x_train).float().to(device)
    for i in range(0, len(x_train), 128):
        img = x_train[i:i + 128]

        img = img.to(device)
        tmp = model(img)
        tmp_sum = torch.sum(abs(img - tmp), dim=data_obj['dims'])
        L1_train_recon_error.append(tmp_sum.data.cpu().detach().numpy())

        model.load_state_dict(torch.load(L2_ae_dir))
        model.eval()
        model = model.to(device)
        L2_output = model(img)
        L2_train_output = L2_output
        # TODO: Chjanged sqrt(img-L2out)^2)
        L2_train_recon_error.append(torch.sum((img - L2_train_output).pow(2), dim=data_obj['dims']).data.cpu().detach().numpy())

    #del L2_train_output, x_train
    sorted_L1 = np.sort(np.concatenate(L1_train_recon_error))
    sorted_L2 = np.sort(np.concatenate(L2_train_recon_error))
    L1_thresh = sorted_L1[int((len(sorted_L1)-1) * (1-percent))]
    L2_thresh = sorted_L2[int((len(sorted_L2)-1) * (1 - percent))]
    return L1_thresh, L2_thresh


def evaluate_attack_impact(classifier_type, x_test_adv, y_test, classifier_name):
    classifier = get_classifier(classifier_name, classifier_type)
    if classifier_type == 'NN':
        classifier.set_learning_phase(False)
        predictions = np.argmax(classifier.predict(x_test_adv), axis=1)
        del classifier
    elif classifier_type == 'SVM':
        predictions = classifier.predict(x_test_adv)
    accuracy = np.sum(predictions == np.argmax(y_test, axis=1)) / len(y_test)
    print('Accuracy on adversarial test examples: {}%'.format(accuracy * 100))
    return accuracy, predictions

def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

def get_JSD_mark(JSD_test, JSD_train, percent):
    sort_train = np.sort(JSD_train)
    JSD_thresh = sort_train[int((len(sort_train)-1) * (1 - percent))]
    return JSD_test > JSD_thresh

def get_conf_mark(conf_test, conf_train, percent):
    sort_train = np.sort(conf_train)
    conf_thresh = sort_train[int((len(sort_train)-1) * (1 - percent))]
    return conf_test > conf_thresh

def save_attacked_data(data_obj, x_test_adv, attack_name):
    pwd = os.getcwd()

    np.save(os.path.join(pwd, 'AttackedData/Evasion/' + data_obj['name'] + '/' + data_obj['classifier'] + '/All/' + attack_name),
            x_test_adv)

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   print("done")