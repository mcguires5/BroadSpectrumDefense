"""
The script demonstrates a simple example of using ART with PyTorch. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import os
import torch
if torch.cuda.device_count() == 1 and torch.cuda.current_device() == 0:
    cuda_name = "0"
    print("Found CUDA 0")
else:
    cuda_name = "4"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_name
    print(torch.cuda.current_device())
print("using GPU " + str(cuda_name))
device = torch.device("cuda:0")


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

from Networks.CIFAR10 import Net, CNN, StandardCIFAR10, MagNet
from Networks.CIFAR100 import SimpleCIFAR100
from Networks.MNIST import Net, StandardMNIST
from Networks.ResNet import ResNet18
from AttackedData.CWAttack.pytorch_cw2 import AttackCarliniWagnerL2
from AttackedData.CWAttack.pytorch_custom import AttackCarliniWagnerCustom
from torchsummary import summary
import foolbox
import advertorch
import wandb
from helpers import *
from evaluate_defense import evaluate_defense_impact
from Datasets.occupancy_dataset import occupancyDataset
from Networks.occupancy import occupancy_classifier
from datasets_objects import generate_struct


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

class Normalize(object):
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
            t.sub_(m).div_(s)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def train_netowrk(model, data_obj, num_epochs, dataloader, dataset, schedule, testloader, x_train, y_train):
    model = model.to(device)
    optimizer = None
    if schedule:
        #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=150, gamma=0.1)
        #scheduler = StepLR(optimizer, step_size=10, gamma=0.01)
    else:
        #optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


    no_epochs = num_epochs
    train_loss = list()
    val_loss = list()
    best_val_loss = 1
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.to(device)
    print('Moved to GPU')
    for epoch in range(no_epochs):
        total_train_loss = 0
        total_val_loss = 0
        correct = 0
        t = 0
        model.train()
        # training
        total = 0
        for itr, (image, label) in enumerate(dataloader):
            optimizer.zero_grad()
            image = image.to(device)
            label = label.to(device)

            x = model(image)

            loss = criterion(x, label)

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            #pred = torch.nn.functional.softmax(x, dim=1)
            _, predicted = torch.max(x.data, 1)
            for i, p in enumerate(predicted):
                if label[i] == torch.max(p.data, 0)[1]:
                    correct = correct + 1

            total += label.size(0)
        total_train_loss = total_train_loss / (len(dataloader.dataset))
        train_loss.append(total_train_loss)

        if schedule:
            scheduler.step()
            '''
            if epoch % 80 == 0 and epoch != 0:
                optimizer.param_groups[0]['momentum'] = optimizer.param_groups[0]['momentum'] * .5
            print('====> Epoch: {} Train Accuracy: {:.2f} LR: {:.5f} Mom: {}'.format(
                epoch + 1, total / t * 100, scheduler.get_lr()[0], optimizer.param_groups[0]['momentum']))
                '''
            print('====> Epoch: {} Train Accuracy: {:.2f} LR: {:.5f}'.format(
                epoch + 1, correct / total , scheduler.get_lr()[0]))
        else:
            print('====> Epoch: {} Train Accuracy: {:.2f}'.format(
                epoch + 1, correct / total))


        # validation
        model.eval()
        total = 0
        for itr, (image, label) in enumerate(testloader):

            image = image.to(device)
            label = label.to(device)

            x = model(image)

            loss = criterion(x, label)
            total_val_loss += loss.item()

            pred = torch.nn.functional.softmax(x, dim=1)
            for i, p in enumerate(pred):
                if label[i] == torch.max(p.data, 0)[1]:
                    total = total + 1

        accuracy = total / len(testloader.dataset) * 100

        total_val_loss = total_val_loss / (len(testloader.dataset))
        val_loss.append(total_val_loss)

        print('Epoch: {}/{}, Train Loss: {:.8f}, Val Loss: {:.8f}, Val Accuracy: {:.8f}'.format(epoch + 1, no_epochs,
                                                                                                  total_train_loss,
                                                                                                  total_val_loss,
                                                                                                  accuracy))

    pwd = os.getcwd()
    save_dir = os.path.join(pwd, 'Classifiers/Evasion/NN/' + str(dataset) + ".pth")
    if epoch == round(num_epochs/2) or epoch == num_epochs:
        torch.save(model.state_dict(), save_dir)
    torch.save(model.state_dict(), save_dir)


    fig = plt.figure(figsize=(20, 10))
    plt.plot(np.arange(1, no_epochs + 1), train_loss, label="Train loss")
    plt.plot(np.arange(1, no_epochs + 1), val_loss, label="Validation loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Loss Plots")
    plt.legend(loc='upper right')
    plt.show()

    return model
# Step 2: Create the model
def create_classifier(data_obj, X_train, y_train, y_test, classifier_name, dataloader, testloader):
    if data_obj['name'] == 'MNIST':
        #model = Net(784,10)
        model = StandardMNIST()
        model.train()
        model = train_netowrk(model, data_obj, 50, dataloader, classifier_name, False, testloader)
    if data_obj['name'] == 'CIFAR10':
        '''
        model = resnet18(pretrained=True)
        model.load_state_dict(torch.load('Networks/state_dicts/resnet18.pt', map_location='cuda:0'))
        '''
        model = MagNet().to(device)
        #model = ResNet18().to(device)
        model.train()
        model = train_netowrk(model, data_obj, 400, dataloader, classifier_name, True, testloader, X_train, y_train)
        #model.load_state_dict(torch.load('Classifiers/Evasion/NN/CIFAR10_standard_classifier.pth'))
    if data_obj['name'] == 'CIFAR100':
        model = SimpleCIFAR100().to(device)
        model.train()
        model = train_netowrk(model, data_obj, 300, dataloader, classifier_name, True, testloader, X_train, y_train)
        #model.load_state_dict(torch.load('Classifiers/Evasion/NN/CIFAR10_standard_classifier.pth'))
    if data_obj['name'] == 'occupancy':
        model = occupancy_classifier(5, 10).to(device)
        model.train()
        model = train_netowrk(model, data_obj, 200, dataloader, classifier_name, True, testloader)
        # model.load_state_dict(torch.load('Classifiers/Evasion/NN/CIFAR10_standard_classifier.pth'))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-6)
    classifier = PyTorchClassifier(model=model, clip_values=(data_obj['min_pixel_value'], data_obj['max_pixel_value']), loss=criterion,
                                   optimizer=optimizer, input_shape=data_obj['input_size'], nb_classes=data_obj['output_size'])
    # TODO Maybe there wasnt enough epochs on the classifier?
    state = classifier.__getstate__()

    pwd = os.getcwd()

    with open(os.path.join(pwd, 'Classifiers/Evasion/NN/' + classifier_name + '.pkl'), 'wb') as output:
        pickle.dump(classifier, output, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(pwd, 'Classifiers/Evasion/NN/' + classifier_name + '_state.pkl'), 'wb') as output2:
        pickle.dump(state, output2, pickle.HIGHEST_PROTOCOL)
    correct = 0

    predictions = []
    classifier.set_learning_phase(False)
    for batch_idx, data in enumerate(testloader):
        img, label = data
        #img = img.view(img.size(0), -1)
        i = img.data.cpu().detach().numpy()
        predictions.append(classifier.predict(i))
    predictions = np.concatenate(predictions, axis=0)
    if len(y_test.shape) > 1:
        y_test = np.argmax(y_test, axis=1)
    correct += np.sum(np.equal(np.argmax(predictions, axis=1), y_test))
    accuracy = correct/len(y_test)
    print('Accuracy on benign test examples: {}%'.format(accuracy * 100))


def generate_attack_ART(data_obj, testloader, y_test, c):

    '''
    class_dict = classifier._model.state_dict()
    new_state_dict = OrderedDict()
    for k, v in class_dict.items():
        name = k[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    ae_model = CIFARVAE()
    # model.load_state_dict(torch.load('autoencoder_VAE_loss.pth'))
    ae_model.load_state_dict(torch.load(os.path.join(pwd, 'Autoencoders/CIFAR10/1000ep_MAE_loss_1-e5_LR.pth')))
    auto_encoder_dict = ae_model.state_dict()
    defended_classifier = DefendedClassifier()
    model_dict = defended_classifier.state_dict()
    pretrained_dict = {k: v for k, v in auto_encoder_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    pretrained_dict.update({k: v for k, v in new_state_dict.items() if k in model_dict})
    model_dict.update(pretrained_dict)
    defended_classifier.load_state_dict(pretrained_dict)
    optimizer = optim.SGD(defended_classifier.parameters(), lr=0.01, momentum=0.5)
    art_classifier = PyTorchClassifier(model=defended_classifier, clip_values=(min_pixel_value, max_pixel_value), loss=nn.CrossEntropyLoss(),
                      optimizer=optimizer, input_shape=(1, 28 * 28), nb_classes=10)
    '''
    classifier = get_torch_classifier(data_obj, device)
    classifier.set_learning_phase(False)
    print(classifier.nb_classes)

    pwd = os.getcwd()


    x_test_adv = []
    #attack = attacks.evasion.ZooAttack(classifier=classifier, confidence=c)
    #attack = attacks.evasion.ShadowAttack(estimator=classifier)
    attack = attacks.evasion.HopSkipJump(classifier)
    attack_name = 'HopSkipJump_ART_logits_1k'
    #attack_name = 'Zoo_ART_logits_1k_c' + str(c).zfill(3)

    print(attack_name)

    for batch_idx, data in enumerate(testloader):
        if batch_idx >= 8:
            break
        img, label = data
        #targets = np.zeros(np.shape(label))
        #img = img.view(img.size(0), -1)
        #img = img.data.cpu().detach().numpy()
        for i in img:
                x_test_adv.append(attack.generate(x=i[None,:,:,:]))
        #x_test_adv.append(attack.generate(x=img))
        print(str((testloader.batch_size * batch_idx) / len(testloader.dataset)*100) +'% complete')
    del testloader


    x_test_adv = np.concatenate(x_test_adv, axis=0)

    save_attacked_data(data_obj, x_test_adv, attack_name)
    empty_temp(data_obj)
    return x_test_adv, attack_name

def generate_attack_Foolbox(data_obj, testloader, y_test, c):
    pwd = os.getcwd()
    model = get_model(data_obj)
    model = model.eval()
    model = model.to(device)

    #defense = get_defended_autoencoder(data_obj)
    #defended_model = nn.Sequential(defense, model)
    #defended_model.to(device)
    #defended_model.eval()
    fmodel = foolbox.models.PyTorchModel(model, bounds=(data_obj['min_pixel_value'], data_obj['max_pixel_value']), device="cuda:" + cuda_name)
    #attack = foolbox.attacks.EADAttack(confidence=c)
    #attack = foolbox.attacks.L2BrendelBethgeAttack()
    #attack = foolbox.attacks.L2CarliniWagnerAttack(confidence=c)
    #attack = foolbox.attacks.L2BrendelBethgeAttack(fmodel, foolbox.criteria.Misclassification())
    #attack = foolbox.attacks.SaltAndPepperNoiseAttack(fmodel, foolbox.criteria.Misclassification())
    #attack = foolbox.attacks.SaltAndPepperNoiseAttack()
    #attack = foolbox.attacks.EADAttack(confidence=c)
    #attack = foolbox.attacks.L2CarliniWagnerAttack()
    #attack = foolbox.attacks.SpatialAttack()
    #attack = foolbox.attacks.LinfFastGradientAttack()
    #attack = foolbox.attacks.LinfBasicIterativeAttack()
    attack = foolbox.attacks.L2BrendelBethgeAttack()
    #attack = foolbox.attacks.BoundaryAttack()
    #attack = foolbox.attacks.LinfFastGradientAttack()
    #attack_name = attack.__class__.__name__ + '_Foolbox_Logits_512_c_' + str(c).zfill(3)
    attack_name = attack.__class__.__name__ + '_Foolbox_Logits_512'
    #attack_name = attack.__class__.__name__ + '_Foolbox_Logits_steps_' + str(c)
    '''
    load = True
    if load:
        directory = os.path.join(pwd, 'AttackedData/Evasion/' + data_obj[
            'name'] + '/standard_classifier/temp/')
        d = np.load(directory + attack_name + '39.npy')
        templst = d.tolist()
        x_test_adv = []
        for item in templst:
            x_test_adv.append(np.asarray(item))
    else:
        x_test_adv = []
    '''
    x_test_adv = []
    print(attack_name)
    for batch_idx, data in enumerate(testloader):
        if batch_idx >= 8:
            break
        '''
        if batch_idx <= 39:
            continue
        '''
        img, label = data
        img = img.to(device)
        label = label.to(device)
        # img = img.view(img.size(0), -1)
        #i = img.data.cpu().detach().numpy()
        #label = label.data.cpu().detach().numpy()
        #targets = np.zeros(np.shape(label))
        #x_test_adv.append(attack.run(fmodel, i, foolbox.criteria.Misclassification(y_test)))
        # Untargeted
        epsilons = [0.0006, 0.0015, 0.03, 0.1, 0.15, 0.33, 0.5]
        # Targeted
        # epsilons = [0.002, 0.04, 0.06, 0.15, 0.2, 0.35]
        __, adv_ex, success = attack(fmodel, img, label, epsilons=epsilons)

        #TODO: Fix data writting/formating
        robust_accuracy = 1 - success.type(torch.float32).mean(axis=-1)
        # TODO: TEST WITH MIN INSEAD OF MAX MAY HAVE UNDERSTOOD ROBUST ACC WRONG

        if type(adv_ex) == list:
            adv_ex = adv_ex[torch.argmin(robust_accuracy)]

        x_test_adv.append(adv_ex.data.cpu().detach().numpy())
        print(str((testloader.batch_size * batch_idx) / len(testloader.dataset) * 100) + '% complete')
        directory = os.path.join(pwd, 'AttackedData/Evasion/' + data_obj[
            'name'] + '/' + data_obj['classifier'] + '/temp/')
        if not os.path.exists(directory):
            os.makedirs(directory)
        #np.save(directory + attack_name + str(batch_idx), np.concatenate(x_test_adv, axis=0))
    print(np.shape(x_test_adv))
    x_test_adv = np.concatenate(x_test_adv, axis=0)
    print(np.shape(x_test_adv))
    save_attacked_data(data_obj, x_test_adv, attack_name)
    empty_temp(data_obj)
    return x_test_adv, attack_name

def generate_attack_AdverTorch(data_obj, testloader, y_test):
    pwd = os.getcwd()
    model = get_model(data_obj)
    model.to(device)
    model.eval()
    attack = advertorch.attacks.CarliniWagnerL2Attack(model, data_obj['output_size'], clip_min=data_obj['min_pixel_value'], clip_max=data_obj['max_pixel_value'])

    BPDA = False
    #BPDA = True
    x_test_adv = []
    if BPDA:
        from advertorch.bpda import BPDAWrapper
        defense = get_defended_autoencoder(data_obj)
        defense.to(device)
        defense_withbpda = BPDAWrapper(defense, forwardsub=lambda x: x)
        defended_model = nn.Sequential(defense_withbpda, model)
        defended_model.eval()
        bpda_adversary = advertorch.attacks.LBFGSAttack(model,clip_min=data_obj['min_pixel_value'], clip_max=data_obj['max_pixel_value'],targeted=False)
        attack_name = 'LBFGS_AdverTorch_logits'
        print(attack_name)

        for batch_idx, data in enumerate(testloader):
            img, label = data
            # img = img.view(img.size(0), -1)
            #label = label.data.cpu().detach().numpy()
            img = img.to(device)
            label = label.to(device)
            bpda_adv = bpda_adversary.perturb(img, label)
            bpda_adv = bpda_adv.data.cpu().detach().numpy()
            #bpda_adv_defended = defense(bpda_adv)
            x_test_adv.append(bpda_adv)
            print(str((testloader.batch_size * batch_idx) / len(testloader.dataset) * 100) + '% complete')
            directory = os.path.join(pwd, 'AttackedData/Evasion/' + data_obj[
                'name'] + '/standard_classifier/temp/')
            if not os.path.exists(directory):
                os.makedirs(directory)
            np.save(directory + attack_name + str(batch_idx), np.concatenate(x_test_adv, axis=0))

    else:
        defense = get_defended_autoencoder(data_obj)
        defended_model = nn.Sequential(defense, model)
        defended_model.to(device)
        defended_model.eval()
        attack = advertorch.attacks.JacobianSaliencyMapAttack(model, clip_min=data_obj['min_pixel_value'],
                                                 clip_max=data_obj['max_pixel_value'], num_classes=data_obj['output_size'])
        attack_name = 'White_JSMA_AdverTorch_logits'
        print(attack_name)
        for batch_idx, data in enumerate(testloader):
            img, label = data
            img = img.to(device)
            label = label.to(device)
            exp = attack.perturb(img, label)
            exp = exp.data.cpu().detach().numpy()
            x_test_adv.append(exp)
            print(str((testloader.batch_size * batch_idx) / len(testloader.dataset) * 100) + '% complete')
            directory = os.path.join(pwd, 'AttackedData/Evasion/' + data_obj[
                'name'] + '/standard_classifier/temp/')
            if not os.path.exists(directory):
                os.makedirs(directory)
            np.save(directory + attack_name + str(batch_idx), np.concatenate(x_test_adv, axis=0))

    #np.save(os.path.join(pwd, 'Attacked-Data/Evasion/' + data_obj['name'] + '/standard_classifier/' + attack_name + '_list'), x_test_adv)
    x_test_adv = np.concatenate(x_test_adv, axis=0)

    save_attacked_data(data_obj, x_test_adv, attack_name)
    empty_temp(data_obj)
    return x_test_adv, attack_name

def generate_attack_CW(data_obj, testloader, y_test):
    model = get_model(data_obj)
    model.to(device)
    model.eval()
    attack = AttackCarliniWagnerL2(
        targeted=False,
        num_classes=10,
        max_steps=10000,
        search_steps=5,
        clip_min=data_obj['min_pixel_value'],
        clip_max=data_obj['max_pixel_value'],
        confidence=0,
        learning_rate=1e-2,
        cuda=True,
        debug=False)
    x_test_adv = []
    attack_name = 'CWL2_Strong_imp_c0'
    print(attack_name)
    pwd = os.getcwd()
    for batch_idx, data in enumerate(testloader):
        if batch_idx >= 8:
            break
        img, label = data
        # img = img.view(img.size(0), -1)
        #label = label.data.cpu().detach().numpy()
        img = img.to(device)
        label = label.to(device)
        input_adv = attack.run(model, img, label, batch_idx)
        x_test_adv.append(input_adv)
        print(str((testloader.batch_size * batch_idx) / len(testloader.dataset) * 100) + '% complete')
        directory = os.path.join(pwd, 'AttackedData/Evasion/' + data_obj[
            'name'] + '/standard_classifier/temp/')
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(directory + attack_name + str(batch_idx), np.concatenate(x_test_adv, axis=0))

    #np.save(os.path.join(pwd, 'Attacked-Data/Evasion/' + data_obj['name'] + '/standard_classifier/' + attack_name + '_list'), x_test_adv)
    x_test_adv = np.concatenate(x_test_adv, axis=0)

    save_attacked_data(data_obj, x_test_adv, attack_name)
    empty_temp(data_obj)
    return x_test_adv, attack_name

def generate_attack_custom(data_obj, testloader, y_test):
    model = get_model(data_obj)
    model.to(device)
    model.eval()
    defense = get_defended_autoencoder(data_obj)
    targeted = False
    attack = AttackCarliniWagnerCustom(ae_model=defense,
                                       targeted=targeted,
                                       num_classes=10,
                                       max_steps=10000,
                                       search_steps=5,
                                       clip_min=data_obj['min_pixel_value'],
                                       clip_max=data_obj['max_pixel_value'],
                                       confidence=10,
                                       learning_rate=1e-2,
                                       cuda=True,
                                       debug=False)
    x_test_adv = []
    attack_name = 'Custom_Attack_Logits_noConst_smartindex_Conf_10'
    print(attack_name)
    pwd = os.getcwd()
    for batch_idx, data in enumerate(testloader):
        img, label = data
        # img = img.view(img.size(0), -1)
        #label = label.data.cpu().detach().numpy()
        img = img.to(device)
        label = label.to(device)
        if targeted:
            label = get_targeted(label)
            label = label.to(device)
        input_adv = attack.run(model, img, label, batch_idx)
        x_test_adv.append(input_adv)
        print(str((testloader.batch_size * batch_idx) / len(testloader.dataset) * 100) + '% complete')
        '''
        directory = os.path.join(pwd, 'AttackedData/Evasion/' + data_obj[
            'name'] + '/standard_classifier/temp/')
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(directory + attack_name + str(batch_idx), np.concatenate(x_test_adv, axis=0))
        '''
    #np.save(os.path.join(pwd, 'Attacked-Data/Evasion/' + data_obj['name'] + '/standard_classifier/' + attack_name + '_list'), x_test_adv)
    x_test_adv = np.concatenate(x_test_adv, axis=0)
    save_attacked_data(data_obj, x_test_adv, attack_name)
    empty_temp(data_obj)
    return x_test_adv, attack_name

def generate_attack_cw_magnet(data_obj, testloader, y_test):
    from EAD_magnet_vs_BSD import setup_MagNet
    operator, thresh, detector_dict, reformer = setup_MagNet(dataloader, 0.01, data_obj)
    model = get_model(data_obj)
    model.to(device)
    model.eval()
    #defense = get_defended_autoencoder(data_obj)
    targeted = False
    from PytorchMagNet.pytorch_cw2_MagNet import AttackCarliniWagnerL2

    attack = AttackCarliniWagnerL2(model, reformer, detector_dict, thresh, targeted=False, search_steps=10)
    x_test_adv = []
    attack_name = 'CarliniWagnerMageNetImp'
    print(attack_name)
    pwd = os.getcwd()
    for batch_idx, data in enumerate(testloader):
        img, label = data
        # img = img.view(img.size(0), -1)
        #label = label.data.cpu().detach().numpy()
        img = img.to(device)
        label = label.to(device)
        if targeted:
            label = get_targeted(label)
            label = label.to(device)
        input_adv = attack.run(img, label, batch_idx)
        x_test_adv.append(input_adv)
        print(str((testloader.batch_size * batch_idx) / len(testloader.dataset) * 100) + '% complete')
        '''
        directory = os.path.join(pwd, 'AttackedData/Evasion/' + data_obj[
            'name'] + '/standard_classifier/temp/')
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(directory + attack_name + str(batch_idx), np.concatenate(x_test_adv, axis=0))
        '''
    #np.save(os.path.join(pwd, 'Attacked-Data/Evasion/' + data_obj['name'] + '/standard_classifier/' + attack_name + '_list'), x_test_adv)
    x_test_adv = np.concatenate(x_test_adv, axis=0)
    save_attacked_data(data_obj, x_test_adv, attack_name)
    empty_temp(data_obj)
    return x_test_adv, attack_name

def get_targeted(label):
    targets = []
    for i in label:
        for j in range(0, torch.max(label)):
            if (j == i):
                continue
            targets.append(j)
            break

    targets = torch.LongTensor(targets)
    return targets

def get_defended_autoencoder(data_obj):
    ae1 = get_autoencoder(data_obj)
    ae1.load_state_dict(torch.load(data_obj['L1_dir']))
    ae2 = get_autoencoder(data_obj)
    ae2.load_state_dict(torch.load(data_obj['L2_dir']))
    if data_obj['name'] == 'CIFAR10':
        defended_ae = defendedConvAutoencoder()
        ae1_dict = ae1.state_dict()
        ae2_dict = ae2.state_dict()
        new_ae1_dict = OrderedDict()
        for k, v in ae1_dict.items():
            name = k[:7] + '1' + k[7:]  # remove 'module.' of dataparallel
            new_ae1_dict[name] = v
        new_ae2_dict = OrderedDict()
        for k, v in ae2_dict.items():
            name = k[:7] + '2' + k[7:]  # remove 'module.' of dataparallel
            new_ae2_dict[name] = v
    elif data_obj['name'] == 'MNIST':
        defended_ae = defendedVAE()
        ae1_dict = ae1.state_dict()
        ae2_dict = ae2.state_dict()
        new_ae1_dict = OrderedDict()
        for k, v in ae1_dict.items():
            name = k[:k.rfind('.')] + '_1' + k[k.rfind('.'):] # remove 'module.' of dataparallel
            new_ae1_dict[name] = v
        new_ae2_dict = OrderedDict()
        for k, v in ae2_dict.items():
            name = k[:k.rfind('.')] + '_2' + k[k.rfind('.'):]  # remove 'module.' of dataparallel
            new_ae2_dict[name] = v

    defended_ae_dict = defended_ae.state_dict()
    pretrained_dict = {k: v for k, v in new_ae1_dict.items() if k in defended_ae_dict}
    defended_ae_dict.update(pretrained_dict)
    pretrained_dict.update({k: v for k, v in new_ae2_dict.items() if k in defended_ae_dict})
    defended_ae_dict.update(pretrained_dict)
    defended_ae.load_state_dict(pretrained_dict)
    return defended_ae



# Step 8: Apply Defense



def generate_autoencoded_classifiers(data_obj, x_train, y_train, y_test, dataloader, testloader):
    import os
    # For a single device (GPU 5)
    #os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    model = newConvAutoencoderTanh()
    #model.load_state_dict(torch.load('autoencoder_VAE_loss.pth'))
    model.load_state_dict(torch.load(data_obj['L1_dir']))
    model.eval()
    model = model.to(device)

    x = torch.from_numpy(x_train).float().to(device)
    outputL1 = []
    for i in range(0, len(x), 128):
        outputL1.append(model(x[i:i + 128]).data.cpu().detach().numpy())
    outputL1 = np.concatenate(outputL1, axis=0)

    model.load_state_dict(torch.load(data_obj['L2_dir']))
    model.eval()
    model = model.to(device)

    outputL2 = []
    for i in range(0, len(x), 128):
        outputL2.append(model(x[i:i + 128]).data.cpu().detach().numpy())
    outputL2 = np.concatenate(outputL2, axis=0)

    print(np.min(outputL2))
    print(np.max(outputL2))
    output = (outputL1 + outputL2)/2
    norm = Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    output = norm(torch.tensor(output))
    create_classifier(data_obj, output, y_train, y_test, "AE_Class_Sig_CIFAR10", dataloader, testloader)


def store_img(img, data_obj, attack_type):
    if data_obj['name'] == 'CIFAR10':
        unorm = UnNormalize(mean=(.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        img = torch.from_numpy(img)
        img = unorm(img).data.cpu().detach().numpy() * 255
        plt.figure(1)
        plt.imshow(np.moveaxis(img, 0, -1).astype('uint8'), interpolation='nearest')
        plt.title(attack_type)
    elif data_obj['name'] == 'MNIST':
        img = np.squeeze(img)
        plt.imshow((img*255).astype('uint8'), interpolation='nearest')
        plt.title(attack_type)
    return plt


def main(data_obj, x_test, y_test, x_train, y_train, dataloader, testloader, scale, percent):
    avg_acc = 0
    c = 0
    results = []
    #create_classifier(data_obj, y_test, 'CIFAR10_deep_classifier', dataloader, testloader)
    #create_classifier(data_obj, 0, y_train, y_test, "SimpleCIFAR100-1to1", dataloader, testloader)
    #generate_autoencoded_classifiers(data_obj, x_train, y_train, y_test, dataloader, testloader)
    generate_attack=False
    generate_attack=True
    if generate_attack:
        #confs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        confs = [10]
        #confs = [0, 50, 100]
        for c in confs:
            #create_classifier(data_obj, y_test, 'CIFAR10_standard_classifier', dataloader, testloader)
            #x_test_adv, attack_name = generate_attack_ART(data_obj, testloader, y_test, c)
            #x_test_adv, attack_name = generate_attack_Foolbox(data_obj, testloader, y_test, c)
            #x_test_adv, attack_name = generate_attack_cw_magnet(data_obj, testloader, y_test)
            #x_test_adv, attack_name = generate_attack_custom(data_obj, testloader, y_test)
            #x_test_adv, attack_name = generate_attack_AdverTorch(data_obj, testloader, y_test)
            x_test_adv, attack_name = generate_attack_CW(data_obj, testloader, y_test)
            #generate_autoencoded_classifiers(data_obj, x_train, y_test, dataloader, testloader)
            NaN_idx = np.unique(np.where(np.isnan(x_test_adv))[0])
            x_test_adv = np.delete(x_test_adv, NaN_idx, axis=0)
            x_test_prune = np.delete(x_test, NaN_idx, axis=0)
            x_test_prune = x_test_prune[:len(x_test_adv)]
            y_test_adv = np.delete(y_test, NaN_idx, axis=0)
            y_test_adv = y_test_adv[:len(x_test_adv)]

            if (x_test_adv.shape[1:] != (3, 32, 32) and data_obj['name'] == 'CIFAR10') or (x_test_adv.shape[1:] != (3, 32, 32) and data_obj['name'] == 'CIFAR100'):
                x_test_adv = np.moveaxis(x_test_adv, 3, -3)

            #ret, ret2, ret3, det, JSD_det = evaluate_defense_impact('NN', data_obj, x_train, 1, x_test_adv, y_test_adv, attack_name,
            #                                          dataloader, x_test_prune)
            #print(ret)
            #print(ret2)
            #print(ret3)

    evaluate = False
    #evaluate = True
    if evaluate:

        # Default values for hyper-parameters we're going to sweep over
        run = wandb.init(project='BroadSpectrumDefense', group=data_obj['name']+str(scale)+'_FP_'+str(percent) + '_' + data_obj['classifier'])
        #wandb.config.update(allow_val_change=True)
        #wandb.config.update({'s':s}, allow_val_change=True)
        #print(wandb.config)
        avg = 0
        all_data = []
        d = []
        c = 0
        pwd = os.getcwd()

        if data_obj['classifier'] == 'CIFAR10_deep_classifier':
            path = os.listdir(os.path.join(pwd, 'AttackedData/Evasion/' + data_obj['name'] + '/deep_classifier/'))
        else:
            path = os.listdir(os.path.join(pwd, 'AttackedData/Evasion/' + data_obj['name'] + '/' + data_obj['classifier'] + '/'))

        for attack_type in path:
            # attack_type = "SinglePixelFoolbox.npy"
            isfile = False
            if data_obj['classifier'] == 'CIFAR10_deep_classifier':
                p = os.path.join(pwd, 'AttackedData/Evasion/' + data_obj['name'] + '/deep_classifier/' + attack_type)
                if os.path.isfile(p):
                    isfile = True
                    x_test_adv = np.load(p)
            else:
                p = os.path.join(pwd, 'AttackedData/Evasion/' + data_obj['name'] + '/' + data_obj['classifier'] + '/' + attack_type)
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

                if x_test_adv.shape[1:] != (3, 32, 32) and data_obj['name'] == 'CIFAR10':
                    x_test_adv = np.moveaxis(x_test_adv, 3, -3)
            

                ret, ret2, ret3, det, JSD_det = evaluate_defense_impact('NN', data_obj, x_train, scale, x_test_adv,
                                                          y_test_adv, attack_type, dataloader, x_test_prune)

                d.append(
                    [ret['No_defense_accuracy'], ret['Total-detected-acc'], attack_type, ret['Comb_Raw-Accuracy'],
                     ret['Comb-detected'], ret['Class_diff_samples'], ret['Average-L2-Dist'], ret['Total-detected-diff'], ret['Number-Attack-Samples'],ret['Total-JSD-detected-acc'], ret['newJSD']])
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
                else:
                    plt = store_img(x_test_adv[0], data_obj, attack_type)
                    wandb.log(
                        {"Table": wandb.Table(data=d, columns=["Original Accuracy", "Defended Accuracy", "Attack Type",
                                                               "Autoencoded Accuracy", "Distance Based Detection",
                                                               "Class Difference Detected", "Average L2 Distance", 'Total Removed Samples', 'Attack Samples Generated', 'JSD Samples Accuracy', 'Additional JSD Removed']),
                         "Image": plt, 'Original Accuracy': ret['No_defense_accuracy'],
                         'Autoencoded Accuracy': ret['Comb_Raw-Accuracy'],
                         'Defended Accuracy': ret['Total-detected-acc'],
                         'Average L2 Distance': ret['Average-L2-Dist'],
                         'Total Removed Samples': ret['Total-detected-diff'],
                         'Distance Based Detection': ret['Comb-detected'],
                         'Attack Samples Generated': ret['Number-Attack-Samples'],
                         'JSD Samples Removed': ret['JSD_rejected_samples'],
                         'Total Acc With JSD': ret['Total-JSD-detected-acc'],
                         'New JSD Detections': ret['newJSD']})
                all_data.append(ret)
                avg += ret['Total-detected-acc']
                c += 1
                print(ret)
                print(ret2)
                print(ret3)
                print(det)
                print()
        print("Average Accuracy = " + str(avg / c))

        ret, ret2, ret3, det, JSD_det = evaluate_defense_impact('NN', data_obj, x_train, scale, x_test, y_test,
                                                  "Clean Data",
                                                  dataloader, x_test)
        print(ret)
        print(ret2)
        print(ret3)
        print()
        all_data.append(ret)
        if data_obj['name'] != 'occupancy':
            plt = store_img(x_test[0], data_obj, 'Clean Data')
            wandb.log(
                {"Image": plt, 'Defended Clean Test Accuracy':  ret['Total-detected-acc'], 'Total Acc With JSD': ret['Total-JSD-detected-acc'], 'Total Removed Clean Samples': ret['Total-detected-diff'], "Clean Distance Based Detection": ret['Comb-detected'], "Clean Adjusted Accuracy Conf": ret['Clean_Adj_Acc_conf'], "Clean Adjusted Accuracy JSD": ret['Clean_Adj_Acc_JSD'], "Clean JSD Removed": ret['JSD_rejected_samples']}
            )
        else:
            wandb.log(
                {'Defended Clean Test Accuracy': ret['Total-detected-acc'],
                 'Total Acc With JSD': ret['Total-JSD-detected-acc'],
                 'Total Removed Clean Samples': ret['Total-detected-diff'],
                 "Clean Distance Based Detection": ret['Comb-detected'],
                 "Clean Adjusted Accuracy Conf": ret['Clean_Adj_Acc_conf'],
                 "Clean Adjusted Accuracy JSD": ret['Clean_Adj_Acc_JSD'],
                 "Clean JSD Removed": ret['JSD_rejected_samples']}
            )
        '''
        with open(os.path.join(pwd, 'Results/OUTPUT' + data_obj['name'] + '.csv'), 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(all_data[0].keys())
            for line in all_data:
                wr.writerow(line.values())

        '''
    '''
    scale = [.75, 1, 1.25, 1.5, 1.75, 2]
    for s in scale:
        run_def(s)
    '''
if __name__ == "__main__":
    # Count the arguments
    import argparse
    parser = argparse.ArgumentParser(description='Process wandb sweep.')
    parser.add_argument('--scale')
    parser.add_argument('--FP')
    parser.add_argument('--set')
    a = parser.parse_args()
    '''
    AllSets = ['MNIST', 'CIFAR10']
    AllFP = [0, 0.01, 0.03, 0.05]
    for set in AllSets:
        for FP in AllFP:
            data_obj, x_test, y_test, x_train, y_train, dataloader, testloader = generate_struct(float(a.scale), set, float(FP), cuda_name)
            main(data_obj, x_test, y_test, x_train, y_train, dataloader, testloader, float(a.scale), float(FP))
    '''
    data_obj, x_test, y_test, x_train, y_train, dataloader, testloader = generate_struct(float(a.scale), a.set, float(a.FP),
                                                                                         cuda_name)
    main(data_obj, x_test, y_test, x_train, y_train, dataloader, testloader, float(a.scale), float(a.FP))