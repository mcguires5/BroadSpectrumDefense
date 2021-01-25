import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy
from numpy.linalg import norm
import os
import copy
from scipy.ndimage import median_filter
import scipy

def to_img(x):
    x = x.clamp(0, 1)
    return x

class CDDetector:
    def __init__(self, model, classifier, device):
        """
        Error based detector.
        Marks examples for filtering decisions.

        model : pytorch model class
        device : torch.device
        path: Path to the autoencoder used.
        p: Distance measure to use.
        """
        self.model = model

        self.device = device
        self.classifier = classifier
        self.model.eval()

    def mark(self, X):

        if torch.is_tensor(X):
            X_torch = X.to(self.device)
        else:
            X_torch = torch.from_numpy(X).to(self.device)

        X_hat = self.model(X_torch)
        Y_hat = self.classifier.model(X_hat)
        Y = self.classifier.model(X_torch)
        marks = torch.eq(torch.argmax(Y_hat,dim=1), torch.argmax(Y,dim=1))*-1

        return marks

    def print(self):
        return "CDDetector:" + self.path.split("/")[-1]

class CDDetectorNoisy:
    def __init__(self, model, classifier, device, minimum, maximum):
        """
        Error based detector.
        Marks examples for filtering decisions.

        model : pytorch model class
        device : torch.device
        path: Path to the autoencoder used.
        p: Distance measure to use.
        """
        self.model = model

        self.device = device
        self.classifier = classifier

        self.model.eval()
        self.min = minimum
        self.max = maximum

    def mark(self, X):

        if torch.is_tensor(X):
            X_torch = X.to(self.device)
        else:
            X_torch = torch.from_numpy(X).to(self.device)

        noise = 0.1 * np.random.normal(size=np.shape(X_torch))
        noise = torch.from_numpy(noise).to(self.device)

        noisy_data = X_torch.double() + noise

        # noisy_train_data.clamp(-1.0, 1.0)
        noisy_data = noisy_data.clamp(self.min, self.max)

        noisy_data = noisy_data.to(self.device).float()

        noisy_hat = self.model(noisy_data).detach()
        noisy_Y_hat = self.classifier.model(noisy_hat)

        X_hat = self.model(X_torch).detach()
        Y_hat = self.classifier.model(X_hat)
        Y = self.classifier.model(X_torch)

        noisy_or_ref = torch.eq(torch.argmax(Y_hat,dim=1), torch.argmax(noisy_Y_hat,dim=1))
        ref_or_real = torch.eq(torch.argmax(Y_hat, dim=1), torch.argmax(Y, dim=1))
        marks = torch.eq(noisy_or_ref, ref_or_real)*-1

        return marks

    def print(self):
        return "CDDetector:" + self.path.split("/")[-1]

class CDDetectorMedian:
    def __init__(self, model, classifier, device, minimum, maximum):
        """
        Error based detector.
        Marks examples for filtering decisions.

        model : pytorch model class
        device : torch.device
        path: Path to the autoencoder used.
        p: Distance measure to use.
        """
        self.model = model

        self.device = device
        self.classifier = classifier

        self.model.eval()
        self.min = minimum
        self.max = maximum

    def mark(self, X):

        if torch.is_tensor(X):
            X_torch = X.to(self.device)
        else:
            X_torch = torch.from_numpy(X).to(self.device)

        #X_median = median_filter(np.asarray(X.cpu()), size=3)
        #print(X_torch.shape())
        # Add extra dimension
        #print(np.full((3, 3, 3), 1.0/27))
        Y = self.classifier.model(X_torch)
        X_hat = self.model(X_torch).detach()
        length = X_torch.size()[0]
        del X_torch

        X_median = []
        for idx in range(length):
            #X_median.append(scipy.ndimage.filters.convolve(X[idx].cpu(), np.full((3, 3, 3), 1.0/(27))))
            #X_median.append(scipy.ndimage.median_filter(X[idx].cpu(), 3))
            X_median.append(scipy.ndimage.gaussian_filter(X[idx].cpu(), sigma=0.5))
        X_median = np.stack(X_median)
        X_median = torch.as_tensor(X_median)

        if torch.is_tensor(X_median):
            X_median = X_median.to(self.device)
        else:
            X_median = torch.from_numpy(X_median).to(self.device)

        median_hat = self.model(X_median).detach()
        del X_median
        medain_Y_hat = self.classifier.model(median_hat)
        del median_hat
        torch.cuda.empty_cache()
        print(torch.cuda.memory_summary())
        Y_hat = self.classifier.model(X_hat)

        noisy_or_ref = torch.eq(torch.argmax(Y_hat,dim=1), torch.argmax(medain_Y_hat,dim=1))
        ref_or_real = torch.eq(torch.argmax(Y_hat, dim=1), torch.argmax(Y, dim=1))
        marks = torch.eq(noisy_or_ref, ref_or_real)*-1

        return marks

    def print(self):
        return "CDDetector:" + self.path.split("/")[-1]

class CDDetectorC:
    def __init__(self, model_list, classifier, device, minimum, maximum):
        """
        Error based detector.
        Marks examples for filtering decisions.

        model : pytorch model class
        device : torch.device
        path: Path to the autoencoder used.
        p: Distance measure to use.
        """
        self.model_list = model_list

        self.device = device
        self.classifier = classifier
        self.min = minimum
        self.max = maximum

    def mark(self, X):

        if torch.is_tensor(X):
            X_torch = X.to(self.device)
        else:
            X_torch = torch.from_numpy(X).to(self.device)

        noise = 0.1 * np.random.normal(size=np.shape(X_torch))
        noise = torch.from_numpy(noise).to(self.device)

        noisy_data = X_torch.double() + noise

        del noise
        # noisy_train_data.clamp(-1.0, 1.0)
        noisy_data = noisy_data.clamp(self.min, self.max)

        noisy_data = noisy_data.to(self.device).float()

        X_hat = torch.zeros(X_torch.size()).to(self.device)
        noisy_hat = torch.zeros(X_torch.size()).to(self.device)
        for ae in self.model_list:
            X_hat += ae(X_torch).to(self.device)
            noisy_hat += ae(noisy_data).to(self.device)

        del noisy_data

        X_hat = X_hat/len(self.model_list)
        noisy_hat = noisy_hat / len(self.model_list)
        noisy_Y_hat = self.classifier.model(noisy_hat)
        Y_hat = self.classifier.model(X_hat)
        Y = self.classifier.model(X_torch)

        noisy_or_ref = torch.eq(torch.argmax(Y_hat,dim=1), torch.argmax(noisy_Y_hat,dim=1))
        ref_or_real = torch.eq(torch.argmax(Y_hat, dim=1), torch.argmax(Y, dim=1))
        marks = torch.eq(noisy_or_ref, ref_or_real)*-1
        return marks

    def print(self):
        return "CDDetectorC:" + self.path.split("/")[-1]


class AEDetector:
    def __init__(self, model, device, p=2, dims=(1, 2, 3)):
        """
        Error based detector.
        Marks examples for filtering decisions.

        model : pytorch model class
        device : torch.device
        path: Path to the autoencoder used.
        p: Distance measure to use.
        """
        self.model = model
        self.p = p
        self.dims = dims

        self.device = device
        self.model = self.model.to(self.device)
        self.model.eval()

    def mark(self, X):

        if torch.is_tensor(X):
            X_torch = X.to(self.device)
        else:
            X_torch = torch.from_numpy(X).to(self.device)

        diff = torch.abs(X_torch -
                         self.model(X_torch).detach())


        marks = torch.mean(torch.pow(diff, self.p), dim=self.dims)

        return marks.cpu()

    def print(self):
        return "AEDetector:" + self.path.split("/")[-1]

class AEDetectorC:
    def __init__(self, ae_list, device, p, minimum, maximum, dims=(1, 2, 3)):
        """
        Error based detector.
        Marks examples for filtering decisions.

        model : pytorch model class
        device : torch.device
        path: Path to the autoencoder used.
        p: Distance measure to use.
        """
        self.ae_list = ae_list
        self.p = p
        self.device = device
        self.dims = dims
        self.min = minimum
        self.max = maximum

    def mark(self, X):
        out = []
        for ae in self.ae_list:
            if torch.is_tensor(X):
                X_torch = X.to(self.device)
            else:
                X_torch = torch.from_numpy(X).to(self.device)

            out.append(ae(X_torch).detach())


        avg_img = torch.clamp(sum(out)/len(out), self.min, self.max)
        marks = torch.mean(torch.pow(X_torch - avg_img, self.p), dim=self.dims)

        return marks.cpu()

    def print(self):
        return "AEDetector:" + self.path.split("/")[-1]

class DBDetector:
    def __init__(self, reconstructor, prober, classifier, option="jsd", T=1):
        """
        Divergence-Based Detector.

        reconstructor: One autoencoder.
        prober: Another autoencoder.
        classifier: Classifier object.
        option: Measure of distance, jsd as default.
        T: Temperature to soften the classification decision.
        """
        self.prober = prober
        self.reconstructor = reconstructor
        self.classifier = classifier
        self.option = option
        self.T = T

    def mark(self, X):
        return self.mark_jsd(X)

    def JSD(self, P, Q):
        _P = P / norm(P, ord=1)
        _Q = Q / norm(Q, ord=1)
        _M = 0.5 * (_P + _Q)
        return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

    def mark_jsd(self, X):
        Xp = self.prober.model(X)
        Xr = self.reconstructor.model(X)
        softmax = torch.nn.Softmax(dim=1)
        comb_prob = softmax(self.classifier.model(Xp) / self.T).cpu().data.numpy()
        og_prob = softmax(self.classifier.model(Xr) / self.T).cpu().data.numpy()
        marks = [(self.JSD(comb_prob[i], og_prob[i])) for i in range(len(Xp))]
        marks = torch.as_tensor(marks)
        return marks

    def print(self):
        return "Divergence-Based Detector"

class IdReformer:
    def __init__(self, path="IdentityFunction"):
        """
        Identity reformer.
        Reforms an example to itself.
        """
        self.path = path
        self.heal = lambda X: X
        self.model = lambda X: X

    def print(self):
        return "IdReformer:" + self.path


class SimpleReformer:
    def __init__(self, model, device, min, max):
        """
        Reformer.
        Reforms examples with autoencoder. Action of reforming is called heal.
        path: Path to the autoencoder used.
        """
        self.model = model

        self.device = device
        self.model.eval()
        self.model = self.model.to(device)

        self.min = min
        self.max = max

    def heal(self, X):
        # X = self.model.predict(X)
        # return np.clip(X, 0.0, 1.0)

        if torch.is_tensor(X):
            X_torch = X
        else:
            X_torch = torch.from_numpy(X)

        X = self.model(X_torch.to(self.device)).detach().cpu()

        return torch.clamp(X, self.min, self.max)

    def print(self):
        return "SimpleReformer:" + self.path.split("/")[-1]


class SimpleReformerC:
    def __init__(self, model_list, device, min, max):
        """
        Reformer.
        Reforms examples with autoencoder. Action of reforming is called heal.
        path: Path to the autoencoder used.
        """
        self.model_lst = model_list
        self.device = device

        self.min = min
        self.max = max

    def heal(self, X):
        # X = self.model.predict(X)
        # return np.clip(X, 0.0, 1.0)

        if torch.is_tensor(X):
            X_torch = X
        else:
            X_torch = torch.from_numpy(X)

        X = torch.zeros(X_torch.size()).to(self.device)
        for ae in self.model_lst:
            ae.to(self.device)
            ae.eval()
            X += ae(X_torch.to(self.device))
        X = (X) / len(self.model_lst)

        return torch.clamp(X, self.min, self.max)

    def print(self):
        return "SimpleReformer:" + self.path.split("/")[-1]


class Classifier:
    def __init__(self, model, device, device_ids=[0]):
        """
        Keras classifier wrapper.
        Note that the wrapped classifier should spit logits as output.
        model : pytorch model class
        device : torch.device
        classifier_path: Path to Keras classifier file.
        """
        self.model = model


        self.softmax = nn.Softmax(dim=1)

        self.device = device
        self.model = self.model.to(device)
        self.model.eval()

    def classify(self, X, option="logit", T=1):

        if torch.is_tensor(X):
            X_torch = X
        else:
            X_torch = torch.from_numpy(X)

        X_torch = X_torch.to(self.device)

        if option == "logit":
            return self.model(X_torch).detach().cpu()
        if option == "prob":
            logits = self.model(X_torch) / T
            logits = self.softmax(logits)
            return logits.detach().cpu()

    def print(self):
        return "Classifier:" + self.path.split("/")[-1]


class AttackData:
    def __init__(self, examples, labels, name=""):
        """
        Input data wrapper. May be normal or adversarial.
        examples: object of input examples.
        labels: Ground truth labels.
        """

        self.data = examples
        self.labels = labels
        self.name = name

    def print(self):
        return "Attack:" + self.name

class Operator:
    def __init__(self, classifier, det_dict, reformer, device, dims=(1, 2, 3)):
        """
        Operator.
        Describes the classification problem and defense.

        data: Standard problem dataset. Including train, test, and validation.
        classifier: Target classifier.
        reformer: Reformer of defense.
        det_dict: Detector(s) of defense.
        """
        self.device = device
        self.classifier = classifier
        self.det_dict = det_dict
        self.reformer = reformer
        self.dims = dims

    def no_def_acc(self, inputs):
        X = inputs
        X = X.to(self.device)
        Y = self.classifier.classify(X)
        return Y

    def operate(self, inputs, filtered=True):
        X = inputs

        if not torch.is_tensor(X):
            X = torch.from_numpy(X)

        X = X.to(self.device)
        if filtered:
            Y_prime = []
            for i in range(0, len(X), 128):
                X_prime = self.reformer.heal(X[i:i+128])
                Y_prime.append(self.classifier.classify(X_prime))
            Y_prime = torch.cat(Y_prime, dim=0)
        else:
            Y_prime = []
            for i in range(0, len(X), 128):
                Y_prime.append(self.classifier.classify(X[i:i+128]))
            Y_prime = torch.cat(Y_prime, dim=0)
        return Y_prime

    def get_thrs(self, dataloader, det_dict, drop_rate):
        """
        Get filtering threshold by marking validation set.
        """
        thrs = dict()
        for name, detector in self.det_dict.items():
            marks = []
            number_seen = 0
            for batch_idx, data in enumerate(dataloader):
                img, label = data
                marks.append(detector.mark(img.to(self.device)).cpu().numpy())
                number_seen += len(img)
                if number_seen > 100000:
                    break

            if drop_rate[name] == -1:
                thrs[name] = -0.5
            else:
                # TODO SAVE MEMORY HERE OVERFLOWING
                num = int(number_seen * drop_rate[name])
                marks = np.concatenate(marks, axis=0)

                marks = np.sort(marks)
                thrs[name] = marks[-num]
        return thrs

    def filters(self, data, thrs):
        """
        untrusted_obj: Untrusted input to test against.
        thrs: Thresholds.
        return:
        all_pass: Index of examples that passed all detectors.
        collector: Number of examples that escaped each detector.
        """
        collector = dict()
        all_pass = np.array(range(len(data)))
        data = data.to(self.device)
        for name, detector in self.det_dict.items():
            marks = detector.mark(data)
            np_marks = marks.cpu().numpy()

            if len(np_marks.shape) == 4:
                np_marks = np.mean(np.power(np_marks, 1), axis=self.dims)
            idx_pass = np.argwhere(np_marks < thrs[name])
            collector[name] = len(idx_pass)
            all_pass = np.intersect1d(all_pass, idx_pass)

        return all_pass, collector
