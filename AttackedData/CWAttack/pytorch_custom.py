"""PyTorch Carlini and Wagner L2 attack algorithm.
Based on paper by Carlini & Wagner, https://arxiv.org/abs/1608.04644 and a reference implementation at
https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks_tf.py
"""
import os
import sys
import torch
import numpy as np
from torch import optim
from torch import autograd
from .helpers import *


class AttackCarliniWagnerCustom:

    def __init__(self, ae_model, targeted=True, search_steps=None, max_steps=None, cuda=True, num_classes=10, confidence=0, clip_min=-1, clip_max=1, learning_rate=1e-2, debug=False):
        self.ae_model = ae_model.cuda()
        self.debug = debug
        self.targeted = targeted
        self.num_classes = num_classes
        self.confidence = confidence  # FIXME need to find a good value for this, 0 value used in paper not doing much...
        self.initial_const = 0.01  # bumped up from default of .01 in reference code
        self.binary_search_steps = search_steps
        self.repeat = self.binary_search_steps >= 10
        self.max_steps = max_steps or 1000
        self.c_loss3 = 0.001
        self.abort_early = True
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.learning_rate = learning_rate
        self.cuda = cuda
        self.clamp_fn = 'tanh'  # set to something else perform a simple clamp instead of tanh
        self.init_rand = False  # an experiment, does a random starting point help?
        self.boxmul = (clip_max - clip_min) / 2.
        self.boxplus = (clip_min + clip_max) / 2.
    def _compare(self, output, target):
        if not isinstance(output, (float, int, np.int64)):
            output = np.copy(output)
            if self.targeted:
                output[target] -= self.confidence
            else:
                output[target] += self.confidence
            output = np.argmax(output)
        if self.targeted:
            return output == target
        else:
            return output != target

    def _loss(self, input_adv, output, ae_output, target, dist, scale_const, const_2):
        # compute the probability of the label class versus the maximum other
        # In untargeted the "target is the real class"
        real = (target * output).sum(1)
        real_ae = (target * ae_output).sum(1)
        other = ((1. - target) * output - target * 10000.).max(1)[0]
        # OK for targeted
        other_ae = ((1. - target) * ae_output - target * 10000.).max(1)[0]
        # TODO: Figure out how to get Autoencodered sample class to equal Real class
        if self.targeted:
            # if targeted, optimize for making the other class most likely
            loss1 = torch.clamp(other - real + self.confidence, min=0.)  # equiv to max(..., 0.)
            loss4 = torch.clamp(other_ae - real + self.confidence, min=0.)
        else:
            # if non-targeted, optimize for making this class least likely.
            loss1 = torch.clamp(real - other + self.confidence, min=0.)  # equiv to max(..., 0.)
            # Have to take the most likely for regular and optimize autoencoded towards it
            new_target = ((1. - target) * output - target * 10000.).max(1)[1]
            new_target_onehot = torch.zeros(new_target.size() + (self.num_classes,))
            if self.cuda:
                new_target_onehot = new_target_onehot.cuda()
            new_target_onehot.scatter_(1, new_target.unsqueeze(1), 1.)
            other_ae = (new_target_onehot * ae_output).max(1)[0]

            loss4 = torch.clamp(real_ae - other_ae + self.confidence, min=0.)


        loss1 = torch.sum(scale_const * (loss1 + loss4))

        detected = [detector.mark(input_adv) - self.thrs[name] for name, detector in self.detector_dict.items()]
        detected = torch.cat(detected, dim=0)
        detected = torch.max(torch.zeros(len(detected)), detected)

        self.detected = torch.sum(detected, dim=0).cuda()

        loss3 = torch.sum(const_2 * self.detected)

        loss2 = dist.sum()

        #Scale_const loss3 = torch.sum(loss3 * scale_const)
        #OLD loss3 = torch.sum(torch.clamp(torch.abs(real_ae - real), min=0.)) * self.c_loss3
        # print('loss1 ' + str(loss1.item()) + 'loss2 ' + str(loss2.item()) + 'loss3 ' + str(loss3.item()))
        loss = loss1 + loss2 + loss3
        return loss

    def _optimize(self, optimizer, model, input_var, modifier_var, target_var, scale_const_var, const_2, input_orig=None):
        # apply modifier and clamp resulting image to keep bounded from clip_min to clip_max
        if self.clamp_fn == 'tanh':
            input_adv = tanh_rescale(modifier_var + input_var, self.clip_min, self.clip_max)
        else:
            input_adv = torch.clamp(modifier_var + input_var, self.clip_min, self.clip_max)

        output = model(input_adv)
        ae_img = self.ae_model(input_adv)
        ae_output = model(ae_img)
        # distance to the original input data
        if input_orig is None:
            dist = l2_dist(input_adv, input_var, keepdim=False)
        else:
            dist = l2_dist(input_adv, input_orig, keepdim=False)
        #print(dist)
        loss = self._loss(input_adv, output, ae_output, target_var, dist, scale_const_var, const_2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # CHANGED THIS
        #loss_np = loss.data[0]
        loss_np = loss.item()
        dist_np = dist.data.cpu().numpy()
        output_np = output.data.cpu().numpy()
        input_adv_np = input_adv.data.permute(0, 2, 3, 1).cpu().numpy()  # back to BHWC for numpy consumption
        return loss_np, dist_np, output_np, input_adv_np

    def run(self, model, input, target, batch_idx=0):
        batch_size = input.size(0)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        scale_const = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        # python/numpy placeholders for the overall best l2, label score, and adversarial image
        o_best_l2 = [1e10] * batch_size
        o_best_score = [-1] * batch_size
        o_best_attack = input.permute(0, 2, 3, 1).cpu().numpy()

        # setup input (image) variable, clamp/scale as necessary
        if self.clamp_fn == 'tanh':
            # convert to tanh-space, input already int -1 to 1 range, does it make sense to do
            # this as per the reference implementation or can we skip the arctanh?

            # input_var = autograd.Variable(torch_arctanh(input), requires_grad=False)
            input_var = autograd.Variable(torch.atan((input - self.boxplus) / self.boxmul * 0.999999), requires_grad=False)
            input_orig = tanh_rescale(input_var, self.clip_min, self.clip_max)
        else:
            input_var = autograd.Variable(input, requires_grad=False)
            input_orig = None

        # setup the target variable, we need it to be in one-hot form for the loss function
        target_onehot = torch.zeros(target.size() + (self.num_classes,))
        if self.cuda:
            target_onehot = target_onehot.cuda()
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        target_var = autograd.Variable(target_onehot, requires_grad=False)

        # setup the modifier variable, this is the variable we are optimizing over
        modifier = torch.zeros(input_var.size()).float()
        if self.init_rand:
            # Experiment with a non-zero starting point...
            modifier = torch.normal(means=modifier, std=0.001)
        if self.cuda:
            modifier = modifier.cuda()
        modifier_var = autograd.Variable(modifier, requires_grad=True)

        optimizer = optim.Adam([modifier_var], lr=self.learning_rate)

        for search_step in range(self.binary_search_steps):
            print('Batch: {0:>3}, search step: {1}'.format(batch_idx, search_step))
            if self.debug:
                print('Const:')
                for i, x in enumerate(scale_const):
                    print(i, x)
            best_l2 = [1e10] * batch_size
            best_score = [-1] * batch_size

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and search_step == self.binary_search_steps - 1:
                scale_const = upper_bound

            scale_const_tensor = torch.from_numpy(scale_const).float()
            if self.cuda:
                scale_const_tensor = scale_const_tensor.cuda()

            scale_const2_tensor = torch.ones(batch_size) * self.initial_const * 10
            if self.cuda:
                scale_const2_tensor = scale_const2_tensor.cuda()

            scale_const_var = autograd.Variable(scale_const_tensor, requires_grad=False)
            const_2 = autograd.Variable(scale_const2_tensor, requires_grad=False)
            # TODO: Add another grid search over other constant to find propper value
            prev_loss = 1e6
            for step in range(self.max_steps):
                # perform the attack
                loss, dist, output, adv_img = self._optimize(
                    optimizer,
                    model,
                    input_var,
                    modifier_var,
                    target_var,
                    scale_const_var,
                    const_2,
                    input_orig)

                if step % 1000 == 0 or step == self.max_steps - 1:
                    print('Step: {0:>4}, loss: {1:6.4f}, dist: {2:8.5f}, modifier mean: {3:.5e}'.format(
                        step, loss, dist.mean(), modifier_var.data.mean()))

                if self.abort_early and step % (self.max_steps // 10) == 0:
                    if loss > prev_loss * .9999:
                        print('Aborting early...')
                        break
                    prev_loss = loss

                # update best result found
                for i in range(batch_size):
                    target_label = target[i]
                    output_logits = output[i]
                    output_label = np.argmax(output_logits)
                    di = dist[i]
                    if self.debug:
                        if step % 100 == 0:
                            print('{0:>2} dist: {1:.5f}, output: {2:>3}, {3:5.3}, target {4:>3}'.format(
                                i, di, output_label, output_logits[output_label], target_label))
                    if di < best_l2[i] and self._compare(output_logits, target_label):
                        if self.debug:
                            print('{0:>2} best step,  prev dist: {1:.5f}, new dist: {2:.5f}'.format(
                                  i, best_l2[i], di))
                        best_l2[i] = di
                        best_score[i] = output_label
                    if di < o_best_l2[i] and self._compare(output_logits, target_label):
                        if self.debug:
                            print('{0:>2} best total, prev dist: {1:.5f}, new dist: {2:.5f}'.format(
                                  i, o_best_l2[i], di))
                        o_best_l2[i] = di
                        o_best_score[i] = output_label
                        o_best_attack[i] = adv_img[i]

                sys.stdout.flush()
                # end inner step loop

            # adjust the constants
            batch_failure = 0
            batch_success = 0
            for i in range(batch_size):
                if self._compare(best_score[i], target[i]) and best_score[i] != -1:
                    # successful, do binary search and divide const by two
                    upper_bound[i] = min(upper_bound[i], scale_const[i])
                    if upper_bound[i] < 1e9:
                        scale_const[i] = (lower_bound[i] + upper_bound[i]) / 2
                    if self.debug:
                        print('{0:>2} successful attack, lowering const to {1:.3f}'.format(
                            i, scale_const[i]))
                else:
                    # failure, multiply by 10 if no solution found
                    # or do binary search with the known upper bound
                    lower_bound[i] = max(lower_bound[i], scale_const[i])
                    if upper_bound[i] < 1e9:
                        scale_const[i] = (lower_bound[i] + upper_bound[i]) / 2
                    else:
                        scale_const[i] *= 10
                    if self.debug:
                        print('{0:>2} failed attack, raising const to {1:.3f}'.format(
                            i, scale_const[i]))
                if self._compare(o_best_score[i], target[i]) and o_best_score[i] != -1:
                    batch_success += 1
                else:
                    batch_failure += 1

            print('Num failures: {0:2d}, num successes: {1:2d}\n'.format(batch_failure, batch_success))
            sys.stdout.flush()
            # end outer search loop

        return o_best_attack