import os
import time
import torch
from torch.nn import DataParallel, Sequential, Module
from torchvision.datasets.folder import default_loader
from Architectures.MLP import MLP
from Architectures.ResNet import ResNet
from Third_Party.smoothing_adversarial.attacks import PGD_L2, DDN
from Third_Party.smoothing_adversarial.architectures import get_architecture
import numpy as np
from matplotlib import pyplot as plt
import gc
from bisect import bisect_left
import pandas as pd
from torch.nn.functional import softmax
from scipy.stats import rankdata
from numpy.random import default_rng
from scipy.stats.mstats import mquantiles
from scipy.stats import norm
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple
from torchvision.models.resnet import resnet50
import torchvision
from torchvision import transforms as transforms
from collections import defaultdict
# function to calculate accuracy of the model
def calculate_accuracy(model, dataloader, device):
    model.eval()  # put in evaluation mode
    total_correct = 0
    total_images = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    model_accuracy = total_correct / total_images
    return model_accuracy


def Smooth_Adv(model, x, y, noises, N_steps=20, max_norm=0.125, device='cpu', GPU_CAPACITY=1024, method='PGD'):
    # create attack model
    if method == 'PGD':
        attacker = PGD_L2(steps=N_steps, device=device, max_norm=max_norm)
    elif method == "DDN":
        attacker = DDN(steps=N_steps, device=device, max_norm=max_norm)

    # create container for the adversarial examples
    x_adv = torch.zeros_like(x)

    # get number of data points
    n = x.size()[0]

    # number of permutations to estimate mean
    num_of_noise_vecs = noises.size()[0] // n

    # calculate maximum batch size according to gpu capacity
    batch_size = GPU_CAPACITY // num_of_noise_vecs

    # calculate number of batches
    if n % batch_size != 0:
        num_of_batches = (n // batch_size) + 1
    else:
        num_of_batches = (n // batch_size)

    # start generating examples for each batch
    print("Generating Adverserial Examples:")

    for j in tqdm(range(num_of_batches)):
        #GPUtil.showUtilization()
        # get inputs and labels of batch
        inputs = x[(j * batch_size):((j + 1) * batch_size)]
        labels = y[(j * batch_size):((j + 1) * batch_size)]

        # duplicate batch according to the number of added noises and send to device
        # the first num_of_noise_vecs samples will be duplicates of x[0] and etc.
        tmp = torch.zeros((len(labels) * num_of_noise_vecs, *inputs.shape[1:]))
        x_tmp = inputs.repeat((1, num_of_noise_vecs, 1, 1)).view(tmp.shape).to(device)

        # send labels to device
        y_tmp = labels.to(device).long()

        # generate random Gaussian noise for the duplicated batch
        noise = noises[(j * (batch_size * num_of_noise_vecs)):((j + 1) * (batch_size * num_of_noise_vecs))].to(device)
        # noise = torch.randn_like(x_tmp, device=device) * sigma_adv

        # generate adversarial examples for the batch
        x_adv_batch = attacker.attack(model, x_tmp, y_tmp,
                                      noise=noise, num_noise_vectors=num_of_noise_vecs,
                                      no_grad=False,
                                      )

        # take only the one example for each point
        x_adv_batch = x_adv_batch[::num_of_noise_vecs]

        # move back to CPU
        x_adv_batch = x_adv_batch.to(torch.device('cpu'))

        # put in the container
        x_adv[(j * batch_size):((j + 1) * batch_size)] = x_adv_batch.detach().clone()


    # return adversarial examples
    return x_adv


def evaluate_predictions(S, X, y, conditional=False, coverage_on_label=False, num_of_classes=10):

    # get numbers of points
    #n = np.shape(X)[0]

    # get points to a matrix of the format nxp
    #X = np.vstack([X[i, 0, :, :].flatten() for i in range(n)])

    # Marginal coverage
    marg_coverage = np.mean([y[i] in S[i] for i in range(len(y))])

    # If desired calculate coverage for each class
    if coverage_on_label:
        sums = np.zeros(num_of_classes)
        size_sums = np.zeros(num_of_classes)
        lengths = np.zeros(num_of_classes)
        for i in range(len(y)):
            lengths[y[i]] = lengths[y[i]] + 1
            size_sums[y[i]] = size_sums[y[i]] + len(S[i])
            if y[i] in S[i]:
                sums[y[i]] = sums[y[i]] + 1
        coverage_given_y = sums/lengths
        lengths_given_y = size_sums/lengths

    # Conditional coverage not implemented
    wsc_coverage = None

    # Size and size conditional on coverage
    size = np.mean([len(S[i]) for i in range(len(y))])
    idx_cover = np.where([y[i] in S[i] for i in range(len(y))])[0]
    size_cover = np.mean([len(S[i]) for i in idx_cover])

    # Combine results
    out = pd.DataFrame({'Coverage': [marg_coverage], 'Conditional coverage': [wsc_coverage],
                        'Size': [size], 'Size cover': [size_cover]})

    # If desired, save coverage for each class
    if coverage_on_label:
        for i in range(num_of_classes):
            out['Coverage given '+str(i)] = coverage_given_y[i]
            out['Size given '+str(i)] = lengths_given_y[i]

    return out


# calculate accuracy of the smoothed classifier
def calculate_accuracy_smooth(model, x, y, noises, num_classes, k=1, device='cpu', GPU_CAPACITY=1024):
    # get size of the test set
    n = x.size()[0]

    # number of permutations to estimate mean
    n_smooth = noises.size()[0] // n

    # create container for the outputs
    smoothed_predictions = torch.zeros((n, num_classes))

    # calculate maximum batch size according to gpu capacity
    batch_size = GPU_CAPACITY // n_smooth

    # calculate number of batches
    if n % batch_size != 0:
        num_of_batches = (n // batch_size) + 1
    else:
        num_of_batches = (n // batch_size)

    # get predictions over all batches
    for j in range(num_of_batches):
        # get inputs and labels of batch
        inputs = x[(j * batch_size):((j + 1) * batch_size)]
        labels = y[(j * batch_size):((j + 1) * batch_size)]

        # duplicate batch according to the number of added noises and send to device
        # the first n_smooth samples will be duplicates of x[0] and etc.
        tmp = torch.zeros((len(labels) * n_smooth, *inputs.shape[1:]))
        x_tmp = inputs.repeat((1, n_smooth, 1, 1)).view(tmp.shape).to(device)

        # generate random Gaussian noise for the duplicated batch
        noise = noises[(j * (batch_size * n_smooth)):((j + 1) * (batch_size * n_smooth))].to(device)

        # add noise to points
        noisy_points = x_tmp + noise

        # get classifier predictions on noisy points
        model.eval()  # put in evaluation mode
        with torch.no_grad():
            noisy_outputs = model(noisy_points).to(torch.device('cpu'))

        # transform the output into probabilities vector
        noisy_outputs = softmax(noisy_outputs, dim=1)

        # get smoothed prediction for each point
        for m in range(len(labels)):
            smoothed_predictions[(j * batch_size) + m, :] = torch.mean(
                noisy_outputs[(m * n_smooth):((m + 1) * n_smooth)], dim=0)

    # transform results to numpy array
    smoothed_predictions = smoothed_predictions.numpy()

    # get label ranks to calculate top k accuracy
    label_ranks = np.array([rankdata(-smoothed_predictions[i, :], method='ordinal')[y[i]] - 1 for i in range(n)])

    # get probabilities of correct labels
    label_probs = np.array([smoothed_predictions[i, y[i]] for i in range(n)])

    # calculate accuracy
    top_k_accuracy = np.sum(label_ranks <= (k - 1)) / float(n)

    # calculate average inverse probability score
    score = np.mean(1 - label_probs)

    # calculate the 90 qunatiule
    quantile = mquantiles(1-label_probs, prob=0.9)
    return top_k_accuracy, score, quantile


def Smooth_Adv_ImageNet(model, x, y, indices, n_smooth, sigma_smooth, N_steps=20, max_norm=0.125, device='cpu', GPU_CAPACITY=1024, method='PGD'):
    # create attack model
    if method == 'PGD':
        attacker = PGD_L2(steps=N_steps, device=device, max_norm=max_norm)
    elif method == "DDN":
        attacker = DDN(steps=N_steps, device=device, max_norm=max_norm)

    # create container for the adversarial examples
    x_adv = torch.zeros_like(x)

    # get number of data points
    n = x.size()[0]

    # get dimension of data
    rows = x.size()[2]
    cols = x.size()[3]
    channels = x.size()[1]

    # number of permutations to estimate mean
    num_of_noise_vecs = n_smooth

    # calculate maximum batch size according to gpu capacity
    batch_size = GPU_CAPACITY // num_of_noise_vecs

    # calculate number of batches
    if n % batch_size != 0:
        num_of_batches = (n // batch_size) + 1
    else:
        num_of_batches = (n // batch_size)

    # start generating examples for each batch
    print("Generating Adverserial Examples:")

    image_index = -1
    for j in tqdm(range(num_of_batches)):
        #GPUtil.showUtilization()
        # get inputs and labels of batch
        inputs = x[(j * batch_size):((j + 1) * batch_size)]
        labels = y[(j * batch_size):((j + 1) * batch_size)]
        curr_batch_size = inputs.size()[0]

        # duplicate batch according to the number of added noises and send to device
        # the first num_of_noise_vecs samples will be duplicates of x[0] and etc.
        tmp = torch.zeros((len(labels) * num_of_noise_vecs, *inputs.shape[1:]))
        x_tmp = inputs.repeat((1, num_of_noise_vecs, 1, 1)).view(tmp.shape).to(device)

        # send labels to device
        y_tmp = labels.to(device).long()

        # generate random Gaussian noise for the duplicated batch
        noise = torch.empty((curr_batch_size * n_smooth, channels, rows, cols))
        # get relevant noises for this batch
        for k in range(curr_batch_size):
            image_index = image_index + 1
            torch.manual_seed(indices[image_index])
            noise[(k * n_smooth):((k + 1) * n_smooth)] = torch.randn(
                (n_smooth, channels, rows, cols)) * sigma_smooth


        #noise = noises[(j * (batch_size * num_of_noise_vecs)):((j + 1) * (batch_size * num_of_noise_vecs))].to(device)
        # noise = torch.randn_like(x_tmp, device=device) * sigma_adv

        noise = noise.to(device)
        # generate adversarial examples for the batch
        x_adv_batch = attacker.attack(model, x_tmp, y_tmp,
                                      noise=noise, num_noise_vectors=num_of_noise_vecs,
                                      no_grad=False,
                                      )

        # take only the one example for each point
        x_adv_batch = x_adv_batch[::num_of_noise_vecs]

        # move back to CPU
        x_adv_batch = x_adv_batch.to(torch.device('cpu'))

        # put in the container
        x_adv[(j * batch_size):((j + 1) * batch_size)] = x_adv_batch.detach().clone()

        del noise, tmp, x_adv_batch
        gc.collect()

    # return adversarial examples
    return x_adv


_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.
      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.means = torch.nn.Parameter(torch.tensor(means).to(device), requires_grad=False)
        self.sds = torch.nn.Parameter(torch.tensor(sds).to(device), requires_grad=False)

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)


def get_scores(model, x, indices, n_smooth, sigma_smooth, num_of_classes, scores_list, base=False,
               device='cpu', GPU_CAPACITY=1024, all_combinations=True, no_noise=False, verbal=True, 
               post_process=None, timer=None, beta=0.001):
    """
    
    Generate scores for given inputs.
    :param list indices: list of seed used for each batch.
    
    """
    n, channels, rows, cols = x.size()

    # create container for the scores
    if base:
        scores_simple = np.zeros((len(scores_list), n, num_of_classes))
    else:
        smoothed_scores = np.zeros((len(scores_list), n, num_of_classes))
        bernstein_bounds = np.zeros((len(scores_list), n, num_of_classes))

    # calculate maximum batch size according to gpu capacity
    batch_size = GPU_CAPACITY // n_smooth

    # calculate number of batches
    if n % batch_size != 0:
        num_of_batches = (n // batch_size) + 1
    else:
        num_of_batches = (n // batch_size)

    # initiate random uniform variables for inverse quantile score
    rng = default_rng()
    uniform_variables = rng.uniform(size=n, low=0.0, high=1.0)

    # create container for smoothed and base classifier outputs
    if base:
        simple_outputs = np.zeros((n, num_of_classes))
    else:
        smooth_outputs = np.zeros((n, num_of_classes))
    image_index = -1
    print("Evaluate predictions:")
    for j in (tqdm(range(num_of_batches)) if verbal else range(num_of_batches)):
        # get inputs of batch
        inputs = x[(j * batch_size):((j + 1) * batch_size)].to(device)
        curr_batch_size = inputs.size()[0]

        if base:
            if not no_noise:
                noises_test_base = torch.empty((curr_batch_size, channels, rows, cols), device=device)
                # get relevant noises for this batch
                for k in range(curr_batch_size):
                    image_index = image_index + 1
                    torch.manual_seed(indices[image_index])
                    noises_test_base[k:(k + 1)] = torch.randn((1, channels, rows, cols), device=device) * sigma_smooth

                noisy_points = inputs.to(device) + noises_test_base
            else:
                noisy_points = inputs.to(device)
        else:
            noises_test = torch.empty((curr_batch_size * n_smooth, channels, rows, cols), device=device)
            # get relevant noises for this batch
            for k in range(curr_batch_size):
                image_index = image_index + 1
                torch.manual_seed(indices[image_index])
                noises_test[(k * n_smooth):(k + 1) * n_smooth] = torch.randn(
                    (n_smooth, channels, rows, cols), device=device) * sigma_smooth
                    
            # duplicate batch according to the number of added noises and send to device
            # the first n_smooth samples will be duplicates of x[0] and etc.
            tmp = torch.zeros((inputs.size()[0] * n_smooth, *inputs.shape[1:]))
            x_tmp = inputs.repeat((1, n_smooth, 1, 1)).view(tmp.shape)

            # add noise to points
            noisy_points = x_tmp + noises_test.to(device)

        # get classifier predictions on noisy points
        model.eval()  # put in evaluation mode
        if timer:
            base_start = time.time()
        with torch.no_grad():
            noisy_outputs = model(noisy_points).to(torch.device('cpu'))
        if timer:
            timer['base'].update(time.time() - base_start)
        # transform the output into probabilities vector
        noisy_outputs = softmax(noisy_outputs, dim=1).numpy()

        if base:
            simple_outputs[(j * batch_size):((j + 1) * batch_size), :] = noisy_outputs
        else:
            # get smoothed scores for each for all points in batch
            batch_uniform = uniform_variables[(j * batch_size):((j + 1) * batch_size)]
            batch_uniform = np.repeat(batch_uniform, n_smooth)
            for p, score_func in enumerate(scores_list):
                if timer:
                    score_start = time.time()
                # get scores for all noisy outputs for all classes
                noisy_scores = score_func(noisy_outputs, np.arange(num_of_classes), batch_uniform, all_combinations=all_combinations)
                # average n_smooth scores for eac points
                smoothed_scores[p, (j * batch_size):((j + 1) * batch_size)] = noisy_scores.reshape(-1, n_smooth, noisy_scores.shape[1]).mean(axis=1)
                bernstein_bounds[p, (j * batch_size):((j + 1) * batch_size)] = calculate_Bern_bound(noisy_scores.reshape(-1, n_smooth, noisy_scores.shape[1]), beta=beta)
                if timer:
                    timer['scores'][p].update(time.time() - score_start)
            # clean
            del batch_uniform, noisy_scores
            gc.collect()

        if base:
            del noisy_points, noisy_outputs
            if not no_noise: del noises_test_base
        else:
            del noisy_points, noisy_outputs, noises_test, tmp
        gc.collect()

    # run over all scores functions and compute scores of smoothed and base classifier
    for p, score_func in enumerate(scores_list):
        if base:
            scores_simple[p, :, :] = score_func(simple_outputs, np.arange(num_of_classes), uniform_variables, all_combinations=all_combinations)

    # return relevant scores
    if base:
        return scores_simple
    else:
        return smoothed_scores, bernstein_bounds

def calibration(scores_simple=None, smoothed_scores=None, alpha=0.1, num_of_scores=2, correction=0, base=False, n_smooth=256):
    # size of the calibration set
    if base:
        n_calib = scores_simple.shape[1]
    else:
        n_calib = smoothed_scores.shape[1]

    # create container for the calibration thresholds
    thresholds = np.zeros((num_of_scores, 2))

    # Compute thresholds
    level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n_calib))
    for p in range(num_of_scores):
        if base:
            thresholds[p, 0] = mquantiles(scores_simple[p, :], prob=level_adjusted)
        else:
            thresholds[p, 1] = mquantiles(smoothed_scores[p, :], prob=level_adjusted)
    return thresholds

def prediction(scores_simple=None, smoothed_scores=None, num_of_scores=2, thresholds=None, 
               threshold_RSCP_plus=None, correction=0, base=False, bounds_bern=None, 
               bound_hoef=None, record_no_correction=False):
    # get number of points
    if base:
        n = scores_simple.shape[1]
    else:
        n = smoothed_scores.shape[1]
    # Generate prediction sets using the thresholds from the calibration
    predicted_sets = []
    for p in range(num_of_scores):
        if base:
            S_hat_simple = [np.where(scores_simple[p, i, :] <= thresholds[p, 0])[0] for i in range(n)]
            predicted_sets.append(S_hat_simple)
        else:
            smoothed_S_hat = [np.where(norm.ppf(smoothed_scores[p, i, :], loc=0, scale=1) <= norm.ppf(thresholds[p, 1], loc=0, scale=1))[0] for i in range(n)]
            smoothed_S_hat_corrected = [np.where(norm.ppf(smoothed_scores[p, i, :], loc=0, scale=1) - correction <= norm.ppf(thresholds[p, 1], loc=0, scale=1))[0] for i in range(n)]
            if bounds_bern is not None and bound_hoef is not None:
                smoothed_S_hat_corrected_MC = [np.where(norm.ppf(np.maximum(smoothed_scores[p, i, :] - bounds_bern[p, i, :], 0), loc=0, scale=1) - correction <= norm.ppf(np.minimum(threshold_RSCP_plus[p, 1] + bound_hoef, 1), loc=0, scale=1))[0] for i in range(n)]
                smoothed_S_hat_corrected_MC_Hoef = [np.where(norm.ppf(np.maximum(smoothed_scores[p, i, :] - bound_hoef, 0), loc=0, scale=1) - correction <= norm.ppf(np.minimum(threshold_RSCP_plus[p, 1] + bound_hoef, 1), loc=0, scale=1))[0] for i in range(n)]
            else:
                smoothed_S_hat_corrected_MC, smoothed_S_hat_corrected_MC_Hoef = None, None
            tmp_list = [smoothed_S_hat_corrected, smoothed_S_hat_corrected_MC]
            if record_no_correction: tmp_list += [smoothed_S_hat, smoothed_S_hat_corrected_MC_Hoef]
            predicted_sets.append(tmp_list)

    # return predictions sets
    return predicted_sets


class running_mean(object):
    def __init__(self):
        self.counter = 0
        self.sum = 0.0

    def update(self, new_value, count=1):
        self.sum += new_value
        self.counter += count
    
    def get_mean(self):
        return self.sum / self.counter


def get_base_model(checkpoint, dataset, data_parallel=False, no_normalization=False, extending=False):
    # Get architecture
    is_MACER = 'net' in checkpoint # MACER model is different from others
    weights_key = 'net' if is_MACER else 'state_dict'
    if dataset == 'CIFAR10':
        norm_layer, backbone = get_architecture('cifar_resnet110', "cifar10")
    elif dataset == 'CIFAR100':
        backbone = ResNet(depth=110, num_classes=100)
        norm_layer = get_normalize_layer("cifar10")
    elif dataset == 'FMNIST':
        in_features = 28 * 28
        out_features = 10
        backbone = MLP([64] * 2, in_features, out_features)
        norm_layer = None
    elif dataset in ['ImageNet', 'imagenet', 'tinyImageNet']:
        if dataset == "tinyImageNet":
            backbone = resnet50(pretrained=False, num_classes=200) 
        else:
            backbone = resnet50(pretrained=False)
        norm_layer = get_normalize_layer("imagenet")

    if is_MACER or no_normalization: norm_layer = None
    # Extending model
    if extending:
        backbone.fc = MLP([128], backbone.fc.in_features, backbone.fc.out_features)
    # Remove weights prefix
    weights = checkpoint[weights_key]
    weights = {(key[2:] if key[:2] == '1.' else key):v for key, v in weights.items()}
    if dataset in ['ImageNet', 'imagenet', 'tinyImageNet']: weights = {(key[7:] if key.startswith("module.") else key):v for key, v in weights.items()} # Remove "module." prefix
    backbone.load_state_dict(weights)
    # Adding data_parallel
    if data_parallel and not isinstance(backbone, DataParallel):
        backbone = DataParallel(backbone)
    model = backbone if norm_layer is None else Sequential(norm_layer, backbone) # No normalization for MACER model
    return model, is_MACER


class Sigmoid_module(torch.nn.Module):
    def __init__(self, temp=1, bias=0.5):
        super(Sigmoid_module, self).__init__()
        self.temp = torch.nn.Parameter(torch.Tensor([temp]))
        self.bias = torch.nn.Parameter(torch.Tensor([bias]))
    
    def forward(self, x, numpy=False):
        if not numpy:
            x = (x - self.bias) / torch.exp(self.temp)
            x = torch.nn.Sigmoid()(x)
        else:
            bias, temp = self.bias.cpu().detach().numpy(), self.temp.cpu().detach().numpy()
            x = (x - bias) / np.exp(temp)
            x = 1 / (1 + np.exp(-x))
        return x


def draw_icdf(scores, methods, labels, savename='cdf.jpg', smoothed=False, cacheit=False):
    for i, method in enumerate(methods):
        score = scores[i]
        score = score[np.arange(len(labels)), labels]
        if cacheit:
            np.save("%s_%s.npy" % (method, savename), score)
        if smoothed:
            score = norm.ppf(score)
        plt.plot(np.linspace(0, 1, len(score)), np.sort(score))
        plt.xlim(0.0, 1)
        plt.savefig("img/%s_%s" % (method, savename))
        plt.close('all')


class Model_register():
    def __init__(self):
        pass

    def record(self, model):
        self.state_dict = model.state_dict()

    def compare(self, model):
        target_states = model.state_dict()
        changed_params = {}
        for name, param in self.state_dict.items():
            diff = torch.norm(param.double() - target_states[name].double())
            if diff > 1e-8:
                changed_params[name] = diff
        print("Model diff:")
        if changed_params:
            for name, diff in changed_params.items():
                print("Diff in parameter %s: %e" % (name, diff))
        else:
            print("None")


def validate_arguments(args):
    # Validate parameters
    assert 0 <= args.alpha <= 1, 'Nominal level must be between 0 to 1'
    assert not(args.n_s & (args.n_s - 1)), 'n_s must be a power of 2.'
    assert not(args.batch_size & (args.batch_size - 1)), 'batch size must be a power of 2.'
    assert args.batch_size >= args.n_s, 'batch size must be larger than n_s'
    assert args.arc == 'ResNet' or args.arc == 'DenseNet' or args.arc == 'VGG', 'Architecture can only be Resnet, ' \
                                                                                    'VGG or DenseNet '
    assert args.sigma_model >= 0, 'std for training the model must be a non negative number.'
    assert args.delta >= 0, 'L2 bound of noise must be non negative.'
    assert isinstance(args.splits, int) and args.splits >= 1, 'number of splits must be a positive integer.'
    assert args.ratio >= 0, 'ratio between sigma and delta must be non negative.'


def load_datasets(dataset_name):
    """
    Normalization is skipped for adversarial attacks.
    """
    if dataset_name == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(root='./datasets/',
                                                    train=True,
                                                    transform=torchvision.transforms.ToTensor(),
                                                    download=True)
        test_dataset = torchvision.datasets.CIFAR10(root='./datasets',
                                                    train=False,
                                                    transform=torchvision.transforms.ToTensor())
    elif dataset_name == "FMNIST":
        train_dataset = torchvision.datasets.FashionMNIST(root='./datasets/',
                                                    train=True,
                                                    transform=torchvision.transforms.ToTensor(),
                                                    download=True)
        test_dataset = torchvision.datasets.FashionMNIST(root='./datasets',
                                                    train=False,
                                                    transform=torchvision.transforms.ToTensor())

    elif dataset_name == "CIFAR100":
        train_dataset = torchvision.datasets.CIFAR100(root='./datasets/',
                                                    train=True,
                                                    transform=torchvision.transforms.ToTensor(),
                                                    download=True)
        test_dataset = torchvision.datasets.CIFAR100(root='./datasets',
                                                    train=False,
                                                    transform=torchvision.transforms.ToTensor())
    elif dataset_name == "ImageNet":
        base_dir = os.environ['DATASET_ROOT_DIR']
        imagenet_dir = os.path.join(base_dir, "imagenet/val")
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

        # load dataset
        train_dataset = None
        test_dataset = torchvision.datasets.ImageFolder(imagenet_dir, transform)

    elif dataset_name == "tinyImageNet":
        base_dir = os.environ['DATASET_ROOT_DIR']
        imagenet_dir = os.path.join(base_dir, "tinyimagenet", "test")
        transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
        train_dataset = None
        test_dataset = torchvision.datasets.ImageFolder(imagenet_dir, transform)
    else:
        print("No such dataset")
        exit(1)
    return train_dataset, test_dataset


def calculate_Bern_bound(scores, beta=0.001):
    # scores.shape = (n_scores, n_gauss_examples, n_inputs, n_classes)
    ns = scores.shape[1]
    square_sum = scores.sum(axis=1) ** 2
    sum_square = (scores ** 2).sum(axis=1)
    sample_variance = (sum_square - square_sum / ns) / (ns - 1)
    sample_variance[sample_variance<0] = 0 # computation stability
    t = np.log(2 / beta)
    bound = np.sqrt(2 * sample_variance * t / ns) + (7 / 3) * t  / (ns - 1)
    return bound


def get_adv_path(args):
    """
    Compute the path to save adversarial examples.
    """
    directory = "./Adversarial_Examples/" + str(args.dataset) + "/epsilon_" + str(args.delta) + "/sigma_model_" + str(
        args.sigma_model) + "/sigma_smooth_" + str(args.sigma_smooth) + "/n_smooth_" + str(args.n_s)

    # normalization layer to my model
    if args.is_RSCP:
        directory = directory + "/Robust"

    # different attacks for different architectures
    if args.arc != 'ResNet':
        directory = directory + "/" + str(args.model_type)
    if args.dataset == "CIFAR10":
        if args.is_RSCP:
            directory = directory + "/My_Model"
        else:
            directory = directory + "/Their_Model"
            if args.model_type == "Salman" or args.model_type == "MACER":
                directory = directory + "/" + args.model_type
    if args.use_conftr:
        directory += '/conftr' if not args.extend else '/conftr-extend'
    return directory


class ImageNetSubset(torchvision.datasets.ImageFolder):
    def __init__(self, root: str,
                 transform= None,
                 target_transform= None,
                 is_valid_file = None,
                 subset_file_path = None):
        with open(subset_file_path, 'r') as f:
            self.classes = sorted(list(f.read().split("\n")))
        classes, class2idx = super().find_classes(root)
        self.class_original_idx = [class2idx[classname] for classname in self.classes]
        super().__init__(root, transform=transform,
                         target_transform=target_transform,
                         is_valid_file=is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        return self.classes, class_to_idx

    def get_fc_mask(self):
        return ClassMask(self.class_original_idx)


class ClassMask(Module):
    def __init__(self, index) -> None:
        super().__init__()
        self.index = index
    
    def forward(self, x):
        return x[..., self.index]
    

class logger():
    def __init__(self) -> None:
        self.current_epoch = 0
        self.epoch_logs = [defaultdict(list)]
    
    def next_epoch(self) -> None:
        self.current_epoch += 1
        self.epoch_logs.append(defaultdict(list))
    
    def log(self, name, metric) -> None:
        current_log = self.epoch_logs[self.current_epoch]
        current_log[name].append(metric)
    
    def get_current_log(self) -> Dict:
        return self.epoch_logs[self.current_epoch]


def get_saving_dir(args, require_normalization):
    directory = "./Results/" + str(args.dataset) + "/epsilon_" + str(args.delta) + "/sigma_model_" + str(
        args.sigma_model) + "/sigma_smooth_" + str(args.sigma_smooth) + "/n_smooth_" + str(args.n_s)

    if require_normalization:
        directory = directory + "/Robust"


    if args.dataset == "CIFAR10":
        if args.is_RSCP:
            directory = directory + "/My_Model"
        else:
            directory = directory + "/Their_Model"
            if args.model_type == "Salman" or "MACER":
                directory = directory + "/" + args.model_type
    if args.use_conftr:
        directory += "/conftr"

    if args.alpha != 0.1:
        directory = directory + "/alpha_" + str(args.alpha)
    directory += "/" + args.exp
    print("Saving results in: " + str(directory))
    return directory