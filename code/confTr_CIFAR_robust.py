"""

Robust ConfTr on CIFAR-10 dataset.

"""
import torch
import torch.nn as nn
import torch.optim as optim


import torchvision
import torchvision.transforms as transforms

import os
import argparse
from tqdm import tqdm
from Architectures.MLP import MLP
from utils import calibration, draw_icdf, evaluate_predictions, get_base_model, get_scores, prediction, running_mean
import numpy as np
import Score_Functions as scores
import confTr
from sklearn.model_selection import train_test_split
from utils import logger


def train(epoch, model, trainloader, optimizer, device, args, post_processing=None, logger=logger()):
    print('Epoch: %d' % epoch)
    train_loss = running_mean()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        loss_mat = torch.eye(n_classes)
        loss_mat = loss_mat.to(device)
        correction = 1 / args.ratio if epoch >= args.warmup else 0
        if not args.disable_robust:
            loss = confTr.compute_robust_loss(model, inputs, targets, 
                args.temp, args.alpha, args.n_train, args.eps * args.ratio, correction, dispersion=args.dispersion,
                n_classes=n_classes, score_post_process=post_processing,
                size_weight=args.size_weight, target_size=args.target_size, loss_matrix=loss_mat,
                tau_correction=args.correction, logger=logger)
        else:
            loss = confTr.compute_loss_and_error(model, inputs, targets, True, args.temp,
                args.alpha, n_classes=n_classes, size_weight=args.size_weight, target_size=args.target_size, loss_matrix=loss_mat)
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item())
    print('Training loss: %.3f' % train_loss.get_mean())
    return train_loss.get_mean()

def test(n_experiments, model, n_smooth, x_test, y_test, device, post_processing=None):
    # score evaluation
    model.eval()
    evaluation_scores_list = ['APS', 'HPS']
    implemented_scores = {'HPS': scores.class_probability_score, 
                      'APS': scores.generalized_inverse_quantile_score, 
                      'RAPS': scores.rank_regularized_score}
    scores_list = []
    for score in evaluation_scores_list:
        if score in implemented_scores:
            score_func = implemented_scores[score]
            if post_processing is not None:
                def post_process(func):
                    def new_func(*args, **kwargs):
                        return post_processing(func(*args, **kwargs), numpy=True)
                    return new_func
                score_func = post_process(score_func)
            scores_list.append(score_func)
    indices = np.arange(x_test.shape[0])
    scores_simple_clean_test = get_scores(model, x_test, indices, 1, 0, n_classes, scores_list, base=True, device=device, GPU_CAPACITY=1024, no_noise=True, verbal=False)
    smoothed_scores_clean_test, scores_smoothed_clean_test = get_scores(model, x_test, indices, n_smooth, args.eps * args.ratio, n_classes, scores_list, base=False, device=device, GPU_CAPACITY=1024)
    draw_icdf(smoothed_scores_clean_test, evaluation_scores_list, y_test, savename="smooth_cdf.jpg")
    avg_size, avg_coverage = {}, {}
    for _ in range(n_experiments):
        test_size = 2 / 3
        idx1, idx2 = train_test_split(indices, test_size=test_size)
        thresholds_base = calibration(scores_simple=scores_simple_clean_test[:, idx1, y_test[idx1]], alpha=args.alpha, num_of_scores=len(scores_list), correction=0, base=True)
        thresholds = calibration(smoothed_scores=smoothed_scores_clean_test[:, idx1, y_test[idx1]], alpha=args.alpha, num_of_scores=len(scores_list), correction=1/args.ratio, base=False)
        thresholds = thresholds + thresholds_base
        predicted_clean_sets_base = prediction(scores_simple=scores_simple_clean_test[:, idx2, :], num_of_scores=len(scores_list), thresholds=thresholds_base, base=True)
        predicted_clean_sets = prediction(smoothed_scores=smoothed_scores_clean_test[:, idx2, :], num_of_scores=len(scores_list),
            thresholds=thresholds, correction=1/args.ratio, base=False)

        for p in range(len(scores_list)):
            score_name = evaluation_scores_list[p]
            methods_list = [score_name + '_smoothed_score_corrected']
            for r, method in enumerate(methods_list):
                res = evaluate_predictions(predicted_clean_sets[p][r], None, y_test[idx2].numpy(),
                                            conditional=False)
                if not method in avg_coverage:
                    avg_coverage[method] = running_mean()
                    avg_size[method] = running_mean()
                avg_coverage[method].update(res['Coverage'])
                avg_size[method].update(res['Size'])
    for method in avg_coverage:
        print("Method: %s" % method)
        print("Average coverage %.3f size %.3f" % (avg_coverage[method].get_mean(), avg_size[method].get_mean()))
    return {method: avg.get_mean() for method, avg in avg_size.items()}


def acc_test(testloader, model, device, sigma=0.25):
    acc = running_mean()
    model.eval()
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            predictions = model(x + torch.randn_like(x) * sigma).argmax(axis=1)
            acc.update(torch.sum(predictions == y), x.shape[0])
    print("Accuracy: %f" % acc.get_mean())
    return acc.get_mean()


def train_robust_CP(args):
    # Set seed and parameters
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if args.device else 'cuda'
    device = torch.device(device)
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    exp_logger = logger()
    # Load dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    if args.dataset == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(
            root='./datasets', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=args.random_shuffle, num_workers=2)
        testset = torchvision.datasets.CIFAR10(
            root='./datasets', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=10000, shuffle=False, num_workers=2)
    elif args.dataset == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(
            root='./datasets', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=args.random_shuffle, num_workers=2)
        testset = torchvision.datasets.CIFAR100(
            root='./datasets', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=10000, shuffle=False, num_workers=2)
    for x, y in testloader:
        x_test, y_test = x, y

    # Model
    print('==> Load pretrained model')
    assert os.path.isdir('checkpoints'), 'Error: no checkpoint directory found!'
    if args.dataset == 'CIFAR10':
        load_path = args.model_path + ('/noise_%.2f/' % (args.ratio * args.eps))
        checkpoint = torch.load(load_path + 'checkpoint.pth.tar')
    elif args.dataset == 'CIFAR100':
        load_path = args.model_path
        weights_name = 'ResNet110_Robust_sigma_' + str(args.ratio * args.eps) + '.pth.tar'
        checkpoint = torch.load(os.path.join(load_path, weights_name))
    model, is_MACER = get_base_model(checkpoint, args.dataset, data_parallel=use_DataParallel)
    model.to(device)
    print('==> Test baseline model')
    # Sanity test
    sanity_test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    if not args.notest:
        acc_test(sanity_test_loader, model, device, sigma=args.eps*args.ratio)
        acc_test(trainloader, model, device, sigma=args.eps*args.ratio)
    # Reinitialize last layer
    backbone = model if is_MACER else model[1]
    if use_DataParallel: backbone = backbone.module
    if not args.no_reinit:
        if not args.extending:
            backbone.fc = nn.Linear(backbone.fc.in_features, backbone.fc.out_features)
        else:
            backbone.fc = MLP([128], backbone.fc.in_features, backbone.fc.out_features)
    training_parameters = backbone.fc.parameters()
    # Freeze the model
    for param in model.parameters():
        param.requires_grad = False
    # Finetuning last layer
    for param in backbone.fc.parameters():
        param.requires_grad = True

    # Add post processing module
    model.to(device)
    post_processing = None
    # Build optimizer
    if args.adam:
        optimizer = optim.Adam(training_parameters, lr=args.lr)
    else:
        optimizer = optim.SGD(training_parameters, lr=args.lr,
            nesterov=True, weight_decay=5e-4, momentum=0.9)
    # Build Scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
        milestones=[int(2 * args.max_epochs / 5), int(3 * args.max_epochs / 5), int(4 * args.max_epochs / 5)])
    # Train model
    for epoch in tqdm(range(start_epoch, args.max_epochs)):
        if epoch % 10 == 0 and not args.notest:
            result = test(5, model, 8, x_test, y_test, device, post_processing=post_processing)
        avg_loss = train(epoch, model, trainloader, optimizer, device, args, post_processing=post_processing, logger=exp_logger)
        acc = acc_test(sanity_test_loader, model, device, sigma=args.eps*args.ratio)
        if post_processing:
            print(post_processing.temp.item(), post_processing.bias.item())
        if not args.no_scheduler:
            scheduler.step()
        exp_logger.next_epoch()
    # Save model
    model_savepath = os.path.join(load_path, args.saving_name)
    torch.save({'state_dict': backbone.state_dict()}, model_savepath)
    # Test after training
    acc_test(sanity_test_loader, model, device, sigma=args.eps*args.ratio)
    results = test(10 if args.finetune else 50, model, 32 if args.finetune else 256, x_test, y_test, device,
        post_processing=post_processing)


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 ConfTraining')
parser.add_argument('--saving_name', type=str, default='Robust_confTr.pth')
parser.add_argument('--model_path', type=str, default='./Pretrained_Models/Cohen/cifar10/resnet110')
parser.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100'], default='CIFAR10')

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for training')
parser.add_argument('--max_epochs', type=int, default=150, help='Max number of epochs to train')
parser.add_argument('--target_size', type=int, default=0, help='Target size')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
parser.add_argument('--temp', type=float, default=0.1, help='Soft thresholding temperature')
parser.add_argument('--dispersion', type=float, default=0.1, help='Regularization strength of soft operation')
parser.add_argument('--size_weight', type=float, default=1, help='Size loss weight')

parser.add_argument('--no_reinit', action='store_true', help='Disable re-initialization')
parser.add_argument('--no_scheduler', action='store_true', help='Disable scheduler')
parser.add_argument('--random_shuffle', action='store_true', help='Random shuffle validation split when training')
parser.add_argument('--adam', action='store_true', help='Use adam')
parser.add_argument('--disable_robust', action='store_true', help='Disable robust training')
parser.add_argument('--extending', action='store_true', help='Extend Resnet')

parser.add_argument('--eps', type=float, default=0.125, help='Scale of adversarial noise')
parser.add_argument('--ratio', type=float, default=2, help='Ratio of smoothing lever and noise level')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
parser.add_argument('--correction', type=float, default=0, help='Correction level')
parser.add_argument('--n-train', type=int, default=4, help='Number of smoothing samples in training')
parser.add_argument('--n-smooth', type=int, default=8, help='Number of smoothing samples in test')
parser.add_argument('--warmup', type=int, default=0, help='Number of warmup epochs')
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--device', type=str, default=None)
parser.add_argument('--project_name', type=str, default='RobustConformalTraining')
args = parser.parse_args()


if args.dataset == 'CIFAR10': n_classes = 10
elif args.dataset == 'CIFAR100': n_classes = 100
use_DataParallel = args.device is None # Use all GPUs when device is not specified
args.notest = args.finetune or args.debug
train_robust_CP(args)
