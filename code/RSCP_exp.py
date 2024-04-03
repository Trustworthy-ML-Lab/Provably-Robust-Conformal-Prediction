# general imports
import gc
import numpy as np
from collections import OrderedDict
from tqdm.auto import tqdm
import random
import torch
import os
import pickle
import sys
import argparse
from sklearn.model_selection import train_test_split
import pandas as pd
import Score_Functions as scores
from utils import draw_icdf, evaluate_predictions,calculate_accuracy_smooth, get_base_model, running_mean, \
     Smooth_Adv_ImageNet, get_scores, calibration, prediction

# My imports
sys.path.insert(0, './')
import utils
# Use comet to log experiments


# parameters
parser = argparse.ArgumentParser(description='Experiments')
parser.add_argument('-a', '--alpha', default=0.1, type=float, help='Desired nominal marginal coverage')
parser.add_argument('-d', '--delta', default=0.125, type=float, help='L2 bound on the adversarial noise')
parser.add_argument('-s', '--splits', default=50, type=int, help='Number of experiments to estimate coverage')
parser.add_argument('-r', '--ratio', default=2, type=float,
                    help='Ratio between adversarial noise bound to smoothing noise')
parser.add_argument('--n-s', default=256, type=int, help='Number of samples used for estimating smoothed score')
parser.add_argument('--seed', default=0, type=int, help='Random seed for experiments')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='Dataset to be used: CIFAR100, CIFAR10, ImageNet')
parser.add_argument('--exp', default='exp', type=str, help='Results save directory.')
parser.add_argument('--arc', default='ResNet', type=str,
                    help='Architecture of classifier : ResNet, DenseNet, VGG. Relevant only of My_model=True')
parser.add_argument('--regenerate-adv', action='store_true', help='Regenerate Adv examples')
parser.add_argument('--batch-size', default=1024, type=int, help='Number of images to send to gpu at once')
parser.add_argument('--sigma-model', default=-1, type=float, help='std of Gaussian noise the model was trained with')
parser.add_argument('--model-type', choices=['RSCP', 'Salman', 'Cohen', 'MACER', 'my_model'], help='Choose what model to use')
parser.add_argument('--coverage-on-label', action='store_true', help='True for getting coverage and size for each label')
# Arguments for RAPS
parser.add_argument('--k-reg', type=int, default=2)
parser.add_argument('--lamda', type=float, default=0.2)
# Arguments for quantile scores
parser.add_argument('--n-quantile', type=int, default=500)
parser.add_argument('--T', type=int, default=400)
parser.add_argument('--bias', type=float, default=0.9)
parser.add_argument('--use-conftr', action='store_true', help="Use RCT model as base model")
parser.add_argument('--extend', action='store_true')
parser.add_argument('--timing', action='store_true')
parser.add_argument('--plot-icdf', action='store_true', help='Plot icdf for clean smoothed score')
# Arguments for RSCP+
parser.add_argument("--beta", type=float, default=0.001, help="1-beta denotes confidence level")
parser.add_argument("--cleanonly", action='store_true', help="Only calculate clean examples")
# Arguments for experiments settings
parser.add_argument("--nocorr", action='store_true', help="Report result without correction in RSCP")
parser.add_argument("--conftr_weight_name", type=str, default="Robust_confTr.pth")
args = parser.parse_args()
# parameters
args.sigma_smooth = args.ratio * args.delta # sigma used fro smoothing
# sigma used for training the model
if args.sigma_model != -1:
    args.sigma_model = args.sigma_model
else:
    args.sigma_model = args.sigma_smooth
args.n_s = args.n_s  # number of samples used for smoothing
args.is_RSCP = (args.model_type == 'RSCP')
args.N_steps = 20  # number of gradiant steps for PGD attack
args.calibration_scores = ['APS', 'HPS', 'PTT_APS', 'PTT_HPS'] # score function to check
args.coverage_on_label = args.coverage_on_label # Whether to calculate coverage and size per class
args.max_plot_size = {'ImageNet': 50, 'CIFAR10': 3, 'CIFAR100':25, 'FMNIST': 3, 'tinyImageNet': 20} # max y-axis range for plot

utils.validate_arguments(args)


require_normalization = args.is_RSCP
GPU_CAPACITY = args.batch_size
GPU_CAPACITY_TRAIN = GPU_CAPACITY // 8

# calculate correction based on the Lipschitz constant
if args.sigma_smooth == 0:
    correction = 10000
else:
    correction = float(args.delta) / float(args.sigma_smooth)

# set random seed
seed = args.seed
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_dataset, test_dataset = utils.load_datasets(args.dataset)
args.n_test = len(test_dataset)

# Save test set in memory
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.n_test//8,
                                          shuffle=False, num_workers=8)
x_list, y_list = [], []
for x_test, y_test in test_loader:
    x_list.append(x_test)
    y_list.append(y_test)
x_test, y_test = torch.cat(x_list), torch.cat(y_list)

# get dimension of data
args.n_test, channels, rows, cols = x_test.shape
if args.dataset == 'ImageNet':
    num_of_classes = 1000
elif args.dataset == 'CIFAR100':
    num_of_classes = 100
else:
    num_of_classes = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

# load RSCP models
# Dict[model_type, dataset]: checkpoint_path
model_checkpoints = {
    ("RSCP", "CIFAR10"): ('./checkpoints', 'CIFAR10_ResNet110_Robust_sigma_%.2f.pth.tar' % args.sigma_model), 
    ("RSCP", "CIFAR100"): ('./checkpoints', 'ResNet110_Robust_sigma_' + str(args.sigma_model) + '.pth.tar'), 
    ("Salman", "CIFAR10"): ('./Pretrained_Models/Salman/cifar10/noise_'+str(args.sigma_model) ,'checkpoint.pth.tar'),
    ("MACER", "CIFAR10"): ('./Pretrained_Models/MACER/cifar10/noise_'+str(args.sigma_model) ,'checkpoint.pth'),
    ("Cohen", "CIFAR10"): ('./Pretrained_Models/Cohen/cifar10/resnet110/noise_%.2f' % args.sigma_model ,'checkpoint.pth.tar'),
}
assert (args.model_type, args.dataset) in model_checkpoints, "Model type and dataset combination not supported"
model_directory, weights_name = model_checkpoints[args.model_type, args.dataset]
if args.use_conftr: # Use weights after RCT
    weights_name = args.conftr_weight_name

state = torch.load(os.path.join(model_directory, weights_name), map_location=device)
model, is_MACER = get_base_model(state, args.dataset, data_parallel=True, no_normalization=(args.model_type=="MACER"),
    extending=args.extend)

model.eval()
model.to(device)
# create indices for the test points
indices = torch.arange(args.n_test)


# Generate adversarial example path
adv_directory = utils.get_adv_path(args)
if not args.cleanonly:
    print("Searching for adversarial examples in: " + str(adv_directory))
    if os.path.exists(adv_directory):
        print("Are there saved adversarial examples: Yes")
    else:
        print("Are there saved adversarial examples: No")
    # If there are no pre created adversarial examples, create new ones
    if args.regenerate_adv or not os.path.exists(os.path.join(adv_directory, 'data.pickle')):
        # Generate adversarial test examples
        print("Generate adversarial test examples for the smoothed model:\n")
        x_test_adv = Smooth_Adv_ImageNet(model, x_test, y_test, indices, args.n_s, args.sigma_smooth, args.N_steps, args.delta, device, GPU_CAPACITY=GPU_CAPACITY_TRAIN)

        # Generate adversarial test examples for the base classifier
        print("Generate adversarial test examples for the base model:\n")
        x_test_adv_base = Smooth_Adv_ImageNet(model, x_test, y_test, indices, 1, args.sigma_model, args.N_steps, args.delta, device,
                                    GPU_CAPACITY=GPU_CAPACITY_TRAIN)

        if not os.path.exists(adv_directory):
            os.makedirs(adv_directory)
        with open(adv_directory + "/data.pickle", 'wb') as f:
            pickle.dump([x_test_adv, x_test_adv_base], f)
    # If there are pre created adversarial examples, load them
    else:
        with open(adv_directory + "/data.pickle", 'rb') as f:
            x_test_adv, x_test_adv_base = pickle.load(f)

# Create holdout set
torch.manual_seed(args.seed)
indices = torch.arange(args.n_test)
test_ratio = args.n_quantile / args.n_test
indices_test, indices_valid = train_test_split(indices, test_size=test_ratio, shuffle=False)
x_test, x_valid = x_test[indices_test], x_test[indices_valid]
y_test, y_valid = y_test[indices_test], y_test[indices_valid]
gc.collect()
if not args.cleanonly:
    x_test_adv, x_valid_adv =  x_test_adv[indices_test], x_test_adv[indices_valid]
    gc.collect()
    x_test_adv_base, x_valid_adv_base = x_test_adv_base[indices_test], x_test_adv_base[indices_valid]
    gc.collect()
args.n_quantile = x_valid.shape[0]

args.n_test = x_test.shape[0]
# Recreate indices for the test points
indices = torch.arange(args.n_test)
# create the noises for the base classifiers only to check its accuracy
noises_base = torch.empty_like(x_test)
for k in range(args.n_test):
    torch.manual_seed(k)
    noises_base[k:(k + 1)] = torch.randn(
        (1, channels, rows, cols)) * args.sigma_model
# Calculate accuracy of classifier on clean test points
acc, _, _ = calculate_accuracy_smooth(model, x_test, y_test, noises_base, num_of_classes, k=1, device=device, GPU_CAPACITY=GPU_CAPACITY)
print("True Model accuracy :" + str(acc * 100) + "%")
gc.collect()

if not args.cleanonly:
    # Calculate accuracy of classifier on adversarial test points
    acc, _, _ = calculate_accuracy_smooth(model, x_test_adv_base, y_test, noises_base, num_of_classes, k=1, device=device, GPU_CAPACITY=GPU_CAPACITY)
    print("True Model accuracy on adversarial examples :" + str(acc * 100) + "%")

del noises_base
gc.collect()

# translate desired scores to their functions and put in a list
scores_list = []
implemented_scores = {'HPS': scores.class_probability_score, 
                      'APS': scores.generalized_inverse_quantile_score, 
                      'RAPS': scores.rank_regularized_score}
validation_scores = OrderedDict() # For non-PTT scores, calculate validation scores and combine them with calib sets
for score in args.calibration_scores:
    if score in implemented_scores:
        score_func = implemented_scores[score]
        scores_list.append(score_func)
        validation_scores[score] = score_func
    elif score.startswith('PTT_'):
        # generate training scores
        base_score = implemented_scores[score[4:]]
        ref_scores = get_scores(model, x_valid, np.arange(args.n_quantile), 1, args.sigma_model, num_of_classes, [base_score], base=True, device=device, GPU_CAPACITY=GPU_CAPACITY).squeeze()
        ref_scores = ref_scores[np.arange(ref_scores.shape[0]), y_valid.numpy()]

        PTT_score = scores.ranking_score(ref_scores, base_score)
        PTT_score = scores.sigmoid_score(PTT_score, T=args.T, bias=args.bias)
        scores_list.append(PTT_score)
    else:
        print("Undefined score function")
        exit(1)

print("Calculating scores for entire dataset:\n")

# Draw empirical pdf
np_ytest = y_test.numpy()

print("Calculating smoothed scores on the clean test points:\n")
if args.timing:
    timer = {}
    timer['base'] = running_mean()
    timer['scores'] = [running_mean() for _ in range(len(args.calibration_scores))]
else:
    timer = None
smoothed_scores_clean_test, Bern_bound_clean = get_scores(model, x_test, indices, args.n_s, args.sigma_smooth, num_of_classes, scores_list, base=False, device=device, GPU_CAPACITY=GPU_CAPACITY, timer=timer)
smoothed_scores_clean_val, _ = get_scores(
    model, x_valid, torch.arange(x_valid.shape[0]),
    args.n_s, args.sigma_smooth, num_of_classes,
    list(validation_scores.values()), base=False, device=device,
    GPU_CAPACITY=GPU_CAPACITY, timer=None)

if args.plot_icdf: draw_icdf(smoothed_scores_clean_test, args.calibration_scores, np_ytest, 'smooth_icdf.jpg', smoothed=True, cacheit=True)

if args.timing:
    print(timer['base'].get_mean())
    for i, s in enumerate(args.calibration_scores):
        print("score %s:" % s, timer['scores'][i].get_mean())
if not args.cleanonly:
    # get smooth scores of whole adversarial test set
    print("Calculating smoothed scores on the adversarial test points:\n")
    smoothed_scores_adv_test, Bern_bound_adv = get_scores(model, x_test_adv, indices, args.n_s, args.sigma_smooth, num_of_classes, scores_list, base=False, device=device, GPU_CAPACITY=GPU_CAPACITY)
    smoothed_scores_adv_val, _= get_scores(
        model, x_valid_adv, torch.arange(x_valid_adv.shape[0]),
        args.n_s, args.sigma_smooth, num_of_classes,
        list(validation_scores.values()), base=False,
        device=device, GPU_CAPACITY=GPU_CAPACITY)
    print(f"{Bern_bound_adv.shape=}")

# clean unnecessary data
del x_test
if not args.cleanonly: del x_test_adv, x_test_adv_base
gc.collect()

# create dataframe for storing results
results = pd.DataFrame()


# run for n_experiments data splittings
print("\nRunning experiments for "+str(args.splits)+" random splits:\n")
for experiment in tqdm(range(args.splits)):

    # Split test data into calibration and test
    idx1, idx2 = train_test_split(indices, test_size=0.5)
    
    # calibrate base model with the desired scores and get the thresholds
    calib_scores = {}
    idx_in_valid_scores = 0
    thresholds, thresholds_RSCP_plus = np.zeros((len(scores_list), 2)), np.zeros((len(scores_list), 2))
    for i, score_name in enumerate(args.calibration_scores):
        calib_scores[score_name] = smoothed_scores_clean_test[i, idx1, y_test[idx1]]
        if score_name in validation_scores:
            calib_scores[score_name] = np.concatenate(
                (calib_scores[score_name],
                 smoothed_scores_clean_val[idx_in_valid_scores, np.arange(y_valid.shape[0]), y_valid])
            )
            idx_in_valid_scores += 1
        thresholds[i, :] = calibration(
            smoothed_scores=calib_scores[score_name][None, :], alpha=args.alpha,
            num_of_scores=1, correction=correction,
            base=False, n_smooth=args.n_s).squeeze()
        thresholds_RSCP_plus[i, :] = calibration(
            smoothed_scores=calib_scores[score_name][None, :], alpha=args.alpha-2*args.beta,
            num_of_scores=1, correction=correction,
            base=False, n_smooth=args.n_s).squeeze()

    # generate prediction sets 
    bound_hoef = np.sqrt(-np.log(args.beta) / 2 / args.n_s)
    predicted_clean_sets = prediction(smoothed_scores=smoothed_scores_clean_test[:, idx2, :], num_of_scores=len(scores_list),
        thresholds=thresholds, threshold_RSCP_plus=thresholds_RSCP_plus, correction=correction, base=False, bounds_bern=Bern_bound_clean,
        bound_hoef=bound_hoef, record_no_correction=args.nocorr)
    if not args.cleanonly:
        predicted_adv_sets = prediction(smoothed_scores=smoothed_scores_adv_test[:, idx2, :], num_of_scores=len(scores_list),
            thresholds=thresholds, threshold_RSCP_plus=thresholds_RSCP_plus, correction=correction, base=False, bounds_bern=Bern_bound_adv,
            bound_hoef=bound_hoef, record_no_correction=args.nocorr)

    # arrange results on clean test set in dataframe
    for p in range(len(scores_list)):
        score_name = args.calibration_scores[p]
        methods_list = [score_name + '_RSCP', score_name + '_RSCP+']
        if args.nocorr: methods_list+= [score_name + '_RSCP_nocorr', score_name + '_RSCP+_Hoeff']
        for r, method in enumerate(methods_list):
            res = evaluate_predictions(predicted_clean_sets[p][r], None, y_test[idx2].numpy(),
                                    conditional=False,coverage_on_label=args.coverage_on_label, num_of_classes=num_of_classes)
            res['Method'] = methods_list[r]
            res['noise_L2_norm'] = 0
            res['Black box'] = 'CNN sigma = ' + str(args.sigma_model)
            # Add results to the list
            results = pd.concat((results, res))

    # arrange results on adversarial test set in dataframe
    if not args.cleanonly:
        for p in range(len(scores_list)):
            score_name = args.calibration_scores[p]
            methods_list = [score_name + '_RSCP', score_name + '_RSCP+']
            if args.nocorr: methods_list+= [score_name + '_RSCP_nocorr', score_name + '_RSCP+_Hoeff']
            for r, method in enumerate(methods_list):
                res = evaluate_predictions(predicted_adv_sets[p][r], None, y_test[idx2].numpy(),
                                        conditional=False, coverage_on_label=args.coverage_on_label, num_of_classes=num_of_classes)
                res['Method'] = methods_list[r]
                res['noise_L2_norm'] = args.delta
                res['Black box'] = 'CNN sigma = ' + str(args.sigma_model)
                # Add results to the list
                results = pd.concat((results, res))

    # clean memory
    del idx1, idx2, predicted_clean_sets, thresholds
    if not args.cleanonly: del predicted_adv_sets
    gc.collect()

# add given y string at the end
if args.coverage_on_label:
    add_string = "_given_y"
else:
    add_string = ""

# directory to save results
directory = utils.get_saving_dir(args, require_normalization)
if not os.path.exists(directory):
    os.makedirs(directory)

# save results
results.to_csv(directory + "/results" + add_string + ".csv")

clean_results = results[(results['noise_L2_norm']==0)]
clean_means = clean_results.groupby("Method").mean()
clean_stds = clean_results.groupby("Method").std()
clean_means.to_csv(directory + "/results_mean_clean.csv")
clean_stds.to_csv(directory + "/results_std_clean.csv")

noisy_results = results[(results['noise_L2_norm']==args.delta)]
noisy_means = noisy_results.groupby("Method").mean()
noisy_stds = noisy_results.groupby("Method").std()
noisy_means.to_csv(directory + "/results_mean_noisy.csv")
noisy_stds.to_csv(directory + "/results_std_noisy.csv")
noisy_results = pd.read_csv(directory + "/results_mean_noisy.csv")
result_dict = noisy_results[['Method', 'Size']].to_dict()
