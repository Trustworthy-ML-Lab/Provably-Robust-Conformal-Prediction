"""

Implementation of ConfTr Paper.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchsort
from torch.distributions.normal import Normal


def smooth_predict_threshold(probabilities: torch.Tensor,
    tau: torch.Tensor, temperature: float) -> torch.Tensor:
    """Smooth implementation of prediction step for Thr."""
    return torch.nn.Sigmoid()((probabilities - tau) / temperature)


def smooth_calibrate_threshold(probabilities: torch.Tensor,
    labels: torch.Tensor, alpha: float, dispersion: float) -> torch.Tensor:
    """Smooth implementation of the calibration step for Thr."""
    n_calib = probabilities.shape[0]
    conformity_scores = probabilities[np.arange(n_calib), labels].unsqueeze(0)
    sorted_scores = torchsort.soft_sort(conformity_scores, regularization_strength=dispersion).squeeze()
    corrected_rank = max(int(alpha * (1.0 + 1.0 / float(n_calib)) * n_calib) - 1, 0)
    return sorted_scores[corrected_rank]


def smooth_coverage(probabilities: torch.Tensor,
    labels: torch.Tensor, tau: torch.Tensor, dispersion: float) -> torch.Tensor:
    """Smooth implementation of coverage calculation"""
    n_calib = probabilities.shape[0]
    conformity_scores = probabilities[np.arange(n_calib), labels].unsqueeze(0)
    soft_rank = torchsort.soft_rank(torch.cat((conformity_scores, tau.unsqueeze(0).unsqueeze(0)), dim=1), regularization_strength=dispersion).squeeze()
    # print(soft_rank)
    return soft_rank[-1] / (n_calib + 1)

    
def compute_general_classification_loss(confidence_sets: torch.Tensor,
    labels: torch.Tensor, loss_matrix: torch.Tensor) -> torch.Tensor:
    """Compute the classification loss Lclass on the given confidence sets."""
    one_hot_labels = F.one_hot(labels, confidence_sets.shape[1])
    l1 = (1 - confidence_sets) * one_hot_labels * loss_matrix[labels]
    l2 = confidence_sets * (1 - one_hot_labels) * loss_matrix[labels]
    loss = torch.sum(torch.maximum(l1 + l2, torch.zeros_like(l1)), axis=1)
    return torch.mean(loss), torch.mean(torch.sum(l1, axis=1))


def compute_size_loss(confidence_sets: torch.Tensor,
    target_size: int, weights: torch.Tensor) -> torch.Tensor:
    """Compute size loss."""
    size_sum = torch.sum(confidence_sets, axis=1) - target_size
    return torch.mean(weights * torch.maximum(size_sum, torch.zeros_like(size_sum)))


def compute_loss_and_error(model, inputs, labels, training, rng, alpha, dispersion=0.1, n_classes=10,
    loss_matrix=None, size_weights=None, size_weight=1, device=torch.device('cuda'), target_size=0):
    """Compute classification and size loss through calibration/prediction."""
    if loss_matrix is None:
        loss_matrix = torch.eye(n_classes).to(device)
    if size_weights is None:
        size_weights = torch.ones((n_classes,)).to(device)

    if training:
        model.train()
        logits = model(inputs)
    else:
        model.eval()
        with torch.no_grad():
            logits = model(inputs)
    
    probabilities = nn.LogSoftmax(dim=1)(logits)
    val_split = int(0.5 * probabilities.shape[0])
    val_probabilities = probabilities[:val_split]
    val_labels = labels[:val_split]
    test_probabilities = probabilities[val_split:]
    test_labels = labels[val_split:]
    if training:
        tau = smooth_calibrate_threshold(val_probabilities, val_labels, alpha, dispersion)
        test_confidence_sets = smooth_predict_threshold(test_probabilities, tau, rng)
    else:
        with torch.no_grad():
            tau = smooth_calibrate_threshold(val_probabilities, val_labels, alpha, dispersion)
            test_confidence_sets = smooth_predict_threshold(test_probabilities, tau, rng)
    classification_loss, train_err = compute_general_classification_loss(
        test_confidence_sets, test_labels, loss_matrix)
    weights = size_weights[test_labels]
    size_loss = size_weight * compute_size_loss(test_confidence_sets, target_size, weights)
    loss = torch.log(classification_loss + size_loss + 1e-8)
    return loss


def compute_robust_loss(model, inputs, labels, rng, alpha, gauss_num, sigma, correction, dispersion=0.1, n_classes=10, score_post_process=None,
    loss_matrix=None, size_weights=None, size_weight=1, device=torch.device('cuda'), target_size=0, tau_correction=0, 
    logger=None):
    """Compute classification and size loss through calibration/prediction."""
    if loss_matrix is None:
        loss_matrix = torch.eye(n_classes).to(device)
    if size_weights is None:
        size_weights = torch.ones((n_classes,)).to(device)
    normal = Normal(0, 1)
    # preparing noisy inputs
    input_size = len(inputs)
    new_shape = [input_size * gauss_num]
    new_shape.extend(inputs[0].shape)
    inputs = inputs.repeat((1, gauss_num, 1, 1)).view(new_shape)
    noise = torch.randn_like(inputs, device=device) * sigma
    noisy_inputs = inputs + noise

    logits = model(noisy_inputs)
    logits = logits.reshape((input_size, gauss_num, -1))

    # Use probability to do training
    probabilities = nn.Softmax(dim=2)(logits)
    if score_post_process is not None:
        probabilities = score_post_process(probabilities)
    # Do smoothing
    probabilities = probabilities.mean(axis=1)
    acc = (torch.argmax(probabilities, dim=1) == labels).float().mean()
    logger.log("train_acc", acc)
    probabilities = normal.icdf(torch.minimum(probabilities + 1e-7, torch.ones_like(probabilities) - 1e-7))
    # Split into validation and test sets
    val_split = int(0.5 * probabilities.shape[0])
    val_probabilities = probabilities[:val_split]
    val_labels = labels[:val_split]
    test_probabilities = probabilities[val_split:]
    test_labels = labels[val_split:]

    hard_threshold = torch.quantile(val_probabilities[torch.arange(val_split), val_labels], alpha)
    prediction_size = (test_probabilities > hard_threshold).float().sum(dim=1).mean()
    logger.log("prediction size", prediction_size)
    # Calculating thresholds
    tau = smooth_calibrate_threshold(val_probabilities, val_labels, alpha, dispersion)
    logger.log("validation original size", (val_probabilities>tau).int().sum().item())
    if correction:
        tau -= correction
    tau -= tau_correction
    # Generate test sets
    logger.log("validation sum", (val_probabilities>tau).int().sum().item())
    logger.log("test sum", (test_probabilities>tau).int().sum().item())
    test_confidence_sets = smooth_predict_threshold(test_probabilities, tau, rng)
    classification_loss, train_err = compute_general_classification_loss(
        test_confidence_sets, test_labels, loss_matrix)
    weights = size_weights[test_labels]
    size_loss = size_weight * compute_size_loss(test_confidence_sets, target_size, weights)
    if logger is not None:
        logger.log("Size loss", size_loss.item() / size_weight)
        logger.log("Classification loss", classification_loss.item())
    loss = torch.log(classification_loss + size_loss + 1e-8)
    return loss

