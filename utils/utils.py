import os
import csv
import torch
import numpy as np
from collections import deque
from typing import Sequence, Union, Tuple


# window the input within two bounds specified by center and width 
def window(image, center, width):
    low_bound = center - width // 2
    up_bound = center + width // 2

    if str(image.dtype).split(".")[0] == "torch":
        windowed_image = image.clone()
    else:
        windowed_image = image.copy()
    
    windowed_image[windowed_image < low_bound] = low_bound
    windowed_image[windowed_image > up_bound] = up_bound
    windowed_image -= windowed_image.min()
    windowed_image /= (windowed_image.max() - windowed_image.min())

    return windowed_image


class Collector:
    def __init__(self, max_len=20):
        self.max_len = max_len
        self.values = deque(maxlen=self.max_len)

    def append(self, x):
        self.values.append(x)

    def __len__(self):
        return len(self.values)

    def mean(self):
        if len(self.values) == 0:
            return 0
        return sum(self.values) / len(self.values)

    def max(self):
        return max(self.values)

    def min(self):
        return min(self.values)

    def first(self):
        return self.values[0]
    
    def last(self):
        return self.values[-1]

    def empty(self):
        self.values = deque(maxlen=self.max_len)

    def var(self):    
        norm_of_mean = [((elem - self.mean()) ** 2).sum() for elem in self.values]
        return torch.mean(torch.stack(norm_of_mean))

    def get_all(self):
        return torch.stack([elem for elem in self.values])


def norm22(x: torch.Tensor):
    return torch.sum(torch.pow(x, 2))

# This is a MONAI library function that I had to readapt to include dilation
def calculate_out_shape(
    in_shape: Union[Sequence[int], int, np.ndarray],
    kernel_size: Union[Sequence[int], int],
    dilation: Union[Sequence[int], int],
    stride: Union[Sequence[int], int],
    padding: Union[Sequence[int], int],
) -> Union[Tuple[int, ...], int]:
    """
    Calculate the output tensor shape when applying a convolution to a tensor of shape `inShape` with kernel size
    `kernel_size`, stride value `stride`, and input padding value `padding`. All arguments can be scalars or multiple
    values, return value is a scalar if all inputs are scalars.
    """
    in_shape_np = np.atleast_1d(in_shape)
    kernel_size_np = np.atleast_1d(kernel_size)
    dilation_np = np.atleast_1d(dilation)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_shape_np = ((in_shape_np - dilation_np * (kernel_size_np - 1) + padding_np + padding_np - 1) // stride_np) + 1
    out_shape = tuple(int(s) for s in out_shape_np)

    return out_shape if len(out_shape) > 1 else out_shape[0]


def log_average(train_loss_list, val_loss_list, type, logger, epoch):
    train_batch_mean = sum(train_loss_list[type]) / len(train_loss_list[type])
    val_batch_mean = sum(val_loss_list[type]) / len(val_loss_list[type])
    logger.add_scalars(type, {"train": train_batch_mean}, global_step=epoch)
    logger.add_scalars(type, {"val": val_batch_mean}, global_step=epoch)


def log_alpha(logger, alphas, predicted_alphas):
    out_dir = os.path.join(logger.save_dir, "output")
    file = os.path.join(out_dir, "alpha_predictions.csv")
    with open(file, "a") as output:
        writer = csv.writer(output)
        for i, gt in enumerate(alphas):
            writer.writerow([round(gt.item(), 4), round(predicted_alphas[i].item(), 4)])


def get_best_threshold(pr_curve):
    precision, recall, threshold = pr_curve
    best_f = 0.
    best_thr = 0.
    
    f_score = np.where((precision + recall) == 0, 0., 2 * (precision * recall) / (precision + recall))
    # loop over threshold to find best one
    for i, thr in enumerate(threshold):
        if f_score[i] > best_f:
            best_f = f_score[i]
            best_thr = thr

    return best_f, best_thr


def get_classification_outcome(score, label, threshold):
    score = score > threshold
    if score and label:
        outcome = 'TP'
    elif not (score or label):
        outcome = 'TN'
    elif score and not label:
        outcome = 'FP'
    elif not score and label:
        outcome = 'FN'
    
    return outcome