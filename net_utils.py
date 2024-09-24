
import os
import time
import random
import numpy as np
import torch


def fix_randomness():
    r"Fix randomness."
    RAND_SEED = 2021
    random.seed(RAND_SEED)
    torch.manual_seed(RAND_SEED)
    np.random.seed(RAND_SEED)


def get_class_weights(
    labels, n_class=2, log=print
):
    r"""Calculate class frequency to calculate class weights.

    In order to calculate loss for imbalanced class.

    Arguments:
        partial_dataset. EcgDataset is extracted to get labels.

    Returns:
        frequency of labels.
        weights of frequency, max_freq/freq.
    """
    freq = np.zeros(n_class)
    for label in labels:
        freq[label] += 1

    # calculate weights
    # max_freq = freq[np.argmax(freq)]
    min_freq = freq[np.argmin(freq)]
    weights = min_freq / freq
    # weights = 1 / freq
    # weights = max_freq // freq
    log(f"freq:{freq}, weights:{weights}")
    return freq, weights


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EarlyStopping():
    '''Early stops the training if validation loss doesn't improve after a
        given patience.'''

    def __init__(
            self, patience=7, verbose=False, delta=0,
            path='checkpoint.pt', log=None, extra_meta=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss
                improved.
            verbose (bool): If True, prints a message for each validation loss
                improvement.
            delta (float): Minimum change in the monitored quantity to qualify
                as an improvement.
            path (str): Path for the checkpoint to be saved to.
            log : log function (TAG, msg).
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_extra = None  # Extra best other scores/info
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.log = log
        self.first_iter = False
        r"extra_meta is {'metric_name':'test_acc', 'max_val':99}"
        self.extra_meta = extra_meta

        self.print_(
            f'[{self.__class__.__name__}] patience:{patience}, delta:{delta}, model-path:{path}')

    def print_(self, msg):
        if self.log is None:
            print(msg)
        else:
            self.log(f"[EarlyStopping] {msg}")

    def __call__(self, val_loss, model, extra=None):
        r"""extra is {'test_acc':90}. The key was passed at c'tor."""
        score = -val_loss

        if self.best_score is None:
            self.first_iter = True
            self.best_score = score
            self.best_extra = extra
            self.save_checkpoint(val_loss, model)
            self.first_iter = False
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.print_(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_extra = extra
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            self.print_(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

        r"If extra is passed in call, and extra_meta exists, terminate \
        training if condition is met."
        if (not self.first_iter
                and self.extra_meta is not None
                and self.best_extra is not None
                and self.extra_meta.get('metric_name') is not None
                and self.extra_meta.get('max_val') is not None
                and self.best_extra.get(self.extra_meta.get('metric_name')) is not None
                and self.best_extra.get(self.extra_meta.get('metric_name')) >= self.extra_meta.get('max_val')
            ):
            self.print_(
                f"{self.extra_meta.get('metric_name')}:"
                f"{self.best_extra.get(self.extra_meta.get('metric_name'))} "
                f">= {self.extra_meta.get('max_val')}")
            self.early_stop = True
