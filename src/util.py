"""Helper definitions"""

import json
import os

import torch
from utils.cityscapes_labels import labels as cityscapes_labels

def compute_params(model):
    """Compute number of parameters"""
    n_total_params = 0
    for name, m in model.named_parameters():
        n_elem = m.numel()
        n_total_params += n_elem
    return n_total_params

# Adopted from https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Saver():
    """Saver class for managing parameters"""
    def __init__(self, args, ckpt_dir, best_val=0, condition=lambda x,y: x > y):
        """
        Args:
            args (dict): dictionary with arguments.
            ckpt_dir (str): path to directory in which to store the checkpoint.
            best_val (float): initial best value.
            condition (function): how to decide whether to save the new checkpoint
                                    by comparing best value and new value (x,y).

        """
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        with open('{}/args.json'.format(ckpt_dir), 'w') as f:
            json.dump({k:v for k,v in args.items() if isinstance(v, (int, float, str))}, f,
                      sort_keys = True, indent = 4, ensure_ascii = False)
        self.ckpt_dir = ckpt_dir
        self.best_val = best_val
        self.condition = condition
        self._counter = 0

    def _do_save(self, new_val):
        """Check whether need to save"""
        return self.condition(new_val, self.best_val)

    def save(self, new_val, dict_to_save, logger):
        """Save new checkpoint"""
        self._counter += 1
        if self._do_save(new_val):
            logger.info(" New best value {:.4f}, was {:.4f}".format(new_val, self.best_val))
            self.best_val = new_val
            dict_to_save['best_val'] = new_val
            torch.save(dict_to_save, '{}/checkpoint.pth.tar'.format(self.ckpt_dir))
            return True
        return False

"""
The following function added to visualize the results following the color labels 
present in the CityScapes GT labels. - Aashish, 14/01/2019, 6:25pm
Color labels reference - https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
"""
def get_palette_cityscapes():
    n = 256
    palette = [0] * (n * 3)
    # Fill all to 'unknowns' by default
    for j in range(0, n):
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
    # Now, map the colors for the valid labels
    for label in cityscapes_labels:
        trainId = label.trainId
        # valid trainIds are from 0-18 (see also dataset/datasets.py)
        if (trainId >=0 and trainId<=18): 
            for ch in range(0, 3):
                palette[trainId * 3 + ch] = label.color[ch]
    return palette  