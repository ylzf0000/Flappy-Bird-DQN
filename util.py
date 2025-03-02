import logging
import os
from datetime import datetime
import random

import numpy as np
import torch
from torch import nn


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed


def count_parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def format_number(num):
        if num >= 1e9:
            return f'{num / 1e9:.2f}B'
        elif num >= 1e6:
            return f'{num / 1e6:.2f}M'
        elif num >= 1e3:
            return f'{num / 1e3:.2f}K'
        else:
            return str(num)

    s = ""
    s += f'Total parameters: {format_number(total_params)}\n'
    s += f'Trainable parameters: {format_number(trainable_params)}\n'
    s += "\nTrainable parameters details:\n"
    for name, param in model.named_parameters():
        if param.requires_grad:
            s += f"\t{name}: {format_number(param.numel())} parameters\n"
    return s


def init_logger(logger_name, prefix):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatted_time = datetime.now().strftime('%Y%m%d-%H%M')
    if not logger.hasHandlers():
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        os.makedirs('./logs', exist_ok=True)
        file_handler = logging.FileHandler(f'./logs/{prefix}_{formatted_time}.log')
        file_handler.suffix = "%Y-%m-%d"
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger, formatted_time
