import numpy as np
import torch
import random
from quora_dataset import get_dataset
import argparse

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

args_dict = dict(
    data_dir="", # path for data files
    output_dir="", # path to save the checkpoints
    model_name_or_path='ramsrigouthamg/t5_squad_v1',
    tokenizer_name_or_path='t5-base',
    max_seq_length=512,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=32,
    eval_batch_size=32,
    num_train_epochs=2,
    gradient_accumulation_steps=32,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)
def main():
    args_dict.update({'data_dir': '/kaggle/working', 'output_dir': '/kaggle/working/output/result', 'num_train_epochs':1,'max_seq_length':50})
    args = argparse.Namespace(**args_dict)
    print(args_dict)
