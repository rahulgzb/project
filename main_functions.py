import numpy as np
import torch
import random
from .quora_dataset import get_dataset
import argparse

from torch.utils.data import Dataset,DataLoader

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

'''total_steps = len(train_dataloader) // gradient_accumulation_steps * epochs
warmup_steps = int(0.1 * total_steps)
'''

args_dict = dict(
    data_dir="", # path for data files
    output_dir="", # path to save the checkpoints
    model_name_or_path='ramsrigouthamg/t5_squad_v1',
    tokenizer_name_or_path='t5-base',
    max_seq_length=128,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=100,    
    train_batch_size=32,
    num_train_epochs=2,
    gradient_accumulation_steps=32,
    n_gpu=1,
    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
    eval_method="rougeL",## value of evaluation option["rougeL","avg_val_loss","bleu"]
    eval_mode="max" ## options "max","min"
)
def data_loading(tokenizer,type_path,args):
    dataset_values = get_dataset(tokenizer=tokenizer, type_path=type_path, args=args)
    return DataLoader(dataset_values, batch_size=args.train_batch_size, drop_last=True, shuffle=True, num_workers=args.n_gpu, pin_memory=True)


def main():
    args_dict.update({'data_dir': '/kaggle/working', 'output_dir': '/kaggle/working/output/result', 'num_train_epochs':1,'max_seq_length':50})
    args = argparse.Namespace(**args_dict)

    print(args_dict)
