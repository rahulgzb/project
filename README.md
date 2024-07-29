
---

# Question-Answering like human Project

This repository contains the code and utilities for a state-of-the-art question-answering model using the Quora Question Answer Dataset. The project uses the 'ramsrigouthamg/t5_squad_v1' model and 't5-base' tokenizer, leveraging the Hugging Face transformers library for model inference. The evaluation metrics include ROUGE, BLEU, and F1 scores.

## Project Structure

### `project/models.py`
- **`t5_model`**: Implements and configures the T5 model for question answering.
- **`utils`**: Contains utility functions for model training and evaluation.

### `project/main_functions.py`
- **`set_seed`**: Sets the random seed for reproducibility.
- **`args_dict`**: Default arguments for various functions.
- **`data_loading`**: Loads and preprocesses data for training and validation.

### `project/generate_plots.py`
- **`save_plot`**: Generates and saves plots for model performance metrics.

## Getting Started

### use of this repo for fine training a llm  in kaggle or google plateform 

To get started, clone the repository and install the required packages in online plateform:

```bash
use these commands in the google colab to install all the dependencies
!git clone https://github.com/rahulgzb/project.git
!pip install -r requirements.txt
```

### Setting Arguments

The following arguments are used across all functions in this project:

```python
args_dict = dict(
    data_dir="",  # Path for data files
    output_dir="",  # Path to save the checkpoints
    model_name_or_path='ramsrigouthamg/t5_squad_v1', # used this model for finetuning the further for traning this model already trained for quora data pair data set 
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
    fp_16=False,  # Enable 16-bit training by setting this to True and installing apex
    max_grad_norm=1.0,  # Set a sensible value if using 16-bit training, 0.5 is a good default
    seed=42,
    eval_method="rougeL",  # Evaluation options: ["rougeL", "avg_val_loss", "bleu"]
    eval_mode="max"  # Options: "max", "min"
)

args = argparse.Namespace(**args_dict)
print(args)
```

### Reproducibility

To ensure reproducibility, set the random seed:

```python
set_seed()
```

### Data Loading

Load the training and validation data using the tokenizer: 
data is already cleaned with post processing 

```python
train_dataloader = data_loading(tokenizer, "train", args)
val_dataloader = data_loading(tokenizer, "val", args)
```

### Model Training

Train the model using the specified arguments and data loaders:

```python
utils.train_model(args, model, tokenizer, train_dataloader, val_dataloader)
```

### Generating Plots

Save performance plots to visualize the training results:

```python
save_plot(args)
```

## Usage Example in kaggle plateform


Here's a complete example of how to run the training:

```python

!git clone https://github.com/rahulgzb/project.git
!pip install -r requirements.txt
from project.kaggle_utils import reload_repo
from project.models import t5_model, utils
from project.main_functions import set_seed, args_dict, data_loading
from project.generate_plots import save_plot

args_dict.update({
    'data_dir': '/kaggle/working/project/data',
    'output_dir': '/kaggle/working/project/result',
    'num_train_epochs': 5,
    'max_seq_length': 168,
    'train_batch_size': 16,
    'n_gpu': 4,
    'fp_16': False,
    'warmup_steps': 10
})

args = argparse.Namespace(**args_dict)
print(args)
set_seed()

train_dataloader = data_loading(tokenizer, "train", args)
val_dataloader = data_loading(tokenizer, "val", args)

utils.train_model(args, model, tokenizer, train_dataloader, val_dataloader)
save_plot(args)
```

## Troubleshooting

If you encounter issues, ensure that:
- All dependencies are correctly installed.
- The paths in `args_dict` are correctly set to your data and output directories.
- You have sufficient GPU resources if `n_gpu` is set to more than 1.
