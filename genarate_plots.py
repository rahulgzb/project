import os
import json
import matplotlib.pyplot as plt
from statistics import mean

def read_json_files(directory):
    metrics_list = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                metrics = json.load(file)
                metrics_list.append(metrics)
    return metrics_list

def aggregate_metrics(metrics_list):
    aggregated_metrics = {
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
        "bleu": [],
        "avg_val_loss": [],
        "avg_train_loss": []
    }
    
    for metrics in metrics_list:
        aggregated_metrics["rouge1"].append(metrics["rouge1"])
        aggregated_metrics["rouge2"].append(metrics["rouge2"])
        aggregated_metrics["rougeL"].append(metrics["rougeL"])
        aggregated_metrics["bleu"].append(metrics["bleu"])
        aggregated_metrics["avg_val_loss"].append(metrics["avg_val_loss"])
        aggregated_metrics["avg_train_loss"].append(metrics["avg_train_loss"])
        
    return aggregated_metrics

def save_plot(args):
    # Directory containing JSON files
    # Read and aggregate JSON files
    directory= args.output_dir
    metrics_list = read_json_files(directory)
    aggregated_metrics = aggregate_metrics(metrics_list)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Plot loss on primary y-axis
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(aggregated_metrics["avg_val_loss"], label='Validation Loss', color='tab:blue')
    ax1.plot(aggregated_metrics["avg_train_loss"], label='train Loss', color='tab:green')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    # Adding legends
    fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))

    # Saving the plot
    plt.title('loss Over Epochs')
    plt.grid(True)
    plt.savefig('loss_plot.jpg', format='jpg')
    plt.show()
    plt.close()



    # Create a second y-axis to plot ROUGE and BLEU scores
    fig, ax2 = plt.subplots(figsize=(10, 6))
    ax2.set_ylabel('Scores')
    ax2.plot(aggregated_metrics["rouge1"], label='F1 Score', color='tab:orange')
    ax2.plot(aggregated_metrics["rouge2"], label='ROUGE-2', color='tab:green')
    ax2.plot(aggregated_metrics["rougeL"], label='ROUGE-L', color='tab:red')
    ax2.plot(aggregated_metrics["bleu"], label='BLEU', color='tab:purple')
    ax2.tick_params(axis='y', labelcolor='black')
    # Adding legends
    fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))

    # Saving the plot
    plt.title('matrix Over Epochs')
    plt.grid(True)
    plt.savefig('matrix_plot.jpg', format='jpg')
    plt.show()
    plt.close()

    
