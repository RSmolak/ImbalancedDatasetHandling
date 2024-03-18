import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score

def plot_losses(train_losses, val_losses):
    """
    Plots training and validation losses. Each set of losses (train and val) 
    is a list of lists, where each inner list contains the losses for a fold.

    Parameters:
    - train_losses: List of lists of training losses.
    - val_losses: List of lists of validation losses.
    """
    plt.figure(figsize=(12, 6))

    # Plot training losses
    for fold_losses in train_losses:
        plt.plot(fold_losses, color='blue', alpha=0.1)  # Slightly transparent
    mean_train_losses = np.mean(train_losses, axis=0)
    plt.plot(mean_train_losses, color='blue', label='Mean Training Loss', linewidth=2)

    # Plot validation losses
    for fold_losses in val_losses:
        plt.plot(fold_losses, color='red', alpha=0.1)  # Slightly transparent
    mean_val_losses = np.mean(val_losses, axis=0)
    plt.plot(mean_val_losses, color='red', label='Mean Validation Loss', linewidth=2)

    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_accuracies(train_accuracies, val_accuracies):
    """
    Plots training and validation accuracies. Each set of accuracies (train and val) 
    is a list of lists, where each inner list contains the accuracies for a fold.

    Parameters:
    - train_accuracies: List of lists of training accuracies.
    - val_accuracies: List of lists of validation accuracies.
    """
    plt.figure(figsize=(12, 6))

    # Plot training accuracies
    for fold_accuracies in train_accuracies:
        plt.plot(fold_accuracies, color='green', alpha=0.1)  # Slightly transparent
    mean_train_accuracies = np.mean(train_accuracies, axis=0)
    plt.plot(mean_train_accuracies, color='green', label='Mean Training Accuracy', linewidth=2)

    # Plot validation accuracies
    for fold_accuracies in val_accuracies:
        plt.plot(fold_accuracies, color='orange', alpha=0.1)  # Slightly transparent
    mean_val_accuracies = np.mean(val_accuracies, axis=0)
    plt.plot(mean_val_accuracies, color='orange', label='Mean Validation Accuracy', linewidth=2)

    plt.title('Training and Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def plot_metric(results, architecture, dataset, imbalance_method, metric='loss'):
    """
    Plots a specified metric for a given architecture, dataset, imbalance method, and phase (train or validation).
    The metric can be either 'loss' or 'accuracy'.
    
    Parameters:
    - results: Nested dictionary containing the metrics organized by architecture, dataset, imbalance method, and fold.
    - architecture: The architecture name as a string.
    - dataset: The dataset name as a string.
    - imbalance_method: The imbalance handling method name as a string.
    - metric: The metric to plot ('loss' or 'accuracy').
    - phase: The phase to plot ('train' or 'validation').
    """
    plt.figure(figsize=(12, 6))
    
    train_metrics = []
    val_metrics = []
    for fold in results[architecture][dataset][imbalance_method].keys():
        #fold_metric = results[architecture][dataset][imbalance_method][fold][phase][metric]
        fold_train_true_labels = results[architecture][dataset][imbalance_method][fold]['train']['labels']
        fold_train_pred_labels = results[architecture][dataset][imbalance_method][fold]['train']['predictions']
        fold_val_true_labels = results[architecture][dataset][imbalance_method][fold]['validation']['labels']
        fold_val_pred_labels = results[architecture][dataset][imbalance_method][fold]['validation']['predictions']

        if metric == 'loss':
            fold_metric_train = results[architecture][dataset][imbalance_method][fold]['train'][metric]
            fold_metric_val = results[architecture][dataset][imbalance_method][fold]['validation'][metric]
        elif metric == 'accuracy':
            fold_metric_train = accuracy_score(fold_train_true_labels, fold_train_pred_labels)
            fold_metric_val = accuracy_score(fold_val_true_labels, fold_val_pred_labels)
        elif metric == 'f1':
            fold_metric_train = f1_score(fold_train_true_labels, fold_train_pred_labels)
            fold_metric_val = f1_score(fold_val_true_labels, fold_val_pred_labels)
        elif metric == 'balanced_accuracy':
            fold_metric_train = balanced_accuracy_score(fold_train_true_labels, fold_train_pred_labels)
            fold_metric_val = balanced_accuracy_score(fold_val_true_labels, fold_val_pred_labels)
        elif metric == 'precision':
            fold_metric_train = precision_score(fold_train_true_labels, fold_train_pred_labels)
            fold_metric_val = precision_score(fold_val_true_labels, fold_val_pred_labels)
        elif metric == 'recall':
            fold_metric_train = recall_score(fold_train_true_labels, fold_train_pred_labels)
            fold_metric_val = recall_score(fold_val_true_labels, fold_val_pred_labels)
            
        train_metrics.append(fold_metric_train)
        val_metrics.append(fold_metric_val)
        
        # Plot individual folds with transparency
        plt.plot(fold_metric_train, alpha=0.1, color='blue')
        plt.plot(fold_metric_val, alpha=0.1, color='red')
    
    # Calculate and plot mean metric across folds for train and validation
    mean_train_metric = np.mean(train_metrics, axis=0)
    mean_val_metric = np.mean(val_metrics, axis=0)
    plt.plot(mean_train_metric, color='blue', label='Mean Train', linewidth=2)
    plt.plot(mean_val_metric, color='red', label='Mean Validation', linewidth=2)

    plt.title(f'{metric.capitalize()} per Epoch\n{architecture} - {dataset} - {imbalance_method}')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.show()