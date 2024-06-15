import pandas as pd
import ast
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


# Function to convert DataFrame to results dictionary


def plot_accuracy_per_epoch(df, model_name, dataset_name, imbalance_method, fold_id):
    # Locate the specific column in the DataFrame
    column = df.loc[:, (df.loc[0] == model_name) & 
                        (df.loc[1] == dataset_name) & 
                        (df.loc[2] == imbalance_method) & 
                        (df.loc[3] == fold_id)].squeeze()

    # Extract training and validation data
    training_data = column[4]
    valid_data = column[5]

    training_data = ast.literal_eval(training_data)
    valid_data = ast.literal_eval(valid_data)

    # print("aa", training_data[0], training_data[0]['TrueLabels'])
    # Calculate accuracy for each epoch
    train_accuracies = []
    val_accuracies = []

    for epoch_data in training_data:
        true_labels = epoch_data['TrueLabels']
        predicted_labels = epoch_data['PredictedLabels']
        accuracy = accuracy_score(true_labels, predicted_labels)
        train_accuracies.append(accuracy)

    for epoch_data in valid_data:
        true_labels = epoch_data['TrueLabels']
        predicted_labels = epoch_data['PredictedLabels']
        accuracy = accuracy_score(true_labels, predicted_labels)
        val_accuracies.append(accuracy)

    # Plot accuracy per epoch
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy per Epoch for {model_name} on {dataset_name} using {imbalance_method} (Fold {fold_id})')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_accuracy_per_epoch_allfolds(df, model_name, dataset_name, imbalance_method):
    all_train_accuracies = []
    all_val_accuracies = []
    
    for fold_id in range(0, 5):
        fold_id = str(fold_id)
        # Locate the specific column in the DataFrame
        column = df.loc[:, (df.loc[0] == model_name) & 
                            (df.loc[1] == dataset_name) & 
                            (df.loc[2] == imbalance_method) & 
                            (df.loc[3] == fold_id)].squeeze()

        # Extract training and validation data
        training_data = column[4]
        valid_data = column[5]

        training_data = ast.literal_eval(training_data)
        valid_data = ast.literal_eval(valid_data)

        # Calculate accuracy for each epoch
        train_accuracies = []
        val_accuracies = []

        for epoch_data in training_data:
            true_labels = epoch_data['TrueLabels']
            predicted_labels = epoch_data['PredictedLabels']
            accuracy = accuracy_score(true_labels, predicted_labels)
            train_accuracies.append(accuracy)

        for epoch_data in valid_data:
            true_labels = epoch_data['TrueLabels']
            predicted_labels = epoch_data['PredictedLabels']
            accuracy = accuracy_score(true_labels, predicted_labels)
            val_accuracies.append(accuracy)
        
        # Store the accuracies for mean calculation
        all_train_accuracies.append(train_accuracies)
        all_val_accuracies.append(val_accuracies)
        
        # Plot accuracy per epoch without legend
        plt.plot(train_accuracies, alpha=0.1, color='orange')
        plt.plot(val_accuracies, alpha=0.1, color='blue')
    
    # Calculate mean accuracies
    mean_train_accuracies = [sum(acc) / len(acc) for acc in zip(*all_train_accuracies)]
    mean_val_accuracies = [sum(acc) / len(acc) for acc in zip(*all_val_accuracies)]
    
    # Plot mean accuracies with legend
    plt.plot(mean_train_accuracies, label='Mean Training Accuracy', linewidth=2, color='orange')
    plt.plot(mean_val_accuracies, label='Mean Validation Accuracy', linewidth=2, color='blue')
    
    # Final plot settings
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy per Epoch for {model_name} on {dataset_name} using {imbalance_method}')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_metric_per_epoch_allfolds(ax, df, model_name, dataset_name, imbalance_method, metric):
    # all_train_metrics = []
    all_val_metrics = []
    metric_score = None

    # plt.figure()

    if metric == 'accuracy':
        metric_score = accuracy_score
    elif metric == 'f1':
        metric_score = f1_score
    elif metric == 'balanced_accuracy':
        metric_score = balanced_accuracy_score
    elif metric == 'precision':
        metric_score = precision_score
    elif metric == 'recall':
        metric_score = recall_score
    elif metric == 'f1':
        metric_score = f1_score

    for fold_id in range(0, 5):
        fold_id = str(fold_id)
        # Locate the specific column in the DataFrame
        column = df.loc[:, (df.loc[0] == model_name) & 
                            (df.loc[1] == dataset_name) & 
                            (df.loc[2] == imbalance_method) & 
                            (df.loc[3] == fold_id)].squeeze()

        # Extract training and validation data
        # training_data = column[4]
        valid_data = column[5]

        # training_data = ast.literal_eval(training_data)
        valid_data = ast.literal_eval(valid_data)

        # Calculate accuracy for each epoch
        # train_metrics = []
        val_metrics = []

        # for epoch_data in training_data:
        #     true_labels = epoch_data['TrueLabels']
        #     predicted_labels = epoch_data['PredictedLabels']
        #     metric_scr = metric_score(true_labels, predicted_labels)
        #     train_metrics.append(metric_scr)

        for epoch_data in valid_data:
            true_labels = epoch_data['TrueLabels']
            predicted_labels = epoch_data['PredictedLabels']
            metric_scr = metric_score(true_labels, predicted_labels)
            val_metrics.append(metric_scr)
        
        # Store the accuracies for mean calculation
        # all_train_metrics.append(train_metrics)
        all_val_metrics.append(val_metrics)
        
        # Plot accuracy per epoch without legend
        # plt.plot(train_metrics, alpha=0.1, color='orange')
        # plt.plot(val_metrics, alpha=0.1)
    
    # Calculate mean accuracies
    # mean_train_metrics = [sum(acc) / len(acc) for acc in zip(*all_train_metrics)]
    mean_val_metrics = [sum(acc) / len(acc) for acc in zip(*all_val_metrics)]
    
    # Plot mean accuracies with legend
    # plt.plot(mean_train_metrics, label=f'Mean Training {metric}', linewidth=2, color='orange')
    ax.plot(mean_val_metrics, label=f'Mean Validation {metric}')
    
    

def plot_last_metric_for_methods(df, model_name, dataset_name, imbalance_methods, metric):
    metric_score = None
    train_scores = []
    valid_scores = []

    plt.figure()
    if metric == 'accuracy':
        metric_score = accuracy_score
    elif metric == 'f1':
        metric_score = f1_score
    elif metric == 'balanced_accuracy':
        metric_score = balanced_accuracy_score
    elif metric == 'precision':
        metric_score = precision_score
    elif metric == 'recall':
        metric_score = recall_score

    for imbalance_method in imbalance_methods:
        print(imbalance_method)
        fold_train_scores = []
        fold_valid_scores = []
        for fold_id in range(0, 5):
            fold_id = str(fold_id)
            print(fold_id)
            # Locate the specific column in the DataFrame
            column = df.loc[:, (df.loc[0] == model_name) & 
                                (df.loc[1] == dataset_name) & 
                                (df.loc[2] == imbalance_method) & 
                                (df.loc[3] == fold_id)].squeeze()

            # Extract training and validation data
            training_data = column[4]
            valid_data = column[5]

            training_data = ast.literal_eval(training_data)
            valid_data = ast.literal_eval(valid_data)

            train_last_epoch_data = training_data[-1]
            train_true_labels = train_last_epoch_data['TrueLabels']
            train_predicted_labels = train_last_epoch_data['PredictedLabels']
            train_metric_scr = metric_score(train_true_labels, train_predicted_labels)
            fold_train_scores.append(train_metric_scr)

            valid_last_epoch_data = valid_data[-1]
            valid_true_labels = valid_last_epoch_data['TrueLabels']
            valid_predicted_labels = valid_last_epoch_data['PredictedLabels']
            valid_metric_scr = metric_score(valid_true_labels, valid_predicted_labels)
            fold_valid_scores.append(valid_metric_scr)
        train_scores.append(sum(fold_train_scores)/5)
        valid_scores.append(sum(fold_valid_scores)/5)
        
    print(train_scores, valid_scores)


def plot_metric_for_imbalanced_ratios(ax, df, model_name, datasets, imbalance_method, metric):
    metric_score = None
    # train_scores = []
    valid_scores = []

    if metric == 'accuracy':
        metric_score = accuracy_score
    elif metric == 'f1':
        metric_score = f1_score
    elif metric == 'balanced_accuracy':
        metric_score = balanced_accuracy_score
    elif metric == 'precision':
        metric_score = precision_score
    elif metric == 'recall':
        metric_score = recall_score

    x_axis = []

    for dataset_name in datasets:
        # fold_train_scores = []
        fold_valid_scores = []
        print(dataset_name)
        for fold_id in range(0, 5):
            fold_id = str(fold_id)
            # Locate the specific column in the DataFrame
            column = df.loc[:, (df.loc[0] == model_name) & 
                                (df.loc[1] == dataset_name) & 
                                (df.loc[2] == imbalance_method) & 
                                (df.loc[3] == fold_id)].squeeze()

            # Extract training and validation data
            # training_data = column[4]
            valid_data = column[5]

            # training_data = ast.literal_eval(training_data)
            valid_data = ast.literal_eval(valid_data)

            # train_last_epoch_data = training_data[-1]
            # train_true_labels = train_last_epoch_data['TrueLabels']
            # train_predicted_labels = train_last_epoch_data['PredictedLabels']
            # train_metric_scr = metric_score(train_true_labels, train_predicted_labels)
            # fold_train_scores.append(train_metric_scr)

            valid_last_epoch_data = valid_data[-1]
            valid_true_labels = valid_last_epoch_data['TrueLabels']
            valid_predicted_labels = valid_last_epoch_data['PredictedLabels']
            valid_metric_scr = metric_score(valid_true_labels, valid_predicted_labels)
            fold_valid_scores.append(valid_metric_scr)
        # train_scores.append(sum(fold_train_scores)/5)
        valid_scores.append(sum(fold_valid_scores)/5)
        
        temp = float(dataset_name.split('_')[-1])
        x_axis.append(temp/(1-temp))

    ax.plot(x_axis, valid_scores, label=imbalance_method)



def create_table(df, metric, datasets, imbalance_methods, model_name):
    metric_score = None

    if metric == 'accuracy':
        metric_score = accuracy_score
    elif metric == 'f1':
        metric_score = f1_score
    elif metric == 'balanced_accuracy':
        metric_score = balanced_accuracy_score
    elif metric == 'precision':
        metric_score = precision_score
    elif metric == 'recall':
        metric_score = recall_score


    columns = {}
    columns['Zbiory danych'] = datasets
    for imbalance_method in imbalance_methods:
        valid_scores = []
        for dataset_name in datasets:
            print(dataset_name)
            fold_valid_scores = []
            for fold_id in range(0, 5):
                fold_id = str(fold_id)
                # Locate the specific column in the DataFrame
                column = df.loc[:, (df.loc[0] == model_name) & 
                                    (df.loc[1] == dataset_name) & 
                                    (df.loc[2] == imbalance_method) & 
                                    (df.loc[3] == fold_id)].squeeze()

                # Extract training and validation data
                valid_data = column[5]

                valid_data = ast.literal_eval(valid_data)

                valid_last_epoch_data = valid_data[-1]
                valid_true_labels = valid_last_epoch_data['TrueLabels']
                valid_predicted_labels = valid_last_epoch_data['PredictedLabels']
                valid_metric_scr = metric_score(valid_true_labels, valid_predicted_labels)
                fold_valid_scores.append(valid_metric_scr)
            valid_scores.append(sum(fold_valid_scores)/5)
            print(valid_scores)
        columns[imbalance_method] = valid_scores
    print(columns)
    result_df = pd.DataFrame(columns)
    return result_df
        
def bold_max_in_row(row):
    # Pominięcie pierwszej kolumny
    subset = row[1:]
    # Znalezienie maksymalnej wartości w wierszu (bez pierwszej kolumny)
    max_value = subset.max()
    # Pogrubienie maksymalnej wartości
    row_str = row.apply(lambda x: '\\textbf{' + str(x) + '}' if x == max_value else str(x) if x != row[0] else str(row[0]))
    return row_str

# df = pd.read_csv("lr001ep200.csv", header=None)
# df2 = pd.read_csv("lr0003ep300.csv", header=None)
df2 = pd.read_csv("experiment_results_final.csv", header=None)
# df3 = pd.read_csv("experiment_results.csv", header=None)
# print(df.head(6))
# plot_accuracy_per_epoch(df, 'MLP_10_10_10', 'ecoli1', 'none', '0')
# plot_accuracy_per_epoch(df, 'MLP_20_20_20', 'winequality-red-4', 'SMOTE', '3')
# plot_accuracy_per_epoch(df, 'MLP_10_10_10', 'synthetic_500_8_0.95', 'KDE-based_loss_weighting', '4')

# plot_accuracy_per_epoch_allfolds(df, 'MLP_10_10_10', 'ecoli1', 'none')
# plot_accuracy_per_epoch_allfolds(df, 'MLP_20_20_20', 'winequality-red-4', 'SMOTE')
# plot_accuracy_per_epoch_allfolds(df, 'MLP_10_10_10', 'synthetic_500_8_0.95', 'KDE-based_loss_weighting')

methods = [
    "none",
    "SMOTE",
    "random_undersampling",
    "batch_balancing",
    "KDE-based_oversampling",
    "KDE-based_loss_weighting",
    "KDE-based_batch_balancing"
]

datasets = [
    'ecoli1',
    'glass4',
    'vowel0',
    'iris0',
    'glass6',
    'winequality-red-4'
]

metrics = [
    'balanced_accuracy',
    'precision',
    'recall',
    'f1'
]


# plot_metric_per_epoch_allfolds(df2, 'MLP_10_10_10', 'vowel0', 'KDE-based_loss_weighting', 'precision')
# plot_metric_per_epoch_allfolds(df2, 'MLP_10_10_10', 'vowel0', 'none', 'precision')
# plot_metric_per_epoch_allfolds(df2, 'MLP_10_10_10', 'vowel0', 'random_undersampling', 'precision')
# plot_metric_per_epoch_allfolds(df2, 'MLP_10_10_10', 'vowel0', 'KDE-based_loss_weighting', 'recall')
# plot_metric_per_epoch_allfolds(df2, 'MLP_10_10_10', 'vowel0', 'none', 'recall')
# plot_metric_per_epoch_allfolds(df2, 'MLP_10_10_10', 'vowel0', 'random_undersampling', 'recall')
# plot_metric_per_epoch_allfolds(df2, 'MLP_10_10_10', 'vowel0', 'KDE-based_loss_weighting', 'f1')
# plot_metric_per_epoch_allfolds(df2, 'MLP_10_10_10', 'vowel0', 'none', 'f1')
# plot_metric_per_epoch_allfolds(df2, 'MLP_10_10_10', 'vowel0', 'random_undersampling', 'f1')

# plot_metric_per_epoch_allfolds(df2, 'MLP_10_10_10', 'ecoli1', 'SMOTE', 'balanced_accuracy')
# plot_metric_per_epoch_allfolds(df2, 'MLP_10_10_10', 'ecoli1', 'random_undersampling', 'balanced_accuracy')
# plot_metric_per_epoch_allfolds(df2, 'MLP_10_10_10', 'ecoli1', 'batch_balancing', 'balanced_accuracy')
# plot_metric_per_epoch_allfolds(df2, 'MLP_10_10_10', 'ecoli1', 'KDE-based_oversampling', 'balanced_accuracy')
# plot_metric_per_epoch_allfolds(df2, 'MLP_10_10_10', 'ecoli1', 'KDE-based_loss_weighting', 'balanced_accuracy')
# plot_metric_per_epoch_allfolds(df2, 'MLP_10_10_10', 'ecoli1', 'KDE-based_batch_balancing', 'balanced_accuracy')

# plt.show()


# plot_metric_per_epoch_allfolds(df2, 'MLP_10_10_10', 'synthetic_500_8_0.95', 'none', 'balanced_accuracy')
# plot_metric_per_epoch_allfolds(df2, 'MLP_10_10_10', 'synthetic_500_8_0.95', 'SMOTE', 'balanced_accuracy')
# plot_metric_per_epoch_allfolds(df2, 'MLP_10_10_10', 'synthetic_500_8_0.95', 'random_undersampling', 'balanced_accuracy')
# plot_metric_per_epoch_allfolds(df2, 'MLP_10_10_10', 'synthetic_500_8_0.95', 'batch_balancing', 'balanced_accuracy')
# plot_metric_per_epoch_allfolds(df2, 'MLP_10_10_10', 'synthetic_500_8_0.95', 'KDE-based_oversampling', 'balanced_accuracy')
# plot_metric_per_epoch_allfolds(df2, 'MLP_10_10_10', 'synthetic_500_8_0.95', 'KDE-based_loss_weighting', 'balanced_accuracy')
# plot_metric_per_epoch_allfolds(df2, 'MLP_10_10_10', 'synthetic_500_8_0.95', 'KDE-based_batch_balancing', 'balanced_accuracy')

# plot_last_metric_for_methods(df2, 'MLP_20_20_20', 'synthetic_500_8_0.97', [
#     "none",
#     "SMOTE",
#     "random_undersampling",
#     "batch_balancing",
#     "KDE-based_oversampling",
#     "KDE-based_loss_weighting",
#     "KDE-based_batch_balancing"
# ], 'balanced_accuracy')






fig, axs = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle("Metrics for different imbalanced ratios")

# First subplot
axs[0, 0].set_title("Balanced Accuracy")
for method in methods:
    plot_metric_for_imbalanced_ratios(axs[0, 0], df2, 'MLP_10_10_10', [
        'synthetic_500_8_0.6',
        'synthetic_500_8_0.7',
        'synthetic_500_8_0.8',
        'synthetic_500_8_0.9',
        'synthetic_500_8_0.92',
        'synthetic_500_8_0.94',
        'synthetic_500_8_0.95',
        'synthetic_500_8_0.96',
        'synthetic_500_8_0.97',
        'synthetic_500_8_0.98'
    ], method, 'balanced_accuracy')

# Second subplot
axs[0, 1].set_title("Precision")
for method in methods:
    plot_metric_for_imbalanced_ratios(axs[0, 1], df2, 'MLP_10_10_10', [
        'synthetic_500_8_0.6',
        'synthetic_500_8_0.7',
        'synthetic_500_8_0.8',
        'synthetic_500_8_0.9',
        'synthetic_500_8_0.92',
        'synthetic_500_8_0.94',
        'synthetic_500_8_0.95',
        'synthetic_500_8_0.96',
        'synthetic_500_8_0.97',
        'synthetic_500_8_0.98'
    ], method, 'precision')

# Third subplot
axs[1, 0].set_title("Recall")
for method in methods:
    plot_metric_for_imbalanced_ratios(axs[1, 0], df2, 'MLP_10_10_10', [
        'synthetic_500_8_0.6',
        'synthetic_500_8_0.7',
        'synthetic_500_8_0.8',
        'synthetic_500_8_0.9',
        'synthetic_500_8_0.92',
        'synthetic_500_8_0.94',
        'synthetic_500_8_0.95',
        'synthetic_500_8_0.96',
        'synthetic_500_8_0.97',
        'synthetic_500_8_0.98'
    ], method, 'recall')

# Fourth subplot
axs[1, 1].set_title("F1")
for method in methods:
    plot_metric_for_imbalanced_ratios(axs[1, 1], df2, 'MLP_10_10_10', [
        'synthetic_500_8_0.6',
        'synthetic_500_8_0.7',
        'synthetic_500_8_0.8',
        'synthetic_500_8_0.9',
        'synthetic_500_8_0.92',
        'synthetic_500_8_0.94',
        'synthetic_500_8_0.95',
        'synthetic_500_8_0.96',
        'synthetic_500_8_0.97',
        'synthetic_500_8_0.98'
    ], method, 'f1')

# Gather handles and labels from subplots for legend
handles, labels = axs[1,1].get_legend_handles_labels()
print(handles, labels)
plt.figlegend(handles, labels, loc='lower center', ncol=4)
axs[0,0].set_ylabel('Balanced Accuracy')
axs[0,1].set_ylabel('Precision')
axs[1,0].set_ylabel('Recall')
axs[1,1].set_ylabel('F1')
axs[0,0].set_xlabel('Imbalanced Ratio')
axs[0,1].set_xlabel('Imbalanced Ratio')
axs[1,0].set_xlabel('Imbalanced Ratio')
axs[1,1].set_xlabel('Imbalanced Ratio')
plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
plt.savefig('results/metrics_for_imbalanced_ratios.png')



plt.show()



# for dataset in datasets:
#     print(dataset)
#     fig, axs = plt.subplots(1, 3, figsize=(12,7))
#     fig.suptitle(f"Balanced Accuracy per epoch for {dataset}")

#     axs[0].set_title("KDE-based_oversampling")
#     plot_metric_per_epoch_allfolds(axs[0], df2, 'MLP_10_10_10', dataset, 'none', 'balanced_accuracy')
#     plot_metric_per_epoch_allfolds(axs[0], df2, 'MLP_10_10_10', dataset, 'SMOTE', 'balanced_accuracy')
#     plot_metric_per_epoch_allfolds(axs[0], df2, 'MLP_10_10_10', dataset, 'random_undersampling', 'balanced_accuracy')
#     plot_metric_per_epoch_allfolds(axs[0], df2, 'MLP_10_10_10', dataset, 'batch_balancing', 'balanced_accuracy')
#     plot_metric_per_epoch_allfolds(axs[0], df2, 'MLP_10_10_10', dataset, 'KDE-based_oversampling', 'balanced_accuracy')
#     axs[0].legend(['none', 'SMOTE', 'random_undersampling', 'batch_balancing', 'KDE-based_oversampling'])
#     axs[0].set_ylabel('Balanced Accuracy')
#     axs[0].set_xlabel('Epoch')
#     axs[0].set_ylim(bottom=0.35)

#     axs[1].set_title("KDE-based_loss_weighting")
#     plot_metric_per_epoch_allfolds(axs[1], df2, 'MLP_10_10_10', dataset, 'none', 'balanced_accuracy')
#     plot_metric_per_epoch_allfolds(axs[1], df2, 'MLP_10_10_10', dataset, 'SMOTE', 'balanced_accuracy')
#     plot_metric_per_epoch_allfolds(axs[1], df2, 'MLP_10_10_10', dataset, 'random_undersampling', 'balanced_accuracy')
#     plot_metric_per_epoch_allfolds(axs[1], df2, 'MLP_10_10_10', dataset, 'batch_balancing', 'balanced_accuracy')
#     plot_metric_per_epoch_allfolds(axs[1], df2, 'MLP_10_10_10', dataset, 'KDE-based_loss_weighting', 'balanced_accuracy')
#     axs[1].legend(['none', 'SMOTE', 'random_undersampling', 'batch_balancing', 'KDE-based_loss_weighting'])
#     axs[1].set_ylabel('Balanced Accuracy')
#     axs[1].set_xlabel('Epoch')
#     axs[1].set_ylim(bottom=0.35)

#     axs[2].set_title("KDE-based_batch_balancing")
#     plot_metric_per_epoch_allfolds(axs[2], df2, 'MLP_10_10_10', dataset, 'none', 'balanced_accuracy')
#     plot_metric_per_epoch_allfolds(axs[2], df2, 'MLP_10_10_10', dataset, 'SMOTE', 'balanced_accuracy')
#     plot_metric_per_epoch_allfolds(axs[2], df2, 'MLP_10_10_10', dataset, 'random_undersampling', 'balanced_accuracy')
#     plot_metric_per_epoch_allfolds(axs[2], df2, 'MLP_10_10_10', dataset, 'batch_balancing', 'balanced_accuracy')
#     plot_metric_per_epoch_allfolds(axs[2], df2, 'MLP_10_10_10', dataset, 'KDE-based_batch_balancing', 'balanced_accuracy')
#     axs[2].legend(['none', 'SMOTE', 'random_undersampling', 'batch_balancing', 'KDE-based_batch_balancing'])
#     axs[2].set_ylabel('Balanced Accuracy')
#     axs[2].set_xlabel('Epoch')
#     axs[2].set_ylim(bottom=0.35)
#     plt.tight_layout()
#     plt.savefig(f'results/Balanced_Accuracy_per_epoch_for_{dataset}.png')



# for metric in metrics:
#     dataframe = create_table(df2, metric, datasets, methods, 'MLP_10_10_10')
#     dataframe = dataframe.round(3)

#     df_bold = dataframe.apply(bold_max_in_row, axis=1)
#     latex_table = df_bold.to_latex(index=False, escape=False)

#     with open(f'results/{metric}_table.txt', 'w') as f:
#         f.write(latex_table)