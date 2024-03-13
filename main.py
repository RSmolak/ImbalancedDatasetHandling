import model 

from sklearn.model_selection import StratifiedKFold

from read_data import read_dataset_file, get_data_and_labels, one_hot_encode_dataframe



models = [
    model.NeuralNet(input_dim=10, hidden_layers=[10, 10, 10], output_dim=3),
    model.NeuralNet(input_dim=10, hidden_layers=[20, 20, 20], output_dim=3),
]

datasets = [
    'abalone19',

]

imbalance_handling = [
    "none",
    #"SMOTE",
    #"random_undersampling",
    #"KDE-based_oversampling",
    #"KDE-based_loss_weighting",
    #"KDE-based_batch_balancing"
]

epochs = 100
batch_size = 32
learning_rate = 0.01




# for id_architecture, architecture in enumerate(models): 

#     for id_dataset, dataset in enumerate(datasets):
#         df = read_dataset_file(f'DATASETS/{dataset}/{dataset}.dat')
#         X, y = get_data_and_labels(df)

#         for id_imbalance, imbalance_method in enumerate(imbalance_handling):

#             cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#             results = []

#             for train_index, val_index in cv.split(X, y):
#                 X_train, X_val = X[train_index], X[val_index]
#                 y_train, y_val = y[train_index], y[val_index]

#                 if imbalance_method != "none":
#                     X_train, y_train = dataset.imbalance_handling(X_train, y_train, imbalance_method)

#                 architecture.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
#                           learning_rate=learning_rate, validation_data=(X_val, y_val))
                
#                 results.append(architecture.evaluate(X_val, y_val)[1])

#             print(f"{architecture.name} {dataset.name} {imbalance_method}: {sum(results)/len(results):.3f}")


df = read_dataset_file('DATASETS/abalone19/abalone19.dat')
print(df.head())
df = one_hot_encode_dataframe(df)
print(df.head())
X,y = get_data_and_labels(df)
print(X, y)
