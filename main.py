import model 

models = [
    model.NeuralNet(input_dim=10, hidden_layers=[10, 10, 10], output_dim=3),
    model.NeuralNet(input_dim=10, hidden_layers=[20, 20, 20], output_dim=3),
]

datasets = [

]

imbalance_handling = [
    "none",
    "SMOTE",
    "random_undersampling",
    "KDE-based_oversampling",
    "KDE-based_loss_weighting",
    "KDE-based_batch_balancing"
]

epochs = 100
batch_size = 32
learning_rate = 0.01


from sklearn.model_selection import StratifiedKFold

for model in models:
    for dataset in datasets:
        for imbalance in imbalance_handling:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            results = []
            for train_index, val_index in cv.split(dataset.X, dataset.y):
                X_train, X_val = dataset.X[train_index], dataset.X[val_index]
                y_train, y_val = dataset.y[train_index], dataset.y[val_index]
                if imbalance != "none":
                    X_train, y_train = dataset.imbalance_handling(X_train, y_train, imbalance)
                model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                          learning_rate=learning_rate, validation_data=(X_val, y_val))
                results.append(model.evaluate(X_val, y_val)[1])
            print(f"{model.name} {dataset.name} {imbalance}: {sum(results)/len(results):.3f}")



