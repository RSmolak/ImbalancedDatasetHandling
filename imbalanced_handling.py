from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def handle_imbalanced(model, X, y, imbalance_method):
    if imbalance_method == "none":
        pass

    elif imbalance_method == "SMOTE":
        X, y = perform_SMOTE(X, y)
        print("Dataset after SMOTE:", X.shape, y.shape)
        return X, y
    
    elif imbalance_method == "random_undersampling":
        X, y = perform_random_undersampling(X, y)
        print("Dataset after random undersampling:", X.shape, y.shape)
        return X, y
    
    elif imbalance_method == "batch_balancing":
        pass
    elif imbalance_method == "KDE-based_oversampling":
        pass
    elif imbalance_method == "KDE-based_loss_weighting":
        pass
    elif imbalance_method == "KDE-based_batch_balancing":
        pass

def perform_SMOTE(X, y):
    sm = SMOTE(random_state=42)
    X, y = sm.fit_resample(X, y)
    return X, y

def perform_random_undersampling(X, y):
    rus = RandomUnderSampler(random_state=42)
    X, y = rus.fit_resample(X, y)
    return X, y