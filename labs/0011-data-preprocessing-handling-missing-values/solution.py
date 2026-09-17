import numpy as np

def impute(X: np.ndarray) -> np.ndarray:
    '''
    Fill in missing values (NaN) in the input array.
    
    Args:
        X: Array with possible NaN values, shape (n_samples, n_features)
    
    Returns:
        X_clean: Array with no NaN values, same shape as X
    '''
    X_clean = np.array(X, dtype=float, copy=True)

    # TODO: Fill in NaN values
    for col_idx in range(X_clean.shape[1]):
        col = X_clean[:, col_idx]
        valid = np.isfinite(col)

        if np.any(valid):
            fill_val = np.median(col[valid])
        else:
            fill_val = 0.0

        col[~valid] = fill_val
    
    return X_clean
