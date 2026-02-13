"""Function that calculate data shapley"""
import numpy as np
import shap
import xgboost as xgb


def data_shapley(dataset_size, x_train, y_train, v0, v1):
    """
    Function to compute kernel shap coefficients with closed form solution
    of Shapley from equation (7) in
    https://proceedings.mlr.press/v130/covert21a/covert21a.pdf

    Args:
    ----
        dataset_size: length of reference dataset size
        x_train: indices of subset, n x d
        y_train: model behavior, n x 1
        v0: model behavior of null subset
        v1: model behavior with all data presented

    Return:
    ------
        coef: coefficients for kernel shap
    """

    train_size = len(x_train)

    a_hat = np.dot(x_train.T, x_train) / train_size
    b_hat = np.dot(x_train.T, (y_train - v0).reshape(-1, 1)) / train_size

    # Using np.linalg.pinv instead of np.linalg.inv in case of singular matrix
    # rond: Cutoff for small singular values. The default is 1e-15.
    # Singular values less than or equal to rcond * largest_singular_value
    # are set to zero.

    a_hat_inv = np.linalg.pinv(a_hat)
    one = np.ones((dataset_size, 1))

    c = one.T @ a_hat_inv @ b_hat - v1 + v0
    d = one.T @ a_hat_inv @ one

    coef = a_hat_inv @ (b_hat - one @ (c / d))

    coef[np.abs(coef) < 1e-10] = 0
    # coef[np.abs(coef) > 50] = 50

    return coef


def surrogateshap(
    x_train,
    y_train,
    v0,
    v1,
    max_depth=5,
    n_estimators=900,
    learning_rate=0.05,
    reg_alpha=0.0,
    subsample=0.8,
    colsample_bytree=1.0,
):
    """
    Train an XGBoost on coalition indicators
    and compute TreeSHAP main + interaction values.

    Args:
    ----
        x_train: Coalition indicators, n x d
        y_train: Model behaviors, n x 1
        v0: Null model behavior (empty coalition)
        v1: Full model behavior (grand coalition)
        max_depth: Maximum tree depth {3, 5, 10}
        n_estimators: Number of trees {100, 300, 900}
        learning_rate: Learning rate {0.01, 0.05}
        reg_alpha: L1 regularization λ {0, 10}
        subsample: Row subsample ratio
        colsample_bytree: Column subsample ratio

    Return:
    ------
        main: Main effect Shapley values
        inter: Interaction Shapley values
        expected_value: Expected value at baseline
    """

    # --- inputs as arrays ---
    X = np.asarray(x_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.float32)
    n = X.shape[1]
    X = np.vstack(
        [X, np.zeros((1, n), dtype=np.float32), np.ones((1, n), dtype=np.float32)]
    )
    y = np.concatenate([y, [float(v0), float(v1)]], axis=0)

    m, n = X.shape

    # Train model with specified hyperparameters
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        reg_alpha=reg_alpha,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
    )
    model.fit(X, y)

    # --- explainer ---
    explainer = shap.TreeExplainer(
        model,
        data=np.zeros((1, n), dtype=np.float32),
        feature_perturbation="interventional",
        model_output="raw",
    )

    X_eval = np.ones((1, x_train.shape[1]), dtype=np.float32)

    # main effects (shape: p x n)
    main = explainer.shap_values(X_eval)
    # interactions (shape: p x n x n), diag == main
    inter = explainer.shap_interaction_values(X_eval)
    expected_value = float(explainer.expected_value)

    return main, inter, expected_value


def predict_quadratic(X, v0, phi, Phi):
    """Predict with model behavior with interaction Shapley values"""
    # ensure symmetry and diag=phi (TreeSHAP inter already satisfies this)
    Phi = 0.5 * (Phi + Phi.T)
    np.fill_diagonal(Phi, phi)
    # y = v0 + X@phi + 1/2 * sum_{i!=j} Phi_ij x_i x_j
    # einsum does the quadratic term efficiently
    # Implement off-diagonal only:
    off = np.einsum("bi,ij,bj->b", X, Phi - np.diag(np.diag(Phi)), X) * 0.5
    return v0 + X @ phi + off
