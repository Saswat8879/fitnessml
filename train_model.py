# train_model.py (fixed imports, sklearn-wrapper LGBM, GPU-aware, robust)
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# show versions for debugging
import lightgbm as lgb
import sklearn
print("LightGBM version:", getattr(lgb, "__version__", "not-found"))
print("scikit-learn version:", sklearn.__version__)
print("pandas version:", pd.__version__)
print("numpy version:", np.__version__)
from feature_engineer import engineer

def train_sklearn_lgb(X_train, y_train, X_val, y_val, use_gpu=False):
    """
    Train using the sklearn API wrapper (LGBMClassifier). This is more portable
    across different LightGBM builds and avoids signature differences of lgb.train.
    """
    params = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 64,
        "random_state": 42,
        "verbosity": -1
    }


    if use_gpu:
        params["device"] = "gpu"
        print("Attempting to train with device='gpu' via LGBMClassifier (requires GPU-enabled LightGBM).")
    else:
        params["device"] = "cpu"

    clf = lgb.LGBMClassifier(**params)

    try:
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            early_stopping_rounds=50,
            verbose=50
        )
    except TypeError as te:
        print("TypeError during fit with early_stopping_rounds:", te)
        print("Retrying fit without early_stopping_rounds (will still use eval_set for monitoring if supported).")
        try:
            clf.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                verbose=50
            )
        except Exception as e:
            print("Fit without early stopping also failed; doing final fallback fit with no eval_set. Error:", e)
            clf.fit(X_train, y_train)
    except Exception as e:
        print("Unexpected exception during fit:", e)
        print("Retrying fallback (no early stopping / eval_set).")
        clf.fit(X_train, y_train)

    return clf

def main():
    if not os.path.exists("training_data.csv"):
        raise RuntimeError("training_data.csv not found. Run generate_synthetic_data.py first.")

    df = pd.read_csv("training_data.csv")
    X, y, _df = engineer(df)

    X = X.fillna(0)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    prefer_gpu = os.environ.get("PREFER_GPU", "0") in ("1", "true", "True")
    model = None

    if prefer_gpu:
        try:
            model = train_sklearn_lgb(X_train, y_train, X_val, y_val, use_gpu=True)
            print("Trained LGBMClassifier with GPU param (if supported).")
        except Exception as e:
            print("GPU attempt failed; falling back to CPU. Error:", e)
            model = None

    if model is None:
        model = train_sklearn_lgb(X_train, y_train, X_val, y_val, use_gpu=False)
        print("Trained LGBMClassifier on CPU (or used fallback).")

    try:
        preds = model.predict_proba(X_val)[:, 1]
    except Exception:
        preds = model.predict(X_val).astype(float)
    auc = roc_auc_score(y_val, preds)
    print("Validation AUC:", auc)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/plate_ranker_lgb.pkl")
    print("Saved model to models/plate_ranker_lgb.pkl")

if __name__ == "__main__":
    main()
