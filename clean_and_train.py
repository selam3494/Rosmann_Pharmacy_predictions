# clean_and_train.py
import pandas as pd, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from feature_utils import feature_engineer
from path_utils import DATA_DIR, MODEL_DIR

SEED      = 42
MODELPATH = MODEL_DIR / "rossmann_lgbm.pkl"

def rmspe(y_true, y_pred):
    mask = y_true != 0
    return (((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2).mean() ** 0.5

def load_merged():
    train  = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["Date"])
    store  = pd.read_csv(DATA_DIR / "store.csv")
    merged = train.merge(store, on="Store", how="left")
    return merged[merged["Open"] == 1]

def main():
    df, cat_cols = feature_engineer(load_merged())
    X = df.drop(columns=["Sales", "Customers", "Date"])
    y = df["Sales"]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, stratify=df["Store"], test_size=0.2, random_state=SEED
    )

    model = LGBMRegressor(
        n_estimators=2000, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, metric="None", objective="regression"
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric=lambda yt, yp: ("rmspe", rmspe(yt, yp), False),
        categorical_feature=cat_cols,
    )

    print("Validation RMSPE:", rmspe(y_val, model.predict(X_val)))

    feature_names = [c for c in X_tr.columns if c.lower() != "id"]
    print("Feature names saved:", feature_names)

    MODELPATH.parent.mkdir(exist_ok=True)
    joblib.dump((model, cat_cols, feature_names), MODELPATH)
    print("Model saved →", MODELPATH)

if __name__ == "__main__":
    main()
