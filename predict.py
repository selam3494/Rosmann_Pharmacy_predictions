# predict.py
import pandas as pd, joblib
from feature_utils import feature_engineer
from path_utils import DATA_DIR, MODEL_DIR

MODELPATH = MODEL_DIR / "rossmann_lgbm.pkl"
OUTDIR    = MODEL_DIR / "outputs"
OUTDIR.mkdir(exist_ok=True)

def main():
    model, _, _ = joblib.load(MODELPATH)
    needed_cols = model.feature_name_

    test  = pd.read_csv(DATA_DIR / "test.csv",  parse_dates=["Date"])
    store = pd.read_csv(DATA_DIR / "store.csv")
    df, _ = feature_engineer(test.merge(store, on="Store", how="left"))

    for col in needed_cols:
        if col not in df.columns:
            df[col] = 0
    X_test = df.drop(columns=["Date"])[needed_cols]

    preds = model.predict(X_test).clip(min=0)
    sub   = pd.DataFrame({"Id": test["Id"], "Sales": preds})
    path  = OUTDIR / "submission.csv"
    sub.to_csv(path, index=False)
    print("Submission written to:", path)

if __name__ == "__main__":
    main()
