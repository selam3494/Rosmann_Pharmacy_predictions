# app.py
import streamlit as st, pandas as pd, joblib
from feature_utils import feature_engineer
from path_utils import DATA_DIR, MODEL_DIR

MODELPATH = MODEL_DIR / "rossmann_lgbm.pkl"

st.set_page_config(page_title="Rossmann Forecaster", layout="wide")
st.title("💊 Rossmann Pharmaceuticals – 6‑week Sales Forecaster")

@st.cache_data
def load_artifacts():
    return joblib.load(MODELPATH)           # model, cat_cols, feature_names

model, cat_cols, feature_names = load_artifacts()
needed_cols = model.feature_name_           # canonical 22‑column list

# ---------------- Batch uploader ----------------
st.sidebar.header("Batch prediction")
csv_file = st.sidebar.file_uploader("Upload a CSV in the *test.csv* schema")

if csv_file:
    test_df  = pd.read_csv(csv_file, parse_dates=["Date"])
    store_df = pd.read_csv(DATA_DIR / "store.csv")
    df, _    = feature_engineer(test_df.merge(store_df, on="Store", how="left"))

    # insert any missing training columns with 0, drop extras
    for col in needed_cols:
        if col not in df.columns:
            df[col] = 0
    X_test = df.drop(columns=["Date"])[needed_cols]

    preds = model.predict(X_test).clip(min=0)
    out   = pd.DataFrame({"Id": test_df["Id"], "Sales": preds})

    st.write("Preview", out.head())
    st.download_button(
        "Download submission.csv",
        out.to_csv(index=False).encode(),
        "submission.csv",
        "text/csv",
    )

# ---------------- Single‑day what‑if ------------
st.sidebar.header("🛒 Single‑day what‑if")
with st.sidebar.form("single"):
    sid     = st.number_input("Store ID", 1, 1115, 1)
    day     = st.date_input("Date")
    promo   = st.checkbox("Promo", value=False)
    school  = st.checkbox("SchoolHoliday", value=False)
    holiday = st.selectbox(
        "StateHoliday", ["0", "a", "b", "c"], index=0,
        format_func=lambda x: {"0":"None","a":"Public","b":"Easter","c":"Christmas"}[x]
    )
    submit = st.form_submit_button("Predict")

    if submit:
        base = {
            "Store": sid,
            "Date": pd.to_datetime(day),
            "DayOfWeek": pd.to_datetime(day).dayofweek + 1,
            "Open": 1,
            "Promo": int(promo),
            "SchoolHoliday": int(school),
            "StateHoliday": holiday,
        }
        static = (
            pd.read_csv(DATA_DIR / "store.csv")
              .query("Store == @sid")
              .iloc[0]
              .to_dict()
        )
        sample, _ = feature_engineer(pd.DataFrame([{**base, **static}]))

        for col in needed_cols:
            if col not in sample.columns:
                sample[col] = 0
        X_single = sample.drop(columns=["Date"])[needed_cols]

        pred = model.predict(X_single)[0]
        st.metric(f"Predicted Sales for store {sid}", f"€{pred:,.0f}")
