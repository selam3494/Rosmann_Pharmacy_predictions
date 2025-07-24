# Rossmann Pharmaceuticals – 6‑Week Sales Forecaster

This repository contains an end‑to‑end pipeline for predicting daily sales for 1,115 Rossmann drugstores.  
What you can do:

1. **Train** a LightGBM model on the original Kaggle dataset.  
2. **Generate** a `submission.csv` for Kaggle scoring.  
3. **Explore** forecasts interactively with a Streamlit dashboard.

---

## Project layout

```

Rosmann\_Pharmacy\_prediction/
├── data/                 # train.csv, test.csv, store.csv
├── models/               # rossmann\_lgbm.pkl (created by training)
├── outputs/              # submission.csv (created by predict.py)
├── path\_utils.py
├── feature\_utils.py
├── clean\_and\_train.py
├── predict.py
└── app.py

````

(`models/` and `outputs/` are created automatically if missing.)

---

## 1  Set up the environment

```bash
conda create -n rossmann python=3.10 -y
conda activate rossmann
pip install -r requirements.txt
````

<details>
<summary><code>requirements.txt</code> content</summary>

```text
pandas>=2.1.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
streamlit>=1.34.0
joblib>=1.3.0
```

</details>

---

## 2  Train the model

```bash
# run from the repo root
python clean_and_train.py
```

The script

* merges **train.csv** and **store.csv**
* engineers calendar/promo features
* fits LightGBM
* saves `models/rossmann_lgbm.pkl`

Screenshot of a successful run:

<img src="Screenshot from 2025-07-24 15-59-46.png" width="700">

---

## 3  Create a Kaggle submission (optional)

```bash
python predict.py
```

A ready‑to‑upload `submission.csv` appears in `models/outputs/`.

---

## 4  Launch the Streamlit dashboard

```bash
streamlit run app.py
```

Streamlit typically starts at [http://localhost:8501](http://localhost:8501).
What you’ll see:

| Sidebar (batch + what‑if)                                       | Blank main panel                                                | Example single‑store result     |
| --------------------------------------------------------------- | --------------------------------------------------------------- | ------------------------------- |
| <img src="Screenshot from 2025-07-24 16-12-00.png" width="230"> | <img src="Screenshot from 2025-07-24 16-12-34.png" width="230"> | *(prediction shown in sidebar)* |

### 4a  Batch mode

Upload a CSV that follows the **test.csv** schema (same columns minus `Sales`).
Download the generated `submission.csv`.

### 4b  Single‑day what‑if

Choose store, date, and flags (promo, holiday) to see an instant forecast.

---

## Troubleshooting

| Problem                                 | Fix                                                                                                                    |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `FileNotFoundError ... data/*.csv`      | Ensure **train.csv**, **test.csv**, **store.csv** are in the `data/` folder.                                           |
| `LightGBMError: number of features ...` | Retrain with `clean_and_train.py`, or keep input columns consistent; the app now auto‑adds missing columns with zeros. |
| `ModuleNotFoundError: lightgbm`         | `pip install lightgbm` or use Python ≤ 3.10 where wheels are available.                                                |

---

## License

Provided as‑is for educational purposes. Rossmann dataset © Kaggle competition authors.

````

### `requirements.txt`

A minimal file has been included in the README. Save that text to `requirements.txt` at repo root so others can run:

```bash
pip install -r requirements.txt
````

Everything is now documented and reproducible.
