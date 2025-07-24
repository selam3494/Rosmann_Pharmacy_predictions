Below is a **drop‑in `README.md`** you can copy to the root of **`Rosmann_Pharmacy_prediction/`**.
It walks a new user from environment setup → training → prediction → Streamlit UI, and it embeds the three screenshots you showed.

---

```markdown
# 💊 Rossmann Pharmaceuticals – 6‑Week Sales Forecaster

End‑to‑end repo for predicting daily sales of 1 115 Rossmann drugstores
using LightGBM.  
You can:

1. **Train** a model from the original Kaggle CSVs.  
2. **Generate** a submission file (`submission.csv`).  
3. **Explore** forecasts with a Streamlit dashboard.

---

## Project layout

```

Rosmann\_Pharmacy\_prediction/
├── data/                 ← train.csv, test.csv, store.csv
├── models/               ← rossmann\_lgbm.pkl (created by training)
├── outputs/              ← submission.csv (created by predict.py)
├── path\_utils.py
├── feature\_utils.py
├── clean\_and\_train.py
├── predict.py
└── app.py

````

(*`models/` and `outputs/` are auto‑created if missing.*)

---

## 1. Set up the environment

```bash
conda create -n rossmann python=3.10 -y
conda activate rossmann
pip install pandas scikit-learn lightgbm streamlit joblib
````

---

## 2. Train the model

```bash
# run from repo root
python clean_and_train.py
```

The script:

* merges **train.csv** + **store.csv**
* engineers calendar / promo features
* fits LightGBM
* writes `models/rossmann_lgbm.pkl`

<p align="center">
  <img src="docs/training_rmspe.png" width="600">
</p>

*(RMSPE \~ 0.128 on a CPU laptop – your log will look similar.)*

---

## 3. Build a Kaggle submission (optional)

```bash
python predict.py
```

Creates `models/outputs/submission.csv` – ready to upload.

---

## 🌐 4. Launch the dashboard

```bash
streamlit run app.py
```

Open the URL Streamlit prints (default **[http://localhost:8501](http://localhost:8501)**) and you’ll see:

| Batch upload & what‑if sidebar             | Blank main panel (waiting for a batch file) | Example single‑store prediction                 |
| ------------------------------------------ | ------------------------------------------- | ----------------------------------------------- |
| <img src="docs/sidebar.png"   width="220"> | <img src="docs/blank_main.png" width="220"> | <img src="docs/predicted_4200.png" width="220"> |

### 4‑a. Batch mode

Upload a CSV in the **test.csv** schema (same columns, no `Sales`).
Download the auto‑generated `submission.csv`.

### 4‑b. Single‑day what‑if

Pick a store, date, promo / holiday flags – get an instant forecast.

---

## ❓ Troubleshooting

| Problem                                 | Fix                                                                                                                                                                               |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `FileNotFoundError ... data/*.csv`      | Make sure **train.csv**, **test.csv**, **store.csv** are inside `data/`.                                                                                                          |
| `LightGBMError: number of features ...` | You trained on different columns; re‑run `clean_and_train.py` **or** ensure your input file contains the same 22 features (the app now auto‑adds any missing columns with zeros). |
| `ModuleNotFoundError: lightgbm`         | `pip install lightgbm` or use Python ≤3.10 where wheels are available.                                                                                                            |

---

## 📜 License

This repo is provided **as‑is** for educational purposes. Rossmann dataset © Kaggle competition authors.

```

---

### Where to put the images

1. Create a folder `docs/` at repo root.
2. Save the three screenshots there with the exact names used above:

```

docs/
├── training\_rmspe.png      ← terminal training log
├── sidebar.png             ← Streamlit sidebar screenshot
├── blank\_main.png          ← empty main screen
└── predicted\_4200.png      ← single‑day forecast example

```

*(Rename your PNGs or adjust the `<img src="...">` paths if you want different filenames.)*

After that, GitHub will render the README exactly like the preview above.
::contentReference[oaicite:0]{index=0}
```
