# Lung Cancer Detection Dashboard — Fortune Hospital

A full-stack pulmonary histopathology screening dashboard: Flask API + CNN
(TensorFlow/Keras) inference + a clinical-style, hospital dashboard frontend.
Deployable on Render.

> **Educational project.** This tool is not validated for clinical use and must
> never be used for real diagnosis. Every API response is labeled with its
> `mode` (`trained_model` or `demo_heuristic`) so the UI can show an honest
> disclaimer.

## What's inside

```
lung-cancer-dashboard/
├── app.py                 # Flask app: routes + REST API
├── model/
│   ├── cnn_model.py        # CNN architecture (4 conv blocks) + train()
│   └── predict.py          # Inference: trained model OR demo-mode fallback
├── templates/index.html    # Dashboard shell
├── static/css/style.css    # Fortune Hospital visual theme
├── static/js/dashboard.js  # Nav, upload/predict flow, Chart.js graphs
├── data/history.json       # Auto-created scan history (small JSON store)
├── uploads/                # Uploaded scan images
├── requirements.txt
├── render.yaml              # Render Blueprint
└── Procfile
```

## How inference works

- 3 classes, matching the public **LC25000** lung histopathology dataset:
  `lung_benign_tissue`, `lung_adenocarcinoma`, `lung_squamous_cell_carcinoma`.
- If `model/lung_cnn.h5` exists, the Flask app loads it and serves real CNN
  predictions.
- If it doesn't (e.g. fresh clone, no GPU/training done yet), the app runs in
  **demo mode**: a deterministic, image-statistics-based heuristic (texture,
  edge density, intensity) produces varied, plausible-looking results so the
  whole dashboard — charts, history, confidence ring — is explorable end to
  end without needing a trained model on hand. Every demo result is clearly
  flagged as `demo_heuristic` in the JSON and in the UI's disclaimer box.

## Running locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Visit `http://localhost:5000`.

## Training a real model (optional)

1. Download the **LC25000** lung histopathology image set (search "LC25000
   lung and colon cancer histopathological images" — a common Kaggle
   dataset). It contains `lung_n` (benign), `lung_aca` (adenocarcinoma),
   `lung_scc` (squamous cell carcinoma) folders.
2. Arrange it as:
   ```
   data/
     train/
       lung_benign_tissue/*.jpeg
       lung_adenocarcinoma/*.jpeg
       lung_squamous_cell_carcinoma/*.jpeg
     val/
       lung_benign_tissue/*.jpeg
       lung_adenocarcinoma/*.jpeg
       lung_squamous_cell_carcinoma/*.jpeg
   ```
3. Uncomment `tensorflow-cpu` in `requirements.txt` and install it.
4. Train:
   ```bash
   python -m model.cnn_model
   ```
   This saves the best checkpoint to `model/lung_cnn.h5`. Restart the Flask
   app — it will auto-detect the checkpoint and switch to `trained_model`
   mode.

## Deploying to Render

**Option A — Blueprint (recommended)**
1. Push this folder to a GitHub repo.
2. In Render: **New → Blueprint**, point it at the repo. `render.yaml` sets
   up the web service automatically (`pip install -r requirements.txt`,
   `gunicorn app:app`).

**Option B — Manual web service**
1. New → Web Service → connect the repo.
2. Build command: `pip install -r requirements.txt`
3. Start command: `gunicorn app:app --bind 0.0.0.0:$PORT`
4. Runtime: Python 3.11

Note: the free Render tier's filesystem is ephemeral — uploaded images and
`data/history.json` reset on redeploy/restart. For persistent scan history in
production, swap the JSON store in `app.py` for a real database (e.g.
Render's managed Postgres) or Render's persistent disk add-on.

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/predict` | POST | multipart form: `image` (file), `patient_ref` (optional). Returns prediction + saves to history. |
| `/api/history?limit=N` | GET | Most recent N scans (default 25). |
| `/api/stats` | GET | Aggregate counts, average confidence, 14-day trend. |
| `/api/system-status` | GET | Whether a trained model or demo heuristic is currently serving. |

## Disclaimer

This project is for portfolio/educational purposes (deep learning + full-stack
deployment demonstration). It is **not a medical device**, has not been
clinically validated, and must not be used to inform real diagnostic or
treatment decisions.
