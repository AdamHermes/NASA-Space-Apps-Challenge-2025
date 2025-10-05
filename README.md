# Light Curve ML & Visualization — Quick Start

Short instructions to get the project running locally. This README is intentionally brief and non-technical.
The Web Interface relating to this project is available at: https://github.com/lammhuyy/NASA-Space-Apps-Challenge-2025-frontend

1) Install

Create and activate a Python environment, then install requirements:

```bash
conda create -n lc-env python=3.13 -y
conda activate lc-env
pip install -r requirements.txt
```

2) Start the server

Run the FastAPI app locally (development mode):

```bash
uvicorn app.main:app --reload
```

Open your browser to http://127.0.0.1:8000. The API docs are at http://127.0.0.1:8000/docs.

3) Typical workflow (high level)

- Upload one or more CSV files via the Data endpoint (these are stored in `app/storage/uploaded_csvs`).
- Process the uploaded CSV(s) using the provided processing endpoint; this generates cleaned files in `app/storage/processed_csvs`.
- Use the ML endpoints to run predictions on processed CSVs or to trigger model retraining.
- Use the Visualization endpoints to query hostnames, Kepler IDs, and to stream or generate light-curve videos.

4) Where files live

- Uploaded CSVs: `app/storage/uploaded_csvs`
- Processed CSVs and splits: `app/storage/processed_csvs`
- Saved models: `app/storage/weights`
- Saved scalers: `app/storage/scalers`
- Videos: `app/videos`

5) If something goes wrong

- Check server logs printed where you ran `uvicorn`.
- Make sure the `app/storage/*` directories exist and are writable.
- If an endpoint returns a parsing error, try cleaning CSV headers or using a small sample file first.

6) Want more detail?

The project contains code under `app/` (routers, services, models). If you want a more detailed developer README (endpoints, examples, Dockerfile), tell me and I will expand this file.

---

Happy hacking — tell me what example requests or extra docs you'd like next.
# NASA Space Apps Challenge — Light Curve ML & Visualization Service

This repository contains a FastAPI-based backend and related ML/processing utilities used for ingesting, processing, training, and running inference on exoplanet light curve datasets. It was created as part of a NASA Space Apps Challenge project.

Key features
- FastAPI server with endpoints for data upload, processing, ML inference, model training, and visualization (light curves & videos).
- Support for merging and processing CSV datasets, scaling features, and running pre-trained sklearn models.
- Utilities and models for training ensemble classifiers and a PyTorch CNN+Attention model for light-curve classification.

Directory overview

- `app/` — main application code
  - `main.py` — FastAPI application entrypoint and router wiring
  - `routes.py` — small root router (/, /ping)
  - `routers/` — API routers grouped by responsibility
    - `data.py` — endpoints to upload/process CSVs and list available CSVs/models
    - `ml_routers.py` — inference endpoints (CSV upload + inference APIs)
    - `train_routers.py` — endpoints to trigger retraining on selected processed CSVs
    - `visualization.py` — find hostnames/kepid, return lightcurve data and videos
    - `light_curve.py` — utilities to fetch and render light-curve videos (MAST DV data)
  - `models/` — model training and dataset code
    - `models.py` — sklearn-based training helpers for ensemble models
    - `train.py` — PyTorch training CLI for CNN+Attention model
    - `dataset_lightcurve.py` — dataset loader for light-curve CSV files
    - `cnn_attention.py` — PyTorch model architecture
  - `service/` — business logic and helpers
    - `data/` — CSV merging, processing helpers (process_koi.py, data_manage.py)
    - `ml/` — inference code, model loading
- `app/storage/` — persisted artifacts and working data
  - `uploaded_csvs/` — user-uploaded CSV files
  - `processed_csvs/` — processed CSV outputs and train/test splits
  - `scalers/` — saved feature scalers per model family
  - `weights/` — saved trained model pickles grouped by algorithm
  - `tces/`, `videos/`, other helper folders
- `scripts/` — helper scripts for dataset download & prep
- `data/` — example data files (e.g., `cumulative.csv`)

Quickstart (development)

1. Create a Python environment (recommended: conda)

```bash
conda create -n lc-env python=3.13 -y
conda activate lc-env
pip install -r requirements.txt
```

2. Run the FastAPI development server (reload enabled)

```bash
uvicorn app.main:app --reload
```

3. The API will be available at `http://127.0.0.1:8000` and automatic OpenAPI docs at `http://127.0.0.1:8000/docs`.

Important endpoints (summary)

- GET `/` — simple health/message endpoint
- GET `/ping` — simple ping/status

- Data router (`/data`)
  - POST `/data/upload_csv/` — upload a CSV file (multipart form). Stores file in `app/storage/uploaded_csvs`.
  - POST `/data/process_csv/` — process one or many uploaded CSVs (calls `process_koi`).
  - GET `/data/current_models` — list available model folders under `app/storage/weights`.
  - GET `/data/current_csvs` — list CSV files in `app/storage/uploaded_csvs`.

- ML router (`/ml`)
  - POST `/ml/predict` — accepts CSV upload for batch prediction (CSV content read and passed to inference).
  - POST `/ml/inference/` — run inference by providing model_type, model_name, and a list of processed CSV filenames.
  - POST `/ml/inference_new_csv_files/` — similar to inference but for a different CSV location/name param.

- Training router (`/train`)
  - POST `/train/retrain/` — trigger model retraining using processed CSVs. Supply form fields for train/test filenames, scaler path, model name & hyperparams.

- Visualization (`/visualization`) and Light-curve (`/light_curve`) routers
  - `/visualization/hostnames`, `/visualization/hostids`, `/visualization/find_by_hostname/{hostname}`, `/visualization/find_by_kepid/{kepid}` — CSV-based lookups on cumulative dataset.
  - `/visualization/lightcurve/{star_id}/{tce_num}` — fetch lightcurve data from MAST for a given star & TCE (may require network access).
  - `/light_curve/get_video_light_curve/{kepler_id}` and related helpers — create or stream mp4 visualizations based on DV JSONs and processed light curves.

Notes and developer tips

- File locations: the API assumes relative paths such as `app/storage/uploaded_csvs`, `app/storage/processed_csvs`, and `app/storage/weights`. Make sure these directories exist and have correct permissions.
- Models and scalers: trained sklearn models are saved as pickles in `app/storage/weights/<model_name>/`. Example model pickles and scaler files are present under `app/storage/scalers/` and `app/storage/weights/`.
- Large files: video generation and training may be resource intensive. Run those operations on machines with sufficient CPU/RAM, and for training enable GPU if using the PyTorch training path.

Testing tips

- The repo includes `test.py` (quick checks) and `scripts/` for dataset preparation. Use small CSVs for quick validation.
- Use the interactive docs (`/docs`) to try uploading CSVs and calling endpoints.

Troubleshooting

- Common issues
  - Missing model files: ensure `app/storage/weights/<model_name>` contains model pickle(s) expected by `service/ml/inference.py`.
  - CSV parsing errors: `pd.read_csv(..., comment="#")` is used in several places; ensure files do not contain unexpected formats or use proper comment lines.
  - Python version: project was developed with Python 3.13 in mind; ensure your environment matches.

Next steps / improvements

- Add unit tests for routers and service functions (pytest).
- Add a small CLI wrapper to run common workflows (process -> train -> inference -> visualize).
- Add Dockerfile and compose for easier local deployment.

License

This project is provided as-is for the NASA Space Apps Challenge. Add a license file if you intend to open-source it publicly.

Contact

For questions about the code, see the files under `app/` or contact the repository owner.
