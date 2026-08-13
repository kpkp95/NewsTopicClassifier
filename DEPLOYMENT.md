# Optional Google Cloud Run Deployment

Deployment is optional and is not required for the course submission. Complete
and submit the academic project first. The steps below can be used afterward to
publish the Gradio application.

## What the deployment contains

The container includes only:

- `app.py`
- The inference modules in `src/`
- The saved files in `models/best_model/`
- The packages in `requirements-deploy.txt`

Training, EDA, notebooks, baseline files, and generated figures are excluded
from the deployment image.

## 1. Confirm the saved model exists

The following files must be available before building the image:

```text
models/best_model/encoder/model.safetensors
models/best_model/encoder/config.json
models/best_model/tokenizer/tokenizer.json
models/best_model/classifier.pt
models/best_model/project_config.json
```

## 2. Test the container locally

Run these commands from the project root:

```powershell
docker build -t ag-news-classifier .
docker run --rm -p 8080:8080 -e PORT=8080 ag-news-classifier
```

Open `http://localhost:8080` and test at least one prediction.

## 3. Prepare Google Cloud

Install the Google Cloud CLI, sign in, select a project, and enable the required
services:

```powershell
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

Replace `YOUR_PROJECT_ID` with the ID of your Google Cloud project.

## 4. Deploy from the project source

```powershell
gcloud run deploy ag-news-classifier `
    --source . `
    --region YOUR_REGION `
    --allow-unauthenticated `
    --memory 2Gi `
    --cpu 2 `
    --timeout 300
```

Replace `YOUR_REGION` with the Cloud Run region you want to use. The memory and
CPU values are reasonable starting settings for CPU inference and can be
adjusted after testing.

Cloud Run supplies a `PORT` environment variable. `app.py` detects it, listens
on that port, and binds to `0.0.0.0`. Local execution continues to use
`127.0.0.1:7860` when `PORT` is not present.

## 5. Check the deployed service

After deployment, the command prints the public service URL. Open it and test:

1. One clear Sports example
2. One clear World example
3. One Business/Sci-Tech ambiguous example
4. The displayed confidence and four class probabilities

## Official Google documentation

- Container runtime contract: https://cloud.google.com/run/docs/container-contract
- Deploy from source: https://cloud.google.com/run/docs/deploying-source-code
- Build containers: https://cloud.google.com/run/docs/building/containers
