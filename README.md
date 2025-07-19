# 🍔 Food Segmentation Model

## Live Demo 💻
![Demo](reports/figures/demo.gif)


> **Delicious pixels, smartly segmented!** Welcome to the lively repository of our Food Segmentation project! This end-to-end machine learning project segments food items from images with precision, offers an interactive API, and a frontend to boot.

Try out the live app: [Streamlit App 🔗](https://segmentation-frontend-289925381630.us-central1.run.app/)

---
## 🛂 Project Architecture

![Model Architecture](reports/figures/Architecture.jpeg)

This architecture represents the full pipeline:

* **Local Side**: Code versioning (Git), data/model tracking (DVC), PyTorch app orchestration via Hydra & Typer, debugging/profiling, and W\&B logging.

* **Cloud Side**: CI/CD via GitHub Actions → GCP Build → Docker artifact → Cloud Run hosting.

* **API & Load Test**: FastAPI app is hosted on Cloud Run, exposed to the end-user. Locust performs load testing.

* **Monitoring**: GCP Logging tracks logs, errors, and performance.

* **Prediction Flow**: End-user hits API → Prediction → Stored in GCP Bucket.

---

## 🚀 Project Overview

This project is a production-ready, MLOps-enabled food segmentation model built using a [cookie-cutter MLOps template](https://github.com/kkkamur07/cookie-cutter). It:

* Segments food items from images using deep learning
* Deploys a FastAPI / BentoML backend
* Serves predictions on a Streamlit frontend

 📄 **Here is our report for the exam:** [View Report](https://github.com/kkkamur07/food103seg-calories/blob/main/reports/README.md)
 
 🌐 **HTML version of the report:** [View HTML Report](https://github.com/kkkamur07/food103seg-calories/blob/main/reports/report.html)  

---

##  Project Structure

```
root/
├── configs/              # Configs for models, datasets, sweeps
├── src/
│   ├── app/              # FastAPI, BentoML, Streamlit
│   ├── segmentation/     # Core training logic
│   └── tests/            # Unit & integration tests
├── saved/                # Model weights, logs and figures.
├── notebooks/            # Jupyter notebooks for experiments
├── report/               # Report, figures, results
├── .github/              # CI/CD pipelines (GitHub Actions)
├── Dockerfile.*, docker-compose.yml  # Containerization
├── data.dvc              # Data versioning and tracking
├── wandb_runner.py       # W&B hyperparameter sweeping.
├── tasks.py              # Automation of CLI using invoke
├── pyproject.toml        # Python project metadata + build system
└── README.md             # You're here
```


---

## 🧵 How It Works

1. **Model Training**

   * We trained a **UNet** model using our `Food103Seg` dataset
   * The dataset contains **104 food classes**
   * Images are preprocessed and fed into the UNet model
   * Trained model W&B is versioned using **DVC** and exported via FastAPI

2. **API Development**

   * FastAPI / BentoML serves the model
   * `\segment` endpoint handles image uploads and returns segmentation masks

3. **Frontend**

   * Streamlit UI lets users upload images and see segmented output in real time

4. **Docs & CI/CD**

   * MkDocs for documentation.
   * GitHub Actions handle CI/CD workflows
   * DVC handles data and model versioning across development cycles

---

## 🚧 Installation

```bash
git clone https://github.com/kkkamur07/food103seg-calories
cd food103seg-calories
```

To run API:
```bash
uvicorn src.app.service:app --host 0.0.0.0 --port 8000 --reload
```

To run frontend:
```bash
streamlit run src/app/frontend.py
```

**Alternatively you can build the docker containers using**

```bash
docker-compose up --build
```
---

## 📊 Model Results

| Metric         | Value          |
| -------------- | -------------- |
| Accuracy       | 65%            |
| Inference Time | 100ms/image    |
| Classes        | 104 food items |


---

## 📑 Documentation

Full API and usage documentation available at: [Documentation](https://kkkamur07.github.io/food103seg-calories/)

---

## 🛠️ Tech Stack

* **Backend**: FastAPI, BentoML
* **Frontend**: Streamlit
* **Model**: UNet (PyTorch)
* **Dataset**: [Food103Seg](https://datasetninja.com/food-seg-103)
* **MLOps**: Cookie-cutter, Docker, GitHub Actions, **DVC**, GCP, WandB
* **Docs**: MkDocs

---

## ✅ CI/CD & Versioning

* **GitHub Actions** for automated testing and building of docker containers
* **DVC** for tracking datasets and model weights and biases
* **Docker** to ensure reproducibility and scalability of our work.
* **Pre-commit** hooks for code quality and consistency.
* **W\&B** for experiment tracking and sweeping
* **GCP** for building & deploying our docker containers

---
