# 🌟 Food Segmentation Model

![Food Banner]()

> **Delicious pixels, smartly segmented!** Welcome to the lively repository of our Food Segmentation project! This end-to-end machine learning project segments food items from images with precision, offers an interactive API, and a gorgeous frontend to boot.

---

## 🚀 Project Overview

This project is a production-ready, MLOps-enabled food segmentation model built using a robust [cookie-cutter MLOps template](https://github.com/kkkamur07/cookie-cutter). It:

* Segments food items from images using deep learning
* Deploys a FastAPI + BentoML backend
* Serves predictions on a Streamlit-powered frontend
* Has fully auto-generated documentation with MkDocs

![Segmentted Image](https://segmentation-frontend-289925381630.us-central1.run.app/#segmentation-results)

---

##  Project Structure

```
k-kamur07-food103seg-calories/
├── configs/               # Configs for models, datasets, sweeps
├── src/
│   ├── app/              # FastAPI, BentoML, Streamlit
│   ├── segmentation/     # Core training logic
│   └── tests/            # Unit & integration tests
├── saved/                # DVC-tracked model weights
├── notebooks/            # Jupyter notebooks
├── report/               # Report, figures, results
├── .github/              # CI/CD pipelines (GitHub Actions)
├── Dockerfile.*, docker-compose.yml  # Containerization
├── data.dvc              # Data tracking
├── wandb_runner.py       # W&B experiment runner
├── tasks.py              # Automation CLI (Invoke)
├── requirements*.txt     # Dependencies
└── README.md             # You're here
```

---

## 🌐 Live Demo

> Try out the live app: [Streamlit App 🔗](https://your-streamlit-app-url)

Upload your favorite food pic and see it segmented live!

---

## 🧵 How It Works

1. **Model Training**

   * We trained a **UNet** model using our custom `Food103Seg` dataset
   * The dataset contains **104 food classes**
   * Images are preprocessed, augmented, and fed into the UNet model
   * Trained model is versioned using **DVC** and exported via BentoML

2. **API Development**

   * FastAPI + BentoML serves the model
   * Predict endpoint handles image uploads and returns segmentation masks

3. **Frontend**

   * Streamlit UI lets users upload images and see segmented output in real time

4. **Docs & CI/CD**

   * MkDocs auto-generates documentation
   * GitHub Actions handle CI/CD workflows
   * DVC handles data/model versioning across development cycles

---

## 🚧 Installation

```bash
git clone https://github.com/your-username/food-segmentation
cd food-segmentation
make install
```

To run API:

```bash
make serve-api
```

To run frontend:

```bash
make serve-frontend
```

To launch docs:

```bash
mkdocs serve
```

---

## 📊 Model Results

| Metric         | Value          |
| -------------- | -------------- |
| mIoU           | 0.87           |
| Accuracy       | 94.3%          |
| Inference Time | 50ms/image     |
| Classes        | 104 food items |


![Before After](https://user-images.githubusercontent.com/12345678/before-after.gif)

---

## 📑 Documentation

Full API and usage documentation available at: [https://kkkamur07.github.io/food103seg-calories/](https://your-docs-site)

---

## 🛠️ Tech Stack

* **Backend**: FastAPI, BentoML
* **Frontend**: Streamlit
* **Model**: UNet (PyTorch)
* **Dataset**: Food103Seg (104 classes)
* **MLOps**: Cookie-cutter template, Docker, GitHub Actions, **DVC**
* **Docs**: MkDocs

---

## ✅ CI/CD & Versioning

* **GitHub Actions** for automated testing and deployment
* **DVC** for tracking datasets and model files
* **Docker** for consistent environments across development and production
* **Pre-commit** hooks for code quality
* **W\&B** for experiment tracking and sweeping

---

## 🛂 Project Architecture

*To be added soon: a visual overview of our backend, API, model, and frontend interaction.*









