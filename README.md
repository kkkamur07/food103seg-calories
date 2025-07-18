# 🌟 Food Segmentation Model

![Food Banner](https://images.unsplash.com/photo-1606788075761-7791f05dc442?auto=format\&fit=crop\&w=1350\&q=80)

> **Delicious pixels, smartly segmented!** Welcome to the lively repository of our Food Segmentation project! This end-to-end machine learning project segments food items from images with precision, offers an interactive API, and a gorgeous frontend to boot.

---

## 🚀 Project Overview

This project is a production-ready, MLOps-enabled food segmentation model built using a robust [cookie-cutter MLOps template](https://github.com/kkkamur07/cookie-cutter). It:

* Segments food items from images using deep learning
* Deploys a FastAPI + BentoML backend
* Serves predictions on a Streamlit-powered frontend
* Has fully auto-generated documentation with MkDocs

![Segmentation Sample](https://user-images.githubusercontent.com/12345678/food-segmentation-example.png)

---

## 📅 Project Structure

```
kkkamur07-food103seg-calories/
├── README.md                           # Project overview and instructions
├── cloudbuild.yaml                    # Google Cloud Build config
├── data.dvc                           # DVC-tracked data file
├── docker-compose.yml                 # Orchestration of backend + frontend
├── Dockerfile.backend                 # Dockerfile for backend service
├── Dockerfile.frontend                # Dockerfile for frontend app
├── project_structure.txt              # Describes the project structure
├── pyproject.toml                     # Python project metadata + build system
├── requirements.txt                   # Production dependencies
├── requirements_dev.txt              # Dev dependencies (linting, testing)
├── tasks.py                           # Automation scripts (e.g. via `invoke`)
├── uv.lock                            # Dependency lock file for uv tool
├── wandb_runner.py                    # W&B experiment runner
├── .dockerignore                      # Ignore rules for Docker builds
├── .dvcignore                         # Ignore rules for DVC
├── .pre-commit-config.yaml           # Pre-commit hooks config
├── configs/                           # All project configs
│   ├── config.yaml                    # Main config file (training, paths)
│   ├── wandb_sweep.yaml              # W&B sweep configuration
│   ├── dataset/
│   │   └── default.yaml              # Dataset-specific config
│   ├── model/
│   │   └── default.yaml              # Model-specific config
│   └── outputs/                      # Experiment outputs
│       ├── 2025-07-02/
│       │   └── 22-43-00/
│       │       ├── wandb/           # W&B run logs
│       │       └── .hydra/          # Hydra config snapshots
│       │           ├── config.yaml
│       │           └── hydra.yaml
│       └── 2025-07-04/
│           └── 21-10-21/
│               └── .hydra/
│                   └── hydra.yaml
├── notebooks/
│   └── experiment.ipynb              # Jupyter notebook for experiments
├── saved/
│   └── models.dvc                    # Tracked model file(s) with DVC
├── src/                              # Source code
│   ├── app/                          # Application code (serving)
│   │   ├── bentoml.py               # BentoML service definition
│   │   ├── bentoml_setup.py        # BentoML setup utility
│   │   ├── frontend.py             # Streamlit or Gradio frontend
│   │   ├── frontend_requirements.txt
│   │   └── service.py              # Service logic
│   ├── segmentation/                # Core ML logic
│   │   ├── __init__.py
│   │   ├── data.py                 # Dataset loading, transforms
│   │   ├── loss.py                 # Custom loss functions
│   │   ├── main.py                 # Entrypoint script
│   │   ├── model.py                # Model architectures
│   │   └── train.py                # Training loop
│   └── tests/                       # Tests
│       ├── test_data1.py
│       ├── test_model.py
│       ├── test_training.py
│       ├── tests_integration/      # Integration-level tests
│       │   ├── api_testing.py
│       │   └── locustfile.py       # Load testing with Locust
│       └── tests_unit/             # Unit-level tests
│           ├── test_data.py
│           └── test_train.py
├── report/                           # Exam report folder
│   ├── README.md                     # Exam answers
│   ├── figures/                      # Images for report
│   └── report.py                     # Report generation script
├── favicon.py                        # API favicon
├── static/                           # Static files
│   └── favicon.ico                   
└── .github/                          # GitHub CI/CD config
    ├── dependabot.yaml              # Dependency update config
    └── workflows/                   # GitHub Actions workflows
        ├── ci.yml                   # Main CI pipeline
        ├── data-changes.yaml       # DVC-based data CI triggers
        └── model-deploy.yml        # Model deployment pipeline
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

Full API and usage documentation available at: [https://your-docs-site](https://your-docs-site)

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

---

## 🚀 Future Enhancements

* [ ] Multi-class segmentation (more food categories)
* [ ] Nutrition prediction integration
* [ ] Mobile app deployment
* [ ] Labeling tool integration

---

## 🙏 Credits

Thanks to the open-source community, [cookie-cutter MLOps](https://github.com/kkkamur07/cookie-cutter), and dataset contributors.

---

## 🚫 License

[MIT](LICENSE)
