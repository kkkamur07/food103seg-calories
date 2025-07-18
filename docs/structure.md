```
kkkamur07-food103seg-calories/
├── README.md                          # Project overview and instructions
├── cloudbuild.yaml                    # Google Cloud Build config
├── data.dvc                           # DVC-tracked data file
├── docker-compose.yml                 # Orchestration of backend + frontend
├── Dockerfile.backend                 # Dockerfile for backend service
├── Dockerfile.frontend                # Dockerfile for frontend app
├── pyproject.toml                     # Python project metadata + build system
├── requirements.txt                   # Production dependencies
├── requirements_dev.txt               # Dev dependencies (linting, testing)
├── tasks.py                           # Automation scripts (e.g. via `invoke`)
├── uv.lock                            # Dependency lock file for uv tool
├── wandb_runner.py                    # W&B experiment runner for hyperparameter sweep
├── .dockerignore                      # Ignore rules for Docker builds
├── .dvcignore                         # Ignore rules for DVC
├── .pre-commit-config.yaml            # Pre-commit hooks config
├── configs/                           # All project configs
│   ├── config.yaml                    # Main config file (training, paths)
│   ├── wandb_sweep.yaml               # W&B sweep configuration
│   ├── dataset/
│   │   └── default.yaml               # Dataset-specific config
│   ├── model/
│   │   └── default.yaml               # Model-specific config
│   └── outputs/                       # Experiment outputs
│       ├── 2025-07-02/
│       │   └── 22-43-00/
│       │       ├── wandb/             # W&B run logs
│       │       └── .hydra/            # Hydra config snapshots
│       │           ├── config.yaml
│       │           └── hydra.yaml
│       └── 2025-07-04/
│           └── 21-10-21/
│               └── .hydra/
│                   └── hydra.yaml
├── notebooks/
│   └── experiment.ipynb              # Jupyter notebook for experiments
├── saved/
│   └── models.dvc                    # Tracked model weights & biases file(s) with DVC
├── src/                              # Source code
│   ├── app/                          # Application code (serving)
│   │   ├── bentoml.py                # BentoML service definition
│   │   ├── bentoml_setup.py          # BentoML setup utility
│   │   ├── frontend.py               # Streamlit or Gradio frontend
│   │   ├── frontend_requirements.txt
│   │   └── service.py                # Service logic
│   ├── segmentation/                 # Core ML logic
│   │   ├── __init__.py
│   │   ├── data.py                   # Dataset loading, transforms
│   │   ├── loss.py                   # Custom loss functions
│   │   ├── main.py                   # Entrypoint script
│   │   ├── model.py                  # Model architectures
│   │   └── train.py                  # Training loop
│   └── tests/                        # Tests
│       ├── test_data1.py
│       ├── test_model.py
│       ├── test_training.py
│       ├── tests_integration/        # Integration-level tests
│       │   ├── api_testing.py
│       │   └── locustfile.py         # Load testing with Locust
│       └── tests_unit/               # Unit-level tests
│           ├── test_data.py
│           └── test_train.py
│── report/                           # Exam report folder
│   ├── README.md                     # Exam answers
│   ├── figures/                      # Images for report
│   └── report.py                     # Report generation script
├── favicon.py                        # API favicon
├── static/                           # Static files
│   ├── favicon.ico
└── .github/                          # GitHub CI/CD config
    ├── dependabot.yaml               # Dependency update config
    └── workflows/                    # GitHub Actions workflows
        ├── ci.yml                    # Main CI pipeline
        ├── data-changes.yaml         # DVC-based data CI triggers
        └── model-deploy.yml          # DVC-based model W&B CI triggers
```
