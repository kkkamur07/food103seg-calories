# Data Module

This module handles the data loading from the datasets and data version control and preprocessing for food segmentation.

## DVC Setup and Configuration

### Data Storage Architecture

The project uses **DVC (Data Version Control)** with Google Cloud Storage for data versioning and management:

- **Data storage**: `gs://dvc-storage-sensor/`
- **Model storage**: `gs://food-segmentation-models/`

### Initial DVC Setup

```bash
# Install required tools
pip install dvc-gs

# List available GCS buckets
gsutil ls

# Add remote storage
dvc remote add -d remote_storage gs://dvc-storage-sensor/

# Configure version-aware storage
dvc remote modify remote_storage version_aware true

# Pull data from remote storage
dvc pull
```

### Common DVC Issues and Solutions

#### Known Problems Encountered

1. **Push Command Failures**
   ```bash
   # This command often fails
   dvc push --no-cache
   ```

2. **Remote Configuration Issues**
   ```bash
   # Remove problematic remotes
   dvc remote remove gcp_storage

   # Set correct default remote
   dvc remote default remote_storage
   ```

3. **Authentication Problems**
   - Ensure Google Cloud SDK is properly configured
   - Verify bucket permissions and access rights

#### Troubleshooting Commands

```bash
# List configured remotes
dvc remote list

# Check remote configuration
dvc remote list --show-origin

# Verify data status
dvc status

# Force refresh from remote
dvc fetch --all-commits
```

### Alternative Data Access

If DVC setup encounters persistent issues, you can:

1. **Direct GCS Access**
   ```bash
   # Access data directly from GCS buckets
   gsutil -m cp -r gs://dvc-storage-sensor/data/ ./data/
   ```

2. **Manual Dataset Download**
   - Download Food103 segmentation dataset from: https://paperswithcode.com/dataset/foodseg103
   - Extract and place in `data/` directory

### Data Module Integration


::: src.segmentation.loss

The data module integrates with DVC by:
- Automatically checking for DVC-tracked files
- Handling both local and remote data sources
- Providing fallback mechanisms when DVC is unavailable
- Managing data preprocessing pipelines with version control

### Best Practices

- Always verify DVC remote configuration before starting work
- Use `dvc status` to check data synchronization
- Keep `.dvc` files in version control
- Test data access in development environment before production deployment

This setup ensures reproducible data management while providing flexibility when DVC encounters configuration issues.
