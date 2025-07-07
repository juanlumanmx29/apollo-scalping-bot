# Model Setup for Deployment

## Important Note

The trained ensemble model (`models/ensemble_model.joblib`) is too large for GitHub (162MB).

## For Local Development

The model exists locally and the backend will load it automatically when running locally.

## For Deployment

### Option 1: Upload Model Manually
1. After deploying to Render, use their file upload feature or SSH access to upload the model file to `/app/models/ensemble_model.joblib`

### Option 2: Create Simpler Model
1. Train a smaller model with fewer features
2. Use feature selection to reduce model size
3. Compress the model using joblib compression

### Option 3: Model Storage Service
1. Upload model to cloud storage (AWS S3, Google Cloud Storage)
2. Download model on startup in the backend
3. Cache locally for performance

## Current Workaround

The backend includes error handling for missing model files. If no model is found, the API will:
- Return a default probability of 0.5
- Log appropriate error messages
- Continue functioning for testing purposes

## Training New Model

To create a deployable model:

```bash
# Run locally with your data
python scripts/train_models.py
python scripts/ensemble_models.py

# The model will be saved to models/ensemble_model.joblib
# Upload this file to your deployed backend
```