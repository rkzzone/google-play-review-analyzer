# Setup Instructions for Deployment

## Model Files Required

This application requires a trained RoBERTa sentiment model. Since model files are too large for GitHub, you need to:

### Option 1: Train the model yourself
1. Download the dataset from Kaggle
2. Run `training_sentiment.ipynb` notebook
3. Model will be saved in `saved_model/` directory

### Option 2: Download pre-trained model
Upload the `saved_model/` folder to your repository or use Git LFS:
```bash
git lfs track "saved_model/*.safetensors"
git lfs track "saved_model/*.bin"
git add .gitattributes
git add saved_model/
git commit -m "Add model files"
git push
```

### Option 3: Use Streamlit secrets (Recommended for deployment)
Host your model on Hugging Face and load it dynamically in the app.

## Local Development
```bash
pip install -r requirements.txt
streamlit run app.py
```
