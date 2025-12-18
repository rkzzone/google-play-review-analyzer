# Setup Instructions for Deployment

## Model Files Required

This application uses an **Indonesian sentiment model** fine-tuned on SMSA dataset. Since model files are too large for GitHub, you have these options:

### Option 1: Use HuggingFace Hub (Recommended for deployment)
The app automatically loads the model from HuggingFace Hub:
- **Model**: [rkkzone/roberta-sentiment-indonesian-playstore](https://huggingface.co/rkkzone/roberta-sentiment-indonesian-playstore)
- **Language**: Indonesian (Bahasa Indonesia)
- **Size**: ~475MB
- **No manual download needed** - app handles it automatically

### Option 2: Train the model yourself
1. Download the SMSA dataset from [GitHub](https://github.com/IndoNLP/indonlu)
2. Run `training_sentiment_id.ipynb` notebook
3. Model will be saved in `saved_model_id/` directory
4. Upload to your own HuggingFace repository using the upload cells in the notebook

### Option 3: Use local model files
Upload the `saved_model_id/` folder to your repository using Git LFS:
```bash
git lfs track "saved_model_id/*.safetensors"
git lfs track "saved_model_id/*.bin"
git add .gitattributes
git add saved_model_id/
git commit -m "Add Indonesian model files"
git push
```

## Local Development
```bash
pip install -r requirements.txt
streamlit run app.py
```
