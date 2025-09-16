
# 🚀 JobMatch AI – Career Recommendation System

![JobMatch AI – Thumbnail](thumbnail/thumbnail_image)


A simple end-to-end demo for job title recommendations using classical ML. Train a `RandomForestClassifier` on text features, then serve predictions via a Streamlit app.

## ✨ Features
- **Training**: Build model from tabular/text data (company, skills, description)
- **Serving**: Interactive UI with Streamlit
- **Artifacts**: Saves `job_recommendation_model.pkl`, `vectorizer.pkl`, `label_encoder.pkl`

## 🗂️ Project Structure
```
app/
  app1.py           # Streamlit UI
model/
  al.py            # Training script (creates .pkl artifacts)
data/
  raw_data/
    fake_news_dataset.csv  # Sample placeholder (not used by current training script)
README.md
```

## 📦 Requirements
See `requirements.txt`. Install with:
```bash
pip install -r requirements.txt
```

Contents:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
streamlit
```

## 🧰 Setup
1. Ensure Python 3.9+ is installed.
2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
# Windows PowerShell
. .venv\Scripts\Activate.ps1
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📝 Data Notes
- The training script `model/al.py` currently reads a dataset from an absolute path:
  - `C:\Users\indur\Downloads\job_recommendation_dataset.csv`
- Update this path inside `model/al.py` to point to your dataset, or place your CSV in the repo and use a relative path (e.g., `data/raw_data/job_recommendation_dataset.csv`).
- Required columns expected by the script:
  - `Job Title`, `Company`, `Skills Required`, `Description`.

## 🏋️ Train the Model
Run the training script to produce the model artifacts:
```bash
python model/al.py
```
This will save:
- `job_recommendation_model.pkl`
- `vectorizer.pkl`
- `label_encoder.pkl`

## 🖥️ Run the App
Start the Streamlit app:
```bash
streamlit run app/app1.py
```
Then open the URL shown in the terminal (usually `http://localhost:8501`).

## 🧪 Usage
- Enter optional Company, comma-separated Skills, and an optional Description.
- Click "Recommend Job" to see the predicted job title.

## ⚠️ Tips & Troubleshooting
- If you see file-not-found errors for `.pkl` files, run the training step first.
- If you see a CSV path error, fix the dataset path in `model/al.py`.
- For best results, ensure your dataset contains the expected columns and encoding (UTF-8).

## 📜 License
This project is for educational/demo purposes. Add a license if you plan to distribute.

## 🙌 Acknowledgements
- Built with `pandas`, `scikit-learn`, and `streamlit`.