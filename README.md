# Game of Thrones Death Prediction

Predicting character survival, scene-level deaths, and episode-level death counts using machine learning on the *Game of Thrones* dataset.

---

## 📌 Project Overview
This project explores whether machine learning models can predict on-screen deaths in the HBO series *Game of Thrones*.  
The pipeline covers:
- **Character-level modeling** → predicting survival until the end of the series  
- **Scene-level modeling** → predicting whether a scene contains a death  
- **Episode-level modeling** → aggregating scene predictions to forecast total deaths per episode  

The project highlights **data wrangling, feature engineering, class imbalance handling, and model comparison**.

---

## 📊 Dataset
Source: [Game of Thrones Datasets and Visualizations](https://github.com/jeffreylancaster/game-of-thrones/tree/master/data)  
- **episodes.json** — metadata and detailed data (location, time, characters, deaths, etc.) for episodes and scenes  
- **characters.json** — character attributes (house, relationships, killers/victims, royal status, etc.)  

### Data Wrangling
- Extracted deeply nested JSON into structured pandas DataFrames  
- One-hot encoded categorical features  
- Derived new features:  
  - Scene length  
  - Character counts  
  - Total screen time per character/scene  
  - Flashback flags, human/non-human, etc.  

---

## ⚙️ Methods
### Models Explored
- Decision Tree  
- Random Forest  
- Logistic Regression  
- Support Vector Classifier (SVC)  
- Naive Bayes  

### Techniques
- Baselines (majority-class predictors)  
- Train/validation/test splits  
- Class imbalance handling (class weights, resampling)  
- Feature engineering for predictive signal  

---

## 📈 Results

### Character Survival
- Baseline (majority class): ~0.51 accuracy  
- Random Forest: **0.80 accuracy** on test set  
- Other models: 0.66–0.75  

### Scene Deaths
- Baseline: ~0.95 accuracy (due to class imbalance)  
- Random Forest (with balancing): **0.998 validation accuracy**  
- Other models: mixed results; Naive Bayes underperformed  

### Episode Death Counts
- Aggregated scene predictions → predicted episode-level deaths  
- Average error per episode inflated by low base rate (~2.75 deaths/episode)  
- Total error across series: ~126 deaths (~62.7% of true total)  

---

## 📊 Visualizations
- Class balance per dataset (characters vs. scenes)  
- Precision/Recall curves  
- Confusion matrices  
- Actual vs. predicted episode deaths  

*(Insert figures here from `/reports/figures`)*

---

## 🚀 How to Run
```bash
# Clone repo
git clone https://github.com/yourusername/got-death-prediction.git
cd got-death-prediction

# Setup environment
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run models
python -m src.models --model rf --target character
python -m src.models --model rf --target scene

# Generate episode-level predictions
python -m src.infer --model rf
