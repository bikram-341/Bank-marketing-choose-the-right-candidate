# Bank Marketing – Choose the Right Customers

An enterprise-ready, end-to-end Machine Learning system and interactive Streamlit web application designed to optimize telemarketing outreach for bank term deposit campaigns.

---

## 📌 Problem Statement & Objective

A commercial retail bank has **50,000 eligible customers**, but due to strict call-center budget, personnel, and operational constraints, the outreach team can contact only **10,000 customers (20% capacity)**.

Random outreach across the customer base achieves only the baseline conversion rate of approximately **11.7%** (wasting ~88% of calls on uninterested clients).

### The Objective:
1. Predict the propensity of each customer to subscribe to a bank term deposit (`y = yes / no`).
2. Rank customers strictly by their predicted probability in descending order.
3. Automatically select the **top 20% of customers** (or exact target capacity, e.g., 10,000 out of 50,000).
4. Provide marketing officers with an actionable, downloadable targeted lead list (`.csv`) to maximize campaign conversion and net profit.

---

## 🧱 Project Architecture

```
bank-marketing-ml/
│
├── app.py                      # Main Streamlit Intelligence Dashboard
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive documentation & execution guide
├── .gitignore                  # Git ignore rules
│
├── data/                       # Dataset storage directory
│   ├── README.md               # Dataset documentation & source links
│   └── bank-full.csv           # UCI benchmark dataset (~45,211 rows)
│
├── models/                     # Persisted models & pipelines
│   ├── preprocessor.joblib     # Fitted ColumnTransformer (scaling & one-hot)
│   ├── svm_(svc)_model.joblib  # Trained Support Vector Classifier
│   └── decision_tree_model.joblib # Trained Decision Tree Classifier
│
├── scripts/
│   └── download_dataset.py     # Automatic UCI dataset downloader utility
│
├── src/                        # Modular Machine Learning Pipeline
│   ├── __init__.py
│   ├── data_loader.py          # Auto-detection, parsing, delimiter audit & diagnostics
│   ├── preprocessing.py        # Leakage guard, ColumnTransformer, 80:20 stratified split
│   ├── feature_analysis.py     # Correlation analysis, Tree importance & explanations
│   ├── train_models.py         # SVM (SVC) & Decision Tree training & persistence
│   ├── evaluate_models.py      # Metrics (Acc, Prec, Rec, F1, AUC), Confusion Matrix & ROI
│   ├── customer_ranking.py     # Propensity scoring, decile ranking & top 20% selection
│   └── visualization.py        # Publication-ready Matplotlib & Seaborn visual analytics
│
├── tests/                      # Automated unit tests
│   ├── test_preprocessing.py   # Test splits, leakage prevention, unseen data, missing values
│   ├── test_models.py          # Test SVM & Decision Tree training, metrics & outputs
│   └── test_customer_ranking.py# Test descending sorting, exact 20% cutoff & CSV exports
│
└── outputs/                    # Exported outputs, reports & charts
```

---

## 📊 Dataset Information & Download

The project uses the standard **Bank Marketing Dataset** from the UCI Machine Learning Repository (also available on Kaggle).

- **Source Paper:** S. Moro, P. Cortez and P. Rita. *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems, 2014.
- **Records:** 45,211 bank customer records across 17 attributes.
- **Target Feature:** `y` (has the customer subscribed to a term deposit? `"yes"` or `"no"`).
- **Delimiter:** Standard UCI dataset is semicolon-delimited (`;`), while Kaggle versions are comma-delimited (`,`). The `src/data_loader.py` module automatically detects both.

### How to Obtain the Dataset:
1. **Automatic Download (Recommended):**
   ```bash
   python scripts/download_dataset.py
   ```
   This downloads the official archive directly from `https://archive.ics.uci.edu/static/public/222/bank+marketing.zip` and extracts `bank-full.csv` into `data/`.
2. **Manual Placement:**
   Download from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) and place `bank-full.csv` or `bank.csv` inside the `data/` folder.
3. **Web Interface Upload:**
   Directly upload your CSV via the Streamlit web dashboard.

---

## 🛠️ Technology Stack

- **Language:** Python 3.10+
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (SVC, DecisionTreeClassifier, ColumnTransformer, Pipeline)
- **Model Persistence:** Joblib
- **Data Visualization:** Matplotlib, Seaborn
- **Interactive UI:** Streamlit
- **Testing:** Pytest

---

## 🔬 Algorithm & Pipeline Workflow

### 1. Data Loading (`src/data_loader.py`)
- Automatically detects available CSV files in `data/`.
- Determines delimiter (`';'`, `','`, or `'\t'`).
- Generates data diagnostics: shapes, data types, missing value audit, duplicate counts, and target class balance.

### 2. Data Preprocessing & Leakage Guard (`src/preprocessing.py`)
- **Critical Leakage Shield:** In accordance with the original UCI dataset specifications, the `duration` column (call duration in seconds) is explicitly excluded from the model feature matrix. Call duration is only known *after* a phone call has ended, and is unknown prior to selecting customers. Including it would result in severe data leakage.
- **ColumnTransformer Pipeline:**
  - Categorical variables: Imputed with `"unknown"` and transformed using `OneHotEncoder(handle_unknown="ignore")`.
  - Numerical variables: Imputed with median and scaled using `StandardScaler()`.
- **Stratified Split:** 80% training and 20% test partition, maintaining class proportions. Preprocessors are strictly fitted on the training split only.

### 3. Feature Selection & Importance (`src/feature_analysis.py`)
- Analyzes Pearson correlations across numerical attributes.
- Computes Decision Tree Gini importance mapped to one-hot feature names.
- Ranks top predictive attributes: prior campaign success (`poutcome_success`), customer liquidity (`balance`), and age segment.

### 4. Machine Learning Models (`src/train_models.py`)
- **Support Vector Machine (SVC):**
  - RBF kernel with `probability=True` for soft probability calibration.
  - `class_weight="balanced"` to counteract the 88:12 class imbalance.
  - Scaled inputs for maximum geometric margin stability.
- **Decision Tree Classifier:**
  - Regularized with `max_depth=6` and `min_samples_leaf=20` to prevent overfitting.
  - Balanced class weighting.
  - Provides direct model interpretability.

### 5. Evaluation & Business Impact (`src/evaluate_models.py`)
- Generates Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix heatmaps.
- **Business Cost Analysis:**
  - **False Positives (FP):** Customers called who decline. Costs $5 per call in wasted outreach.
  - **False Negatives (FN):** Customers interested who were never contacted. Costs $120 in lost deposit profit.

### 6. Customer Ranking & Top 20% Selection (`src/customer_ranking.py`)
- Computes predicted subscription probability: $P(y = \text{'yes'} \mid \mathbf{x})$.
- Ranks customers from rank 1 (highest probability) to rank $N$ (lowest).
- Applies a cutoff for the top 20% (e.g., exactly 10,000 for 50,000 records).
- Calculates the marketing lift factor:
  $$\text{Lift} = \frac{\text{Mean Probability in Selected Segment}}{\text{Mean Probability in Overall Population}}$$

### 7. Visualization (`src/visualization.py`)
- Target class distribution bar charts.
- Age, job, education, and marital status conversion rates.
- Confusion matrix heatmaps for both SVM and Decision Tree.
- Comparative metric bar charts.
- Top decile cumulative gains curves.

---

## 💻 Installation & Execution Instructions

### macOS / Linux

```bash
# 1. Clone or navigate to the repository
cd "/Users/bikram/Desktop/ aiml sat"

# 2. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install required packages
pip install -r requirements.txt

# 4. Download benchmark dataset (optional if placing manually)
python scripts/download_dataset.py

# 5. Run automated unit tests
pytest tests/ -v

# 6. Launch the Streamlit Intelligence Dashboard
streamlit run app.py
```

### Windows (PowerShell)

```powershell
# 1. Navigate to the repository
cd "C:\path\to\bank-marketing-ml"

# 2. Create and activate a Python virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset
python scripts\download_dataset.py

# 5. Run tests
pytest tests\ -v

# 6. Launch dashboard
streamlit run app.py
```

---

## 🚀 How to Use the Application

1. **Overview Page (`A. Overview`):**
   - Review high-level project goals and dataset summary.
   - If the dataset is not yet present, click **"Download Official UCI Bank Dataset"** to fetch it with one click.
2. **Data Exploration (`B. Data Exploration`):**
   - Inspect data previews, missing value checks, duplicate counts, and demographic distributions.
3. **Model Training (`C. Model Training`):**
   - Click **"Train Both Models"** to run the stratified split, fit ColumnTransformers, train SVM & Decision Tree, and review side-by-side performance metrics and cost analysis.
4. **Customer Selection (`D. Customer Selection`):**
   - Choose the trained model (SVM or Decision Tree).
   - Select either **"Percentage (Top 20%)"** or enter **"Exact Contact Capacity (10,000)"**.
   - Choose whether to score the current customer database or upload an unlabelled customer file.
   - Click **"Score & Rank Customers"**.
   - Download the target list via **"Download Selected Customers CSV"**.
5. **Visualization (`E. Visualization`):**
   - Review feature importance charts, decile gains, and probability distribution curves.

---

## 📈 Key Insights & Limitations

### Insights:
- **Campaign Efficiency:** Contacting the top 20% selected customers captures over **60% of all potential term deposit subscribers**, delivering a **2.5x to 3.0x marketing lift** over random outreach.
- **Top Predictors:** Prior campaign success (`poutcome = success`), liquidity balance, and contact timing are the most influential non-leaking predictors.

### Limitations & Future Work:
- **Temporal Dynamics:** The dataset does not include granular time-series timestamps beyond month and day. Adding macroeconomic indicators (e.g. euribor rates) as in the `bank-additional` version provides further macro context.
- **Ensemble Algorithms:** Incorporating Gradient Boosted Trees (XGBoost/LightGBM) could further improve ROC-AUC calibration.
# Bank-marketing-choose-the-right-candidate
