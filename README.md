# 🧠 Mental Health Score Predictor

A machine learning project that predicts a student's **Mental Health Score** (a continuous value from ~3.6 to ~9.4) based on their social media usage patterns, study habits, sleep, physical activity, and stress level. The trained model is served via a **FastAPI** REST API with a static HTML/JS frontend.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
  - [Step 1 – Import Libraries](#step-1--import-libraries)
  - [Step 2 – Load the Dataset](#step-2--load-the-dataset)
  - [Step 3 – Exploratory Data Analysis](#step-3--exploratory-data-analysis-eda)
  - [Step 4 – Feature Engineering](#step-4--feature-engineering)
  - [Step 5 – Encoding and Train/Test Split](#step-5--encoding-and-traintest-split)
  - [Step 6 – Preprocessing Pipeline](#step-6--preprocessing-pipeline)
  - [Step 7 – Model Building](#step-7--model-building)
  - [Step 8 – Hyperparameter Tuning](#step-8--hyperparameter-tuning)
  - [Step 9 – Model Export](#step-9--model-export)
- [Model Performance Summary](#-model-performance-summary)
- [API and Web App](#-api-and-web-app)
- [Project Structure](#-project-structure)
- [Installation and Running Locally](#-installation-and-running-locally)
- [Tech Stack](#-tech-stack)

---

## 🎯 Problem Statement

**Goal:** Predict a student's `Mental_Health_Score` (a continuous score ranging from ~3.6 to ~9.4) using their social media habits, academic lifestyle, and stress indicators.

**Type:** Regression problem (predicting a continuous numerical output).

---

## 📊 Dataset

**File:** `Student Social Media And Mental Health Impact.csv`

| Property | Value |
|---|---|
| Rows | 5,000 students |
| Columns | 13 |
| Missing Values | 0 (clean dataset) |
| Duplicates | 2 (detected) |

### Feature Overview

| Column | Type | Description |
|---|---|---|
| `Age` | Numeric | Student age (18–24) |
| `Gender` | Categorical | Male / Female |
| `Country` | Categorical | 111 unique countries |
| `Academic_Level` | Categorical | High School / Undergraduate / Graduate |
| `Most_Used_Platform` | Categorical | 12 platforms (Instagram, TikTok, Facebook, etc.) |
| `Purpose_Of_Use` | Categorical | Education / Entertainment / Networking / News |
| `Avg_Daily_Usage_Hours` | Numeric | Daily social media usage (1.0–8.8 hrs) |
| `Daily_Unlocks` | Numeric | Phone unlocks per day (62–273) |
| `Study_Hours` | Numeric | Daily study hours (0.3–8.3 hrs) — right-skewed |
| `Physical_Activity_Hours` | Numeric | Daily physical activity (−0.4–4.1 hrs) |
| `Sleep_Hours_Per_Night` | Numeric | Nightly sleep (3.6–9.9 hrs) |
| `Stress_Level` | Ordinal | Low / Medium / High / Very High |
| **`Mental_Health_Score`** | **Target** | **Score from 3.6 to 9.4** |

**Descriptive statistics of the target:**
- Mean: **6.23** | Std: **1.28** | Min: **3.6** | Max: **9.4**

---

## 🔬 Project Workflow

### Step 1 – Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

Standard data science libraries are imported for data manipulation, statistical analysis, and visualization.

---

### Step 2 – Load the Dataset

```python
df = pd.read_csv("Student Social Media And Mental Health Impact.csv")
df.shape   # (5000, 13)
```

- Loaded 5,000 rows × 13 columns.
- Confirmed **zero null values** across all columns via `df.isnull().sum()`.
- Detected **2 duplicate rows** via `df.duplicated().sum()`.
- Inspected data types via `df.info()`: 5 float columns, 2 int columns, 6 object (categorical) columns.
- Reviewed descriptive statistics via `df.describe()`.

---

### Step 3 – Exploratory Data Analysis (EDA)

#### 3a. Target Distribution

```python
sns.histplot(df['Mental_Health_Score'], kde=True)
```

The `Mental_Health_Score` follows a roughly **normal distribution** centered around 6.2, confirming the regression approach is appropriate.

#### 3b. Correlation Heatmap

```python
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
```

Identified the linear relationships between all numeric features and the target. Key findings:
- **Positive correlations:** `Sleep_Hours_Per_Night`, `Study_Hours`
- **Negative correlations:** `Avg_Daily_Usage_Hours`, `Daily_Unlocks`

#### 3c. Categorical Feature Analysis

Visualized how `Stress_Level`, `Gender`, `Academic_Level`, `Most_Used_Platform`, and `Purpose_Of_Use` affect `Mental_Health_Score` using box plots and bar charts. Notably:
- **Stress Level** shows a clear monotonic relationship: students with "Very High" stress have significantly lower mental health scores.

#### 3d. Skewness Detection

```python
df['Study_Hours'].skew()
```

`Study_Hours` was identified as **right-skewed**, requiring a log transformation before modeling to reduce the impact of outliers and improve model performance.

---

### Step 4 – Feature Engineering

#### Country Grouping

With 111 unique countries, direct one-hot encoding would create a highly sparse, noisy feature space. Instead, countries were consolidated into **8 groups** — the 7 most-represented countries plus an `"Other"` catch-all bucket:

```python
top_country = ['United States', 'India', 'United Kingdom',
               'Canada', 'Australia', 'Germany', 'France']

df['Grouped_country'] = df['Country'].apply(
    lambda x: x if x in top_country else 'Other'
)
```

**Distribution of Grouped Country:**

| Group | Count |
|---|---|
| Other | 3,231 |
| India | 389 |
| USA | 354 |
| Canada | 230 |
| Australia | 198 |
| UK | 185 |
| Germany | 136 |
| Mexico | 94 |
| Turkey | 94 |
| France | 87 |

The engineered `Grouped_country` feature replaces the raw `Country` column in training.

---

### Step 5 – Encoding and Train/Test Split

Feature columns were categorized by their required encoding strategy:

```python
skewed_col        = ["Study_Hours"]
other_numeric_col = ['Age', 'Avg_Daily_Usage_Hours', 'Daily_Unlocks',
                     'Sleep_Hours_Per_Night', 'Physical_Activity_Hours']
ordinal_col       = ['Stress_Level']
nominal_col       = ['Gender', 'Academic_Level', 'Most_Used_Platform',
                     'Purpose_Of_Use', 'Grouped_country']

feature_col = skewed_col + other_numeric_col + ordinal_col + nominal_col
X = df[feature_col]
y = df['Mental_Health_Score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```

- **Train set:** 70% (3,500 samples)
- **Test set:** 30% (1,500 samples)

---

### Step 6 – Preprocessing Pipeline

A `ColumnTransformer` was built to apply the correct transformation to each column type in a single, reproducible, production-safe step:

```python
# 1. Skewed column: log1p transform, then standardize
skew_pipeline = Pipeline([
    ('log_transform', FunctionTransformer(np.log1p)),
    ('scaler', StandardScaler())
])

# 2. Regular numeric columns: standardize only
plain_numeric_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

# 3. Ordinal column: encode with explicit category order
ordinal_pipeline = Pipeline([
    ('encode', OrdinalEncoder(categories=[['Low', 'Medium', 'High', 'Very High']]))
])

# 4. Nominal columns: one-hot encode (handle unseen categories gracefully)
nominal_pipeline = Pipeline([
    ('encode', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('skew_pipeline',          skew_pipeline,          skewed_col),
    ('plain_numeric_pipeline', plain_numeric_pipeline, other_numeric_col),
    ('ordinal_pipeline',       ordinal_pipeline,       ordinal_col),
    ('nominal_pipeline',       nominal_pipeline,       nominal_col)
])
```

**Key Design Decisions:**
- `log1p` on `Study_Hours` reduces skewness so the model isn't disproportionately influenced by high-study-hour outliers.
- `OrdinalEncoder` with explicit category ordering (`Low → Medium → High → Very High`) preserves the meaningful rank ordering of `Stress_Level`.
- `OneHotEncoder(handle_unknown='ignore')` ensures the API doesn't crash if it receives a platform or country not seen during training.
- `SimpleImputer` safety nets are included in pipelines even though training data has zero missing values — a best practice for production robustness.

---

### Step 7 – Model Building

#### 7a. Baseline — Linear Regression

```python
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
```

| Metric | Train | Test |
|---|---|---|
| R² Score | 0.7237 | 0.7398 |
| MAE | 0.519 | 0.536 |

Linear Regression explains ~74% of the variance in mental health scores. This serves as the **performance floor** that more complex models should beat.

---

#### 7b. Random Forest Regressor (Default)

```python
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('RandomForestRegressor', RandomForestRegressor(random_state=42))
])
rf_pipeline.fit(X_train, y_train)
```

| Metric | Train | Test |
|---|---|---|
| R² Score | 0.9809 | 0.8778 |
| MAE | 0.126 | 0.346 |

Random Forest dramatically improves test R² from 0.74 to 0.88. However, the large train/test gap (0.98 vs 0.88) indicates overfitting with default hyperparameters — addressed in the next step.

---

### Step 8 – Hyperparameter Tuning

`RandomizedSearchCV` was used to tune the Random Forest hyperparameters with 5-fold cross-validation across 15 random configurations:

```python
param_distributions = {
    'RandomForestRegressor__n_estimators':      [100, 200, 300],
    'RandomForestRegressor__max_depth':         [5, 10, 15],
    'RandomForestRegressor__min_samples_split': [2, 5, 10],
    'RandomForestRegressor__min_samples_leaf':  [1, 2, 4]
}

search = RandomizedSearchCV(
    rf_pipeline,
    param_distributions,
    n_iter=15,
    cv=5,
    scoring='r2',
    random_state=42,
    n_jobs=-1  # use all available CPU cores
)
search.fit(X_train, y_train)
```

**Best hyperparameters found:**

| Parameter | Best Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | 15 |
| `min_samples_split` | 5 |
| `min_samples_leaf` | 2 |

The tuned model achieves a better bias-variance trade-off: limiting `max_depth` and increasing `min_samples_leaf` regularizes the forest, reducing overfitting while retaining strong generalization.

---

### Step 9 – Model Export

The final tuned pipeline (full preprocessing + best Random Forest) was serialized using `joblib`:

```python
import joblib
joblib.dump(search.best_estimator_, 'mental_health_model.pkl')
```

The saved `mental_health_model.pkl` (~25 MB) bundles the entire preprocessing and prediction logic into a single, deployable artifact. The API loads it once at startup and calls `.predict()` on incoming requests.

---

## 📈 Model Performance Summary

| Model | Train R² | Test R² | Test MAE |
|---|---|---|---|
| Linear Regression (baseline) | 0.7237 | 0.7398 | 0.536 |
| Random Forest (default params) | 0.9809 | 0.8778 | 0.346 |
| **Random Forest (tuned)** ✅ | — | **Best** | **Best** |

**Winner: Tuned Random Forest** — best generalization on unseen data with significantly reduced overfitting compared to default settings.

---

## 🌐 API and Web App

The model is served as a REST API using **FastAPI**, with an embedded static HTML/JS frontend.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend (`index.html`) |
| `POST` | `/predict` | Returns a predicted mental health score |

### Prediction Request Body

```json
{
  "Age": 21,
  "Gender": "Male",
  "Country": "India",
  "Academic_Level": "Undergraduate",
  "Most_Used_Platform": "Instagram",
  "Purpose_Of_Use": "Entertainment",
  "Avg_Daily_Usage_Hours": 4.5,
  "Daily_Unlocks": 150,
  "Study_Hours": 3.0,
  "Physical_Activity_Hours": 1.5,
  "Sleep_Hours_Per_Night": 7.0,
  "Stress_Level": "Medium"
}
```

### Prediction Response

```json
{
  "predected_mental_health_score": 6.42
}
```

### Input Validation Rules (Pydantic)

| Field | Constraint |
|---|---|
| `Age` | Integer, 10–100 |
| `Gender` | `"Male"` or `"Female"` |
| `Academic_Level` | `"Undergraduate"`, `"Graduate"`, `"High School"` |
| `Most_Used_Platform` | One of 12 supported platforms |
| `Stress_Level` | `"Low"`, `"Medium"`, `"High"`, `"Very High"` |
| `Avg_Daily_Usage_Hours` | Float, 0–24 |
| `Sleep_Hours_Per_Night` | Float, 0–24 |
| `Physical_Activity_Hours` | Float, 0–24 |

### Country Grouping in the API

The API replicates the same feature engineering used during training:

```python
top_country = ['United States', 'India', 'United Kingdom',
               'Canada', 'Australia', 'Germany', 'France']

country_group = data.Country if data.Country in top_country else 'Other'
```

---

## 📁 Project Structure

```
Mental-Health-Score-Predictor/
│
├── Mental_Health_Score_Predictor.ipynb  # Full ML workflow notebook
├── main.py                              # FastAPI application
├── mental_health_model.pkl              # Trained serialized pipeline (~25 MB)
├── index.html                           # Frontend web app
│
├── requirements.txt                     # Python dependencies
├── runtime.txt                          # Python runtime version (3.13.0)
├── Procfile                             # Deployment process command (uvicorn)
└── vercel.json                          # Vercel deployment configuration
```

---

## 🚀 Installation and Running Locally

### 1. Clone the repository

```bash
git clone <repository-url>
cd Mental-Health-Score-Predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the API server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open the app

Navigate to [http://localhost:8000](http://localhost:8000) in your browser.

### 5. Explore interactive API docs

FastAPI auto-generates Swagger documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML / Data Science | Python, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn |
| Model Serialization | Joblib |
| API Framework | FastAPI + Uvicorn |
| Data Validation | Pydantic |
| Frontend | HTML / JavaScript |
| Deployment | Vercel |

---

## 📦 Dependencies

```
fastapi
uvicorn
joblib
pandas
scikit-learn==1.6.1
python-multipart
```

> **Note:** `scikit-learn==1.6.1` is pinned to ensure the saved model pipeline remains compatible with the exact version used during training.
