# Student Dropout Prediction

A machine learning project that predicts whether a student is at risk of dropping out, using academic, demographic, socioeconomic, and enrollment-related data. Built as part of an ML internship project.

**Live app:** _[add your deployed Streamlit URL here once live]_

---

## 1. Problem

Educational institutions often identify at-risk students too late, after they've already withdrawn. This project builds a classification model that flags at-risk students early, using data the institution already collects, so advisors can intervene with support (tutoring, financial aid, counseling) before a student leaves.

- **Type of problem:** Binary classification
- **Target variable:** `Target_binary` — 1 = Dropout, 0 = Not Dropout (Enrolled/Graduate)
- **Why it matters:** limited support resources are better spent on students who actually need them, and early intervention is far more effective than late intervention.

## 2. Dataset

- **Source:** [UCI Machine Learning Repository — Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- **Records:** 4,424 students
- **Features:** 36 input features across four categories:
  - Academic (grades, approved units, evaluations)
  - Demographic (age, gender, nationality)
  - Socioeconomic (scholarship status, debtor status, tuition payment status)
  - Enrollment-related (course, attendance mode, application details)
- **Original target:** 3 classes (Dropout / Enrolled / Graduate), collapsed to binary (Dropout vs. Not Dropout) for this project.

Loaded via the `ucimlrepo` package:
```python
from ucimlrepo import fetch_ucirepo
dataset = fetch_ucirepo(id=697)
```

## 3. Data Cleaning & Preprocessing

- Checked for missing values with `df.isnull().sum()` — none found.
- Checked for duplicate rows with `df.duplicated().sum()` — removed if present.
- Verified data types with `df.dtypes` — all features numeric except the original `Target` column.
- Encoded the target: `Target_binary` = 1 if `Target == 'Dropout'`, else 0.
- Scaled numeric features using `StandardScaler` before training (Logistic Regression performs better on scaled data).

## 4. Exploratory Data Analysis

Key findings from EDA (see notebook for full charts):

- **Academic performance** (units approved, 2nd semester grade) is the strongest signal — dropout students show far fewer approved units and lower grades than graduates.
- **Financial status** matters a lot — students behind on tuition payments or classified as debtors show meaningfully higher dropout rates; scholarship holders show lower dropout rates.
- **Age at enrollment** skews slightly older for dropouts, possibly reflecting non-traditional students balancing more responsibilities.
- **Gender** shows some difference in dropout rate, but weaker than academic/financial features.
- A correlation heatmap confirmed the 1st/2nd semester curricular features cluster together and correlate most strongly with the target.

**Modeling implication:** curricular performance and financial-status features were expected to carry the most weight in the model; demographic features like nationality were expected to contribute little.

## 5. Model

- **Algorithm:** Logistic Regression (`sklearn.linear_model.LogisticRegression`, `max_iter=1000`)
- **Train/test split:** 80/20, stratified on the target to preserve class balance
- **Features:** all 36 original features (scaled)

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)
```

## 6. Evaluation

Evaluated on the held-out test set using:

- **Confusion Matrix** — breaks down True Positives, True Negatives, False Positives, and False Negatives
- **Classification Report** — Precision, Recall, F1-score per class
- **Accuracy, Precision, Recall, F1 Score**
- **ROC-AUC** — overall ability to separate the two classes across thresholds

_Fill in your actual numbers here once you have them, e.g.:_
| Metric | Score |
|---|---|
| Accuracy | _e.g. 0.87_ |
| Precision | _e.g. 0.78_ |
| Recall | _e.g. 0.72_ |
| F1 Score | _e.g. 0.75_ |
| ROC-AUC | _e.g. 0.91_ |

**Error analysis:** Given the use case, a False Negative (missing a real at-risk student) is more costly than a False Positive (flagging a student who turns out fine) — a missed student doesn't get help, while a flagged student just gets unnecessary outreach. Recall was prioritized over raw accuracy for this reason, especially given the original class imbalance (Dropout being the minority class).

## 7. Prediction / Application

A prediction function takes new student data (or user input from a simple UI), applies the same scaling used in training, and returns:
- Predicted class (Dropout / Not Dropout)
- Dropout probability
- Risk level (Low / Medium / High, thresholded on probability)

This was built as a small interactive web app (Streamlit) so a non-technical user (e.g. an advisor) could enter a student's information and get an instant risk read.

## 8. Real-World Application

This mirrors early-warning systems already used by some universities: instead of waiting for a formal withdrawal, the model flags risk during the semester so staff can proactively offer tutoring, financial aid guidance, or academic advising — when intervention can still make a difference.

## 9. Reproducing This Project

1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook (`student_dropout.ipynb`) top to bottom to reproduce data loading, cleaning, EDA, training, and evaluation.
4. To run the web app locally: `streamlit run app.py` (requires `dropout_model.pkl`, `scaler.pkl`, `feature_columns.pkl` generated by the notebook).

## 10. Project Structure

```
├── student_dropout.ipynb      # Full workflow: data loading → cleaning → EDA → training → evaluation
├── app.py                     # Streamlit web app for live predictions
├── dropout_model.pkl          # Trained Logistic Regression model
├── scaler.pkl                 # Fitted StandardScaler
├── feature_columns.pkl        # Feature column order (needed for correct predictions)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

**Author:** Emaan Rana
**Internship Project — Student Dropout Prediction (Logistic Regression)**
