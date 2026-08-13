# Term Deposit Subscription Prediction

##### A supervised machine learning project that predicts which bank customers are likely to subscribe to a term deposit, built on direct telemarketing campaign data from a Portuguese banking institution. The model supports two concrete business goals — maximizing the number of subscribers found (customer acquisition) and minimizing wasted outreach (marketing budget optimization) — and compares six classifiers before selecting a deployment model for each goal.

---

## Author

**Aleksandra Vislova**

[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0072b1?&style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/aleksandra-vislova-a51ba9297)
---

## Files in This Repository

| File / Folder | Description |
|---|---|
| `README.md` | Project overview, methodology, and results |
| `requirements.txt` | Python dependencies |
| `notebooks/EDA.ipynb` | Data extraction, cleaning, exploratory analysis, feature engineering, and encoding |
| `notebooks/models.ipynb` | Model training, comparison, selection, and evaluation |
| `data/processed/` | Encoded train/test features and labels (`X_train_encoded.csv`, `X_test_encoded.csv`, `y_train.csv`, `y_test.csv`) |
| `models/preprocessor.pkl` | Fitted `ColumnTransformer` (one-hot, ordinal, and scaling logic) for reuse at inference time |
| `models/acquisition_model.pkl` | Final Random Forest classifier selected for the customer-acquisition use case |
| `img/eda/` | Exploratory data analysis charts |
| `img/outputs/` | Model evaluation charts |

---

## Table of Contents

- [Project Background](#project-background)
- [Data Source](#data-source)
- [Methodology](#methodology)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling & Results](#modeling--results)
- [Business Recommendations](#business-recommendations)
- [Summary](#summary)

---

## Project Background

Portuguese banking institutions run direct telemarketing campaigns to promote term deposit subscriptions — a call center contacts existing clients and offers them a term deposit product. Each call is costly, and not every client is equally likely to say yes. This project builds a classification model that predicts, **before a client is called**, how likely they are to subscribe, so the marketing team can prioritize outreach.

### Objectives
- Predict term deposit subscription (yes/no) from client demographic, financial, and campaign-contact attributes.
- Compare multiple classification algorithms on a consistent train/test split and metric set.
- Translate model performance into two distinct business strategies — acquisition-maximizing vs. budget-efficient — rather than a single "best" model.
- Ship a reusable preprocessing pipeline and a selected model so the recommendation can be deployed against new client lists.

---

## Data Source

| Source | Description |
|---|---|
| **OpenML / UCI Machine Learning Repository** | [Bank Marketing dataset (ID 43718)](https://www.openml.org/search?type=data&status=active&id=43718) |
| **Records** | 11,162 clients |
| **Features** | 16 input variables — demographic (age, job, marital status, education), financial (balance, housing loan, personal loan, default history), and campaign contact details (contact type, month, day, number of contacts, days since last contact, previous outcome) |
| **Target** | `deposit` — whether the client subscribed to a term deposit (`yes` / `no`) |

---

## Methodology

1. **Data extraction** — pulled directly from OpenML via the `openml` Python client.
   
2.  **Data quality checks** — confirmed no missing values and checked for duplicate rows.
   
3. **Leakage prevention** — the `duration` column (call length) was dropped, since it's only known *after* a call ends and would leak future information into a model meant to prioritize calls *before* they happen.

4. **Feature engineering** — derived `was_contacted` (binary) from `pdays`, since `pdays = -1` is a sentinel meaning "never contacted before," not a numeric distance; the raw `pdays` column is dropped afterward to avoid double-counting that signal.

5. **Encoding** — a `ColumnTransformer` applies one-hot encoding to nominal features, ordinal encoding to `education` (with `unknown` imputed to the most frequent known category, standard scaling to numeric features . `day` (day of month contacted) is cyclically encoded as `day_sin`/`day_cos` .

6. **Train/test split** — stratified 70/30 split (7,813 train / 3,349 test rows).
   
7. **Modeling** — six classifiers trained and compared on identical splits: Naive Bayes, K-Nearest Neighbors, Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.

8. **Model selection** — two top performers were mapped to two different business goals (see [Business Recommendations](#business-recommendations)).

---

## Exploratory Data Analysis

### 1. Target Distribution

The dataset is balanced: **5,873 clients (52.6%) did not subscribe, and 5,289 clients (47.4%) did subscribe** to a term deposit.
![Target_Distribution](https://github.com/aleksandra20050404/Customer_Subscription_Prediction/blob/main/img/eda/target_distribution.png)

### 2. Categorical Feature Distributions

![Categorical_Features](https://github.com/aleksandra20050404/Customer_Subscription_Prediction/blob/main/img/eda/eda_categorical.png)


- **`poutcome`**: 8,326 have `unknown` prior outcome — most clients were never part of a previous campaign. Only 1,071 had a prior `success`.
- **`contact`**: 8,042 were contacted by cellular, vs. 21% unknown and 7% telephone.
- **`month`**: heavily concentrated in May (2,824, ~25% of all contacts), tapering sharply through the rest of the year (December has only 110).
- **`default`**: "no" 10,994 vs. 168 "yes" — very little variance in this feature.
- **`education`**: `unknown` accounts for 497 rows (4.5%) —  `unknown` will be imputed the most frequent known category
- **`housing`** and **`loan`** are more balanced binary splits.

### 3. Numeric Features vs. Target (Box Plots)


![Numeric_Features](https://github.com/aleksandra20050404/Customer_Subscription_Prediction/blob/main/img/eda//box_plots.png)


- **`age`**: medians are nearly identical between subscribers (~38) and non-subscribers (~39) with heavily overlapping IQRs — little linear separation by deposit status in this view; 
- **`balance`**: both classes are dominated by extreme right-tail outliers (up to ~€80k); medians sit close together near the low end, with the outliers visually overwhelming the box itself.
- **`campaign`**: both groups cluster at low contact counts (median ~2–3) with a long outlier tail (30+ calls) — no strong separation by median, consistent with a "contact fatigue" story rather than "more calls = more success."
- **`pdays`**: the clearest numeric signal in the set — subscribers ("yes") show a visibly higher box (up to ~100 days) while non-subscribers cluster near 0, reinforcing the `poutcome`/`was_contacted` pattern that prior contact correlates with subscribing again.
- **`previous`**: same direction as `pdays` but weaker.

---

## Modeling & Results

### Model Comparison

![Model Performance Comparison](https://github.com/aleksandra20050404/Customer_Subscription_Prediction/blob/main/img/outputs/models_summary.png)

**Key insight:** Gradient Boosting leads on precision, F1, and ROC-AUC; Random Forest leads on recall. Both tree ensembles clearly outperform the simpler baselines. Naive Bayes is a notable middle case — its precision (0.7529) is second-highest in the table, close to Gradient Boosting's, but its recall (0.4972) is the worst of all six models. That combination is consistent with Naive Bayes' independence assumption being violated by this data (several encoded features are correlated — e.g. `previous`/`was_contacted` at 0.62 — which tends to make Naive Bayes overconfident on the cases it does flag, while missing many true positives entirely). Decision Tree, as expected for a single unpruned tree, trails every ensemble and baseline on nearly every metric — its main value here is as a reference point showing what one high-variance tree looks like next to 100+ trees averaged together in the ensembles above it.

### ROC Curve

![ROC Curve](https://github.com/aleksandra20050404/Customer_Subscription_Prediction/blob/main/img/outputs/roc_curve.png)

AUC = **0.771**, with an optimal threshold (by Youden's J statistic) of **0.480** — this matches the Random Forest row in the table above exactly, confirming the plot is now correctly tied to the deployed model (the earlier notebook revision had a variable-reuse bug that mislabeled Gradient Boosting's curve as Random Forest's — that's fixed and verified here). The curve sits well above the diagonal across the full threshold range, indicating solid separation between classes without being an exceptionally strong discriminator — there's a real but moderate signal in the features, not a near-perfect one.

### Precision-Recall Curve

![Precision-Recall Curve](https://github.com/aleksandra20050404/Customer_Subscription_Prediction/blob/main/img/outputs/precision_recall_curve.png)

Average Precision (AP) = **0.760**, well above the ~0.475 class-imbalance baseline (matching the actual "yes" proportion in the test set). Precision stays above 0.90 up to roughly 15–20% recall, then degrades in a fairly steady, gradual slope rather than a sharp cliff — meaning there isn't a single "obvious" threshold where performance falls off; the acquisition/precision trade-off is continuous, and the choice of operating threshold should be driven by the actual cost of a wasted call vs. the value of a missed subscriber, rather than a visually obvious inflection point.

### Confusion Matrix

![Confusion Matrix](https://github.com/aleksandra20050404/Customer_Subscription_Prediction/blob/main/img/outputs/confusion_matrix.png)

| | Predicted No | Predicted Yes |
|---|---|---|
| **Actual No** | 1,406 (TN) | 356 (FP) |
| **Actual Yes** | 597 (FN) | 990 (TP) |

Accuracy 71.5%, precision 73.6%, recall 62.4% — matching the Random Forest row above. Business reading: **356 false positives** are wasted outreach calls to clients who wouldn't have subscribed, and **597 false negatives** are subscribers the campaign would miss entirely if calls were prioritized by this model alone. The recall gap (missing 597 of 1,587 actual subscribers, ~38%) is the main cost of choosing Random Forest for the acquisition goal — it's the model most tuned to minimize this specific number relative to the other five.

### Calibration Plot

![Calibration Plot](https://github.com/aleksandra20050404/Customer_Subscription_Prediction/blob/main/img/outputs/calibration_plot.png)

The model is well-calibrated across most of the probability range — points track close to the diagonal from roughly 0.55 to 0.95 predicted probability. The main deviation is at the low end (predicted probabilities of 0.07–0.35), where the actual positive rate runs somewhat *higher* than predicted — i.e. the model is mildly underconfident about its lowest-scored clients, some of whom subscribe anyway more often than the raw score suggests. For a marketing use case where the probability score itself is used to rank a call list (not just a yes/no cutoff), this is close enough to trust directly in the mid-to-high range, with a note of caution about taking very low scores as precisely as their numeric value implies.

### Feature Importance

![Feature Importance](https://github.com/aleksandra20050404/Customer_Subscription_Prediction/blob/main/img/outputs/feature_importance.png)

##### `day` (day of month contacted) is encoded cyclically  as `day_sin`/`day_cos` rather than as a single scaled number; feature importance is more than `campaign` and `poutcome_success` combined.  Day-of-month is one of the strongest predictors in the model, sitting between `age` and `balance` — plausibly tied to payday or billing-cycle timing, a legitimate and actionable finding. `balance` and `age` remain the top two individual features either way, and together with the day-of-month signal, numeric/temporal features clearly outweigh the categorical ones — `poutcome_success`, despite being the feature that most directly signals that client responded well before, still ranks behind all of them.
---

## Business Recommendations

Two different business goals point to two different models:

| Goal | Recommended Model | Reason |
|---|---|---|
| **Customer Acquisition** — find as many likely subscribers as possible | **Random Forest** | Highest recall (62.4%) — catches more true subscribers, at the cost of more false positives (wasted calls) |
| **Marketing Budget Optimization** — only call people who will likely say yes | **Gradient Boosting** | Highest precision (78.5%) and ROC-AUC (0.789) — when it predicts "yes," it's right about 4 times out of 5, minimizing wasted call volume |

The deployed model in this repository (`acquisition_model.pkl`) is the **Random Forest**, reflecting a customer-acquisition priority. If the business goal shifts toward budget efficiency, retrain and ship the Gradient Boosting model instead.

Naive Bayes' precision (0.753) is close to Gradient Boosting's, so it might look tempting as a lighter-weight alternative — but its recall (0.497) is by far the worst in the comparison, meaning it would miss roughly half of all actual subscribers. It isn't a reasonable substitute for either business goal here.

---


## Summary

| Metric | Value |
|---|---|
| **Dataset size** | 11,162 clients |
| **Target balance** | 52.6% no / 47.4% yes |
| **Train / test split** | 7,813 / 3,349 (stratified 70/30) |
| **Best precision & ROC-AUC** | Gradient Boosting (78.5% / 0.789) |
| **Best recall** | Random Forest (62.4%) |


| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Gradient Boosting | 0.7348 | **0.7853** | 0.6062 | 0.6842 | **0.7892** |
| Random Forest | 0.7154 | 0.7355 | **0.6238** | 0.6751 | 0.7713 |
| Logistic Regression | 0.7008 | 0.7403 | 0.5677 | 0.6427 | 0.7599 |
| Naive Bayes | 0.6844 | 0.7529 | 0.4972 | 0.5989 | 0.7395 |
| K-Nearest Neighbors | 0.6793 | 0.6876 | 0.5923 | 0.6364 | 0.7179 |
| Decision Tree | 0.6226 | 0.6013 | 0.6043 | 0.6028 | 0.6217 |

**Key insight:** Gradient Boosting leads on precision, F1, and ROC-AUC; Random Forest leads on recall. Both tree ensembles clearly outperform the simpler baselines. Naive Bayes is a notable middle case — its precision (0.7529) is second-highest in the table, close to Gradient Boosting's, but its recall (0.4972) is the worst of all six models. That combination is consistent with Naive Bayes' independence assumption being violated by this data (several encoded features are correlated — e.g. `previous`/`was_contacted` at 0.62 — which tends to make Naive Bayes overconfident on the cases it does flag, while missing many true positives entirely). Decision Tree, as expected for a single unpruned tree, trails every ensemble and baseline on nearly every metric — its main value here is as a reference point showing what one high-variance tree looks like next to 100+ trees averaged together in the ensembles above it.

### Top Predictive Features (Random Forest)
##### Based on feature importances from the deployed model, the strongest predictors of subscription are `balance`, `age`, day-of-month (encoded cyclically as `day_sin`/`day_cos`), and `campaign` (number of contacts) — the numeric client/contact attributes — followed by `poutcome_success`, `education`, and contact method. This is a useful correction to the intuitive assumption that "having succeeded before" would dominate the model; in practice, this Random Forest leans more heavily on financial, demographic, and timing signals. The day-of-month result in particular is worth highlighting: it suggests scheduling outreach around specific points in the month (e.g. paydays) may be as impactful as anything else in the model.
---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
openml
joblib
```
