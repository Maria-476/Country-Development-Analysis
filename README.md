# Country Development Segmentation & Classification

Unsupervised clustering + supervised classification pipeline that groups 167 countries into development tiers (**Developed / Developing / Underdeveloped**) using socio-economic and health indicators, then trains a fast classifier to predict the tier for any new country without re-running the clustering pipeline.

---

## Project Overview

This project answers two questions:

1. **Can countries be objectively grouped by development level using only raw economic and health data — without being told what "developed" means?** (Unsupervised learning — K-Means)
2. **Can we build a fast, deployable model that assigns a new country to the correct tier, without re-running the full clustering pipeline each time?** (Supervised learning — XGBoost)

This mirrors a real-world use case: organizations like the World Bank or aid agencies segmenting countries to prioritize resource allocation (e.g., "which countries need aid most").

---

## Dataset

- **Source:** Country-level socio-economic dataset (`Country-data.csv`)
- **Size:** 167 countries × 10 columns
- **Features:**

| Column | Description |
|---|---|
| `child_mortality_per_1000` | Child deaths under age 5, per 1000 live births |
| `exports_gdp_percent` | Exports as % of GDP |
| `health_spending_gdp_percent` | Health spending as % of GDP |
| `imports_gdp_percent` | Imports as % of GDP |
| `net_income_per_capita_usd` | Net income per person (USD) |
| `inflation_rate_percent` | Annual inflation rate |
| `life_expectancy_years` | Average life expectancy |
| `total_fertility_rate` | Average children per woman |
| `gdp_per_capita_usd` | GDP per capita (USD) |

No missing values or duplicate rows were present in the raw data.

---

## Methodology

### Part 1 — EDA & Clustering (`01_eda_&_clustering.ipynb`)

1. **Exploratory Data Analysis**
   - Histograms to check distribution shape → most features were right-skewed (income, GDP, mortality, trade %), a few mildly skewed (fertility, health spending), one left-skewed (life expectancy).
   - Boxplot to detect outliers → `net_income` and `gdp_per_capita` showed heavy outliers and a completely different scale from other features.
   - Correlation heatmap → confirmed strong relationships: `child_mortality` ↔ `life_expectancy` (-0.89), `net_income` ↔ `gdp_per_capita` (0.90, near-duplicate information).
   - Pairplot → revealed a clear exponential relationship between income/GDP and child mortality, hinting at natural groupings in the data.

2. **Preprocessing**
   - Checked skewness numerically (not just visually) before transforming.
   - Applied `log1p` transformation to 5 heavily skewed columns (`child_mortality`, `net_income`, `gdp_per_capita`, `exports%`, `imports%`).
   - `inflation_rate_percent` was **deliberately excluded** from log transformation — it contains negative values (deflation), which broke `log1p` and produced NaNs; kept in its original scale instead to preserve real data over forcing a transform.
   - Applied `StandardScaler` to bring all features to the same scale (mean 0, std 1) — required for K-Means since it's a distance-based algorithm.
   - Applied PCA, keeping **4 components** (91.4% cumulative variance retained) to reduce dimensionality and handle multicollinearity between correlated features.

3. **Clustering**
   - Used the **Elbow Method** and **Silhouette Score** to evaluate k = 2 to 10.
   - Silhouette score peaked at k=2, but **k=3** was chosen for better real-world interpretability (Developed / Developing / Underdeveloped is a standard, actionable framework), with only a marginal silhouette trade-off.
   - Trained final `KMeans(n_clusters=3, random_state=42, n_init=10)` on the PCA-transformed data.

### Part 2 — Classification (`02_XGBoost_classification.ipynb`)

1. Loaded the clustered dataset saved from Part 1.
2. Used the existing `cluster` column (0/1/2) directly as the target — no label encoding needed, since it was already numeric.
3. Split data 80/20 with `stratify=y` and `random_state=42` (ensures reproducible, class-balanced splits given the small dataset).
4. **No feature scaling applied** — tree-based models like XGBoost split on thresholds, not distances, so scaling has no effect on results (unlike K-Means, where it was essential).
5. Trained `XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)`.
6. Evaluated using accuracy, classification report, confusion matrix, and **5-fold stratified cross-validation** (chosen over a single train/test split, since the dataset covers most of the world's countries — CV gives a far more reliable performance estimate than one lucky/unlucky split).
7. Extracted feature importance to understand what drives predictions.

---

## Results

### Cluster Profiles (K-Means, k=3)

| Feature | Developing (62 countries) | Underdeveloped (46 countries) | Developed (59 countries) |
|---|---|---|---|
| Child mortality (per 1000) | 28.6 | 91.6 | 6.9 |
| Life expectancy (years) | 70.8 | 60.1 | 78.5 |
| Net income per capita (USD) | 11,665 | 2,172 | 34,577 |
| GDP per capita (USD) | 5,536 | 993 | 30,103 |
| Inflation rate | 10.2% | 11.3% | 2.5% |
| Fertility rate | 2.63 | 4.92 | 1.75 |
| Health spending (% GDP) | 5.7% | 6.5% | 8.3% |

**Key insight:** The clustering recovered a real, interpretable economic development spectrum without ever being told what "development" means — it only saw raw health/trade/economic indicators. The 3 groups align closely with frameworks used by real organizations (e.g., World Bank income classifications), which validates that the clusters are meaningful, not arbitrary. The "Underdeveloped" group is the clearest aid-priority segment — highest child mortality, lowest income, lowest life expectancy.

### Classification Performance (XGBoost)

- **Cross-validated accuracy (5-fold, stratified): 92.2% ± 5.3%** — reported as the primary, realistic performance metric.
- Single held-out test set accuracy: 97.1% (not used as the headline number — with only 34 test rows, a single split is too small to be reliable on its own; CV was used instead).
- Training accuracy: 100%, vs. 92.2% CV accuracy — indicates mild overfitting, expected and acceptable given the dataset's small size (167 rows total).

**Top predictive features:**

| Feature | Importance |
|---|---|
| `net_income_per_capita_usd` | 38.0% |
| `child_mortality_per_1000` | 25.7% |
| `gdp_per_capita_usd` | 11.3% |
| `inflation_rate_percent` | 8.8% |
| `life_expectancy_years` | 6.0% |

Income and child mortality dominate — consistent with their strong role in the original clustering and their -0.89 correlation with each other found during EDA.

**Important caveat on accuracy:** The high accuracy is expected, not surprising — the classification target (`cluster`) was itself derived from these same features via K-Means. XGBoost is effectively learning to approximate a rule that already exists in the data, not predicting something genuinely unknown. Its practical value is **speed and deployability**: it lets a new country be classified instantly, without re-running the full scaling → PCA → K-Means pipeline.

---

## Tech Stack

- **Language:** Python
- **Data handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn (StandardScaler, PCA, KMeans, train_test_split, cross_val_score), XGBoost
- **Model persistence:** joblib

---

## Project Structure

```
country-development-analysis/
│
├── data/
│   ├── Country-data.csv                  # raw dataset
│   └── country_data_clustered.csv        # output of Part 1 (used as input to Part 2)
│
├── plots/                                 # all saved visualizations
│   ├── 01_Distribution_Histograms.png
│   ├── 02_boxplot_before_transformation.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_pairplot_relationships.png
│   ├── 05_boxenplot_after_transformation.png
│   ├── 06_elbow_method_&_silhoutte.png
│   ├── 07_clustering_visualization.png
│   ├── 08_confusion_matrix.png
│   └── 09_feature_importance.png
│
├── 01_eda_&_clustering.ipynb              # EDA, preprocessing, K-Means clustering
├── 02_XGBoost_classification.ipynb        # Supervised classification on cluster labels
├── xgboost_country_classifier.pkl         # saved trained model
├── cluster_mapping.json                   # maps 0/1/2 → readable status labels
└── README.md
```

---

## How to Run

1. Clone the repo and install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
   ```
2. Run `01_eda_&_clustering.ipynb` first — this generates `data/country_data_clustered.csv`.
3. Run `02_XGBoost_classification.ipynb` — loads the clustered data, trains and evaluates the classifier.

### Using the saved model on new data

```python
import joblib, json

model = joblib.load('xgboost_country_classifier.pkl')
with open('cluster_mapping.json') as f:
    cluster_mapping = json.load(f)

prediction = model.predict(new_country_data)   # raw feature row(s), same 9 columns
status = cluster_mapping[str(prediction[0])]
print(status)   # e.g. "Developing"
```

---

## Limitations & Honest Notes

- Dataset covers most existing countries (167), not a sample of a larger population — true unseen-data validation isn't fully possible; cross-validation was used as the best available proxy.
- `exports_gdp_percent` and `imports_gdp_percent` remained moderately skewed even after log transformation — accepted as "good enough" rather than over-engineering the preprocessing further.
- k=3 was chosen for interpretability over k=2, which had a marginally higher silhouette score — a deliberate trade-off favoring a more actionable business story.
- XGBoost's high accuracy reflects that it's learning to replicate an existing rule (the K-Means clustering), not predicting an independent unknown outcome — its value is deployment speed, not novel insight.

---

## Future Work

- Try `PowerTransformer` (Yeo-Johnson) as an alternative to log transform for the remaining skewed trade columns.
- Compare K-Means results against hierarchical clustering or DBSCAN.
- Extend the classifier to a regression task (e.g., predicting `gdp_per_capita` directly) for a broader supervised learning demonstration.
- Build a simple Streamlit/Flask app around the saved `.pkl` model for interactive predictions.

---

## Author

Maria Anwar - Aspiring AI/ML Engineer
Portfolio: [github.com/Maria-476](https://github.com/Maria-476)