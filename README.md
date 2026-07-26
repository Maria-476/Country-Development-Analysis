# Country Development Segmentation & Classification

Unsupervised clustering + supervised classification pipeline that groups 167 countries into development tiers (**Developed / Developing / Under Developed**) using socio-economic and health indicators — plus an interactive Streamlit app to explore the results and simulate "what-if" policy scenarios.

**🔗 Live App:** [your-app-url.streamlit.app](https://country-development-analysis-7svbcyvhsyvvcsgnffymww.streamlit.app/)

---

## Project Overview

This project answers three questions:

1. **Can countries be objectively grouped by development level using only raw economic and health data — without being told what "developed" means?** (Unsupervised learning — K-Means)
2. **Can we build a fast, deployable model that assigns a new country to the correct tier, without re-running the full clustering pipeline each time?** (Supervised learning — XGBoost)
3. **Can this be made usable for a non-technical person** — exploring the data, testing new scenarios, and seeing the impact of policy decisions — without touching a notebook? (Streamlit app)

This mirrors a real-world use case: organizations like the World Bank or aid agencies segmenting countries to prioritize resource allocation.

---

## Dataset

- **Source:** Country-level socio-economic dataset (`Country-data.csv`)
- **Size:** 167 countries × 10 columns, no missing values or duplicates

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

---

## Methodology

### Part 1 — EDA & Clustering (`01_eda_&_clustering.ipynb`)

- Histograms, boxplots, correlation heatmap, and pairplots to understand distribution shape, outliers, and feature relationships. Most features were right-skewed; `net_income` and `gdp_per_capita` had heavy outliers and a different scale from everything else.
- Applied `log1p` to 5 heavily skewed columns. `inflation_rate_percent` was deliberately excluded — it contains negative values (deflation), which broke `log1p`.
- `StandardScaler` (required for K-Means, a distance-based algorithm) → PCA, keeping 4 components (91.4% variance retained) to reduce dimensionality and multicollinearity.
- Elbow Method + Silhouette Score evaluated k=2 to 10. Silhouette peaked at k=2, but **k=3** was chosen for real-world interpretability (Developed/Developing/Under Developed is a standard, actionable framework).
- Final model: `KMeans(n_clusters=3, random_state=42, n_init=10)`.

### Part 2 — Classification (`02_XGBoost_classification.ipynb`)

- Used the `cluster` column from Part 1 directly as the target (already numeric, no encoding needed).
- 80/20 split with `stratify=y`, `random_state=42`. **No feature scaling** — tree-based models split on thresholds, not distances.
- `XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)`, evaluated with 5-fold stratified cross-validation (more reliable than a single split, given the small dataset).

### Part 3 — Interactive App (`app.py`)

Built with Streamlit to make both parts of the analysis explorable without opening a notebook.

**📊 Explore Countries tab**
- Interactive scatter plot of all 167 countries (GDP vs. child mortality, colored by tier)
- Cluster profile summary table
- Live, type-ahead search across the full dataset

**🔮 Predict & Simulate tab**
- Prefill inputs from any real country, or build a custom profile from scratch
- Instant tier prediction with class probabilities
- **What-If radar chart** — compare any input profile against its predicted tier's average across all 9 indicators
- **Resource Allocation Calculator** — for a country below "Developed," shows the numeric gap to the next tier and simulates how allocating a hypothetical fund across Health / Education / Economic Infrastructure shifts key indicators and, potentially, the predicted tier (explicitly labeled as an illustrative model, not real economic forecasting)
- **Economic Twins** — finds the 3 real countries most similar to any input profile using cosine similarity, with a side-by-side comparison table

---

## Results

### Cluster Profiles (K-Means, k=3)

| Feature | Developing (62 countries) | Under Developed (46 countries) | Developed (59 countries) |
|---|---|---|---|
| Child mortality (per 1000) | 28.6 | 91.6 | 6.9 |
| Life expectancy (years) | 70.8 | 60.1 | 78.5 |
| Net income per capita (USD) | 11,665 | 2,172 | 34,577 |
| GDP per capita (USD) | 5,536 | 993 | 30,103 |
| Inflation rate | 10.2% | 11.3% | 2.5% |
| Fertility rate | 2.63 | 4.92 | 1.75 |
| Health spending (% GDP) | 5.7% | 6.5% | 8.3% |

**Key insight:** The clustering recovered a real, interpretable economic development spectrum without ever being told what "development" means — it only saw raw health/trade/economic indicators. The 3 groups align closely with frameworks used by real organizations (e.g., World Bank income classifications).

### Classification Performance (XGBoost)

- **Cross-validated accuracy (5-fold, stratified): 92.2% ± 5.3%** — the primary, realistic performance metric.
- Training accuracy: 100% vs. 92.2% CV — indicates mild overfitting, expected given the dataset's small size (167 rows).

**Top predictive features:** `net_income_per_capita_usd` (38.0%), `child_mortality_per_1000` (25.7%), `gdp_per_capita_usd` (11.3%), `inflation_rate_percent` (8.8%), `life_expectancy_years` (6.0%).

**Important caveat:** The high accuracy is expected, not surprising — the classification target (`cluster`) was itself derived from these same features via K-Means. XGBoost is learning to approximate an existing rule, not predicting something genuinely unknown. Its practical value is **speed and deployability** — classifying a new country instantly, without re-running the full clustering pipeline.

---

## Tech Stack

- **Data & ML:** pandas, numpy, scikit-learn (StandardScaler, PCA, KMeans, cosine similarity), XGBoost, joblib
- **Notebook visualization:** matplotlib, seaborn
- **App:** Streamlit, streamlit-keyup (live search), Plotly (interactive charts)

---

## Project Structure

```
country-development-analysis/
│
├── data/
│   ├── Country-data.csv                  # raw dataset
│   └── country_data_clustered.csv        # output of Part 1, input to Parts 2 & 3
│
├── plots/                                 # notebook visualizations
│
├── 01_eda_&_clustering.ipynb              # EDA, preprocessing, K-Means clustering
├── 02_XGBoost_classification.ipynb        # Supervised classification on cluster labels
├── app.py                                 # Streamlit app
├── xgb_country_classifier.pkl             # saved trained model
├── cluster_mapping.json                   # maps 0/1/2 → readable tier labels
├── requirements.txt
└── README.md
```

---

## How to Run

**Notebooks** (regenerate the data/model from scratch):
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```
Run `01_eda_&_clustering.ipynb` first, then `02_XGBoost_classification.ipynb`.

**App:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Limitations & Honest Notes

- Dataset covers most existing countries (167), not a sample of a larger population — true unseen-data validation isn't fully possible; cross-validation was used as the best available proxy.
- `exports_gdp_percent` and `imports_gdp_percent` remained moderately skewed even after log transformation — accepted as "good enough" rather than over-engineering the preprocessing further.
- k=3 was chosen for interpretability over k=2, which had a marginally higher silhouette score.
- The Resource Allocation Calculator's fund-impact estimates are simplified assumptions, not real economic cost data — the app labels this explicitly.

---

## Author

Maria Anwar — Aspiring AI/ML Engineer
[GitHub](https://github.com/Maria-476) · [LinkedIn](https://www.linkedin.com/in/maria-anwar4786)