# Week 4: EDA and Feature Engineering

This submission produces a detailed Word report, reproducible synthetic
transaction and customer datasets, visualizations, and a baseline-versus-
engineered-feature model comparison for the beauty and wellness market.

## Important data note

No source or preprocessed dataset was present when this submission was prepared.
The script therefore generates a **synthetic** dataset of 1,500 customers and
their historical orders with a fixed random seed. All figures, statistical test
results, and model metrics in the report describe this simulated example only;
they are not claims about real customers or market performance.

## Run

```powershell
python -m pip install -r requirements.txt
python eda_feature_engineering.py
```

Outputs are written to `output/`:

- `week4_eda_feature_engineering.docx` — detailed assignment report
- `synthetic_transactions.csv` and `synthetic_customer_features.csv`
- `monthly_revenue.png`, `category_revenue.png`, and `feature_correlation.png`
- `model_comparison.csv` — held-out model metrics

The analysis is seeded and deterministic. To analyze real data, replace the
synthetic-data generation in `build_synthetic_data()` with validated data
loading and map the source columns to the documented transaction schema. Do
not reuse the simulated report results as evidence for a real business.
