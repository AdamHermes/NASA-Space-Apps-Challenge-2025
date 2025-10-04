# Response from `/data/process_csv/`:

```json
{
  "message": "CSV processed with option 'string'",
  "data": {
    "train_filename": "cumulative_train.csv",
    "train_filepath": "app\\storage\\processed_csvs\\cumulative_train.csv",
    "train_stats": {
      "num_samples": 3070,
      "num_features": 42,
      "class_counts": {
        "0": 1590,
        "1": 1480
      },
      "class_percentage": {
        "0": 51.79,
        "1": 48.21
      }
    },
    "test_filename": "cumulative_test.csv",
    "test_filepath": "app\\storage\\processed_csvs\\cumulative_test.csv",
    "test_stats": {
      "num_samples": 1316,
      "num_features": 42,
      "class_counts": {
        "0": 682,
        "1": 634
      },
      "class_percentage": {
        "0": 51.82,
        "1": 48.18
      }
    },
    "train_head": [
      {
        "koi_fpflag_nt": -0.06936134776353386,
        "koi_fpflag_ss": -0.11576315932671022,
        "koi_fpflag_co": -0.03701166050988026,
        "koi_fpflag_ec": -0.015101330108226504,
        "koi_period": -0.3252495221154905,
        "koi_period_err1": -0.21247352280501483,
        "koi_period_err2": 0.21247352280501483,
      }
    ]
  }
}
```