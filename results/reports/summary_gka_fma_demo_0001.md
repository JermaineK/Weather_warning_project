# Pipeline summary – gka_fma_demo (run 0001)

        **Labelled:** `data/grid_labelled_FMA_gka_realthermo.parquet`  
        **Model:** `models/grid_logit_cal.pkl`

        **Leads:** 24, 48, 72, 120, 240  
        **Thresholds column used:** thr_Fbeta (from `results/best_fbeta_thresholds.csv`)

        ## Outputs
        - Lead 24h -> alerts_gka_fma_demo_lead24_thr0.0435_throttled.csv.gz
- Lead 48h -> alerts_gka_fma_demo_lead48_thr0.0435_throttled.csv.gz
- Lead 72h -> alerts_gka_fma_demo_lead72_thr0.0435_throttled.csv.gz
- Lead 120h -> alerts_gka_fma_demo_lead120_thr0.0435_throttled.csv.gz
- Lead 240h -> alerts_gka_fma_demo_lead240_thr0.0435_throttled.csv.gz

        ## Coverage (post-throttle)
        | Lead (h) | Threshold | Hours | Active cells | Total cells | Coverage |
        |---------:|----------:|------:|-------------:|------------:|---------:|
        |      24 |    0.0435 |  2160 |      952,188 |  10,672,560 |   0.089 |
|      48 |    0.0435 |  2160 |      952,188 |  10,672,560 |   0.089 |
|      72 |    0.0435 |  2160 |      952,188 |  10,672,560 |   0.089 |
|     120 |    0.0435 |  2160 |      952,188 |  10,672,560 |   0.089 |
|     240 |    0.0435 |  2160 |      952,188 |  10,672,560 |   0.089 |

        ## Pipeline Diagnostics
        (see `results\reports\diagnostics_gka_fma_demo_0001.csv` for CSV)


| Lead | thr | base_rows | base_bad_time | base_cov | den_cov | thr_cov | keep_denoise | keep_throttle | flags |
|----:|----:|----------:|-------------:|---------:|--------:|--------:|------------:|--------------:|:------|
| 24 | 0.0435 | 10672560 | 0 | 0.098 | 0.099 | 0.089 | 1.006 | 0.901 |  |
| 48 | 0.0435 | 10672560 | 0 | 0.098 | 0.099 | 0.089 | 1.006 | 0.901 |  |
| 72 | 0.0435 | 10672560 | 0 | 0.098 | 0.099 | 0.089 | 1.006 | 0.901 |  |
| 120 | 0.0435 | 10672560 | 0 | 0.098 | 0.099 | 0.089 | 1.006 | 0.901 |  |
| 240 | 0.0435 | 10672560 | 0 | 0.098 | 0.099 | 0.089 | 1.006 | 0.901 |  |
