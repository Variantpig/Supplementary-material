source("xgb_pipeline_template.R")

run_xgb_pipeline(model_number = "model_1a")
run_xgb_pipeline(model_number = "model_1b")
run_xgb_pipeline(model_number = "model_2")
run_xgb_pipeline(model_number = "model_2a")
run_xgb_pipeline(model_number = "model_2b")
summary_files <- c(
  "results/model_1_xgboost_summary.csv",
  "results/model_1a_xgboost_summary.csv",
  "results/model_1b_xgboost_summary.csv",
  "results/model_2_xgboost_summary.csv",
  "results/model_2a_xgboost_summary.csv",
  "results/model_2b_xgboost_summary.csv"
)
xgb_summary_all <- summary_files %>%
  keep(file.exists) %>%
  map_dfr(~ read_csv(.x, show_col_types = FALSE)) %>%
  arrange(model_number)
print(xgb_summary_all)
