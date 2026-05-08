library(readr)
library(dplyr)
library(tidyr)
library(tibble)
library(purrr)

# =========================================================
# 1. Helper function:
#    read feature importance file and get top 10 variables
#    using median importance across repeated runs
# =========================================================

get_top10_importance <- function(file_path, top_n = 10) {
  df <- read_csv(file_path, show_col_types = FALSE)
  
  # 防止有些文件里带了多余索引列
  if ("...1" %in% names(df)) {
    df <- df %>% select(-"...1")
  }
  
  df_long <- df %>%
    pivot_longer(
      cols = everything(),
      names_to = "variable",
      values_to = "importance"
    )
  
  top_df <- df_long %>%
    group_by(variable) %>%
    summarise(
      median_importance = median(importance, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(median_importance)) %>%
    slice_head(n = top_n) %>%
    mutate(rank = row_number()) %>%
    select(rank, variable, median_importance)
  
  return(top_df)
}

# =========================================================
# 2. Helper function:
#    combine before/after into one comparison table
# =========================================================

make_comparison_table <- function(before_file, after_file,
                                  before_label = "Before mandates",
                                  after_label = "After mandates",
                                  top_n = 10) {
  before_top <- get_top10_importance(before_file, top_n = top_n) %>%
    rename(
      rank_before = rank,
      variable_before = variable,
      median_importance_before = median_importance
    )
  
  after_top <- get_top10_importance(after_file, top_n = top_n) %>%
    rename(
      rank_after = rank,
      variable_after = variable,
      median_importance_after = median_importance
    )
  
  comparison_table <- bind_cols(before_top, after_top)
  
  return(comparison_table)
}

# =========================================================
# 3. Create output folder
# =========================================================

dir.create("results/top10_feature_comparison_tables", recursive = TRUE, showWarnings = FALSE)

# =========================================================
# 4. Table 1:
#    XGBoost - mask wearing (model_1a vs model_1b)
# =========================================================

table_1_xgb_mask <- make_comparison_table(
  before_file = "results/model_1a_xgboost_feature_importance.csv",
  after_file  = "results/model_1b_xgboost_feature_importance.csv"
)

write_csv(
  table_1_xgb_mask,
  "results/top10_feature_comparison_tables/table_1_xgb_mask_before_after.csv"
)

print(table_1_xgb_mask)

# =========================================================
# 5. Table 2:
#    RF - mask wearing (model_1a vs model_1b)
# =========================================================

table_2_rf_mask <- make_comparison_table(
  before_file = "results/model_1a_rf_feature_importance.csv",
  after_file  = "results/model_1b_rf_feature_importance.csv"
)

write_csv(
  table_2_rf_mask,
  "results/top10_feature_comparison_tables/table_2_rf_mask_before_after.csv"
)

print(table_2_rf_mask)

# =========================================================
# 6. Table 3:
#    XGBoost - general protective behaviour (model_2a vs model_2b)
# =========================================================

table_3_xgb_protective <- make_comparison_table(
  before_file = "results/model_2a_xgboost_feature_importance.csv",
  after_file  = "results/model_2b_xgboost_feature_importance.csv"
)

write_csv(
  table_3_xgb_protective,
  "results/top10_feature_comparison_tables/table_3_xgb_protective_before_after.csv"
)

print(table_3_xgb_protective)

# =========================================================
# 7. Table 4:
#    RF - general protective behaviour (model_2a vs model_2b)
# =========================================================

table_4_rf_protective <- make_comparison_table(
  before_file = "results/model_2a_rf_feature_importance.csv",
  after_file  = "results/model_2b_rf_feature_importance.csv"
)

write_csv(
  table_4_rf_protective,
  "results/top10_feature_comparison_tables/table_4_rf_protective_before_after.csv"
)

print(table_4_rf_protective)