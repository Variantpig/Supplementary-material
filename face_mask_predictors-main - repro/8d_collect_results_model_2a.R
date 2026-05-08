library(readr)
library(dplyr)
library(purrr)
library(tibble)

# =========================================================
# 1. Settings
# =========================================================

file_count <- "04"
model_number <- "model_2a"

model_types <- c(
  "logistic_reg",
  "binary_tree",
  "xgboost",
  "rf"
)

# =========================================================
# 2. Helper function
# =========================================================

collect_one_model_result <- function(model_number, model_type) {
  rds_path <- paste0("results/", model_number, "_", model_type, ".rds")
  csv_path <- paste0("results/", model_number, "_", model_type, "_summary.csv")
  
  if (file.exists(rds_path)) {
    M <- readRDS(rds_path)
    
    if ("test_precision" %in% names(M)) {
      precision_vec <- M$test_precision
      recall_vec <- M$test_recall
      roc_auc_vec <- M$test_roc_auc
      accuracy_vec <- M$test_accuracy
      f1_vec <- M$test_f1
    } else {
      precision_vec <- M$precision
      recall_vec <- M$recall
      roc_auc_vec <- M$roc_auc
      accuracy_vec <- M$accuracy
      f1_vec <- M$f1
    }
    
    model_df <- tibble(
      model_number = model_number,
      model_type = model_type,
      precision = mean(precision_vec, na.rm = TRUE),
      precision_std = sd(precision_vec, na.rm = TRUE) / sqrt(sum(!is.na(precision_vec))),
      recall = mean(recall_vec, na.rm = TRUE),
      recall_std = sd(recall_vec, na.rm = TRUE) / sqrt(sum(!is.na(recall_vec))),
      roc_auc = mean(roc_auc_vec, na.rm = TRUE),
      roc_auc_std = sd(roc_auc_vec, na.rm = TRUE) / sqrt(sum(!is.na(roc_auc_vec))),
      accuracy = mean(accuracy_vec, na.rm = TRUE),
      accuracy_std = sd(accuracy_vec, na.rm = TRUE) / sqrt(sum(!is.na(accuracy_vec))),
      f1 = mean(f1_vec, na.rm = TRUE),
      f1_std = sd(f1_vec, na.rm = TRUE) / sqrt(sum(!is.na(f1_vec)))
    )
    
    return(model_df)
  }
  
  if (file.exists(csv_path)) {
    M <- read_csv(csv_path, show_col_types = FALSE)
    
    model_df <- tibble(
      model_number = model_number,
      model_type = model_type,
      precision = M$mean_precision[[1]],
      precision_std = NA_real_,
      recall = M$mean_recall[[1]],
      recall_std = NA_real_,
      roc_auc = M$mean_roc_auc[[1]],
      roc_auc_std = NA_real_,
      accuracy = M$mean_accuracy[[1]],
      accuracy_std = NA_real_,
      f1 = M$mean_f1[[1]],
      f1_std = NA_real_
    )
    
    return(model_df)
  }
  
  warning(paste("No result file found for", model_number, model_type))
  return(NULL)
}

# =========================================================
# 3. Collect all model results
# =========================================================

final_df <- map_dfr(
  model_types,
  ~ collect_one_model_result(model_number, .x)
)

print(final_df)

# =========================================================
# 4. Save
# =========================================================

write_csv(
  final_df,
  paste0("results/", file_count, "_", model_number, "_final_results.csv")
)