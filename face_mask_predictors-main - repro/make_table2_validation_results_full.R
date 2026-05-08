library(readr)
library(dplyr)
library(jsonlite)
library(ranger)
library(xgboost)
library(yardstick)
library(tibble)
library(stringr)
library(gt)
library(webshot2)
library(htmltools)

# =========================================================
# 1. Helpers
# =========================================================

find_existing_file <- function(candidates) {
  hit <- candidates[file.exists(candidates)]
  if (length(hit) == 0) {
    stop(
      paste0(
        "File not found. Tried:\n",
        paste(candidates, collapse = "\n")
      )
    )
  }
  hit[1]
}

get_data_paths <- function(model_number) {
  list(
    x_train = find_existing_file(c(
      paste0("data/X_train_", model_number, ".csv"),
      paste0("../data/X_train_", model_number, ".csv")
    )),
    y_train = find_existing_file(c(
      paste0("data/y_train_", model_number, ".csv"),
      paste0("../data/y_train_", model_number, ".csv")
    )),
    x_test = find_existing_file(c(
      paste0("data/X_test_", model_number, ".csv"),
      paste0("../data/X_test_", model_number, ".csv")
    )),
    y_test = find_existing_file(c(
      paste0("data/y_test_", model_number, ".csv"),
      paste0("../data/y_test_", model_number, ".csv")
    ))
  )
}

load_best_params <- function(model_number, model_type) {
  json_path <- find_existing_file(c(
    paste0("results/", model_number, "_", model_type, "_best_within_one.json"),
    paste0("../results/", model_number, "_", model_type, "_best_within_one.json")
  ))
  
  params <- read_json(json_path)
  
  params$number <- NULL
  params$value <- NULL
  params$std_err <- NULL
  params$threshold <- NULL
  
  params$n_estimators <- 250
  
  if ("max_depth" %in% names(params)) {
    params$max_depth <- as.integer(params$max_depth)
  }
  if ("min_node_size" %in% names(params)) {
    params$min_node_size <- as.integer(params$min_node_size)
  }
  if ("min_child_weight" %in% names(params)) {
    params$min_child_weight <- as.integer(params$min_child_weight)
  }
  
  params
}

max_features_to_mtry <- function(mode, p) {
  if (is.null(mode)) return(as.integer(p))
  
  mode <- as.character(mode)
  
  if (mode %in% c("all", "None", "none")) {
    return(as.integer(p))
  } else if (mode == "sqrt") {
    return(max(1L, floor(sqrt(p))))
  } else if (mode == "log2") {
    return(max(1L, floor(log2(p))))
  } else {
    return(as.integer(p))
  }
}

fit_xgb_model <- function(x_train, y_train, params, seed = 2026) {
  dtrain <- xgb.DMatrix(
    data = as.matrix(x_train),
    label = as.numeric(y_train)
  )
  
  xgb_params <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eta = as.numeric(params$learning_rate),
    max_depth = as.integer(params$max_depth),
    subsample = as.numeric(params$subsample),
    colsample_bytree = as.numeric(params$colsample_bytree),
    seed = seed,
    nthread = 4
  )
  
  if ("min_child_weight" %in% names(params)) {
    xgb_params$min_child_weight <- as.integer(params$min_child_weight)
  }
  if ("gamma" %in% names(params)) {
    xgb_params$gamma <- as.numeric(params$gamma)
  }
  if ("reg_lambda" %in% names(params)) {
    xgb_params$lambda <- as.numeric(params$reg_lambda)
  }
  
  xgb.train(
    params = xgb_params,
    data = dtrain,
    nrounds = as.integer(params$n_estimators),
    verbose = 0
  )
}

fit_rf_model <- function(x_train, y_train, params, seed = 2026) {
  train_df <- bind_cols(
    x_train,
    outcome = factor(as.character(y_train), levels = c("0", "1"))
  )
  
  p <- ncol(x_train)
  
  ranger(
    dependent.variable.name = "outcome",
    data = train_df,
    num.trees = as.integer(params$n_estimators),
    mtry = max_features_to_mtry(params$max_features_mode, p),
    min.node.size = as.integer(params$min_node_size),
    max.depth = as.integer(params$max_depth),
    probability = TRUE,
    replace = TRUE,
    importance = "impurity",
    seed = seed,
    num.threads = 4
  )
}

predict_xgb_prob <- function(model, x_test) {
  dtest <- xgb.DMatrix(data = as.matrix(x_test))
  as.numeric(predict(model, dtest))
}

predict_rf_prob <- function(model, x_test) {
  pred <- predict(model, data = x_test)$predictions
  as.numeric(pred[, "1"])
}

evaluate_binary_metrics <- function(y_true, prob_1, threshold = 0.5) {
  truth <- factor(as.character(y_true), levels = c("0", "1"))
  pred_class <- factor(ifelse(prob_1 >= threshold, "1", "0"), levels = c("0", "1"))
  
  eval_df <- tibble(
    truth = truth,
    pred_class = pred_class,
    .pred_1 = prob_1
  )
  
  tibble(
    auc = roc_auc(eval_df, truth = truth, .pred_1, event_level = "second")$.estimate,
    precision = precision(eval_df, truth = truth, estimate = pred_class, event_level = "second")$.estimate,
    recall = recall(eval_df, truth = truth, estimate = pred_class, event_level = "second")$.estimate,
    accuracy = accuracy(eval_df, truth = truth, estimate = pred_class)$.estimate,
    f1 = f_meas(eval_df, truth = truth, estimate = pred_class, event_level = "second")$.estimate
  )
}

clean_model_name <- function(x) {
  recode(
    x,
    "xgboost" = "XGBoost",
    "rf" = "random forest",
    .default = x
  )
}

format_metric <- function(x) {
  sprintf("%.3f", x)
}

make_section_block <- function(df_section) {
  section_label <- unique(df_section$section)
  
  section_row <- tibble(
    row_type = "section",
    section = section_label,
    model_label = section_label,
    auc_disp = "",
    precision_disp = "",
    recall_disp = "",
    accuracy_disp = "",
    f1_disp = ""
  )
  
  model_rows <- df_section %>%
    mutate(
      row_type = "model",
      auc_disp = format_metric(auc),
      precision_disp = format_metric(precision),
      recall_disp = format_metric(recall),
      accuracy_disp = format_metric(accuracy),
      f1_disp = format_metric(f1)
    ) %>%
    select(
      row_type, section, model_label,
      auc_disp, precision_disp, recall_disp, accuracy_disp, f1_disp
    )
  
  bind_rows(section_row, model_rows)
}

# =========================================================
# 2. One model_number => evaluate XGBoost + RF on test set
# =========================================================

run_validation_for_one_model <- function(model_number) {
  cat("\n====================================\n")
  cat("Running validation for:", model_number, "\n")
  cat("====================================\n")
  
  paths <- get_data_paths(model_number)
  
  x_train <- read_csv(paths$x_train, show_col_types = FALSE)
  y_train <- read_csv(paths$y_train, show_col_types = FALSE)[[1]]
  
  x_test <- read_csv(paths$x_test, show_col_types = FALSE)
  y_test <- read_csv(paths$y_test, show_col_types = FALSE)[[1]]
  
  # ---- XGBoost ----
  xgb_params <- load_best_params(model_number, "xgboost")
  xgb_fit <- fit_xgb_model(x_train, y_train, xgb_params)
  xgb_prob <- predict_xgb_prob(xgb_fit, x_test)
  xgb_metrics <- evaluate_binary_metrics(y_test, xgb_prob) %>%
    mutate(model_type = "xgboost")
  
  # ---- Random Forest ----
  rf_params <- load_best_params(model_number, "rf")
  rf_fit <- fit_rf_model(x_train, y_train, rf_params)
  rf_prob <- predict_rf_prob(rf_fit, x_test)
  rf_metrics <- evaluate_binary_metrics(y_test, rf_prob) %>%
    mutate(model_type = "rf")
  
  validation_df <- bind_rows(xgb_metrics, rf_metrics) %>%
    select(model_type, auc, precision, recall, accuracy, f1)
  
  out_path <- paste0("results/", model_number, "_validation_results.csv")
  write_csv(validation_df, out_path)
  
  cat("Saved:", out_path, "\n")
  validation_df
}

# =========================================================
# 3. Run all four model groups
# =========================================================

val_1a <- run_validation_for_one_model("model_1a")
val_1b <- run_validation_for_one_model("model_1b")
val_2a <- run_validation_for_one_model("model_2a")
val_2b <- run_validation_for_one_model("model_2b")

# =========================================================
# 4. Prepare Table 2 data
# =========================================================

before_mask <- val_1a %>%
  mutate(
    section = "before mandates—face mask wearing",
    model_label = clean_model_name(model_type)
  )

after_mask <- val_1b %>%
  mutate(
    section = "after mandates—face mask wearing",
    model_label = clean_model_name(model_type)
  )

before_prot <- val_2a %>%
  mutate(
    section = "before mandates—general protective behaviour",
    model_label = clean_model_name(model_type)
  )

after_prot <- val_2b %>%
  mutate(
    section = "after mandates—general protective behaviour",
    model_label = clean_model_name(model_type)
  )

table_df <- bind_rows(
  make_section_block(before_mask),
  make_section_block(after_mask),
  make_section_block(before_prot),
  make_section_block(after_prot)
) %>%
  mutate(row_id = row_number())

# =========================================================
# 5. Render Table 2
# =========================================================

gt_tbl <- table_df %>%
  gt(rowname_col = "model_label") %>%
  cols_hide(columns = c(row_type, section, row_id)) %>%
  cols_label(
    auc_disp = "AUC",
    precision_disp = "precision",
    recall_disp = "recall",
    accuracy_disp = "accuracy",
    f1_disp = "F1"
  ) %>%
  tab_stubhead(label = "") %>%
  tab_header(
    title = md("**Table 2.** Metric evaluation on an independent validation set for the optimal models predicting face mask wearing and general protective health behaviours before and after face mask mandates are enacted. Metric scores can range from 0 (low) to 1 (high).")
  ) %>%
  tab_options(
    table.width = pct(100),
    heading.align = "left",
    table.font.names = c("Arial", "Helvetica", "sans-serif"),
    table.font.size = px(16),
    data_row.padding = px(9),
    heading.padding = px(6),
    column_labels.padding = px(10),
    row_group.padding = px(4),
    table.border.top.width = px(0),
    table.border.bottom.width = px(0),
    column_labels.border.top.width = px(0),
    column_labels.border.bottom.width = px(0)
  ) %>%
  tab_style(
    style = list(
      cell_fill(color = "black"),
      cell_text(color = "white", weight = "bold", align = "center", size = px(17))
    ),
    locations = cells_column_labels(columns = everything())
  ) %>%
  tab_style(
    style = list(
      cell_text(weight = "normal", size = px(17), align = "left")
    ),
    locations = cells_stub(rows = row_type == "section")
  ) %>%
  tab_style(
    style = list(
      cell_text(weight = "normal", size = px(17), align = "left")
    ),
    locations = cells_stub(rows = row_type == "model")
  ) %>%
  tab_style(
    style = cell_borders(
      sides = "bottom",
      color = "#444444",
      style = "dotted",
      weight = px(1.2)
    ),
    locations = cells_body(
      columns = everything(),
      rows = row_type %in% c("section", "model")
    )
  ) %>%
  tab_style(
    style = cell_borders(
      sides = "bottom",
      color = "#444444",
      style = "dotted",
      weight = px(1.2)
    ),
    locations = cells_stub(
      rows = row_type %in% c("section", "model")
    )
  ) %>%
  cols_align(
    align = "center",
    columns = c(auc_disp, precision_disp, recall_disp, accuracy_disp, f1_disp)
  )

# =========================================================
# 6. Save outputs
# =========================================================

gtsave(gt_tbl, "results/table2_validation_results.html")
gtsave(gt_tbl, "results/table2_validation_results.png", zoom = 2)



gt_tbl