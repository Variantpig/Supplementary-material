library(xgboost)
library(readr)
library(dplyr)
library(jsonlite)
library(ggplot2)
library(ranger)
library(yardstick)
library(purrr)
library(tibble)

# =========================================================
# 0. Global settings
# =========================================================

XGB_MODEL_TYPE <- "xgboost"
XGB_SEED_CV <- 20240627
XGB_SEED_TUNE <- 2020

dir.create("results", recursive = TRUE, showWarnings = FALSE)
dir.create("figures/6aa_hyperparameter_importance", recursive = TRUE, showWarnings = FALSE)

# =========================================================
# 1. Helpers: load data + basic utilities
# =========================================================

load_model_xy <- function(model_number) {
  x <- read_csv(
    paste0("data/X_train_", model_number, ".csv"),
    show_col_types = FALSE
  )
  
  y <- read_csv(
    paste0("data/y_train_", model_number, ".csv"),
    show_col_types = FALSE
  )[[1]]
  
  list(
    x = x,
    y = as.numeric(y)
  )
}

compute_scale_pos_weight <- function(obj) {
  if (is.data.frame(obj)) {
    if (!"outcome" %in% names(obj)) {
      stop("If a data frame is supplied, it must contain a column named 'outcome'.")
    }
    y <- obj$outcome
  } else {
    y <- obj
  }
  
  y <- as.numeric(as.character(y))
  sum(1 - y) / sum(y)
}

sample_log_uniform <- function(n, low, high) {
  exp(runif(n, log(low), log(high)))
}

# =========================================================
# 2. StratifiedShuffleSplit-like resampling
# =========================================================

make_stratified_splits <- function(y, n_splits = 5, test_size = 0.2, seed = XGB_SEED_CV) {
  set.seed(seed)
  
  y <- as.numeric(y)
  idx_0 <- which(y == 0)
  idx_1 <- which(y == 1)
  
  n_test_0 <- max(1, round(length(idx_0) * test_size))
  n_test_1 <- max(1, round(length(idx_1) * test_size))
  
  splits <- vector("list", n_splits)
  
  for (i in seq_len(n_splits)) {
    test_idx <- c(
      sample(idx_0, size = n_test_0, replace = FALSE),
      sample(idx_1, size = n_test_1, replace = FALSE)
    )
    train_idx <- setdiff(seq_along(y), test_idx)
    
    splits[[i]] <- list(
      train_idx = train_idx,
      test_idx = test_idx
    )
  }
  
  splits
}

# =========================================================
# 3. Metrics
# =========================================================

make_metric_df <- function(y_true, y_prob, threshold = 0.5) {
  y_pred <- ifelse(y_prob >= threshold, 1, 0)
  
  tibble(
    truth = factor(as.character(y_true), levels = c("0", "1")),
    .pred_1 = as.numeric(y_prob),
    .pred_class = factor(as.character(y_pred), levels = c("0", "1"))
  )
}

calc_metrics <- function(y_true, y_prob) {
  pred_df <- make_metric_df(y_true, y_prob)
  
  tibble(
    precision = yardstick::precision(
      pred_df, truth = truth, estimate = .pred_class, event_level = "second"
    )$.estimate,
    recall = yardstick::recall(
      pred_df, truth = truth, estimate = .pred_class, event_level = "second"
    )$.estimate,
    roc_auc = yardstick::roc_auc(
      pred_df, truth = truth, .pred_1, event_level = "second"
    )$.estimate,
    accuracy = yardstick::accuracy(
      pred_df, truth = truth, estimate = .pred_class
    )$.estimate,
    f1 = yardstick::f_meas(
      pred_df, truth = truth, estimate = .pred_class, event_level = "second"
    )$.estimate
  )
}

# =========================================================
# 4. Core fitter: one parameter set across 5 splits
# =========================================================

fit_predict_xgb_cv <- function(x, y, param_list, n_estimators = 250, seed = XGB_SEED_CV) {
  splits <- make_stratified_splits(
    y = y,
    n_splits = 5,
    test_size = 0.2,
    seed = seed
  )
  
  fold_metrics <- map_dfr(seq_along(splits), function(i) {
    tr_idx <- splits[[i]]$train_idx
    te_idx <- splits[[i]]$test_idx
    
    x_train <- as.matrix(x[tr_idx, , drop = FALSE])
    x_test  <- as.matrix(x[te_idx, , drop = FALSE])
    
    y_train <- y[tr_idx]
    y_test  <- y[te_idx]
    
    dtrain <- xgb.DMatrix(data = x_train, label = y_train)
    dtest  <- xgb.DMatrix(data = x_test, label = y_test)
    
    params <- c(
      list(
        booster = "gbtree",
        objective = "binary:logistic",
        eval_metric = "auc",
        scale_pos_weight = compute_scale_pos_weight(y_train)
      ),
      param_list
    )
    
    model <- xgb.train(
      params = params,
      data = dtrain,
      nrounds = n_estimators,
      verbose = 0
    )
    
    pred_prob <- predict(model, dtest)
    
    calc_metrics(y_true = y_test, y_prob = pred_prob) %>%
      mutate(fold = i)
  })
  
  fold_metrics
}

# =========================================================
# 5. Parameter samplers
# =========================================================


sample_params_6aa <- function(n, seed = XGB_SEED_TUNE) {
  set.seed(seed)
  
  tibble(
    learning_rate = runif(n, 0.01, 1),
    max_depth = sample(2:10, n, replace = TRUE),
    min_child_weight = sample(1:10, n, replace = TRUE),
    subsample = runif(n, 0.1, 1),
    colsample_bytree = runif(n, 0.5, 0.9),
    gamma = sample_log_uniform(n, 1e-8, 1.0),
    reg_lambda = sample_log_uniform(n, 1e-8, 1.0)
  )
}

sample_params_6ab <- function(n, seed = XGB_SEED_TUNE) {
  set.seed(seed)
  
  tibble(
    learning_rate = runif(n, 0.01, 1),
    max_depth = sample(2:10, n, replace = TRUE),
    subsample = runif(n, 0.1, 1),
    colsample_bytree = runif(n, 0.5, 0.9)
  )
}

# =========================================================
# 6. Save helpers
# =========================================================

save_json_lines <- function(df, file_path, phase = "6ab") {
  lines_out <- lapply(seq_len(nrow(df)), function(i) {
    if (phase == "6aa") {
      obj <- list(
        number = df$number[i],
        value = df$value[i],
        params = list(
          learning_rate = df$learning_rate[i],
          max_depth = df$max_depth[i],
          min_child_weight = df$min_child_weight[i],
          subsample = df$subsample[i],
          colsample_bytree = df$colsample_bytree[i],
          gamma = df$gamma[i],
          reg_lambda = df$reg_lambda[i]
        ),
        user_attrs = list(
          std_err = df$std_err[i]
        )
      )
    } else {
      obj <- list(
        number = df$number[i],
        value = df$value[i],
        params = list(
          learning_rate = df$learning_rate[i],
          max_depth = df$max_depth[i],
          subsample = df$subsample[i],
          colsample_bytree = df$colsample_bytree[i]
        ),
        user_attrs = list(
          std_err = df$std_err[i]
        )
      )
    }
    
    jsonlite::toJSON(obj, auto_unbox = TRUE, null = "null")
  })
  
  writeLines(unlist(lines_out), con = file_path)
}

save_best_trial_json <- function(best_trial_df, file_path) {
  obj <- list(
    number = best_trial_df$number[[1]],
    value = best_trial_df$value[[1]],
    params = best_trial_df %>%
      select(-number, -value, -std_err) %>%
      as.list(),
    user_attrs = list(
      std_err = best_trial_df$std_err[[1]]
    )
  )
  
  write_json(obj, file_path, auto_unbox = TRUE, pretty = TRUE)
}

# =========================================================
# 7. 6aa: exploratory tuning + importance
# =========================================================

run_6aa <- function(model_number, trials_6aa = 50) {
  dat <- load_model_xy(model_number)
  x <- dat$x
  y <- dat$y
  
  param_grid <- sample_params_6aa(trials_6aa)
  
  cat("\n[6aa] Starting exploratory tuning for", model_number, "\n")
  cat("[6aa] Total trials:", trials_6aa, "\n")
  flush.console()
  
  results_list <- vector("list", nrow(param_grid))
  
  for (i in seq_len(nrow(param_grid))) {
    cat(sprintf("[6aa] Trial %d / %d ...\n", i, nrow(param_grid)))
    flush.console()
    
    params <- list(
      eta = param_grid$learning_rate[i],
      max_depth = as.integer(param_grid$max_depth[i]),
      min_child_weight = as.integer(param_grid$min_child_weight[i]),
      subsample = param_grid$subsample[i],
      colsample_bytree = param_grid$colsample_bytree[i],
      gamma = param_grid$gamma[i],
      lambda = param_grid$reg_lambda[i]
    )
    
    fold_metrics <- fit_predict_xgb_cv(
      x = x,
      y = y,
      param_list = params,
      n_estimators = 250,
      seed = XGB_SEED_CV
    )
    
    one_result <- tibble(
      number = i,
      value = mean(fold_metrics$roc_auc, na.rm = TRUE),
      learning_rate = param_grid$learning_rate[i],
      max_depth = param_grid$max_depth[i],
      min_child_weight = param_grid$min_child_weight[i],
      subsample = param_grid$subsample[i],
      colsample_bytree = param_grid$colsample_bytree[i],
      gamma = param_grid$gamma[i],
      reg_lambda = param_grid$reg_lambda[i],
      std_err = sd(fold_metrics$roc_auc, na.rm = TRUE) / sqrt(nrow(fold_metrics))
    )
    
    results_list[[i]] <- one_result
    
    cat(sprintf(
      "[6aa] Trial %d / %d finished | roc_auc = %.4f\n",
      i, nrow(param_grid), one_result$value[[1]]
    ))
    flush.console()
  }
  
  trial_results <- bind_rows(results_list) %>%
    arrange(desc(value))
  
  write_csv(
    trial_results,
    paste0("results/", model_number, "_", XGB_MODEL_TYPE, "_trials_6aa.csv")
  )
  
  save_json_lines(
    trial_results,
    paste0("results/", model_number, "_", XGB_MODEL_TYPE, "_trials_6aa.json"),
    phase = "6aa"
  )
  
  importance_fit <- ranger::ranger(
    value ~ learning_rate + max_depth + min_child_weight + subsample + colsample_bytree + gamma + reg_lambda,
    data = as.data.frame(trial_results),
    num.trees = 1000,
    importance = "permutation",
    seed = XGB_SEED_TUNE
  )
  
  importance_df <- tibble(
    parameter = names(importance_fit$variable.importance),
    importance = as.numeric(importance_fit$variable.importance)
  ) %>%
    arrange(desc(importance))
  
  write_csv(
    importance_df,
    paste0("results/", model_number, "_", XGB_MODEL_TYPE, "_hyperparameter_importance_6aa.csv")
  )
  
  p_importance <- ggplot(
    importance_df,
    aes(x = importance, y = reorder(parameter, importance))
  ) +
    geom_col(fill = "#4E79A7", width = 0.68) +
    geom_text(
      aes(label = round(importance, 3)),
      hjust = -0.1,
      size = 3.3,
      colour = "#222222"
    ) +
    labs(
      title = paste("Hyperparameter Importance for", model_number, "XGBoost"),
      subtitle = "Exploratory tuning importance based on mean ROC AUC",
      x = "Importance",
      y = "Hyperparameter"
    ) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
    theme_classic(base_size = 12)
  
  ggsave(
    paste0(
      "figures/6aa_hyperparameter_importance/",
      model_number, "_", XGB_MODEL_TYPE, "_hyperparameter_importance.png"
    ),
    plot = p_importance,
    width = 8,
    height = 5,
    dpi = 300
  )
  
  cat("[6aa] Completed for", model_number, "\n")
  flush.console()
  
  list(
    trial_results = trial_results,
    importance_df = importance_df
  )
}

# =========================================================
# 8. 6ab: focused tuning
# =========================================================


run_6ab <- function(model_number, trials_6ab = 100) {
  dat <- load_model_xy(model_number)
  x <- dat$x
  y <- dat$y
  
  param_grid <- sample_params_6ab(trials_6ab)
  
  cat("\n[6ab] Starting focused tuning for", model_number, "\n")
  cat("[6ab] Total trials:", trials_6ab, "\n")
  flush.console()
  
  results_list <- vector("list", nrow(param_grid))
  
  for (i in seq_len(nrow(param_grid))) {
    cat(sprintf("[6ab] Trial %d / %d ...\n", i, nrow(param_grid)))
    flush.console()
    
    params <- list(
      eta = param_grid$learning_rate[i],
      max_depth = as.integer(param_grid$max_depth[i]),
      subsample = param_grid$subsample[i],
      colsample_bytree = param_grid$colsample_bytree[i]
    )
    
    fold_metrics <- fit_predict_xgb_cv(
      x = x,
      y = y,
      param_list = params,
      n_estimators = 250,
      seed = XGB_SEED_CV
    )
    
    one_result <- tibble(
      number = i,
      value = mean(fold_metrics$roc_auc, na.rm = TRUE),
      learning_rate = param_grid$learning_rate[i],
      max_depth = param_grid$max_depth[i],
      subsample = param_grid$subsample[i],
      colsample_bytree = param_grid$colsample_bytree[i],
      std_err = sd(fold_metrics$roc_auc, na.rm = TRUE) / sqrt(nrow(fold_metrics))
    )
    
    results_list[[i]] <- one_result
    
    cat(sprintf(
      "[6ab] Trial %d / %d finished | roc_auc = %.4f\n",
      i, nrow(param_grid), one_result$value[[1]]
    ))
    flush.console()
  }
  
  trial_results <- bind_rows(results_list) %>%
    arrange(desc(value))
  
  write_csv(
    trial_results,
    paste0("results/", model_number, "_", XGB_MODEL_TYPE, "_trials_6ab.csv")
  )
  
  save_json_lines(
    trial_results,
    paste0("results/", model_number, "_", XGB_MODEL_TYPE, "_trials.json"),
    phase = "6ab"
  )
  
  best_trial <- trial_results %>%
    filter(!is.na(value)) %>%
    slice_max(value, n = 1, with_ties = FALSE)
  
  save_best_trial_json(
    best_trial,
    paste0("results/", model_number, "_", XGB_MODEL_TYPE, "_trial_best.json")
  )
  
  cat(sprintf("[6ab] Best roc_auc = %.4f\n", best_trial$value[[1]]))
  cat("[6ab] Completed for", model_number, "\n")
  flush.console()
  
  list(
    trial_results = trial_results,
    best_trial = best_trial
  )
}

# =========================================================
# 9. 6ac: best within one std err
# =========================================================

run_6ac <- function(model_number) {
  trial_results <- read_csv(
    paste0("results/", model_number, "_", XGB_MODEL_TYPE, "_trials_6ab.csv"),
    show_col_types = FALSE
  )
  
  best_trial <- read_json(
    paste0("results/", model_number, "_", XGB_MODEL_TYPE, "_trial_best.json")
  )
  
  within_one_std_err <- best_trial$value - best_trial$user_attrs$std_err
  
  best_shots <- trial_results %>%
    filter(!is.na(value)) %>%
    filter(value >= within_one_std_err)
  
  if (nrow(best_shots) == 0) {
    warning("No trial found within one standard error; using the best trial directly.")
    
    best_within_one <- tibble(
      number = best_trial$number,
      value = best_trial$value,
      learning_rate = best_trial$params$learning_rate,
      max_depth = best_trial$params$max_depth,
      subsample = best_trial$params$subsample,
      colsample_bytree = best_trial$params$colsample_bytree,
      std_err = best_trial$user_attrs$std_err
    )
  } else {
    best_within_one <- best_shots %>%
      arrange(
        learning_rate,
        subsample,
        max_depth,
        colsample_bytree
      ) %>%
      slice(1)
  }
  
  best_within_one_json <- list(
    number = best_within_one$number[[1]],
    value = best_within_one$value[[1]],
    learning_rate = best_within_one$learning_rate[[1]],
    max_depth = best_within_one$max_depth[[1]],
    subsample = best_within_one$subsample[[1]],
    colsample_bytree = best_within_one$colsample_bytree[[1]],
    std_err = best_within_one$std_err[[1]],
    threshold = within_one_std_err
  )
  
  write_json(
    best_within_one_json,
    paste0("results/", model_number, "_", XGB_MODEL_TYPE, "_best_within_one.json"),
    auto_unbox = TRUE,
    pretty = TRUE
  )
  
  list(
    threshold = within_one_std_err,
    candidate_count = nrow(best_shots),
    best_within_one = best_within_one
  )
}

# =========================================================
# 10. 6ad: final CV with selected params
# =========================================================

run_6ad <- function(model_number) {
  dat <- load_model_xy(model_number)
  x <- dat$x
  y <- dat$y
  
  selected_params <- read_json(
    paste0("results/", model_number, "_", XGB_MODEL_TYPE, "_best_within_one.json")
  )
  
  params <- list(
    eta = as.numeric(selected_params$learning_rate),
    max_depth = as.integer(selected_params$max_depth),
    subsample = as.numeric(selected_params$subsample),
    colsample_bytree = as.numeric(selected_params$colsample_bytree)
  )
  
  fold_metrics <- fit_predict_xgb_cv(
    x = x,
    y = y,
    param_list = params,
    n_estimators = 250,
    seed = XGB_SEED_CV
  )
  
  saveRDS(
    fold_metrics,
    file = paste0("results/", model_number, "_", XGB_MODEL_TYPE, ".rds")
  )
  
  summary_df <- tibble(
    model_number = model_number,
    mean_precision = mean(fold_metrics$precision, na.rm = TRUE),
    mean_recall = mean(fold_metrics$recall, na.rm = TRUE),
    mean_roc_auc = mean(fold_metrics$roc_auc, na.rm = TRUE),
    mean_accuracy = mean(fold_metrics$accuracy, na.rm = TRUE),
    mean_f1 = mean(fold_metrics$f1, na.rm = TRUE)
  )
  
  write_csv(
    summary_df,
    paste0("results/", model_number, "_", XGB_MODEL_TYPE, "_summary.csv")
  )
  
  cat("\n", XGB_MODEL_TYPE, "-", model_number, "\n", sep = "")
  cat("Mean recall: ", round(summary_df$mean_recall, 3), "\n", sep = "")
  cat("Mean roc: ", round(summary_df$mean_roc_auc, 3), "\n", sep = "")
  cat("Mean accuracy: ", round(summary_df$mean_accuracy, 3), "\n", sep = "")
  
  list(
    fold_metrics = fold_metrics,
    summary_df = summary_df
  )
}

# =========================================================
# 11. One-click full pipeline
# =========================================================

run_xgb_pipeline <- function(model_number,
                             trials_6aa = 50,
                             trials_6ab = 100) {
  cat("\n==============================\n")
  cat("Running XGBoost pipeline for:", model_number, "\n")
  cat("==============================\n")
  flush.console()
  
  cat("[Pipeline] Step 6aa begins...\n")
  flush.console()
  out_6aa <- run_6aa(model_number, trials_6aa = trials_6aa)
  
  cat("[Pipeline] Step 6ab begins...\n")
  flush.console()
  out_6ab <- run_6ab(model_number, trials_6ab = trials_6ab)
  
  cat("[Pipeline] Step 6ac begins...\n")
  flush.console()
  out_6ac <- run_6ac(model_number)
  
  cat("[Pipeline] Step 6ad begins...\n")
  flush.console()
  out_6ad <- run_6ad(model_number)
  
  cat("[Pipeline] All steps completed for ", model_number, "\n", sep = "")
  flush.console()
  
  invisible(list(
    step_6aa = out_6aa,
    step_6ab = out_6ab,
    step_6ac = out_6ac,
    step_6ad = out_6ad
  ))
}