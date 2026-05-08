library(ranger)
library(readr)
library(dplyr)
library(jsonlite)
library(ggplot2)
library(yardstick)
library(purrr)
library(tibble)

# =========================================================
# 0. Global settings
# =========================================================

RF_MODEL_TYPE <- "rf"
RF_SEED_CV <- 20240627
RF_SEED_TUNE <- 2013

dir.create("results", recursive = TRUE, showWarnings = FALSE)
dir.create("figures/7aa_hyperparameter_importance", recursive = TRUE, showWarnings = FALSE)

# =========================================================
# 1. Helpers: load data
# =========================================================

load_model_xy_rf <- function(model_number) {
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

# =========================================================
# 2. StratifiedShuffleSplit-like helper
# =========================================================

make_stratified_splits <- function(y, n_splits = 5, test_size = 0.2, seed = RF_SEED_CV) {
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
# 4. Helper: parameter samplers
# =========================================================

sample_int_log_uniform <- function(n, low, high) {
  vals <- round(exp(runif(n, log(low), log(high))))
  vals <- pmax(low, pmin(high, vals))
  as.integer(vals)
}

max_features_to_mtry <- function(mode, p) {
  if (mode == "sqrt") {
    return(max(1L, floor(sqrt(p))))
  } else if (mode == "log2") {
    return(max(1L, floor(log2(p))))
  } else {
    return(as.integer(p))
  }
}

balanced_class_weights <- function(y_factor) {
  counts <- table(y_factor)
  n <- sum(counts)
  k <- length(counts)
  as.numeric(n / (k * counts))
}

sample_params_7aa <- function(n, seed = RF_SEED_TUNE) {
  set.seed(seed)
  
  tibble(
    max_depth = sample_int_log_uniform(n, 2, 32),
    num_trees = sample(50:1000, n, replace = TRUE),
    splitrule = sample(c("gini", "extratrees"), n, replace = TRUE),
    min_node_size = sample(1:200, n, replace = TRUE),
    max_features_mode = sample(c("sqrt", "log2", "all"), n, replace = TRUE),
    class_weight_mode = sample(c("none", "balanced"), n, replace = TRUE)
  )
}

sample_params_7ab <- function(n, seed = RF_SEED_TUNE) {
  set.seed(seed)
  
  tibble(
    max_depth = sample_int_log_uniform(n, 2, 32),
    min_node_size = sample(1:200, n, replace = TRUE),
    max_features_mode = sample(c("sqrt", "log2", "all"), n, replace = TRUE)
  )
}

# =========================================================
# 5. Core fitter: one parameter set across 5 splits
# =========================================================

fit_predict_rf_cv <- function(x, y,
                              num_trees,
                              max_depth,
                              min_node_size,
                              max_features_mode,
                              splitrule = "gini",
                              class_weight_mode = "none",
                              seed = RF_SEED_CV) {
  splits <- make_stratified_splits(
    y = y,
    n_splits = 5,
    test_size = 0.2,
    seed = seed
  )
  
  p <- ncol(x)
  
  fold_metrics <- map_dfr(seq_along(splits), function(i) {
    tr_idx <- splits[[i]]$train_idx
    te_idx <- splits[[i]]$test_idx
    
    x_train <- x[tr_idx, , drop = FALSE]
    x_test  <- x[te_idx, , drop = FALSE]
    
    y_train <- factor(as.character(y[tr_idx]), levels = c("0", "1"))
    y_test  <- y[te_idx]
    
    train_df <- bind_cols(x_train, outcome = y_train)
    
    class_weights <- NULL
    if (class_weight_mode == "balanced") {
      class_weights <- balanced_class_weights(y_train)
    }
    
    rf_fit <- ranger(
      dependent.variable.name = "outcome",
      data = train_df,
      num.trees = as.integer(num_trees),
      mtry = max_features_to_mtry(max_features_mode, p),
      min.node.size = as.integer(min_node_size),
      max.depth = as.integer(max_depth),
      splitrule = splitrule,
      probability = TRUE,
      bootstrap = TRUE,
      class.weights = class_weights,
      seed = seed + i,
      num.threads = 4
    )
    
    pred <- predict(rf_fit, data = x_test)$predictions
    
    prob1 <- if ("1" %in% colnames(pred)) pred[, "1"] else pred[, 2]
    
    calc_metrics(y_true = y_test, y_prob = prob1) %>%
      mutate(fold = i)
  })
  
  fold_metrics
}

# =========================================================
# 6. Save helpers
# =========================================================

save_rf_json_lines <- function(df, file_path, phase = "7ab") {
  lines_out <- lapply(seq_len(nrow(df)), function(i) {
    if (phase == "7aa") {
      obj <- list(
        number = df$number[i],
        value = df$value[i],
        params = list(
          max_depth = df$max_depth[i],
          n_estimators = df$num_trees[i],
          splitrule = df$splitrule[i],
          min_node_size = df$min_node_size[i],
          max_features_mode = df$max_features_mode[i],
          class_weight_mode = df$class_weight_mode[i]
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
          max_depth = df$max_depth[i],
          min_node_size = df$min_node_size[i],
          max_features_mode = df$max_features_mode[i]
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

save_rf_best_trial_json <- function(best_trial_df, file_path) {
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
# 7. 7aa: exploratory tuning + importance
# =========================================================


run_7aa <- function(model_number, trials_7aa = 50) {
  dat <- load_model_xy_rf(model_number)
  x <- dat$x
  y <- dat$y
  
  param_grid <- sample_params_7aa(trials_7aa)
  
  cat("\n[7aa] Starting exploratory tuning for", model_number, "\n")
  cat("[7aa] Total trials:", trials_7aa, "\n")
  flush.console()
  
  results_list <- vector("list", nrow(param_grid))
  
  for (i in seq_len(nrow(param_grid))) {
    cat(sprintf("[7aa] Trial %d / %d ...\n", i, nrow(param_grid)))
    flush.console()
    
    fold_metrics <- fit_predict_rf_cv(
      x = x,
      y = y,
      num_trees = param_grid$num_trees[i],
      max_depth = param_grid$max_depth[i],
      min_node_size = param_grid$min_node_size[i],
      max_features_mode = param_grid$max_features_mode[i],
      splitrule = param_grid$splitrule[i],
      class_weight_mode = param_grid$class_weight_mode[i],
      seed = RF_SEED_CV
    )
    
    one_result <- tibble(
      number = i,
      value = mean(fold_metrics$roc_auc, na.rm = TRUE),
      max_depth = param_grid$max_depth[i],
      num_trees = param_grid$num_trees[i],
      splitrule = param_grid$splitrule[i],
      min_node_size = param_grid$min_node_size[i],
      max_features_mode = param_grid$max_features_mode[i],
      class_weight_mode = param_grid$class_weight_mode[i],
      std_err = sd(fold_metrics$roc_auc, na.rm = TRUE) / sqrt(nrow(fold_metrics))
    )
    
    results_list[[i]] <- one_result
    
    cat(sprintf(
      "[7aa] Trial %d / %d finished | roc_auc = %.4f\n",
      i, nrow(param_grid), one_result$value[[1]]
    ))
    flush.console()
  }
  
  trial_results <- bind_rows(results_list) %>%
    arrange(desc(value))
  
  write_csv(
    trial_results,
    paste0("results/", model_number, "_", RF_MODEL_TYPE, "_trials_7aa.csv")
  )
  
  save_rf_json_lines(
    trial_results,
    paste0("results/", model_number, "_", RF_MODEL_TYPE, "_trials_7aa.json"),
    phase = "7aa"
  )
  
  importance_data <- trial_results %>%
    mutate(
      splitrule = factor(splitrule),
      max_features_mode = factor(max_features_mode),
      class_weight_mode = factor(class_weight_mode)
    )
  
  importance_fit <- ranger(
    value ~ max_depth + num_trees + splitrule + min_node_size + max_features_mode + class_weight_mode,
    data = as.data.frame(importance_data),
    num.trees = 1000,
    importance = "permutation",
    seed = RF_SEED_TUNE
  )
  
  importance_df <- tibble(
    parameter = names(importance_fit$variable.importance),
    importance = as.numeric(importance_fit$variable.importance)
  ) %>%
    arrange(desc(importance))
  
  write_csv(
    importance_df,
    paste0("results/", model_number, "_", RF_MODEL_TYPE, "_hyperparameter_importance_7aa.csv")
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
      title = paste("Hyperparameter Importance for", model_number, "Random Forest"),
      subtitle = "Exploratory tuning importance based on mean ROC AUC",
      x = "Importance",
      y = "Hyperparameter"
    ) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
    theme_classic(base_size = 12)
  
  ggsave(
    paste0(
      "figures/7aa_hyperparameter_importance/",
      model_number, "_", RF_MODEL_TYPE, "_hyperparameter_importance.png"
    ),
    plot = p_importance,
    width = 8,
    height = 5,
    dpi = 300
  )
  
  cat("[7aa] Completed for", model_number, "\n")
  flush.console()
  
  list(
    trial_results = trial_results,
    importance_df = importance_df
  )
}

# =========================================================
# 8. 7ab: focused tuning
# =========================================================


run_7ab <- function(model_number, trials_7ab = 100) {
  dat <- load_model_xy_rf(model_number)
  x <- dat$x
  y <- dat$y
  
  param_grid <- sample_params_7ab(trials_7ab)
  
  cat("\n[7ab] Starting focused tuning for", model_number, "\n")
  cat("[7ab] Total trials:", trials_7ab, "\n")
  flush.console()
  
  results_list <- vector("list", nrow(param_grid))
  
  for (i in seq_len(nrow(param_grid))) {
    cat(sprintf("[7ab] Trial %d / %d ...\n", i, nrow(param_grid)))
    flush.console()
    
    fold_metrics <- fit_predict_rf_cv(
      x = x,
      y = y,
      num_trees = 250,
      max_depth = param_grid$max_depth[i],
      min_node_size = param_grid$min_node_size[i],
      max_features_mode = param_grid$max_features_mode[i],
      splitrule = "gini",
      class_weight_mode = "none",
      seed = RF_SEED_CV
    )
    
    one_result <- tibble(
      number = i,
      value = mean(fold_metrics$roc_auc, na.rm = TRUE),
      max_depth = param_grid$max_depth[i],
      min_node_size = param_grid$min_node_size[i],
      max_features_mode = param_grid$max_features_mode[i],
      std_err = sd(fold_metrics$roc_auc, na.rm = TRUE) / sqrt(nrow(fold_metrics))
    )
    
    results_list[[i]] <- one_result
    
    cat(sprintf(
      "[7ab] Trial %d / %d finished | roc_auc = %.4f\n",
      i, nrow(param_grid), one_result$value[[1]]
    ))
    flush.console()
  }
  
  trial_results <- bind_rows(results_list) %>%
    arrange(desc(value))
  
  write_csv(
    trial_results,
    paste0("results/", model_number, "_", RF_MODEL_TYPE, "_trials_7ab.csv")
  )
  
  save_rf_json_lines(
    trial_results,
    paste0("results/", model_number, "_", RF_MODEL_TYPE, "_trials.json"),
    phase = "7ab"
  )
  
  best_trial <- trial_results %>%
    filter(!is.na(value)) %>%
    slice_max(value, n = 1, with_ties = FALSE)
  
  save_rf_best_trial_json(
    best_trial,
    paste0("results/", model_number, "_", RF_MODEL_TYPE, "_trial_best.json")
  )
  
  cat(sprintf("[7ab] Best roc_auc = %.4f\n", best_trial$value[[1]]))
  cat("[7ab] Completed for", model_number, "\n")
  flush.console()
  
  list(
    trial_results = trial_results,
    best_trial = best_trial
  )
}

# =========================================================
# 9. 7ac: best within one std err
# =========================================================

run_7ac <- function(model_number) {
  trial_results <- read_csv(
    paste0("results/", model_number, "_", RF_MODEL_TYPE, "_trials_7ab.csv"),
    show_col_types = FALSE
  )
  
  best_trial <- read_json(
    paste0("results/", model_number, "_", RF_MODEL_TYPE, "_trial_best.json")
  )
  
  within_one_std_err <- best_trial$value - best_trial$user_attrs$std_err
  
  best_shots <- trial_results %>%
    filter(!is.na(value)) %>%
    filter(value >= within_one_std_err) %>%
    mutate(
      max_features_rank = case_when(
        max_features_mode == "sqrt" ~ 1,
        max_features_mode == "log2" ~ 2,
        TRUE ~ 3
      )
    )
  
  if (nrow(best_shots) == 0) {
    warning("No trial found within one standard error; using the best trial directly.")
    
    best_within_one <- tibble(
      number = best_trial$number,
      value = best_trial$value,
      max_depth = best_trial$params$max_depth,
      min_node_size = best_trial$params$min_node_size,
      max_features_mode = best_trial$params$max_features_mode,
      std_err = best_trial$user_attrs$std_err
    )
  } else {
    best_within_one <- best_shots %>%
      arrange(
        max_depth,
        desc(min_node_size),
        max_features_rank
      ) %>%
      select(-max_features_rank) %>%
      slice(1)
  }
  
  best_within_one_json <- list(
    number = best_within_one$number[[1]],
    value = best_within_one$value[[1]],
    max_depth = best_within_one$max_depth[[1]],
    min_node_size = best_within_one$min_node_size[[1]],
    max_features_mode = best_within_one$max_features_mode[[1]],
    std_err = best_within_one$std_err[[1]],
    threshold = within_one_std_err
  )
  
  write_json(
    best_within_one_json,
    paste0("results/", model_number, "_", RF_MODEL_TYPE, "_best_within_one.json"),
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
# 10. 7ad: final CV with selected params
# =========================================================

run_7ad <- function(model_number) {
  dat <- load_model_xy_rf(model_number)
  x <- dat$x
  y <- dat$y
  
  selected_params <- read_json(
    paste0("results/", model_number, "_", RF_MODEL_TYPE, "_best_within_one.json")
  )
  
  fold_metrics <- fit_predict_rf_cv(
    x = x,
    y = y,
    num_trees = 250,
    max_depth = as.integer(selected_params$max_depth),
    min_node_size = as.integer(selected_params$min_node_size),
    max_features_mode = as.character(selected_params$max_features_mode),
    splitrule = "gini",
    class_weight_mode = "none",
    seed = RF_SEED_CV
  )
  
  saveRDS(
    fold_metrics,
    file = paste0("results/", model_number, "_", RF_MODEL_TYPE, ".rds")
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
    paste0("results/", model_number, "_", RF_MODEL_TYPE, "_summary.csv")
  )
  
  cat("\n", RF_MODEL_TYPE, "-", model_number, "\n", sep = "")
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

run_rf_pipeline <- function(model_number,
                            trials_7aa = 50,
                            trials_7ab = 100) {
  cat("\n==============================\n")
  cat("Running Random Forest pipeline for:", model_number, "\n")
  cat("==============================\n")
  flush.console()
  
  cat("[Pipeline] Step 7aa begins...\n")
  flush.console()
  out_7aa <- run_7aa(model_number, trials_7aa = trials_7aa)
  
  cat("[Pipeline] Step 7ab begins...\n")
  flush.console()
  out_7ab <- run_7ab(model_number, trials_7ab = trials_7ab)
  
  cat("[Pipeline] Step 7ac begins...\n")
  flush.console()
  out_7ac <- run_7ac(model_number)
  
  cat("[Pipeline] Step 7ad begins...\n")
  flush.console()
  out_7ad <- run_7ad(model_number)
  
  cat("[Pipeline] All steps completed for ", model_number, "\n", sep = "")
  flush.console()
  
  invisible(list(
    step_7aa = out_7aa,
    step_7ab = out_7ab,
    step_7ac = out_7ac,
    step_7ad = out_7ad
  ))
}