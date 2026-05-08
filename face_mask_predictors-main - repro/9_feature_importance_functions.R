library(readr)
library(dplyr)
library(jsonlite)
library(tibble)
library(purrr)
library(ranger)
library(xgboost)
library(tidyr)

# =========================================================
# Global settings
# =========================================================

global_seed <- 2026
rf_num_threads <- 4
xgb_num_threads <- 4

dir.create("results", recursive = TRUE, showWarnings = FALSE)

# =========================================================
# 1. Random oversampling
#    Only this step changes across repeats
# =========================================================

random_oversample <- function(x, y, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  
  y <- as.numeric(y)
  class_counts <- table(y)
  target_n <- max(class_counts)
  
  resampled_idx <- c()
  
  for (cls in names(class_counts)) {
    idx <- which(y == as.numeric(cls))
    n_current <- length(idx)
    
    if (n_current < target_n) {
      extra_idx <- sample(idx, size = target_n - n_current, replace = TRUE)
      idx <- c(idx, extra_idx)
    }
    
    resampled_idx <- c(resampled_idx, idx)
  }
  
  resampled_idx <- sample(resampled_idx)
  
  list(
    x = x[resampled_idx, , drop = FALSE],
    y = y[resampled_idx]
  )
}

# =========================================================
# 2. Convert RF max_features style to ranger mtry
# =========================================================

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

# =========================================================
# 3. Load best params
# =========================================================

load_best_params <- function(model_number, model_type) {
  params <- read_json(
    paste0("results/", model_number, "_", model_type, "_best_within_one.json")
  )
  
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
  
  return(params)
}

# =========================================================
# 4. Fit model with upsampling
# =========================================================

fit_model_with_upsample <- function(model_number, model_type, params, repeat_id) {
  x <- read_csv(
    paste0("data/X_train_", model_number, ".csv"),
    show_col_types = FALSE
  )
  
  y <- read_csv(
    paste0("data/y_train_", model_number, ".csv"),
    show_col_types = FALSE
  )[[1]]
  
  upsampled <- random_oversample(x, y, seed = global_seed + repeat_id)
  x_up <- upsampled$x
  y_up <- upsampled$y
  
  if (model_type == "xgboost") {
    dtrain <- xgb.DMatrix(data = as.matrix(x_up), label = y_up)
    
    xgb_params <- list(
      booster = "gbtree",
      objective = "binary:logistic",
      eta = as.numeric(params$learning_rate),
      max_depth = as.integer(params$max_depth),
      subsample = as.numeric(params$subsample),
      colsample_bytree = as.numeric(params$colsample_bytree),
      seed = global_seed,         # fixed model seed
      nthread = xgb_num_threads
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
    
    model_fitted <- xgb.train(
      params = xgb_params,
      data = dtrain,
      nrounds = as.integer(params$n_estimators),
      verbose = 0
    )
    
    return(model_fitted)
  }
  
  if (model_type == "rf") {
    train_df <- bind_cols(
      x_up,
      outcome = factor(as.character(y_up), levels = c("0", "1"))
    )
    
    p <- ncol(x_up)
    
    rf_fit <- ranger(
      dependent.variable.name = "outcome",
      data = train_df,
      num.trees = as.integer(params$n_estimators),
      mtry = max_features_to_mtry(params$max_features_mode, p),
      min.node.size = as.integer(params$min_node_size),
      max.depth = as.integer(params$max_depth),
      probability = TRUE,
      replace = TRUE,
      importance = "impurity",
      seed = global_seed,         # fixed model seed
      num.threads = rf_num_threads
    )
    
    return(rf_fit)
  }
  
  stop("Unsupported model_type.")
}

# =========================================================
# 5. Extract feature importance
# =========================================================

extract_feature_importance <- function(model_fitted, model_type, feature_names) {
  importance_dict <- setNames(rep(0, length(feature_names)), feature_names)
  
  if (model_type == "xgboost") {
    imp <- xgb.importance(model = model_fitted)
    
    if (!is.null(imp) && nrow(imp) > 0) {
      for (i in seq_len(nrow(imp))) {
        feature_name <- imp$Feature[i]
        gain_value <- imp$Gain[i]
        
        if (feature_name %in% names(importance_dict)) {
          importance_dict[feature_name] <- gain_value
        }
      }
    }
  }
  
  if (model_type == "rf") {
    imp <- model_fitted$variable.importance
    
    if (!is.null(imp)) {
      for (nm in names(imp)) {
        if (nm %in% names(importance_dict)) {
          importance_dict[nm] <- imp[[nm]]
        }
      }
    }
  }
  
  return(as.list(importance_dict))
}

# =========================================================
# 6. Summarise feature importance
# =========================================================

summarise_feature_importance <- function(raw_df) {
  raw_df %>%
    pivot_longer(
      cols = everything(),
      names_to = "variable",
      values_to = "importance"
    ) %>%
    group_by(variable) %>%
    summarise(
      median_importance = median(importance, na.rm = TRUE),
      q1 = quantile(importance, 0.25, na.rm = TRUE),
      q3 = quantile(importance, 0.75, na.rm = TRUE),
      iqr = q3 - q1,
      mean_importance = mean(importance, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(median_importance))
}

# =========================================================
# 7. Run one combination
# =========================================================

run_one_feature_importance <- function(model_number, model_type, num_perm = 200) {
  cat("\n====================================\n")
  cat("Running Step 9 feature importance for:", model_number, "-", model_type, "\n")
  cat("Repeats:", num_perm, "\n")
  cat("====================================\n")
  
  params <- load_best_params(model_number, model_type)
  
  x_train <- read_csv(
    paste0("data/X_train_", model_number, ".csv"),
    show_col_types = FALSE
  )
  feature_names <- names(x_train)
  
  results_list <- vector("list", num_perm)
  
  for (i in seq_len(num_perm)) {
    if (i %% 10 == 0 || i == 1 || i == num_perm) {
      cat("Repeat", i, "of", num_perm, "\n")
      flush.console()
    }
    
    model_fitted <- fit_model_with_upsample(
      model_number = model_number,
      model_type = model_type,
      params = params,
      repeat_id = i
    )
    
    importance_dict <- extract_feature_importance(
      model_fitted = model_fitted,
      model_type = model_type,
      feature_names = feature_names
    )
    
    results_list[[i]] <- as_tibble(importance_dict)
  }
  
  raw_df <- bind_rows(results_list)
  
  raw_file <- paste0(
    "results/", model_number, "_", model_type, "_feature_importance.csv"
  )
  
  summary_file <- paste0(
    "results/", model_number, "_", model_type, "_feature_importance_summary.csv"
  )
  
  write_csv(raw_df, raw_file)
  
  summary_df <- summarise_feature_importance(raw_df)
  write_csv(summary_df, summary_file)
  
  cat("Saved raw importance to: ", raw_file, "\n", sep = "")
  cat("Saved summary importance to: ", summary_file, "\n", sep = "")
  
  invisible(list(
    raw_df = raw_df,
    summary_df = summary_df
  ))
}

# =========================================================
# 8. Batch runner
# =========================================================

run_feature_importance_batch <- function(num_perm = 200) {
  target_models <- c("model_1a", "model_1b", "model_2a", "model_2b")
  target_types  <- c("xgboost", "rf")
  
  start_time <- Sys.time()
  
  for (mn in target_models) {
    for (mt in target_types) {
      run_one_feature_importance(
        model_number = mn,
        model_type = mt,
        num_perm = num_perm
      )
    }
  }
  
  cat("\nAll 8 feature importance jobs completed.\n")
  cat("Total time taken:", Sys.time() - start_time, "\n")
}