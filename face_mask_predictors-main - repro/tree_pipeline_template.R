library(tidymodels)
library(readr)
library(dplyr)
library(ggplot2)
library(jsonlite)
library(ranger)
library(purrr)

# =========================================================
# 0. Global settings
# =========================================================

TREE_MODEL_TYPE <- "binary_tree"
TREE_SEED_CV <- 20240627
TREE_SEED_TUNE <- 2020

dir.create("results", recursive = TRUE, showWarnings = FALSE)
dir.create("figures/5aa_hyperparameter_importance", recursive = TRUE, showWarnings = FALSE)

# =========================================================
# 1. Metrics
# =========================================================

roc_auc_ev   <- metric_tweak("roc_auc_ev",   roc_auc,   event_level = "second")
precision_ev <- metric_tweak("precision_ev", precision, event_level = "second")
recall_ev    <- metric_tweak("recall_ev",    recall,    event_level = "second")
f_meas_ev    <- metric_tweak("f_meas_ev",    f_meas,    event_level = "second")

metric_roc_only <- metric_set(roc_auc_ev)
metric_final_cv <- metric_set(precision_ev, recall_ev, roc_auc_ev, accuracy, f_meas_ev)

# =========================================================
# 2. Helper: load model data
# =========================================================

load_model_train_df <- function(model_number) {
  x <- read_csv(
    paste0("data/X_train_", model_number, ".csv"),
    show_col_types = FALSE
  )
  
  y <- read_csv(
    paste0("data/y_train_", model_number, ".csv"),
    show_col_types = FALSE
  )[[1]]
  
  bind_cols(
    x,
    outcome = factor(as.character(y), levels = c("0", "1"))
  )
}

# =========================================================
# 3. Helper: resampling
# =========================================================


make_tree_resamples <- function(train_df, times = 5, prop = 0.8, seed = TREE_SEED_CV) {
  set.seed(seed)
  mc_cv(
    train_df,
    prop = prop,
    times = times,
    strata = outcome
  )
}

# =========================================================
# 4. Helper: workflow and parameter spaces
# =========================================================

make_tree_workflow <- function(train_df) {
  tree_spec <- decision_tree(
    mode = "classification",
    tree_depth = tune(),
    min_n = tune(),
    cost_complexity = tune()
  ) %>%
    set_engine("rpart")
  
  tree_recipe <- recipe(outcome ~ ., data = train_df)
  
  workflow() %>%
    add_model(tree_spec) %>%
    add_recipe(tree_recipe)
}

make_tree_params_5aa <- function() {
  parameters(
    tree_depth(),
    min_n(),
    cost_complexity()
  ) %>%
    update(
      tree_depth = tree_depth(c(1L, 20L)),
      min_n = min_n(c(2L, 20L)),
      cost_complexity = cost_complexity(c(-4, -0.3))
    )
}

make_tree_params_5ab <- function() {
  parameters(
    tree_depth(),
    min_n(),
    cost_complexity()
  ) %>%
    update(
      tree_depth = tree_depth(c(8L, 12L)),
      min_n = min_n(c(8L, 20L)),
      cost_complexity = cost_complexity(c(-4.1, -3.3))
    )
}

# =========================================================
# 5. Helper: save trial JSON lines
# =========================================================

save_trial_json_lines <- function(trial_df, file_path) {
  trial_json_lines <- lapply(seq_len(nrow(trial_df)), function(i) {
    list(
      number = trial_df$number[i],
      value = trial_df$value[i],
      params = list(
        tree_depth = trial_df$tree_depth[i],
        min_n = trial_df$min_n[i],
        cost_complexity = trial_df$cost_complexity[i]
      ),
      user_attrs = list(
        std_err = trial_df$std_err[i]
      )
    )
  })
  
  writeLines(
    vapply(
      trial_json_lines,
      function(x) jsonlite::toJSON(x, auto_unbox = TRUE, null = "null"),
      character(1)
    ),
    con = file_path
  )
}

# =========================================================
# 6. Helper: save best trial JSON
# =========================================================

save_best_trial_json <- function(best_trial_df, file_path) {
  best_trial_json <- list(
    number = best_trial_df$number[[1]],
    value = best_trial_df$value[[1]],
    params = list(
      tree_depth = best_trial_df$tree_depth[[1]],
      min_n = best_trial_df$min_n[[1]],
      cost_complexity = best_trial_df$cost_complexity[[1]]
    ),
    user_attrs = list(
      std_err = best_trial_df$std_err[[1]]
    )
  )
  
  write_json(
    best_trial_json,
    path = file_path,
    auto_unbox = TRUE,
    pretty = TRUE
  )
}

# =========================================================
# 7. 5aa: exploratory tuning + hyperparameter importance
# =========================================================

run_5aa <- function(model_number, trials_5aa = 250) {
  train_df <- load_model_train_df(model_number)
  resamples <- make_tree_resamples(train_df)
  tree_workflow <- make_tree_workflow(train_df)
  tree_params <- make_tree_params_5aa()
  
  set.seed(TREE_SEED_TUNE)
  
  param_grid <- grid_random(
    tree_params,
    size = trials_5aa
  )
  
  tune_res <- tune_grid(
    tree_workflow,
    resamples = resamples,
    grid = param_grid,
    metrics = metric_roc_only,
    control = control_grid(
      verbose = TRUE,
      save_workflow = TRUE
    )
  )
  
  trial_results <- collect_metrics(tune_res) %>%
    filter(.metric == "roc_auc_ev") %>%
    transmute(
      tree_depth = tree_depth,
      min_n = min_n,
      cost_complexity = cost_complexity,
      mean = mean,
      std_err = std_err
    ) %>%
    arrange(desc(mean))
  
  write_csv(
    trial_results,
    paste0("results/", model_number, "_", TREE_MODEL_TYPE, "_trials_5aa.csv")
  )
  
  importance_fit <- ranger::ranger(
    mean ~ tree_depth + min_n + cost_complexity,
    data = as.data.frame(trial_results),
    num.trees = 1000,
    importance = "permutation",
    seed = TREE_SEED_TUNE
  )
  
  importance_df <- tibble(
    parameter = names(importance_fit$variable.importance),
    importance = as.numeric(importance_fit$variable.importance)
  ) %>%
    arrange(desc(importance))
  
  write_csv(
    importance_df,
    paste0("results/", model_number, "_", TREE_MODEL_TYPE, "_hyperparameter_importance_5aa.csv")
  )
  
  journal_theme <- theme_classic(base_size = 12) +
    theme(
      plot.title = element_text(size = 14, face = "bold", colour = "#222222"),
      plot.subtitle = element_text(size = 11, colour = "#555555"),
      axis.title = element_text(size = 12, colour = "#222222"),
      axis.text = element_text(size = 10, colour = "#333333"),
      axis.line = element_line(colour = "#333333", linewidth = 0.5),
      axis.ticks = element_line(colour = "#333333", linewidth = 0.4),
      panel.grid.major.x = element_line(colour = "#E6E6E6", linewidth = 0.3),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank()
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
      title = paste("Hyperparameter Importance for", model_number, "Decision Tree"),
      subtitle = "Exploratory tuning importance based on mean ROC AUC",
      x = "Importance",
      y = "Hyperparameter"
    ) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
    journal_theme
  
  ggsave(
    paste0(
      "figures/5aa_hyperparameter_importance/",
      model_number, "_", TREE_MODEL_TYPE, "_hyperparameter_importance.png"
    ),
    plot = p_importance,
    width = 8,
    height = 5,
    dpi = 300
  )
  
  list(
    trial_results = trial_results,
    importance_df = importance_df
  )
}

# =========================================================
# 8. 5ab: Bayesian tuning
# =========================================================

run_5ab <- function(model_number, trials_5ab = 100, initial_evals = 20) {
  train_df <- load_model_train_df(model_number)
  resamples <- make_tree_resamples(train_df)
  tree_workflow <- make_tree_workflow(train_df)
  tree_params <- make_tree_params_5ab()
  
  bayes_iter <- trials_5ab - initial_evals
  if (bayes_iter < 1) {
    stop("trials_5ab must be larger than initial_evals.")
  }
  
  set.seed(TREE_SEED_TUNE)
  
  tune_res <- tune_bayes(
    tree_workflow,
    resamples = resamples,
    param_info = tree_params,
    initial = initial_evals,
    iter = bayes_iter,
    metrics = metric_roc_only,
    control = control_bayes(
      verbose = TRUE,
      no_improve = Inf,
      save_workflow = TRUE
    )
  )
  
  trial_results <- collect_metrics(tune_res) %>%
    filter(.metric == "roc_auc_ev") %>%
    mutate(
      number = row_number(),
      value = mean
    ) %>%
    select(
      number,
      value,
      tree_depth,
      min_n,
      cost_complexity,
      std_err,
      .config
    )
  
  write_csv(
    trial_results,
    paste0("results/", model_number, "_", TREE_MODEL_TYPE, "_trials_5ab.csv")
  )
  
  save_trial_json_lines(
    trial_results,
    paste0("results/", model_number, "_", TREE_MODEL_TYPE, "_trials.json")
  )
  
  best_trial <- trial_results %>%
    filter(!is.na(value)) %>%
    slice_max(value, n = 1, with_ties = FALSE)
  
  save_best_trial_json(
    best_trial,
    paste0("results/", model_number, "_", TREE_MODEL_TYPE, "_trial_best.json")
  )
  
  list(
    tune_res = tune_res,
    trial_results = trial_results,
    best_trial = best_trial
  )
}

# =========================================================
# 9. 5ac: best within one std err
# =========================================================


run_5ac <- function(model_number) {
  trial_results <- read_csv(
    paste0("results/", model_number, "_", TREE_MODEL_TYPE, "_trials_5ab.csv"),
    show_col_types = FALSE
  )
  
  best_trial <- jsonlite::read_json(
    paste0("results/", model_number, "_", TREE_MODEL_TYPE, "_trial_best.json")
  )
  
  within_one_std_err <- best_trial$value - best_trial$user_attrs$std_err
  
  best_shots <- trial_results %>%
    filter(!is.na(value)) %>%
    filter(value >= within_one_std_err)
  
  if (nrow(best_shots) == 0) {
    warning("No trial found within one standard error; using the best trial directly.")
    
    best_within_one <- tibble::tibble(
      number = best_trial$number,
      value = best_trial$value,
      tree_depth = best_trial$params$tree_depth,
      min_n = best_trial$params$min_n,
      cost_complexity = best_trial$params$cost_complexity,
      std_err = best_trial$user_attrs$std_err
    )
  } else {
    best_within_one <- best_shots %>%
      arrange(
        desc(cost_complexity),
        tree_depth,
        desc(min_n)
      ) %>%
      slice(1)
  }
  
  best_within_one_json <- list(
    number = best_within_one$number[[1]],
    value = best_within_one$value[[1]],
    tree_depth = best_within_one$tree_depth[[1]],
    min_n = best_within_one$min_n[[1]],
    cost_complexity = best_within_one$cost_complexity[[1]],
    std_err = best_within_one$std_err[[1]],
    threshold = within_one_std_err
  )
  
  write_json(
    best_within_one_json,
    path = paste0("results/", model_number, "_", TREE_MODEL_TYPE, "_best_within_one.json"),
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
# 10. 5ad: final cross validation with selected params
# =========================================================

run_5ad <- function(model_number) {
  train_df <- load_model_train_df(model_number)
  resamples <- make_tree_resamples(train_df)
  
  selected_params <- jsonlite::read_json(
    paste0("results/", model_number, "_", TREE_MODEL_TYPE, "_best_within_one.json")
  )
  
  final_spec <- decision_tree(
    mode = "classification",
    tree_depth = as.integer(selected_params$tree_depth),
    min_n = as.integer(selected_params$min_n),
    cost_complexity = as.numeric(selected_params$cost_complexity)
  ) %>%
    set_engine("rpart")
  
  final_recipe <- recipe(outcome ~ ., data = train_df)
  
  final_workflow <- workflow() %>%
    add_model(final_spec) %>%
    add_recipe(final_recipe)
  
  split_metrics <- map_dfr(seq_along(resamples$splits), function(i) {
    split_obj <- resamples$splits[[i]]
    
    analysis_df <- analysis(split_obj)
    assessment_df <- assessment(split_obj)
    
    fitted_wf <- fit(final_workflow, data = analysis_df)
    
    pred_class <- predict(fitted_wf, assessment_df, type = "class")
    pred_prob  <- predict(fitted_wf, assessment_df, type = "prob")
    
    pred_df <- bind_cols(
      assessment_df %>% select(outcome),
      pred_class,
      pred_prob
    )
    
    tibble(
      fold = i,
      test_precision = precision_ev(pred_df, truth = outcome, estimate = .pred_class)$.estimate,
      test_recall = recall_ev(pred_df, truth = outcome, estimate = .pred_class)$.estimate,
      test_roc_auc = roc_auc_ev(pred_df, truth = outcome, .pred_1)$.estimate,
      test_accuracy = accuracy(pred_df, truth = outcome, estimate = .pred_class)$.estimate,
      test_f1 = f_meas_ev(pred_df, truth = outcome, estimate = .pred_class)$.estimate
    )
  })
  
  saveRDS(
    split_metrics,
    file = paste0("results/", model_number, "_", TREE_MODEL_TYPE, ".rds")
  )
  
  summary_df <- tibble(
    model_number = model_number,
    mean_precision = mean(split_metrics$test_precision, na.rm = TRUE),
    mean_recall = mean(split_metrics$test_recall, na.rm = TRUE),
    mean_roc_auc = mean(split_metrics$test_roc_auc, na.rm = TRUE),
    mean_accuracy = mean(split_metrics$test_accuracy, na.rm = TRUE),
    mean_f1 = mean(split_metrics$test_f1, na.rm = TRUE)
  )
  
  write_csv(
    summary_df,
    paste0("results/", model_number, "_", TREE_MODEL_TYPE, "_summary.csv")
  )
  
  cat("\n", TREE_MODEL_TYPE, "-", model_number, "\n", sep = "")
  cat("Mean recall: ", round(summary_df$mean_recall, 3), "\n", sep = "")
  cat("Mean roc: ", round(summary_df$mean_roc_auc, 3), "\n", sep = "")
  cat("Mean accuracy: ", round(summary_df$mean_accuracy, 3), "\n", sep = "")
  
  list(
    fold_metrics = split_metrics,
    summary_df = summary_df
  )
}

# =========================================================
# 11. One-click full pipeline
# =========================================================

run_tree_pipeline <- function(model_number,
                              trials_5aa = 250,
                              trials_5ab = 100,
                              initial_evals_5ab = 20) {
  cat("\n==============================\n")
  cat("Running tree pipeline for:", model_number, "\n")
  cat("==============================\n")
  
  out_5aa <- run_5aa(model_number, trials_5aa = trials_5aa)
  out_5ab <- run_5ab(model_number,
                     trials_5ab = trials_5ab,
                     initial_evals = initial_evals_5ab)
  out_5ac <- run_5ac(model_number)
  out_5ad <- run_5ad(model_number)
  
  invisible(list(
    step_5aa = out_5aa,
    step_5ab = out_5ab,
    step_5ac = out_5ac,
    step_5ad = out_5ad
  ))
}