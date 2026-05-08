library(tidymodels)
library(readr)
library(dplyr)
library(ggplot2)
library(jsonlite)

# =========================================================
# 0. Create output folders
# =========================================================

dir.create("results", recursive = TRUE, showWarnings = FALSE)

# =========================================================
# 1. Read training data
# =========================================================

model_number <- "model_1"
model_type <- "binary_tree"

x <- read_csv(
  paste0("data/X_train_", model_number, ".csv"),
  show_col_types = FALSE
)

y <- read_csv(
  paste0("data/y_train_", model_number, ".csv"),
  show_col_types = FALSE
)[[1]]

# For classification in tidymodels
train_df <- bind_cols(
  x,
  outcome = factor(as.character(y), levels = c("0", "1"))
)

# =========================================================
# 2. Resampling setup
#    Python used StratifiedShuffleSplit(n_splits = 5, test_size = 0.2)
#    Closest tidymodels equivalent: mc_cv(prop = 0.8, times = 5, strata = outcome)
# =========================================================

set.seed(20240627)

resamples <- mc_cv(
  train_df,
  prop = 0.8,
  times = 5,
  strata = outcome
)

# =========================================================
# 3. Model specification
#    In the R/tidymodels version we continue tuning the 3 key tree parameters:
#      tree_depth
#      min_n
#      cost_complexity
#    These play the same logical role as the important complexity-related
#    parameters identified in 5aa.
# =========================================================

tree_spec <- decision_tree(
  mode = "classification",
  tree_depth = tune(),
  min_n = tune(),
  cost_complexity = tune()
) %>%
  set_engine("rpart")

tree_recipe <- recipe(outcome ~ ., data = train_df)

tree_workflow <- workflow() %>%
  add_model(tree_spec) %>%
  add_recipe(tree_recipe)

# =========================================================
# 4. Parameter ranges
#    Keep the same broad ranges as in 5aa
# =========================================================

tree_params <- parameters(
  tree_depth(),
  min_n(),
  cost_complexity()
) %>%
  update(
    tree_depth = tree_depth(c(1L, 20L)),
    min_n = min_n(c(2L, 20L)),
    cost_complexity = cost_complexity(c(-4, -0.3))
  )

# =========================================================
# 5. Bayesian tuning
#    Python 5ab used Optuna with n_trials = 1000.
#    In tidymodels, tune_bayes() = initial random evaluations + Bayesian iterations.
#    Total evaluations here = initial_evals + bayes_iter = 1000.
# =========================================================

n_trials_total <- 1000
initial_evals <- 20
bayes_iter <- n_trials_total - initial_evals

roc_metric <- metric_set(roc_auc)

set.seed(2020)

tune_res <- tune_bayes(
  tree_workflow,
  resamples = resamples,
  param_info = tree_params,
  initial = initial_evals,
  iter = bayes_iter,
  metrics = roc_metric,
  control = control_bayes(
    verbose = TRUE,
    no_improve = Inf,
    save_workflow = TRUE
  )
)

# =========================================================
# 6. Collect trial results
#    Equivalent to Python 5ab's study.trials
# =========================================================

trial_results_raw <- collect_metrics(tune_res) %>%
  filter(.metric == "roc_auc") %>%
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

# Save a convenient CSV version
write_csv(
  trial_results_raw,
  paste0("results/", model_number, "_", model_type, "_trials_5ab.csv")
)

print(head(arrange(trial_results_raw, desc(value)), 10))

# =========================================================
# 7. Save line-delimited JSON trial file
#    This mimics the Python structure more closely
# =========================================================

trial_json_lines <- lapply(seq_len(nrow(trial_results_raw)), function(i) {
  list(
    number = trial_results_raw$number[i],
    value = trial_results_raw$value[i],
    params = list(
      tree_depth = trial_results_raw$tree_depth[i],
      min_n = trial_results_raw$min_n[i],
      cost_complexity = trial_results_raw$cost_complexity[i]
    ),
    user_attrs = list(
      std_err = trial_results_raw$std_err[i]
    )
  )
})

writeLines(
  vapply(
    trial_json_lines,
    function(x) jsonlite::toJSON(x, auto_unbox = TRUE, null = "null"),
    character(1)
  ),
  con = paste0("results/", model_number, "_", model_type, "_trials.json")
)

# =========================================================
# 8. Save best trial
#    Equivalent to Python's study.best_trial
# =========================================================

best_trial <- trial_results_raw %>%
  slice_max(value, n = 1, with_ties = FALSE)

best_trial_json <- list(
  number = best_trial$number[[1]],
  value = best_trial$value[[1]],
  params = list(
    tree_depth = best_trial$tree_depth[[1]],
    min_n = best_trial$min_n[[1]],
    cost_complexity = best_trial$cost_complexity[[1]]
  ),
  user_attrs = list(
    std_err = best_trial$std_err[[1]]
  )
)

write_json(
  best_trial_json,
  path = paste0("results/", model_number, "_", model_type, "_trial_best.json"),
  auto_unbox = TRUE,
  pretty = TRUE
)

# =========================================================
# 9. Print summary
# =========================================================

cat("\n5ab completed.\n")
cat("All trials saved to: ",
    paste0("results/", model_number, "_", model_type, "_trials.json"), "\n", sep = "")
cat("CSV trial summary saved to: ",
    paste0("results/", model_number, "_", model_type, "_trials_5ab.csv"), "\n", sep = "")
cat("Best trial saved to: ",
    paste0("results/", model_number, "_", model_type, "_trial_best.json"), "\n", sep = "")

print(best_trial)