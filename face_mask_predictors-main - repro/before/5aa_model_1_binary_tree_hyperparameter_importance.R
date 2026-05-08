library(tidymodels)
library(ranger)
library(readr)
library(dplyr)
library(ggplot2)

# =========================================================
# 0. Create output folders
# =========================================================

dir.create("results", recursive = TRUE, showWarnings = FALSE)
dir.create("figures/5aa_hyperparameter_importance", recursive = TRUE, showWarnings = FALSE)

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

# outcome must be a factor for classification
train_df <- bind_cols(
  x,
  outcome = factor(as.character(y), levels = c("0", "1"))
)

# =========================================================
# 2. Resampling setup
#    Python used StratifiedShuffleSplit with:
#    n_splits = 5, test_size = 0.2
#    Here the closest tidymodels equivalent is mc_cv with strata
# =========================================================

set.seed(20240627)

resamples <- mc_cv(
  train_df,
  prop = 0.8,      # 80% analysis, 20% assessment
  times = 5,
  strata = outcome
)

# =========================================================
# 3. Model specification
#    Pure R tree model: rpart via tidymodels
#    Closest main tunable analogues:
#      tree_depth        ~ max_depth
#      min_n             ~ min_samples_split / min_samples_leaf
#      cost_complexity   ~ min_impurity_decrease-like complexity control
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
#    Broad exploratory space, matching 5aa's role
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
# 5. Exploratory tuning
#    250 random combinations, analogous to the Python exploratory search
# =========================================================

set.seed(2020)

param_grid <- grid_random(
  tree_params,
  size = 250
)

roc_metric <- metric_set(roc_auc)

tune_res <- tune_grid(
  tree_workflow,
  resamples = resamples,
  grid = param_grid,
  metrics = roc_metric,
  control = control_grid(
    verbose = TRUE,
    save_workflow = TRUE
  )
)

# =========================================================
# 6. Collect trial results
# =========================================================

trial_results <- collect_metrics(tune_res) %>%
  filter(.metric == "roc_auc") %>%
  select(tree_depth, min_n, cost_complexity, mean, std_err) %>%
  arrange(desc(mean))

write_csv(
  trial_results,
  paste0("results/", model_number, "_", model_type, "_trials_5aa.csv")
)

print(head(trial_results, 10))

# =========================================================
# 7. Hyperparameter importance
#    Use a surrogate random forest:
#    parameters -> mean ROC AUC
# =========================================================

importance_fit <- ranger::ranger(
  mean ~ tree_depth + min_n + cost_complexity,
  data = as.data.frame(trial_results),
  num.trees = 1000,
  importance = "permutation",
  seed = 2020
)

importance_df <- tibble(
  parameter = names(importance_fit$variable.importance),
  importance = as.numeric(importance_fit$variable.importance)
) %>%
  arrange(desc(importance))

write_csv(
  importance_df,
  paste0("results/", model_number, "_", model_type, "_hyperparameter_importance_5aa.csv")
)

print(importance_df)

# =========================================================
# 8. Plot hyperparameter importance
# =========================================================

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
    title = "Hyperparameter Importance for Model 1 Decision Tree",
    subtitle = "Exploratory tuning importance based on mean ROC AUC",
    x = "Importance",
    y = "Hyperparameter"
  ) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
  journal_theme

print(p_importance)

ggsave(
  paste0(
    "figures/5aa_hyperparameter_importance/",
    model_number, "_", model_type, "_hyperparameter_importance.png"
  ),
  plot = p_importance,
  width = 8,
  height = 5,
  dpi = 300
)

cat("\n5aa completed.\n")
cat("Trial results saved to: ",
    paste0("results/", model_number, "_", model_type, "_trials_5aa.csv"), "\n", sep = "")
cat("Importance table saved to: ",
    paste0("results/", model_number, "_", model_type, "_hyperparameter_importance_5aa.csv"), "\n", sep = "")
cat("Importance figure saved to: ",
    paste0("figures/5aa_hyperparameter_importance/",
           model_number, "_", model_type, "_hyperparameter_importance.png"), "\n", sep = "")

readr::read_csv("results/model_1_binary_tree_trials_5aa.csv", show_col_types = FALSE) |> head()
readr::read_csv("results/model_1_binary_tree_hyperparameter_importance_5aa.csv", show_col_types = FALSE)