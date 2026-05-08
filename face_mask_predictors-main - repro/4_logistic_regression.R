library(readr)
library(dplyr)
library(ggplot2)
library(pROC)

# =========================================================
# 0. Create output folders
# =========================================================

dir.create("results", recursive = TRUE, showWarnings = FALSE)
dir.create("figures/4_logistic_regression", recursive = TRUE, showWarnings = FALSE)

# =========================================================
# 1. Parameter setup
# =========================================================

n_splits <- 5
seed <- 20240627
model_fitting <- "logistic_reg"

# =========================================================
# 2. Helper functions
# =========================================================

make_stratified_splits <- function(y, n_splits = 5, test_prop = 0.2, seed = 20240627) {
  set.seed(seed)
  
  y <- as.vector(y)
  class_levels <- sort(unique(y))
  
  split_list <- vector("list", n_splits)
  
  class_indices <- lapply(class_levels, function(cls) which(y == cls))
  names(class_indices) <- as.character(class_levels)
  
  for (i in seq_len(n_splits)) {
    test_idx <- integer(0)
    
    for (cls in class_levels) {
      idx <- class_indices[[as.character(cls)]]
      n_test <- round(length(idx) * test_prop)
      
      if (n_test < 1 && length(idx) > 0) {
        n_test <- 1
      }
      if (n_test >= length(idx)) {
        n_test <- max(1, length(idx) - 1)
      }
      
      sampled <- sample(idx, size = n_test, replace = FALSE)
      test_idx <- c(test_idx, sampled)
    }
    
    test_idx <- sort(unique(test_idx))
    train_idx <- setdiff(seq_along(y), test_idx)
    
    split_list[[i]] <- list(train_idx = train_idx, val_idx = test_idx)
  }
  
  split_list
}

random_oversample <- function(X, y) {
  y <- as.vector(y)
  classes <- sort(unique(y))
  class_counts <- table(y)
  max_count <- max(class_counts)
  
  sampled_idx <- integer(0)
  
  for (cls in classes) {
    idx <- which(y == cls)
    extra_idx <- sample(idx, size = max_count, replace = TRUE)
    sampled_idx <- c(sampled_idx, extra_idx)
  }
  
  sampled_idx <- sample(sampled_idx)  # shuffle
  
  list(
    X = X[sampled_idx, , drop = FALSE],
    y = y[sampled_idx]
  )
}

calc_precision <- function(y_true, y_pred) {
  tp <- sum(y_true == 1 & y_pred == 1)
  fp <- sum(y_true == 0 & y_pred == 1)
  if ((tp + fp) == 0) return(NA_real_)
  tp / (tp + fp)
}

calc_recall <- function(y_true, y_pred) {
  tp <- sum(y_true == 1 & y_pred == 1)
  fn <- sum(y_true == 1 & y_pred == 0)
  if ((tp + fn) == 0) return(NA_real_)
  tp / (tp + fn)
}

calc_accuracy <- function(y_true, y_pred) {
  mean(y_true == y_pred)
}

calc_f1 <- function(y_true, y_pred) {
  p <- calc_precision(y_true, y_pred)
  r <- calc_recall(y_true, y_pred)
  if (is.na(p) || is.na(r) || (p + r) == 0) return(NA_real_)
  2 * p * r / (p + r)
}

calc_auc <- function(y_true, y_prob) {
  if (length(unique(y_true)) < 2) return(NA_real_)
  as.numeric(pROC::auc(pROC::roc(response = y_true, predictor = y_prob, quiet = TRUE)))
}

fit_logistic_glm <- function(X_train, y_train, X_val) {
  train_df <- as.data.frame(X_train)
  val_df <- as.data.frame(X_val)
  
  colnames(train_df) <- make.names(colnames(train_df), unique = TRUE)
  colnames(val_df) <- make.names(colnames(val_df), unique = TRUE)
  
  train_df$y <- y_train
  
  fit <- glm(
    y ~ .,
    data = train_df,
    family = binomial(),
    control = glm.control(maxit = 5000)
  )
  
  prob <- predict(fit, newdata = val_df, type = "response")
  pred <- ifelse(prob >= 0.5, 1L, 0L)
  
  list(pred = pred, prob = prob)
}

# =========================================================
# 3. Cross-validation function
# =========================================================

cross_validate_model <- function(model_number, upsample = FALSE) {
  x <- read_csv(
    file.path("data", paste0("X_train_", model_number, ".csv")),
    show_col_types = FALSE
  )
  
  y_df <- read_csv(
    file.path("data", paste0("y_train_", model_number, ".csv")),
    show_col_types = FALSE
  )
  
  y <- as.integer(unlist(y_df[, 1]))
  
  splits <- make_stratified_splits(
    y = y,
    n_splits = n_splits,
    test_prop = 1 / n_splits,
    seed = seed
  )
  
  cv_scores <- tibble(
    fold = integer(),
    test_precision = numeric(),
    test_recall = numeric(),
    test_roc_auc = numeric(),
    test_accuracy = numeric(),
    test_f1 = numeric()
  )
  
  for (fold in seq_along(splits)) {
    train_idx <- splits[[fold]]$train_idx
    val_idx <- splits[[fold]]$val_idx
    
    X_train <- x[train_idx, , drop = FALSE]
    y_train <- y[train_idx]
    
    X_val <- x[val_idx, , drop = FALSE]
    y_val <- y[val_idx]
    
    if (upsample) {
      upsampled <- random_oversample(X_train, y_train)
      X_train <- upsampled$X
      y_train <- upsampled$y
    }
    
    fit_out <- fit_logistic_glm(X_train, y_train, X_val)
    
    preds <- fit_out$pred
    probs <- fit_out$prob
    
    cv_scores <- bind_rows(
      cv_scores,
      tibble(
        fold = fold,
        test_precision = calc_precision(y_val, preds),
        test_recall = calc_recall(y_val, preds),
        test_roc_auc = calc_auc(y_val, probs),
        test_accuracy = calc_accuracy(y_val, preds),
        test_f1 = calc_f1(y_val, preds)
      )
    )
  }
  
  cat("\n", model_number, "\n", sep = "")
  cat("Mean recall: ", round(mean(cv_scores$test_recall, na.rm = TRUE), 3), "\n", sep = "")
  cat("Mean roc: ", round(mean(cv_scores$test_roc_auc, na.rm = TRUE), 3), "\n", sep = "")
  cat("Mean accuracy: ", round(mean(cv_scores$test_accuracy, na.rm = TRUE), 3), "\n", sep = "")
  
  saveRDS(
    cv_scores,
    file = file.path("results", paste0(model_number, "_", model_fitting, ".rds"))
  )
  
  summary_row <- tibble(
    model = model_number,
    mean_recall = mean(cv_scores$test_recall, na.rm = TRUE),
    mean_roc_auc = mean(cv_scores$test_roc_auc, na.rm = TRUE),
    mean_accuracy = mean(cv_scores$test_accuracy, na.rm = TRUE)
  )
  
  list(cv_scores = cv_scores, summary_row = summary_row)
}

# =========================================================
# 4. Run all six models
# =========================================================

results_summary <- bind_rows(
  cross_validate_model("model_1",  upsample = FALSE)$summary_row,
  cross_validate_model("model_1a", upsample = TRUE)$summary_row,
  cross_validate_model("model_1b", upsample = TRUE)$summary_row,
  cross_validate_model("model_2",  upsample = FALSE)$summary_row,
  cross_validate_model("model_2a", upsample = TRUE)$summary_row,
  cross_validate_model("model_2b", upsample = TRUE)$summary_row
)

write_csv(
  results_summary,
  file.path("results", "logistic_reg_cv_summary.csv")
)

print(results_summary)

# =========================================================
# 5. Plot one grouped bar chart:
#    six models × three metrics
# =========================================================

plot_df <- results_summary %>%
  mutate(
    model = c("Model 1", "Model 1a", "Model 1b", "Model 2", "Model 2a", "Model 2b")
  ) %>%
  tidyr::pivot_longer(
    cols = c(mean_recall, mean_roc_auc, mean_accuracy),
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(
    metric = recode(
      metric,
      mean_recall = "Recall",
      mean_roc_auc = "ROC AUC",
      mean_accuracy = "Accuracy"
    )
  )

journal_theme <- theme_classic(base_size = 12) +
  theme(
    plot.title = element_text(size = 14, face = "bold", colour = "#222222"),
    plot.subtitle = element_text(size = 11, colour = "#555555"),
    axis.title = element_text(size = 12, colour = "#222222"),
    axis.text = element_text(size = 10, colour = "#333333"),
    axis.line = element_line(colour = "#333333", linewidth = 0.5),
    axis.ticks = element_line(colour = "#333333", linewidth = 0.4),
    panel.grid.major.y = element_line(colour = "#E6E6E6", linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank()
  )

p_metrics <- ggplot(plot_df, aes(x = model, y = value, fill = metric)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.68) +
  geom_text(
    aes(label = round(value, 3)),
    position = position_dodge(width = 0.75),
    vjust = -0.35,
    size = 3.0,
    colour = "#222222"
  ) +
  labs(
    title = "Cross-Validation Performance Across the Six Models",
    subtitle = "Mean Recall, ROC AUC, and Accuracy from repeated stratified validation",
    x = "Model",
    y = "Mean performance"
  ) +
  scale_y_continuous(limits = c(0, 1.08), expand = expansion(mult = c(0, 0.02))) +
  journal_theme

print(p_metrics)

ggsave(
  file.path("figures/4_logistic_regression", "figure1_cv_metrics_across_six_models.png"),
  plot = p_metrics,
  width = 10,
  height = 5.5,
  dpi = 300
)

cat("\nSummary saved to results/logistic_reg_cv_summary.csv\n")
cat("Figure saved to figures/3_model_performance/figure1_cv_metrics_across_six_models.png\n")