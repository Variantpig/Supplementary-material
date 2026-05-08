library(readr)
library(dplyr)
library(ggplot2)

# =========================
# 0. Helper functions
# =========================

encode_yes_no <- function(x) {
  ifelse(as.character(x) == "Yes", 1L, 0L)
}

save_model_data <- function(model_name, X_train, X_test, y_train, y_test) {
  write_csv(X_train, file.path("data", paste0("X_train_", model_name, ".csv")))
  write_csv(X_test,  file.path("data", paste0("X_test_",  model_name, ".csv")))
  write_csv(tibble(y_train = y_train), file.path("data", paste0("y_train_", model_name, ".csv")))
  write_csv(tibble(y_test  = y_test),  file.path("data", paste0("y_test_",  model_name, ".csv")))
}

make_stratified_split <- function(data, stratify_col, test_size = 0.2, seed = 20240417) {
  set.seed(seed)
  
  groups <- split(seq_len(nrow(data)), data[[stratify_col]])
  
  test_idx <- unlist(
    lapply(groups, function(idx) {
      n_test <- round(length(idx) * test_size)
      if (n_test == 0 && length(idx) > 0) {
        n_test <- 1
      }
      sample(idx, n_test)
    }),
    use.names = FALSE
  )
  
  test_idx <- sort(test_idx)
  
  list(
    train = data[-test_idx, , drop = FALSE],
    test  = data[test_idx,  , drop = FALSE]
  )
}

# =========================
# 1. Read preprocessed data
# =========================

cleaned_df <- read_csv(
  "data/2_cleaned_data_preprocessing.csv",
  show_col_types = FALSE
) %>%
  mutate(
    endtime = as.Date(endtime, format = "%Y-%m-%d")
  )

# =========================
# 2. Train/test split (stratified by within_mandate_period)
# =========================

split_obj <- make_stratified_split(
  data = cleaned_df,
  stratify_col = "within_mandate_period",
  test_size = 0.2,
  seed = 20240417
)

df_train <- split_obj$train
df_test  <- split_obj$test

write_csv(df_train, "data/df_train.csv")
write_csv(df_test,  "data/df_test.csv")

# =========================
# 3. Model 1: Predicting face masks
# =========================

response_col <- "face_mask_behaviour_binary"

feature_cols_model_1 <- setdiff(
  names(cleaned_df),
  c(
    "RecordNo",
    "face_mask_behaviour_scale",
    "protective_behaviour_scale",
    "face_mask_behaviour_binary",
    "protective_behaviour_binary",
    "endtime"
  )
)

X_train_model_1 <- df_train[, feature_cols_model_1, drop = FALSE]
X_test_model_1  <- df_test[,  feature_cols_model_1, drop = FALSE]

y_train_model_1 <- encode_yes_no(df_train[[response_col]])
y_test_model_1  <- encode_yes_no(df_test[[response_col]])

save_model_data("model_1", X_train_model_1, X_test_model_1, y_train_model_1, y_test_model_1)

# =========================
# 4. Model 1a: Predicting face masks in early time
# =========================

mandate_starter <- as.Date("2022-01-01")

feature_cols_model_1a <- setdiff(
  names(cleaned_df),
  c(
    "RecordNo",
    "face_mask_behaviour_scale",
    "protective_behaviour_scale",
    "face_mask_behaviour_binary",
    "protective_behaviour_binary",
    "endtime",
    "within_mandate_period"
  )
)

logic_subsetter_train_1a <- (df_train$endtime < mandate_starter) &
  (df_train$within_mandate_period == 0)

logic_subsetter_test_1a <- (df_test$endtime < mandate_starter) &
  (df_test$within_mandate_period == 0)

X_train_model_1a <- df_train[logic_subsetter_train_1a, feature_cols_model_1a, drop = FALSE]
X_test_model_1a  <- df_test[logic_subsetter_test_1a,  feature_cols_model_1a, drop = FALSE]

y_train_model_1a <- encode_yes_no(df_train[logic_subsetter_train_1a, response_col, drop = TRUE])
y_test_model_1a  <- encode_yes_no(df_test[logic_subsetter_test_1a,  response_col, drop = TRUE])

save_model_data("model_1a", X_train_model_1a, X_test_model_1a, y_train_model_1a, y_test_model_1a)

# =========================
# 5. Model 1b: Predicting face masks in mandate periods
# =========================

feature_cols_model_1b <- feature_cols_model_1a

logic_subsetter_train_1b <- df_train$within_mandate_period == 1
logic_subsetter_test_1b  <- df_test$within_mandate_period == 1

X_train_model_1b <- df_train[logic_subsetter_train_1b, feature_cols_model_1b, drop = FALSE]
X_test_model_1b  <- df_test[logic_subsetter_test_1b,  feature_cols_model_1b, drop = FALSE]

y_train_model_1b <- encode_yes_no(df_train[logic_subsetter_train_1b, response_col, drop = TRUE])
y_test_model_1b  <- encode_yes_no(df_test[logic_subsetter_test_1b,  response_col, drop = TRUE])

save_model_data("model_1b", X_train_model_1b, X_test_model_1b, y_train_model_1b, y_test_model_1b)

# =========================
# 6. Model 2: Predicting protective behaviour
# =========================

response_col <- "protective_behaviour_binary"

feature_cols_model_2 <- setdiff(
  names(cleaned_df),
  c(
    "RecordNo",
    "face_mask_behaviour_scale",
    "protective_behaviour_scale",
    "face_mask_behaviour_binary",
    "protective_behaviour_binary",
    "protective_behaviour_nomask_scale",
    "endtime"
  )
)

X_train_model_2 <- df_train[, feature_cols_model_2, drop = FALSE]
X_test_model_2  <- df_test[,  feature_cols_model_2, drop = FALSE]

y_train_model_2 <- encode_yes_no(df_train[[response_col]])
y_test_model_2  <- encode_yes_no(df_test[[response_col]])

save_model_data("model_2", X_train_model_2, X_test_model_2, y_train_model_2, y_test_model_2)

# =========================
# 7. Model 2a: Predicting protective behaviour in early time
# =========================

feature_cols_model_2a <- setdiff(
  names(cleaned_df),
  c(
    "RecordNo",
    "face_mask_behaviour_scale",
    "protective_behaviour_scale",
    "face_mask_behaviour_binary",
    "protective_behaviour_binary",
    "protective_behaviour_nomask_scale",
    "endtime",
    "within_mandate_period"
  )
)

logic_subsetter_train_2a <- (df_train$endtime < mandate_starter) &
  (df_train$within_mandate_period == 0)

logic_subsetter_test_2a <- (df_test$endtime < mandate_starter) &
  (df_test$within_mandate_period == 0)

X_train_model_2a <- df_train[logic_subsetter_train_2a, feature_cols_model_2a, drop = FALSE]
X_test_model_2a  <- df_test[logic_subsetter_test_2a,  feature_cols_model_2a, drop = FALSE]

y_train_model_2a <- encode_yes_no(df_train[logic_subsetter_train_2a, response_col, drop = TRUE])
y_test_model_2a  <- encode_yes_no(df_test[logic_subsetter_test_2a,  response_col, drop = TRUE])

save_model_data("model_2a", X_train_model_2a, X_test_model_2a, y_train_model_2a, y_test_model_2a)

# =========================
# 8. Model 2b: Predicting protective behaviour in mandate periods
# =========================

feature_cols_model_2b <- feature_cols_model_2a

logic_subsetter_train_2b <- df_train$within_mandate_period == 1
logic_subsetter_test_2b  <- df_test$within_mandate_period == 1

X_train_model_2b <- df_train[logic_subsetter_train_2b, feature_cols_model_2b, drop = FALSE]
X_test_model_2b  <- df_test[logic_subsetter_test_2b,  feature_cols_model_2b, drop = FALSE]

y_train_model_2b <- encode_yes_no(df_train[logic_subsetter_train_2b, response_col, drop = TRUE])
y_test_model_2b  <- encode_yes_no(df_test[logic_subsetter_test_2b,  response_col, drop = TRUE])

save_model_data("model_2b", X_train_model_2b, X_test_model_2b, y_train_model_2b, y_test_model_2b)

# =========================
# 9. Plot total sample size across the six models
# =========================

dir.create("figures/3_model_split", recursive = TRUE, showWarnings = FALSE)

model_sample_size_df <- tibble(
  model = c("Model 1", "Model 1a", "Model 1b", "Model 2", "Model 2a", "Model 2b"),
  total_n = c(
    nrow(X_train_model_1)  + nrow(X_test_model_1),
    nrow(X_train_model_1a) + nrow(X_test_model_1a),
    nrow(X_train_model_1b) + nrow(X_test_model_1b),
    nrow(X_train_model_2)  + nrow(X_test_model_2),
    nrow(X_train_model_2a) + nrow(X_test_model_2a),
    nrow(X_train_model_2b) + nrow(X_test_model_2b)
  ),
  outcome = c(
    "Face mask",
    "Face mask",
    "Face mask",
    "Protective behaviour",
    "Protective behaviour",
    "Protective behaviour"
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

p_model_n <- ggplot(model_sample_size_df, aes(x = model, y = total_n)) +
  geom_col(fill = "#4E79A7", width = 0.68) +
  geom_text(
    aes(label = total_n),
    vjust = -0.35,
    size = 3.6,
    colour = "#222222"
  ) +
  labs(
    title = "Total Sample Size Across Modelling Datasets",
    subtitle = "Total number of observations used in each modelling task",
    x = "Model dataset",
    y = "Total sample size"
  ) +
  journal_theme +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12)))

print(p_model_n)

ggsave(
  "figures/3_model_split/figure1_total_sample_size_across_models.png",
  plot = p_model_n,
  width = 8,
  height = 5,
  dpi = 300
)

print("Train/test split and model-specific datasets have been saved.")
print(model_sample_size_df)
print("Figure saved to figures/3_model_split/figure1_total_sample_size_across_models.png")