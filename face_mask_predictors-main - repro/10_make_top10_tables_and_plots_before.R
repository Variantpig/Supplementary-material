# =========================================================
# Step 10: Make top-10 comparison tables and plots
# =========================================================

library(readr)
library(dplyr)
library(ggplot2)
library(forcats)
library(patchwork)

# =========================================================
# 1. Settings
# =========================================================

top_n <- 10
exclude_state <- TRUE   

dir.create("results/top10_feature_comparison_tables", recursive = TRUE, showWarnings = FALSE)
dir.create("figures/top10_feature_comparison_plots", recursive = TRUE, showWarnings = FALSE)

# =========================================================
# 2. Helper: read one summary file and get top N features
# =========================================================

get_top_features <- function(file_path, top_n = 10, exclude_state = TRUE) {
  df <- read_csv(file_path, show_col_types = FALSE)
  
  if (exclude_state) {
    df <- df %>%
      filter(!grepl("^state_", variable))
  }
  
  df %>%
    arrange(desc(median_importance)) %>%
    slice_head(n = top_n) %>%
    mutate(rank = row_number()) %>%
    select(rank, variable, median_importance, q1, q3, iqr)
}

# =========================================================
# 3. Helper: make side-by-side comparison table
# =========================================================

make_comparison_table <- function(before_file,
                                  after_file,
                                  output_file,
                                  top_n = 10,
                                  exclude_state = TRUE) {
  
  before_df <- get_top_features(before_file, top_n, exclude_state) %>%
    rename(
      rank_before = rank,
      variable_before = variable,
      median_importance_before = median_importance,
      q1_before = q1,
      q3_before = q3,
      iqr_before = iqr
    )
  
  after_df <- get_top_features(after_file, top_n, exclude_state) %>%
    rename(
      rank_after = rank,
      variable_after = variable,
      median_importance_after = median_importance,
      q1_after = q1,
      q3_after = q3,
      iqr_after = iqr
    )
  
  final_table <- bind_cols(before_df, after_df)
  
  write_csv(final_table, output_file)
  
  return(final_table)
}

# =========================================================
# 4. Helper: make before/after comparison plot
#    raw variable names only, with IQR error bars
# =========================================================

plot_comparison <- function(before_file,
                            after_file,
                            output_file,
                            main_title,
                            before_title = "Before mandates",
                            after_title = "After mandates",
                            top_n = 10,
                            exclude_state = TRUE) {
  
  before_df <- get_top_features(before_file, top_n, exclude_state)
  after_df  <- get_top_features(after_file, top_n, exclude_state)
  
  x_max <- max(c(before_df$q3, after_df$q3), na.rm = TRUE) * 1.1
  
  p_before <- ggplot(
    before_df,
    aes(x = median_importance,
        y = fct_reorder(variable, median_importance))
  ) +
    geom_col(fill = "#1F77B4", width = 0.7) +
    geom_errorbarh(aes(xmin = q1, xmax = q3), height = 0.2, linewidth = 0.5) +
    scale_x_continuous(limits = c(0, x_max)) +
    labs(
      title = before_title,
      x = "Median feature importance",
      y = NULL
    ) +
    theme_bw(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      axis.text.y = element_text(size = 10)
    )
  
  p_after <- ggplot(
    after_df,
    aes(x = median_importance,
        y = fct_reorder(variable, median_importance))
  ) +
    geom_col(fill = "#E68613", width = 0.7) +
    geom_errorbarh(aes(xmin = q1, xmax = q3), height = 0.2, linewidth = 0.5) +
    scale_x_continuous(limits = c(0, x_max)) +
    labs(
      title = after_title,
      x = "Median feature importance",
      y = NULL
    ) +
    theme_bw(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      axis.text.y = element_text(size = 10)
    )
  
  final_plot <- (p_before | p_after) +
    plot_annotation(
      title = main_title,
      theme = theme(
        plot.title = element_text(face = "bold", size = 14, hjust = 0.5)
      )
    )
  
  print(final_plot)
  
  ggsave(
    filename = output_file,
    plot = final_plot,
    width = 14,
    height = 6,
    dpi = 300
  )
  
  return(final_plot)
}

# =========================================================
# 5. File definitions
# =========================================================

comparisons <- list(
  
  list(
    name = "table_1_xgb_mask_before_after",
    before_file = "results/model_1a_xgboost_feature_importance_summary.csv",
    after_file  = "results/model_1b_xgboost_feature_importance_summary.csv",
    plot_title  = "XGBoost: Face Mask Behaviour"
  ),
  
  list(
    name = "table_2_rf_mask_before_after",
    before_file = "results/model_1a_rf_feature_importance_summary.csv",
    after_file  = "results/model_1b_rf_feature_importance_summary.csv",
    plot_title  = "Random Forest: Face Mask Behaviour"
  ),
  
  list(
    name = "table_3_xgb_protective_before_after",
    before_file = "results/model_2a_xgboost_feature_importance_summary.csv",
    after_file  = "results/model_2b_xgboost_feature_importance_summary.csv",
    plot_title  = "XGBoost: Protective Behaviour"
  ),
  
  list(
    name = "table_4_rf_protective_before_after",
    before_file = "results/model_2a_rf_feature_importance_summary.csv",
    after_file  = "results/model_2b_rf_feature_importance_summary.csv",
    plot_title  = "Random Forest: Protective Behaviour"
  )
)

# =========================================================
# 6. Run all four comparisons
# =========================================================

all_plots <- list()

for (cmp in comparisons) {
  
  cat("\n====================================\n")
  cat("Processing:", cmp$name, "\n")
  cat("====================================\n")
  
  table_out <- paste0(
    "results/top10_feature_comparison_tables/",
    cmp$name,
    ".csv"
  )
  
  plot_out <- paste0(
    "figures/top10_feature_comparison_plots/",
    sub("^table_", "figure_", cmp$name),
    ".png"
  )
  
  table_df <- make_comparison_table(
    before_file = cmp$before_file,
    after_file = cmp$after_file,
    output_file = table_out,
    top_n = top_n,
    exclude_state = exclude_state
  )
  
  print(table_df)
  
  all_plots[[cmp$name]] <- plot_comparison(
    before_file = cmp$before_file,
    after_file = cmp$after_file,
    output_file = plot_out,
    main_title = cmp$plot_title,
    before_title = "Before mandates",
    after_title = "After mandates",
    top_n = top_n,
    exclude_state = exclude_state
  )
}

cat("\nAll tables and plots have been saved successfully.\n")

# =========================================================
# 7. Optional: save one combined 2x2 figure
# =========================================================

combined_plot <- (all_plots[[1]] / all_plots[[2]]) | (all_plots[[3]] / all_plots[[4]])

ggsave(
  filename = "figures/top10_feature_comparison_plots/combined_top10_feature_comparison.png",
  plot = combined_plot,
  width = 20,
  height = 12,
  dpi = 300
)

cat("Combined 2x2 figure has also been saved.\n")