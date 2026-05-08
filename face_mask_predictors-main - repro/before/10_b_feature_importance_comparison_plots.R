library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(forcats)

dir.create("figures/feature_importance_comparison", recursive = TRUE, showWarnings = FALSE)

prepare_mirror_plot_data <- function(df) {
  
  before_df <- df %>%
    transmute(
      feature = variable_before,
      importance_before = median_importance_before
    ) %>%
    filter(!is.na(feature), !is.na(importance_before))
  
  after_df <- df %>%
    transmute(
      feature = variable_after,
      importance_after = median_importance_after
    ) %>%
    filter(!is.na(feature), !is.na(importance_after))
  
  merged_df <- full_join(before_df, after_df, by = "feature") %>%
    mutate(
      importance_before = ifelse(is.na(importance_before), 0, importance_before),
      importance_after  = ifelse(is.na(importance_after), 0, importance_after)
    ) %>%
    select(feature, importance_before, importance_after)
  
  merged_df <- merged_df %>%
    mutate(max_imp = pmax(importance_before, importance_after)) %>%
    arrange(max_imp) %>%
    mutate(feature = factor(feature, levels = feature))
  
  bind_rows(
    merged_df %>%
      transmute(
        feature = feature,
        period = "Before mandates",
        importance = -importance_before
      ),
    merged_df %>%
      transmute(
        feature = feature,
        period = "After mandates",
        importance = importance_after
      )
  )
}

plot_mirror_importance <- function(input_file,
                                   output_file,
                                   plot_title = NULL,
                                   panel_label = NULL) {
  
  df <- read_csv(input_file, show_col_types = FALSE)
  plot_df <- prepare_mirror_plot_data(df)
  
  max_abs <- max(abs(plot_df$importance), na.rm = TRUE)
  y_top <- length(unique(plot_df$feature)) + 1.2
  
  p <- ggplot(plot_df, aes(x = importance, y = feature, fill = period)) +
    geom_col(width = 0.72) +
    geom_vline(xintercept = 0, linewidth = 0.4, colour = "grey30") +
    
    annotate("text",
             x = -max_abs * 0.55,
             y = y_top,
             label = "Before mandates",
             size = 5) +
    annotate("text",
             x = max_abs * 0.55,
             y = y_top,
             label = "After mandates",
             size = 5) +
    
    {if (!is.null(panel_label))
      annotate("text",
               x = -max_abs * 1.18,
               y = y_top,
               label = panel_label,
               hjust = 0,
               size = 5)} +
    
    scale_fill_manual(values = c(
      "Before mandates" = "#0B78A8",
      "After mandates" = "#E89C2C"
    )) +
    
    scale_x_continuous(
      limits = c(-max_abs * 1.15, max_abs * 1.15),
      labels = function(x) abs(round(x, 3)),
      expand = expansion(mult = c(0.02, 0.02))
    ) +
    
    labs(
      title = plot_title,
      x = "Median feature importance",
      y = NULL,
      fill = NULL
    ) +
    
    theme_bw(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5, size = 14),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position = "bottom",
      axis.text.y = element_text(size = 10),
      axis.text.x = element_text(size = 10),
      axis.title.x = element_text(size = 12),
      plot.margin = margin(15, 20, 10, 20)
    ) +
    coord_cartesian(clip = "off")
  
  print(p)
  
  ggsave(
    filename = output_file,
    plot = p,
    width = 10,
    height = 6,
    dpi = 300
  )
}

plot_mirror_importance(
  input_file  = "results/top10_feature_comparison_tables/table_1_xgb_mask_before_after.csv",
  output_file = "figures/feature_importance_comparison/figure1_xgb_mask_before_after.png",
  plot_title  = "XGBoost: Face Mask Behaviour",
  panel_label = "(a)"
)

plot_mirror_importance(
  input_file  = "results/top10_feature_comparison_tables/table_2_rf_mask_before_after.csv",
  output_file = "figures/feature_importance_comparison/figure2_rf_mask_before_after.png",
  plot_title  = "Random Forest: Face Mask Behaviour",
  panel_label = "(b)"
)

plot_mirror_importance(
  input_file  = "results/top10_feature_comparison_tables/table_3_xgb_protective_before_after.csv",
  output_file = "figures/feature_importance_comparison/figure3_xgb_protective_before_after.png",
  plot_title  = "XGBoost: Protective Behaviour",
  panel_label = "(c)"
)

plot_mirror_importance(
  input_file  = "results/top10_feature_comparison_tables/table_4_rf_protective_before_after.csv",
  output_file = "figures/feature_importance_comparison/figure4_rf_protective_before_after.png",
  plot_title  = "Random Forest: Protective Behaviour",
  panel_label = "(d)"
)

cat("All figures saved in figures/feature_importance_comparison/\n")