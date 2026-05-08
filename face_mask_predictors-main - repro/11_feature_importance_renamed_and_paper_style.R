library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(forcats)
library(stringr)
library(patchwork)
library(grid)

# =========================================================
# 0. Output folders
# =========================================================

dir.create("results/top10_feature_comparison_tables_renamed_v3",
           recursive = TRUE, showWarnings = FALSE)

dir.create("figures/feature_importance_comparison_renamed_v3",
           recursive = TRUE, showWarnings = FALSE)

dir.create("figures/paper_style_feature_importance_renamed_v3",
           recursive = TRUE, showWarnings = FALSE)

# =========================================================
# 1. Variable renaming
# =========================================================

feature_name_map <- c(
  "protective_behaviour_nomask_scale" = "Protective Behaviour",
  "age" = "Age",
  "week_number" = "Week Number",
  "i2_health" = "Non-Household Contacts",
  "r1_1" = "Perceived Severity",
  "r1_2" = "Perceived Susceptibility",
  "cantril_ladder" = "Cantril Ladder",
  "household_size" = "Household Size",
  "d1_comorbidities_Yes" = "Comorbidity(YES)",
  "gender_Male" = "Gender(Male)",
  "i9_health_Yes" = "Isolate If Unwell(YES)",
  "i11_health_Very willing" = "Very Willing To Isolate",
  "i9_health_Not sure" = "Isolate If Unwell(Not Sure)"
)

# =========================================================
# 2. Category mapping
# =========================================================

feature_category_map <- c(
  "Protective Behaviour" = "Self protective behaviours",
  "Very Willing To Isolate" = "Self protective behaviours",
  "Isolate If Unwell(YES)" = "Self protective behaviours",
  "Isolate If Unwell(Not Sure)" = "Self protective behaviours",
  
  "Age" = "Demographics",
  "Gender(Male)" = "Demographics",
  "Household Size" = "Demographics",
  "Comorbidity(YES)" = "Health, mental health and wellbeing",
  
  "Non-Household Contacts" = "Health, mental health and wellbeing",
  "Cantril Ladder" = "Health, mental health and wellbeing",
  
  "Perceived Severity" = "Perception of illness threat",
  "Perceived Susceptibility" = "Perception of illness threat",
  
  "Week Number" = "Time"
)

category_palette <- c(
  "Self protective behaviours" = "#0B6FA4",
  "Demographics" = "#3A92BE",
  "Health, mental health and wellbeing" = "#67A9CF",
  "Perception of illness threat" = "#BEB4A2",
  "Time" = "#E08D2D",
  "Trust in government" = "#C76E00"
)

# =========================================================
# 3. Helper functions
# =========================================================

rename_features <- function(x) {
  recode(x, !!!feature_name_map, .default = x)
}

assign_category <- function(feature_vec) {
  out <- feature_category_map[feature_vec]
  out[is.na(out)] <- "Other"
  out
}

# read raw feature importance file from step 9
# expected format: rows = repetitions, columns = features
read_feature_importance_raw <- function(file_path) {
  df <- read_csv(file_path, show_col_types = FALSE)
  
  # remove possible index column like ...1 / X
  drop_candidates <- names(df)[grepl("^\\.\\.\\.|^X$", names(df)) |
                                 names(df) %in% c("X", "...1")]
  if (length(drop_candidates) > 0) {
    df <- df %>% select(-any_of(drop_candidates))
  }
  
  df_long <- df %>%
    pivot_longer(
      cols = everything(),
      names_to = "feature",
      values_to = "importance"
    ) %>%
    mutate(feature = as.character(feature))
  
  df_long
}

summarise_feature_importance <- function(file_path,
                                         top_n = 10,
                                         exclude_state = TRUE) {
  
  df_long <- read_feature_importance_raw(file_path)
  
  out <- df_long %>%
    group_by(feature) %>%
    summarise(
      median_importance = median(importance, na.rm = TRUE),
      se = sd(importance, na.rm = TRUE) / sqrt(sum(!is.na(importance))),
      .groups = "drop"
    )
  
  if (exclude_state) {
    out <- out %>%
      filter(!str_detect(feature, "^state_"))
  }
  
  out %>%
    mutate(
      feature = rename_features(feature),
      category = assign_category(feature)
    ) %>%
    arrange(desc(median_importance)) %>%
    slice_head(n = top_n)
}


build_top10_comparison <- function(before_file,
                                   after_file,
                                   top_n = 10,
                                   exclude_state = TRUE) {
  
  before_top <- summarise_feature_importance(
    before_file,
    top_n = top_n,
    exclude_state = exclude_state
  ) %>%
    rename(
      median_before = median_importance,
      se_before = se,
      category_before = category
    )
  
  after_top <- summarise_feature_importance(
    after_file,
    top_n = top_n,
    exclude_state = exclude_state
  ) %>%
    rename(
      median_after = median_importance,
      se_after = se,
      category_after = category
    )
  
  union_features <- union(before_top$feature, after_top$feature)
  
  comp <- tibble(feature = union_features) %>%
    left_join(before_top, by = "feature") %>%
    left_join(after_top, by = "feature") %>%
    mutate(
      in_before_top10 = !is.na(median_before),
      in_after_top10 = !is.na(median_after),
      
      median_before = ifelse(is.na(median_before), 0, median_before),
      se_before = ifelse(is.na(se_before), 0, se_before),
      
      median_after = ifelse(is.na(median_after), 0, median_after),
      se_after = ifelse(is.na(se_after), 0, se_after)
    ) %>%
    mutate(
      plot_category = coalesce(category_before, category_after)
    ) %>%
    arrange(desc(pmax(median_before, median_after))) %>%
    mutate(feature = factor(feature, levels = rev(feature)))
  
  comp
}

# =========================================================
# 4. Plot function (legend fixed version)
# =========================================================

make_paper_panel_plot_fixed <- function(comp_df,
                                        main_title = "",
                                        panel_title = NULL,
                                        show_legend = TRUE) {
  
  comp_df <- comp_df %>%
    mutate(feature_chr = as.character(feature))
  
  n_feat <- nrow(comp_df)
  max_x <- max(comp_df$median_before, comp_df$median_after, na.rm = TRUE)
  max_x <- max_x * 1.15
  
  left_df <- comp_df %>%
    transmute(
      feature = factor(feature_chr, levels = rev(unique(feature_chr))),
      value = -median_before,
      xmin = -(median_before + se_before),
      xmax = -(pmax(median_before - se_before, 0)),
      category = plot_category,
      keep_errorbar = in_before_top10
    )
  
  right_df <- comp_df %>%
    transmute(
      feature = factor(feature_chr, levels = rev(unique(feature_chr))),
      value = median_after,
      xmin = pmax(median_after - se_after, 0),
      xmax = median_after + se_after,
      category = plot_category,
      keep_errorbar = in_after_top10
    )
  
  plot_df <- bind_rows(left_df, right_df)
  
  legend_breaks <- intersect(names(category_palette), unique(plot_df$category))
  
  y_bottom <- n_feat + 0.6
  y_top <- n_feat + 1.45
  y_text <- n_feat + 1.02
  
  p <- ggplot(plot_df, aes(x = value, y = feature, fill = category)) +
    geom_col(width = 0.85, colour = NA) +
    
    geom_errorbarh(
      data = plot_df %>% filter(keep_errorbar, value != 0),
      aes(xmin = xmin, xmax = xmax),
      height = 0.18,
      linewidth = 0.7,
      colour = "black"
    ) +
    
    geom_vline(xintercept = 0, linewidth = 0.8, colour = "black") +
    
    annotate("rect",
             xmin = -max_x, xmax = 0,
             ymin = y_bottom, ymax = y_top,
             fill = "#D9D9D9", colour = "black") +
    annotate("rect",
             xmin = 0, xmax = max_x,
             ymin = y_bottom, ymax = y_top,
             fill = "#D9D9D9", colour = "black") +
    
    annotate("text",
             x = -max_x / 2, y = y_text,
             label = "Before mandates",
             family = "serif", fontface = "plain", size = 7) +
    annotate("text",
             x = max_x / 2, y = y_text,
             label = "After mandates",
             family = "serif", fontface = "plain", size = 7) +
    
    scale_fill_manual(
      values = category_palette,
      breaks = legend_breaks,
      drop = TRUE
    ) +
    
    scale_x_continuous(
      limits = c(-max_x, max_x),
      breaks = pretty(c(-max_x, max_x), n = 7),
      labels = function(x) {
        ifelse(abs(x) < 1e-10, "0", format(abs(x), trim = TRUE))
      },
      expand = expansion(mult = c(0.01, 0.01))
    ) +
    
    coord_cartesian(
      ylim = c(0.5, n_feat + 1.7),
      clip = "off"
    ) +
    
    labs(
      title = main_title,
      x = "Median feature importance",
      y = NULL,
      fill = NULL
    ) +
    
    guides(
      fill = guide_legend(
        nrow = 2,
        byrow = TRUE
      )
    ) +
    
    theme_bw(base_size = 16) +
    theme(
      plot.title = element_text(
        hjust = 0.5,
        face = "bold",
        size = 22,
        margin = margin(b = 20)
      ),
      axis.title.x = element_text(size = 18),
      axis.text.x = element_text(size = 13),
      axis.text.y = element_text(size = 13),
      panel.grid.major = element_line(colour = "#D9D9D9"),
      panel.grid.minor = element_blank(),
      legend.position = if (show_legend) "bottom" else "none",
      legend.text = element_text(size = 11),
      legend.key.width = unit(1.2, "cm"),
      legend.spacing.x = unit(0.4, "cm"),
      legend.box.margin = margin(t = 6, r = 6, b = 6, l = 6),
      plot.margin = margin(t = 26, r = 18, b = 28, l = 18)
    )
  
  if (!is.null(panel_title)) {
    p <- p + annotate(
      "text",
      x = -max_x * 1.48,
      y = n_feat + 1.55,
      label = panel_title,
      family = "serif",
      fontface = "plain",
      size = 7,
      hjust = 0
    )
  }
  
  p
}

# =========================================================
# 5. Build the four NEW V3 comparison tables
# =========================================================

comparison_results_v3 <- list()

# 1. XGBoost - Face Mask
comparison_results_v3[["table_1"]] <- build_top10_comparison(
  before_file = "results/model_1a_xgboost_feature_importance.csv",
  after_file  = "results/model_1b_xgboost_feature_importance.csv",
  top_n = 10,
  exclude_state = TRUE
)

write_csv(
  comparison_results_v3[["table_1"]] %>% mutate(feature = as.character(feature)),
  "results/top10_feature_comparison_tables_renamed_v3/table_1_xgb_face_mask_before_after_RENAMED_V3.csv"
)

# 2. RF - Face Mask
comparison_results_v3[["table_2"]] <- build_top10_comparison(
  before_file = "results/model_1a_rf_feature_importance.csv",
  after_file  = "results/model_1b_rf_feature_importance.csv",
  top_n = 10,
  exclude_state = TRUE
)

write_csv(
  comparison_results_v3[["table_2"]] %>% mutate(feature = as.character(feature)),
  "results/top10_feature_comparison_tables_renamed_v3/table_2_rf_face_mask_before_after_RENAMED_V3.csv"
)

# 3. XGBoost - Protective Behaviour
comparison_results_v3[["table_3"]] <- build_top10_comparison(
  before_file = "results/model_2a_xgboost_feature_importance.csv",
  after_file  = "results/model_2b_xgboost_feature_importance.csv",
  top_n = 10,
  exclude_state = TRUE
)

write_csv(
  comparison_results_v3[["table_3"]] %>% mutate(feature = as.character(feature)),
  "results/top10_feature_comparison_tables_renamed_v3/table_3_xgb_protective_before_after_RENAMED_V3.csv"
)

# 4. RF - Protective Behaviour
comparison_results_v3[["table_4"]] <- build_top10_comparison(
  before_file = "results/model_2a_rf_feature_importance.csv",
  after_file  = "results/model_2b_rf_feature_importance.csv",
  top_n = 10,
  exclude_state = TRUE
)

write_csv(
  comparison_results_v3[["table_4"]] %>% mutate(feature = as.character(feature)),
  "results/top10_feature_comparison_tables_renamed_v3/table_4_rf_protective_before_after_RENAMED_V3.csv"
)

# =========================================================
# 6. Save four single figures
# =========================================================

p1 <- make_paper_panel_plot_fixed(
  comparison_results_v3[["table_1"]],
  main_title = "XGBoost: Face Mask Behaviour",
  show_legend = TRUE
)

ggsave(
  "figures/feature_importance_comparison_renamed_v3/xgb_face_mask_RENAMED_V3.png",
  plot = p1, width = 14, height = 10, dpi = 300
)

p2 <- make_paper_panel_plot_fixed(
  comparison_results_v3[["table_2"]],
  main_title = "Random Forest: Face Mask Behaviour",
  show_legend = TRUE
)

ggsave(
  "figures/feature_importance_comparison_renamed_v3/rf_face_mask_RENAMED_V3.png",
  plot = p2, width = 14, height = 10, dpi = 300
)

p3 <- make_paper_panel_plot_fixed(
  comparison_results_v3[["table_3"]],
  main_title = "XGBoost: Protective Behaviour",
  show_legend = TRUE
)

ggsave(
  "figures/feature_importance_comparison_renamed_v3/xgb_protective_RENAMED_V3.png",
  plot = p3, width = 14, height = 10, dpi = 300
)

p4 <- make_paper_panel_plot_fixed(
  comparison_results_v3[["table_4"]],
  main_title = "Random Forest: Protective Behaviour",
  show_legend = TRUE
)

ggsave(
  "figures/feature_importance_comparison_renamed_v3/rf_protective_RENAMED_V3.png",
  plot = p4, width = 14, height = 10, dpi = 300
)

# =========================================================
# 7. Save two combined paper-style figures
# =========================================================

p_mask_xgb <- make_paper_panel_plot_fixed(
  comparison_results_v3[["table_1"]],
  main_title = NULL,
  panel_title = "(a)",
  show_legend = FALSE
)

p_mask_rf <- make_paper_panel_plot_fixed(
  comparison_results_v3[["table_2"]],
  main_title = NULL,
  panel_title = "(b)",
  show_legend = TRUE
)

combined_mask <- (p_mask_xgb / p_mask_rf) +
  plot_layout(guides = "collect", heights = c(1, 1)) +
  plot_annotation(
    theme = theme(
      legend.position = "bottom"
    )
  )

ggsave(
  "figures/paper_style_feature_importance_renamed_v3/paper_style_face_mask_RENAMED_V3.png",
  plot = combined_mask,
  width = 13.5,
  height = 14.5,
  dpi = 300
)

p_protective_xgb <- make_paper_panel_plot_fixed(
  comparison_results_v3[["table_3"]],
  main_title = NULL,
  panel_title = "(a)",
  show_legend = FALSE
)

p_protective_rf <- make_paper_panel_plot_fixed(
  comparison_results_v3[["table_4"]],
  main_title = NULL,
  panel_title = "(b)",
  show_legend = TRUE
)

combined_protective <- (p_protective_xgb / p_protective_rf) +
  plot_layout(guides = "collect", heights = c(1, 1)) +
  plot_annotation(
    theme = theme(
      legend.position = "bottom"
    )
  )

ggsave(
  "figures/paper_style_feature_importance_renamed_v3/paper_style_protective_behaviour_RENAMED_V3.png",
  plot = combined_protective,
  width = 13.5,
  height = 14.5,
  dpi = 300
)

cat("Done.\n")
cat("Tables saved to: results/top10_feature_comparison_tables_renamed_v3/\n")
cat("Single figures saved to: figures/feature_importance_comparison_renamed_v3/\n")
cat("Combined figures saved to: figures/paper_style_feature_importance_renamed_v3/\n")