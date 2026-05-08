library(readr)
library(dplyr)
library(ggplot2)

# =========================
# 0. Helper functions for recording preprocessing process
# =========================

record_stage <- function(data, stage_name) {
  tibble(
    stage = stage_name,
    rows = nrow(data),
    columns = ncol(data)
  )
}

make_missing_summary <- function(data, stage_name) {
  tibble(
    variable_name = names(data),
    missing_count = colSums(is.na(data)),
    missing_percentage = round(colSums(is.na(data)) / nrow(data) * 100, 2),
    stage = stage_name
  )
}

preprocess_stage_dimensions <- tibble(
  stage = character(),
  rows = integer(),
  columns = integer()
)

preprocess_missing_records <- tibble(
  variable_name = character(),
  missing_count = integer(),
  missing_percentage = numeric(),
  stage = character()
)


# =========================
# 1. Helper function: create dummy variables
# =========================

add_dummy_variables <- function(data, col_name) {
  if (!(col_name %in% names(data))) {
    warning(paste("Column not found:", col_name))
    return(data)
  }
  
  x <- as.character(data[[col_name]])
  
  levels <- sort(unique(x[!is.na(x)]))
  
  if (length(levels) <= 1) {
    data <- data %>% select(-all_of(col_name))
    return(data)
  }
  
  dummy_levels <- levels[-1]
  
  for (level in dummy_levels) {
    new_col_name <- paste0(col_name, "_", level)
    
    data[[new_col_name]] <- ifelse(
      is.na(x),
      0L,
      as.integer(x == level)
    )
  }
  
  data <- data %>% select(-all_of(col_name))
  
  return(data)
}


# =========================
# 2. Read cleaned data
# =========================

cleaned_df <- read_csv(
  "data/1_cleaned_data.csv",
  na = character(),
  show_col_types = FALSE
)

cleaned_df <- cleaned_df %>%
  mutate(
    endtime = as.Date(endtime, format = "%Y-%m-%d")
  )


# =========================
# 3. Read mandate start dates
# =========================

mandate_df <- read_csv(
  "data/00_mandate_start_dates.csv",
  show_col_types = FALSE
)

mandate_df <- mandate_df %>%
  mutate(
    Date = as.Date(Date, format = "%Y-%m-%d")
  )

states_date <- mandate_df$Date
names(states_date) <- mandate_df$RegionName


# =========================
# 4. Create within_mandate_period
# =========================

states_missing_from_mandate <- setdiff(
  unique(cleaned_df$state),
  names(states_date)
)

if (length(states_missing_from_mandate) > 0) {
  warning(
    paste(
      "These states are missing from mandate_start_dates.csv:",
      paste(states_missing_from_mandate, collapse = ", ")
    )
  )
}

cleaned_df <- cleaned_df %>%
  mutate(
    mandate_start_date = states_date[state],
    within_mandate_period = ifelse(
      endtime >= mandate_start_date,
      1,
      0
    )
  ) %>%
  select(-mandate_start_date)

preprocess_stage_dimensions <- bind_rows(
  preprocess_stage_dimensions,
  record_stage(cleaned_df, "After adding mandate-period indicator")
)

preprocess_missing_records <- bind_rows(
  preprocess_missing_records,
  make_missing_summary(cleaned_df, "After adding mandate-period indicator")
)


# =========================
# 5. Convert selected categorical variables into dummy variables
# =========================

convert_into_dummy_cols <- c(
  "state",
  "gender",
  "i9_health",
  "employment_status",
  "i11_health",
  "WCRex1",
  "WCRex2",
  "PHQ4_1",
  "PHQ4_2",
  "PHQ4_3",
  "PHQ4_4",
  "d1_comorbidities"
)

for (col in convert_into_dummy_cols) {
  cleaned_df <- add_dummy_variables(cleaned_df, col)
}

preprocess_stage_dimensions <- bind_rows(
  preprocess_stage_dimensions,
  record_stage(cleaned_df, "Final modelling dataset")
)

preprocess_missing_records <- bind_rows(
  preprocess_missing_records,
  make_missing_summary(cleaned_df, "Final modelling dataset")
)


# =========================
# 6. Save preprocessed data
# =========================

write_csv(
  cleaned_df,
  "data/2_cleaned_data_preprocessing.csv"
)

print("Preprocessed data saved to data/2_cleaned_data_preprocessing.csv")
print(dim(cleaned_df))


# =========================
# 7. Read cleaning-stage records
# =========================

cleaning_stage_dimensions <- read_csv(
  "figures/1_cleaning_comparison/cleaning_stage_dimensions.csv",
  show_col_types = FALSE
)

cleaning_missing_records <- read_csv(
  "figures/1_cleaning_comparison/cleaning_missing_summary_records.csv",
  show_col_types = FALSE
)

stage_dimensions <- bind_rows(
  cleaning_stage_dimensions,
  preprocess_stage_dimensions
)

missing_summary_records <- bind_rows(
  cleaning_missing_records,
  preprocess_missing_records
)

print(stage_dimensions)


# =========================
# 8. Create complete cleaning and preprocessing comparison figures
# =========================

dir.create("figures/1_cleaning_comparison", recursive = TRUE, showWarnings = FALSE)

journal_theme <- theme_classic(base_size = 12) +
  theme(
    plot.title = element_text(size = 14, face = "bold", colour = "#222222"),
    plot.subtitle = element_text(size = 11, colour = "#555555"),
    axis.title = element_text(size = 12, colour = "#222222"),
    axis.text = element_text(size = 10, colour = "#333333"),
    axis.text.x = element_text(angle = 25, hjust = 1),
    axis.line = element_line(colour = "#333333", linewidth = 0.5),
    axis.ticks = element_line(colour = "#333333", linewidth = 0.4),
    panel.grid.major.y = element_line(colour = "#E6E6E6", linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank()
  )


# =========================
# Figure 1:
# Rows and columns retained across cleaning and preprocessing steps
# =========================

dimension_long <- bind_rows(
  stage_dimensions %>%
    transmute(stage, measure = "Rows", value = rows),
  stage_dimensions %>%
    transmute(stage, measure = "Columns", value = columns)
)

dimension_long$stage <- factor(
  dimension_long$stage,
  levels = stage_dimensions$stage
)

p1 <- ggplot(dimension_long, aes(x = stage, y = value)) +
  geom_col(fill = "#4E79A7", width = 0.65) +
  geom_text(
    aes(label = value),
    vjust = -0.3,
    size = 3.0,
    colour = "#222222"
  ) +
  facet_wrap(~ measure, scales = "free_y", ncol = 1) +
  labs(
    title = "Rows and Columns Retained Across Data Cleaning and Preprocessing Steps",
    subtitle = "Changes in dataset size from raw data to final modelling dataset",
    x = "Data processing step",
    y = "Count"
  ) +
  journal_theme +
  scale_y_continuous(expand = expansion(mult = c(0, 0.14)))

print(p1)

ggsave(
  "figures/1_cleaning_comparison/figure1_rows_columns_cleaning_preprocessing.png",
  plot = p1,
  width = 11,
  height = 7,
  dpi = 300
)


# =========================
# Figure 2:
# Missingness distribution across selected cleaning and preprocessing stages
# =========================

selected_missing_stages <- c(
  "Raw data",
  "After removing high-missing variables",
  "After filling medical missing values",
  "Final cleaned data",
  "After adding mandate-period indicator",
  "Final modelling dataset"
)

missing_plot_df <- missing_summary_records %>%
  filter(stage %in% selected_missing_stages)

missing_plot_df$stage <- factor(
  missing_plot_df$stage,
  levels = selected_missing_stages
)

p2 <- ggplot(missing_plot_df, aes(x = missing_percentage)) +
  geom_histogram(
    binwidth = 5,
    boundary = 0,
    fill = "#4E79A7",
    colour = "white",
    linewidth = 0.3
  ) +
  facet_wrap(~ stage, ncol = 1) +
  scale_x_continuous(
    limits = c(0, 100),
    breaks = seq(0, 100, 20),
    labels = function(x) paste0(x, "%")
  ) +
  labs(
    title = "Distribution of Variable-Level Missingness Across Cleaning and Preprocessing Steps",
    subtitle = "Each panel shows the distribution of missing percentages across variables",
    x = "Missing percentage",
    y = "Number of variables"
  ) +
  journal_theme +
  theme(
    axis.text.x = element_text(angle = 0, hjust = 0.5)
  )

print(p2)

ggsave(
  "figures/1_cleaning_comparison/figure2_missingness_distribution_cleaning_preprocessing.png",
  plot = p2,
  width = 8,
  height = 10,
  dpi = 300
)

print("Complete cleaning and preprocessing comparison figures saved to figures/1_cleaning_comparison/")