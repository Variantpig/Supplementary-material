library(readr)
library(dplyr)
library(ggplot2)
library(scales)

# =========================
# 1. Read raw data
# =========================

df <- read_csv(
  "raw_data/australia.csv",
  na = c("", " ", "__NA__", "NA", "N/A", "NULL", "NaN"),
  show_col_types = FALSE
)

# =========================
# 2. Missing value summary table
# =========================

missing_value_df <- tibble(
  `Variable Name` = names(df),
  `Missing Value Count` = colSums(is.na(df))
) %>%
  mutate(
    `Missing Percentage` = round(`Missing Value Count` / nrow(df) * 100, 2)
  ) %>%
  arrange(`Missing Value Count`, `Variable Name`)

# Create data folder 
if (!dir.exists("data")) {
  dir.create("data")
}

write_csv(
  missing_value_df,
  "data/0_missing_value_counts.csv"
)

print(head(missing_value_df, 20))

# =========================
# 3. Journal-style theme
# =========================

journal_theme <- theme_classic(base_size = 12, base_family = "Arial") +
  theme(
    plot.title = element_text(size = 14, face = "bold", colour = "#222222"),
    plot.subtitle = element_text(size = 11, colour = "#555555"),
    axis.title = element_text(size = 12, colour = "#222222"),
    axis.text = element_text(size = 10, colour = "#333333"),
    axis.line = element_line(colour = "#333333", linewidth = 0.5),
    axis.ticks = element_line(colour = "#333333", linewidth = 0.4),
    panel.grid.major.y = element_line(colour = "#E6E6E6", linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    plot.margin = margin(12, 16, 12, 12)
  )

# =========================
# 4. Figure 1:
# Distribution of missing percentage across variables
# =========================

p1 <- ggplot(missing_value_df, aes(x = `Missing Percentage`)) +
  geom_histogram(
    bins = 30,
    fill = "#4E79A7",
    colour = "white",
    linewidth = 0.3
  ) +
  scale_x_continuous(
    labels = function(x) paste0(x, "%"),
    expand = expansion(mult = c(0.01, 0.03))
  ) +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0.08))
  ) +
  labs(
    title = "Distribution of Missingness Across Variables",
    subtitle = "Each bar represents the number of variables within a missing-percentage range",
    x = "Missing percentage",
    y = "Number of variables"
  ) +
  journal_theme

print(p1)

ggsave(
  "figures/0_figure_missing_percentage_distribution.png",
  plot = p1,
  width = 7.2,
  height = 4.8,
  dpi = 300
)

# =========================
# 5. Figure 2:
# Number of variables by missingness group
# =========================

missing_group_df <- missing_value_df %>%
  mutate(
    missing_group = cut(
      `Missing Percentage`,
      breaks = c(-Inf, 0, 5, 10, 20, 50, Inf),
      labels = c("0%", "0-5%", "5-10%", "10-20%", "20-50%", ">50%"),
      right = TRUE
    )
  ) %>%
  count(missing_group) %>%
  mutate(
    missing_group = factor(
      missing_group,
      levels = c("0%", "0-5%", "5-10%", "10-20%", "20-50%", ">50%")
    )
  )

p2 <- ggplot(missing_group_df, aes(x = missing_group, y = n)) +
  geom_col(
    fill = "#4E79A7",
    width = 0.68
  ) +
  geom_text(
    aes(label = n),
    vjust = -0.35,
    size = 3.5,
    colour = "#222222"
  ) +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0.12))
  ) +
  labs(
    title = "Variables Grouped by Missingness Level",
    subtitle = "Variables are grouped according to their percentage of missing values",
    x = "Missingness group",
    y = "Number of variables"
  ) +
  journal_theme +
  theme(
    panel.grid.major.y = element_line(colour = "#E6E6E6", linewidth = 0.3),
    axis.line.x = element_line(colour = "#333333"),
    axis.line.y = element_line(colour = "#333333")
  )

print(p2)

ggsave(
  "figures/0_figure_missingness_group_barplot.png",
  plot = p2,
  width = 7.2,
  height = 4.8,
  dpi = 300
)

