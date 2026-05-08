library(readr)
library(dplyr)
library(ggplot2)

# =========================
# 1. Read cleaned data
# =========================

df <- read_csv(
  "data/1_cleaned_data.csv",
  show_col_types = FALSE
)

df <- df %>%
  mutate(
    endtime = as.Date(endtime)
  )

if (!dir.exists("figures")) {
  dir.create("figures")
}

if (!dir.exists("figures/cleaned_data")) {
  dir.create("figures/1_cleaned_data")
}

# =========================
# 2. Journal-style theme
# =========================

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

# =========================
# Figure 1:
# Face mask behaviour binary distribution
# =========================

p1 <- df %>%
  count(face_mask_behaviour_binary) %>%
  ggplot(aes(x = face_mask_behaviour_binary, y = n)) +
  geom_col(fill = "#4E79A7", width = 0.65) +
  geom_text(aes(label = n), vjust = -0.35, size = 3.8) +
  labs(
    title = "Distribution of Face Mask Behaviour",
    subtitle = "Binary outcome derived from the face mask behaviour scale",
    x = "Face mask behaviour",
    y = "Number of respondents"
  ) +
  journal_theme +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12)))

print(p1)

ggsave(
  "figures/1_cleaned_data/1_figure1_face_mask_binary_distribution.png",
  plot = p1,
  width = 7,
  height = 5,
  dpi = 300
)

# =========================
# Figure 2:
# Face mask behaviour scale distribution
# =========================

p2 <- ggplot(df, aes(x = face_mask_behaviour_scale)) +
  geom_histogram(
    binwidth = 0.5,
    fill = "#4E79A7",
    colour = "white",
    linewidth = 0.3
  ) +
  labs(
    title = "Distribution of Face Mask Behaviour Scale",
    subtitle = "Scale calculated as the median of selected face mask behaviour items",
    x = "Face mask behaviour scale",
    y = "Number of respondents"
  ) +
  journal_theme

print(p2)

ggsave(
  "figures/1_cleaned_data/1_figure2_face_mask_scale_distribution.png",
  plot = p2,
  width = 7,
  height = 5,
  dpi = 300
)

# =========================
# Figure 3:
# Face mask behaviour over time
# =========================

time_summary <- df %>%
  group_by(week_number) %>%
  summarise(
    n = n(),
    face_mask_yes_rate = mean(face_mask_behaviour_binary == "Yes") * 100,
    .groups = "drop"
  )

p3 <- ggplot(time_summary, aes(x = week_number, y = face_mask_yes_rate)) +
  geom_line(colour = "#4E79A7", linewidth = 1) +
  geom_point(colour = "#4E79A7", size = 1.8) +
  labs(
    title = "Face Mask Behaviour Over Time",
    subtitle = "Percentage of respondents classified as Yes by two-week period",
    x = "Two-week period",
    y = "Face mask behaviour: Yes (%)"
  ) +
  journal_theme +
  scale_y_continuous(limits = c(0, 100))

print(p3)

ggsave(
  "figures/1_cleaned_data/1_figure3_face_mask_behaviour_over_time.png",
  plot = p3,
  width = 8,
  height = 5,
  dpi = 300
)

# =========================
# Figure 4:
# Face mask behaviour by state
# =========================

state_summary <- df %>%
  group_by(state) %>%
  summarise(
    n = n(),
    face_mask_yes_rate = mean(face_mask_behaviour_binary == "Yes") * 100,
    .groups = "drop"
  ) %>%
  arrange(face_mask_yes_rate)

p4 <- ggplot(
  state_summary,
  aes(x = face_mask_yes_rate, y = reorder(state, face_mask_yes_rate))
) +
  geom_col(fill = "#4E79A7", width = 0.65) +
  geom_text(
    aes(label = paste0(round(face_mask_yes_rate, 1), "%")),
    hjust = -0.15,
    size = 3.5
  ) +
  labs(
    title = "Face Mask Behaviour by State",
    subtitle = "Percentage of respondents classified as Yes in each state",
    x = "Face mask behaviour: Yes (%)",
    y = "State"
  ) +
  journal_theme +
  scale_x_continuous(limits = c(0, 100), expand = expansion(mult = c(0, 0.08)))

print(p4)

ggsave(
  "figures/1_cleaned_data/1_figure4_face_mask_behaviour_by_state.png",
  plot = p4,
  width = 8,
  height = 5,
  dpi = 300
)

# =========================
# Figure 5:
# Protective behaviour without face mask by face mask group
# =========================

p5 <- ggplot(
  df,
  aes(
    x = face_mask_behaviour_binary,
    y = protective_behaviour_nomask_scale
  )
) +
  geom_boxplot(
    fill = "#A0CBE8",
    colour = "#333333",
    width = 0.6,
    outlier.alpha = 0.25
  ) +
  labs(
    title = "Protective Behaviour Without Face Mask by Face Mask Behaviour Group",
    subtitle = "Comparison of non-mask protective behaviour scale between face mask behaviour groups",
    x = "Face mask behaviour",
    y = "Protective behaviour without face mask scale"
  ) +
  journal_theme

print(p5)

ggsave(
  "figures/1_cleaned_data/1_figure5_protective_behaviour_by_face_mask_group.png",
  plot = p5,
  width = 7,
  height = 5,
  dpi = 300
)

# =========================
# Figure 6:
# Protective behaviour binary distribution
# =========================

p6 <- df %>%
  count(protective_behaviour_binary) %>%
  ggplot(aes(x = protective_behaviour_binary, y = n)) +
  geom_col(fill = "#4E79A7", width = 0.65) +
  geom_text(aes(label = n), vjust = -0.35, size = 3.8) +
  labs(
    title = "Distribution of Overall Protective Behaviour",
    subtitle = "Binary outcome derived from the overall protective behaviour scale",
    x = "Overall protective behaviour",
    y = "Number of respondents"
  ) +
  journal_theme +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12)))

print(p6)

ggsave(
  "figures/1_cleaned_data/1_figure6_protective_behaviour_binary_distribution.png",
  plot = p6,
  width = 7,
  height = 5,
  dpi = 300
)


# =========================
# Figure 7:
# Protective behaviour scale distribution
# =========================

p7 <- ggplot(df, aes(x = protective_behaviour_scale)) +
  geom_histogram(
    binwidth = 0.5,
    fill = "#4E79A7",
    colour = "white",
    linewidth = 0.3
  ) +
  labs(
    title = "Distribution of Overall Protective Behaviour Scale",
    subtitle = "Scale calculated as the median of all protective behaviour items",
    x = "Overall protective behaviour scale",
    y = "Number of respondents"
  ) +
  journal_theme

print(p7)

ggsave(
  "figures/1_cleaned_data/1_figure7_protective_behaviour_scale_distribution.png",
  plot = p7,
  width = 7,
  height = 5,
  dpi = 300
)


# =========================
# Figure 8:
# Protective behaviour over time
# =========================

protective_time_summary <- df %>%
  group_by(week_number) %>%
  summarise(
    n = n(),
    protective_yes_rate = mean(protective_behaviour_binary == "Yes") * 100,
    .groups = "drop"
  )

p8 <- ggplot(protective_time_summary, aes(x = week_number, y = protective_yes_rate)) +
  geom_line(colour = "#4E79A7", linewidth = 1) +
  geom_point(colour = "#4E79A7", size = 1.8) +
  labs(
    title = "Overall Protective Behaviour Over Time",
    subtitle = "Percentage of respondents classified as Yes by two-week period",
    x = "Two-week period",
    y = "Overall protective behaviour: Yes (%)"
  ) +
  journal_theme +
  scale_y_continuous(limits = c(0, 100))

print(p8)

ggsave(
  "figures/1_cleaned_data/1_figure8_protective_behaviour_over_time.png",
  plot = p8,
  width = 8,
  height = 5,
  dpi = 300
)


# =========================
# Figure 9:
# Protective behaviour by state
# =========================

protective_state_summary <- df %>%
  group_by(state) %>%
  summarise(
    n = n(),
    protective_yes_rate = mean(protective_behaviour_binary == "Yes") * 100,
    .groups = "drop"
  ) %>%
  arrange(protective_yes_rate)

p9 <- ggplot(
  protective_state_summary,
  aes(x = protective_yes_rate, y = reorder(state, protective_yes_rate))
) +
  geom_col(fill = "#4E79A7", width = 0.65) +
  geom_text(
    aes(label = paste0(round(protective_yes_rate, 1), "%")),
    hjust = -0.15,
    size = 3.5
  ) +
  labs(
    title = "Overall Protective Behaviour by State",
    subtitle = "Percentage of respondents classified as Yes in each state",
    x = "Overall protective behaviour: Yes (%)",
    y = "State"
  ) +
  journal_theme +
  scale_x_continuous(
    limits = c(0, 100),
    expand = expansion(mult = c(0, 0.08))
  )

print(p9)

ggsave(
  "figures/1_cleaned_data/1_figure9_protective_behaviour_by_state.png",
  plot = p9,
  width = 8,
  height = 5,
  dpi = 300
)

print("Cleaned data visualisation completed.")
print("Figures saved to figures/cleaned_data/")