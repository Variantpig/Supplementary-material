library(readr)
library(dplyr)
library(ggplot2)

# =========================
# 1. Load and subset data
# =========================

df <- read_csv(
  "raw_data/OxCGRT_AUS_latest.csv",
  show_col_types = FALSE
)

col_subsets <- c(
  "RegionName",
  "RegionCode",
  "Date",
  "H6M_Facial Coverings"
)

df <- df %>%
  select(all_of(col_subsets)) %>%
  mutate(
    Date = as.Date(as.character(Date), format = "%Y%m%d")
  ) %>%
  filter(!is.na(RegionName))




# =========================
# 2. Define pandas-style rolling mean
# =========================

rolling_mean_pandas_style <- function(x, window) {
  out <- rep(NA_real_, length(x))
  
  for (i in seq_along(x)) {
    if (i >= window) {
      window_values <- x[(i - window + 1):i]
      
      if (sum(!is.na(window_values)) == window) {
        out[i] <- mean(window_values)
      } else {
        out[i] <- NA_real_
      }
    }
  }
  
  return(out)
}


# =========================
# 3. Calculate 14-day rolling average
# =========================

rolling_days <- 14

df_rolling <- df %>%
  group_by(RegionName) %>%
  mutate(
    rolling_facial_coverings = rolling_mean_pandas_style(
      `H6M_Facial Coverings`,
      rolling_days
    )
  ) %>%
  ungroup()


# =========================
# 4. Find first mandate date by region
# =========================

mandate_limit <- 3

df_mandates <- df_rolling %>%
  filter(
    !is.na(rolling_facial_coverings),
    rolling_facial_coverings >= mandate_limit
  ) %>%
  group_by(RegionName) %>%
  slice_head(n = 1) %>%
  ungroup() %>%
  arrange(Date) %>%
  mutate(
    region_label = paste0(RegionName, " (", RegionCode, ")")
  )


# =========================
# 5. Save output
# =========================

if (!dir.exists("data")) {
  dir.create("data")
}

write_csv(
  df_mandates,
  "data/00_mandate_start_dates.csv"
)

print(df_mandates)


# =========================
# 6. Visualisation: mandate start dates by state/territory
# =========================

if (!dir.exists("figures")) {
  dir.create("figures")
}

p1 <- ggplot(
  df_mandates,
  aes(
    x = Date,
    y = reorder(region_label, Date)
  )
) +
  geom_segment(
    aes(
      x = min(Date),
      xend = Date,
      y = reorder(region_label, Date),
      yend = reorder(region_label, Date)
    ),
    colour = "#D9D9D9",
    linewidth = 0.7
  ) +
  geom_point(
    size = 3.2,
    colour = "#4E79A7"
  ) +
  labs(
    title = "Estimated Start Dates of Consistent Face-Covering Mandates",
    subtitle = "First date when the 14-day rolling average of H6M Facial Coverings reached 3 or above",
    x = "Date",
    y = "State / territory"
  ) +
  theme_classic(base_size = 12) +
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

print(p1)

ggsave(
  "figures/00_mandate_start_dates_by_region.png",
  plot = p1,
  width = 8,
  height = 5,
  dpi = 300
)

# =========================
# 7. Visualisation: policy strength heatmap over time
# =========================

df_heatmap <- df_rolling %>%
  mutate(
    region_label = paste0(RegionName, " (", RegionCode, ")")
  )

p2 <- ggplot(
  df_heatmap,
  aes(
    x = Date,
    y = region_label,
    fill = `H6M_Facial Coverings`
  )
) +
  geom_tile(height = 0.9) +
  geom_point(
    data = df_mandates,
    aes(
      x = Date,
      y = region_label
    ),
    inherit.aes = FALSE,
    size = 2.4,
    shape = 21,
    fill = "#F28E2B",
    colour = "white",
    stroke = 0.5
  ) +
  scale_fill_gradient(
    low = "#F7FBFF",
    high = "#08306B",
    na.value = "#F0F0F0",
    name = "Policy\nstrength"
  ) +
  labs(
    title = "Face-Covering Policy Strength Over Time",
    subtitle = "Orange points indicate estimated first consistent mandate dates",
    x = "Date",
    y = "State / territory"
  ) +
  theme_classic(base_size = 12) +
  theme(
    plot.title = element_text(size = 14, face = "bold", colour = "#222222"),
    plot.subtitle = element_text(size = 11, colour = "#555555"),
    axis.title = element_text(size = 12, colour = "#222222"),
    axis.text = element_text(size = 10, colour = "#333333"),
    axis.line = element_blank(),
    axis.ticks = element_blank(),
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9),
    plot.margin = margin(12, 16, 12, 12)
  )

print(p2)

ggsave(
  "figures/00_face_covering_policy_heatmap.png",
  plot = p2,
  width = 9,
  height = 5.2,
  dpi = 300
)

print("Mandate start date analysis completed.")
print("Output saved to data/mandate_start_dates.csv")
print("Figure saved to figures/mandate_start_dates_by_region.png")