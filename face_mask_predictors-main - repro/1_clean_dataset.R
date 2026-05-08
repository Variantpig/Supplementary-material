library(readr)
library(dplyr)

# =========================
# 0. Helper functions for recording cleaning process
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

stage_dimensions <- tibble(
  stage = character(),
  rows = integer(),
  columns = integer()
)

missing_summary_records <- tibble(
  variable_name = character(),
  missing_count = integer(),
  missing_percentage = numeric(),
  stage = character()
)


# =========================
# 1. Helper functions
# =========================

convert_datetime <- function(dt) {
  date_part <- strsplit(as.character(dt), " ")[[1]][1]
  as.Date(date_part, format = "%d/%m/%Y")
}

household_convert <- function(size_str) {
  size_str <- as.character(size_str)
  
  if (size_str %in% as.character(1:7)) {
    return(as.integer(size_str))
  } else if (size_str == "8 or more") {
    return(8)
  } else if (size_str == "Prefer not to say" || size_str == "Don't know") {
    return(NA)
  } else {
    return(NA)
  }
}


# =========================
# 2. Read raw data
# =========================

df <- read_csv(
  "raw_data/australia.csv",
  na = c("", " ", "__NA__", "NA", "N/A", "NULL", "NaN"),
  show_col_types = FALSE
)

stage_dimensions <- bind_rows(
  stage_dimensions,
  record_stage(df, "Raw data")
)

missing_summary_records <- bind_rows(
  missing_summary_records,
  make_missing_summary(df, "Raw data")
)

df$endtime <- as.Date(
  sapply(df$endtime, convert_datetime),
  origin = "1970-01-01"
)


# =========================
# 3. Drop variables with too many missing values
# =========================

thresh_value <- 10781

missing_value_df <- read_csv(
  "data/0_missing_value_counts.csv",
  show_col_types = FALSE
)

columns_to_drop <- missing_value_df %>%
  filter(`Missing Value Count` > thresh_value) %>%
  pull(`Variable Name`)

df <- df %>%
  select(-any_of(columns_to_drop))

stage_dimensions <- bind_rows(
  stage_dimensions,
  record_stage(df, "After removing high-missing variables")
)

missing_summary_records <- bind_rows(
  missing_summary_records,
  make_missing_summary(df, "After removing high-missing variables")
)


# =========================
# 4. Fill medical-question missing values with "N/A"
# =========================

sdate <- as.Date("2021-02-10")
edate <- as.Date("2021-10-18")

mask <- df$endtime <= edate & df$endtime >= sdate

# PHQ4_1 to PHQ4_4
for (i in 1:4) {
  col_name <- paste0("PHQ4_", i)
  
  if (col_name %in% names(df)) {
    df[[col_name]][mask & is.na(df[[col_name]])] <- "N/A"
  }
}

# d1_health_1 to d1_health_13
for (i in 1:13) {
  col_name <- paste0("d1_health_", i)
  
  if (col_name %in% names(df)) {
    df[[col_name]][mask & is.na(df[[col_name]])] <- "N/A"
  }
}

# d1_health_98 and d1_health_99
for (i in 98:99) {
  col_name <- paste0("d1_health_", i)
  
  if (col_name %in% names(df)) {
    df[[col_name]][mask & is.na(df[[col_name]])] <- "N/A"
  }
}

stage_dimensions <- bind_rows(
  stage_dimensions,
  record_stage(df, "After filling medical missing values")
)

missing_summary_records <- bind_rows(
  missing_summary_records,
  make_missing_summary(df, "After filling medical missing values")
)


# =========================
# 5. Remove remaining missing values
# =========================

df <- df[complete.cases(df), ]

stage_dimensions <- bind_rows(
  stage_dimensions,
  record_stage(df, "After complete-case removal")
)

missing_summary_records <- bind_rows(
  missing_summary_records,
  make_missing_summary(df, "After complete-case removal")
)


# =========================
# 6. Convert r1_1 and r1_2 agreement scale to numeric
# =========================

agreement_map <- c(
  "7 - Agree" = 7,
  "6" = 6,
  "5" = 5,
  "4" = 4,
  "3" = 3,
  "2" = 2,
  "1 – Disagree" = 1,
  "1 - Disagree" = 1
)

for (i in 1:2) {
  col_name <- paste0("r1_", i)
  
  if (col_name %in% names(df)) {
    df[[col_name]] <- agreement_map[as.character(df[[col_name]])]
    df[[col_name]] <- as.numeric(df[[col_name]])
  }
}


# =========================
# 7. Convert i12_health_* frequency variables to numeric
# =========================

frequency_dict <- c(
  "Always" = 5,
  "Frequently" = 4,
  "Sometimes" = 3,
  "Rarely" = 2,
  "Not at all" = 1
)

i12_health_cols <- names(df)[startsWith(names(df), "i12_health_")]

for (col_name in i12_health_cols) {
  df[[col_name]] <- frequency_dict[as.character(df[[col_name]])]
  df[[col_name]] <- as.numeric(df[[col_name]])
}


# =========================
# 8. Create face mask behaviour variables
# =========================

face_mask_cols <- c(
  "i12_health_1",
  "i12_health_22",
  "i12_health_23",
  "i12_health_25"
)

df$face_mask_behaviour_scale <- apply(
  df[, face_mask_cols],
  1,
  median
)

df$face_mask_behaviour_binary <- ifelse(
  df$face_mask_behaviour_scale >= 4,
  "Yes",
  "No"
)


# =========================
# 9. Create protective behaviour variables
# =========================

protective_behaviour_cols <- names(df)[startsWith(names(df), "i12_")]

df$protective_behaviour_scale <- apply(
  df[, protective_behaviour_cols],
  1,
  median
)

df$protective_behaviour_binary <- ifelse(
  df$protective_behaviour_scale >= 4,
  "Yes",
  "No"
)

protective_behaviour_nomask_cols <- setdiff(
  protective_behaviour_cols,
  face_mask_cols
)

df$protective_behaviour_nomask_scale <- apply(
  df[, protective_behaviour_nomask_cols],
  1,
  median
)

stage_dimensions <- bind_rows(
  stage_dimensions,
  record_stage(df, "After creating behaviour outcomes")
)

missing_summary_records <- bind_rows(
  missing_summary_records,
  make_missing_summary(df, "After creating behaviour outcomes")
)


# =========================
# 10. Combine comorbidities
# =========================

d1_cols <- names(df)[startsWith(names(df), "d1_")]

df$d1_comorbidities <- "Yes"

if ("d1_health_99" %in% names(df)) {
  df$d1_comorbidities[df$d1_health_99 == "Yes"] <- "No"
  df$d1_comorbidities[df$d1_health_99 == "N/A"] <- "NA"
}

if ("d1_health_98" %in% names(df)) {
  df$d1_comorbidities[df$d1_health_98 == "Yes"] <- "Prefer_not_to_say"
}

df <- df %>%
  select(-any_of(d1_cols))

stage_dimensions <- bind_rows(
  stage_dimensions,
  record_stage(df, "After combining comorbidities")
)

missing_summary_records <- bind_rows(
  missing_summary_records,
  make_missing_summary(df, "After combining comorbidities")
)


# =========================
# 11. Create week_number variable
# =========================

start_date <- min(df$endtime)
end_date <- max(df$endtime)

df$week_number <- floor(as.numeric(df$endtime - start_date) / 14) + 1

stage_dimensions <- bind_rows(
  stage_dimensions,
  record_stage(df, "After creating week number")
)

missing_summary_records <- bind_rows(
  missing_summary_records,
  make_missing_summary(df, "After creating week number")
)


# =========================
# 12. Convert household_size to numeric
# =========================

df$household_size <- sapply(df$household_size, household_convert)

stage_dimensions <- bind_rows(
  stage_dimensions,
  record_stage(df, "After household conversion")
)

missing_summary_records <- bind_rows(
  missing_summary_records,
  make_missing_summary(df, "After household conversion")
)

# Remove induced missing values
df <- df[complete.cases(df), ]

stage_dimensions <- bind_rows(
  stage_dimensions,
  record_stage(df, "After removing induced missing rows")
)

missing_summary_records <- bind_rows(
  missing_summary_records,
  make_missing_summary(df, "After removing induced missing rows")
)


# =========================
# 13. Drop qweek, weight, and original i12 behaviour columns
# =========================

df <- df %>%
  select(
    -any_of(c("qweek", "weight", protective_behaviour_cols))
  )

stage_dimensions <- bind_rows(
  stage_dimensions,
  record_stage(df, "Final cleaned data")
)

missing_summary_records <- bind_rows(
  missing_summary_records,
  make_missing_summary(df, "Final cleaned data")
)


# =========================
# 14. Save cleaned data and cleaning records
# =========================

dir.create("data", recursive = TRUE, showWarnings = FALSE)
dir.create("figures/1_cleaning_comparison", recursive = TRUE, showWarnings = FALSE)

write_csv(
  df,
  "data/1_cleaned_data.csv"
)

write_csv(
  stage_dimensions,
  "figures/1_cleaning_comparison/cleaning_stage_dimensions.csv"
)

write_csv(
  missing_summary_records,
  "figures/1_cleaning_comparison/cleaning_missing_summary_records.csv"
)

print("Cleaned data saved to data/1_cleaned_data.csv")
print("Cleaning records saved to figures/1_cleaning_comparison/")
print(dim(df))
print(stage_dimensions)