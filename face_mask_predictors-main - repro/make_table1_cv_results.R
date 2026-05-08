library(readr)
library(dplyr)
library(stringr)
library(gt)
library(webshot2)

# =========================================================
# 1. File paths
# =========================================================

file_before_mask   <- "results/03_model_1a_final_results.csv"
file_after_mask    <- "results/05_model_1b_final_results.csv"
file_before_prot   <- "results/04_model_2a_final_results.csv"
file_after_prot    <- "results/06_model_2b_final_results.csv"

# =========================================================
# 2. Helper functions
# =========================================================

format_mean_se <- function(mean_val, se_val) {
  sprintf("%.3f (%.3f)", mean_val, se_val)
}

clean_model_name <- function(x) {
  recode(
    x,
    "logistic_reg" = "logistic regression",
    "binary_tree"  = "classification tree",
    "xgboost"      = "XGBoost",
    "rf"           = "random forest",
    .default = x
  )
}

read_one_section <- function(file_path, section_label) {
  df <- read_csv(file_path, show_col_types = FALSE)
  
  df %>%
    mutate(
      model_label = clean_model_name(model_type),
      section = section_label,
      auc_disp = format_mean_se(roc_auc, roc_auc_std),
      precision_disp = format_mean_se(precision, precision_std),
      recall_disp = format_mean_se(recall, recall_std),
      accuracy_disp = format_mean_se(accuracy, accuracy_std),
      f1_disp = format_mean_se(f1, f1_std)
    ) %>%
    select(
      section, model_label,
      roc_auc,
      auc_disp, precision_disp, recall_disp, accuracy_disp, f1_disp
    )
}

make_section_block <- function(df_section) {
  section_label <- unique(df_section$section)
  
  section_row <- tibble(
    row_type = "section",
    section = section_label,
    model_label = section_label,
    roc_auc = NA_real_,
    auc_disp = "",
    precision_disp = "",
    recall_disp = "",
    accuracy_disp = "",
    f1_disp = ""
  )
  
  model_rows <- df_section %>%
    mutate(row_type = "model")
  
  bind_rows(section_row, model_rows)
}

# =========================================================
# 3. Read and combine the four sections
# =========================================================

before_mask <- read_one_section(
  file_before_mask,
  "before mandates—face mask wearing"
)

after_mask <- read_one_section(
  file_after_mask,
  "after mandates—face mask wearing"
)

before_prot <- read_one_section(
  file_before_prot,
  "before mandates—general protective behaviour"
)

after_prot <- read_one_section(
  file_after_prot,
  "after mandates—general protective behaviour"
)

table_df <- bind_rows(
  make_section_block(before_mask),
  make_section_block(after_mask),
  make_section_block(before_prot),
  make_section_block(after_prot)
) %>%
  mutate(row_id = row_number())

# =========================================================
# 4. Determine which AUC cells should be bold
# =========================================================

bold_rows <- bind_rows(
  before_mask %>% filter(roc_auc == max(roc_auc, na.rm = TRUE)),
  after_mask  %>% filter(roc_auc == max(roc_auc, na.rm = TRUE)),
  before_prot %>% filter(roc_auc == max(roc_auc, na.rm = TRUE)),
  after_prot  %>% filter(roc_auc == max(roc_auc, na.rm = TRUE))
) %>%
  select(section, model_label)

bold_row_ids <- table_df %>%
  inner_join(bold_rows, by = c("section", "model_label")) %>%
  pull(row_id)

# =========================================================
# 5. Build gt table
# =========================================================

gt_tbl <- table_df %>%
  gt(rowname_col = "model_label") %>%
  cols_hide(columns = c(section, roc_auc, row_type, row_id)) %>%
  cols_label(
    auc_disp = "AUC",
    precision_disp = "precision",
    recall_disp = "recall",
    accuracy_disp = "accuracy",
    f1_disp = "F1"
  ) %>%
  tab_stubhead(label = "") %>%
  tab_header(
    title = md("**Table 1.** Fivefold cross-validation results comparing four predictive models for predicting face mask wearing and general protective health behaviours before and after face mask mandates are enacted. Values are given as mean (standard error) and can range from 0 (low) to 1 (high). **Bold values show the optimal models based on largest AUC.**")
  ) %>%
  tab_options(
    table.width = pct(100),
    heading.align = "left",
    table.font.names = c("Arial", "Helvetica", "sans-serif"),
    table.font.size = px(16),
    data_row.padding = px(9),
    heading.padding = px(6),
    column_labels.padding = px(10),
    row_group.padding = px(4),
    table.border.top.width = px(0),
    table.border.bottom.width = px(0),
    column_labels.border.top.width = px(0),
    column_labels.border.bottom.width = px(0)
  ) %>%
  tab_style(
    style = list(
      cell_fill(color = "black"),
      cell_text(color = "white", weight = "bold", align = "center", size = px(17))
    ),
    locations = cells_column_labels(columns = everything())
  ) %>%
  tab_style(
    style = list(
      cell_text(weight = "normal", size = px(17), align = "left")
    ),
    locations = cells_stub(rows = row_type == "section")
  ) %>%
  tab_style(
    style = list(
      cell_text(weight = "normal", size = px(17), align = "left")
    ),
    locations = cells_stub(rows = row_type == "model")
  ) %>%
  tab_style(
    style = list(
      cell_text(weight = "bold")
    ),
    locations = cells_body(
      columns = auc_disp,
      rows = row_id %in% bold_row_ids
    )
  ) %>%
  tab_style(
    style = cell_borders(
      sides = "bottom",
      color = "#444444",
      style = "dotted",
      weight = px(1.2)
    ),
    locations = cells_body(
      columns = everything(),
      rows = row_type == "model"
    )
  ) %>%
  tab_style(
    style = cell_borders(
      sides = "bottom",
      color = "#444444",
      style = "dotted",
      weight = px(1.2)
    ),
    locations = cells_stub(
      rows = row_type == "model"
    )
  ) %>%
  tab_style(
    style = cell_borders(
      sides = "bottom",
      color = "#444444",
      style = "dotted",
      weight = px(1.2)
    ),
    locations = cells_stub(
      rows = row_type == "section"
    )
  ) %>%
  tab_style(
    style = cell_borders(
      sides = "bottom",
      color = "#444444",
      style = "dotted",
      weight = px(1.2)
    ),
    locations = cells_body(
      columns = everything(),
      rows = row_type == "section"
    )
  ) %>%
  cols_align(
    align = "center",
    columns = c(auc_disp, precision_disp, recall_disp, accuracy_disp, f1_disp)
  )

# =========================================================
# 6. Save outputs
# =========================================================

gtsave(gt_tbl, "results/table1_cv_results.html")
gtsave(gt_tbl, "results/table1_cv_results.png", zoom = 2)



gt_tbl