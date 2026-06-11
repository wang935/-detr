library(ggplot2)
library(patchwork)
library(jsonlite)
library(dplyr)
library(tidyr)
library(svglite)
library(ragg)

# ---- source and output ----
json_path <- normalizePath("D:/detr_Q3/formal_results/stage5_pv_v2/dfire_rtdetr_seed11/gonogo.json", mustWork = TRUE)
out_dir <- "D:/detr_Q3/figures"
out_prefix <- file.path(out_dir, "detr_seed11_fpr_fppi_fixed_recall")

# ---- missing-package guard (explicit error if unavailable) ----
required_pkgs <- c("ggplot2", "patchwork", "jsonlite", "dplyr", "tidyr", "svglite", "ragg", "purrr")
missing <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  stop(sprintf("Missing R packages: %s", paste(missing, collapse = ", ")))
}

# ---- load data ----
j <- fromJSON(json_path)

df <- purrr::imap_dfr(j, function(arm_block, arm_name) {
  rc <- arm_block$at_fixed_recall
  tibble::tibble(
    arm = arm_name,
    recall = as.numeric(names(rc)),
    FPR = vapply(rc, function(x) as.numeric(x$FPR), numeric(1)),
    FPPI = vapply(rc, function(x) as.numeric(x$FPPI), numeric(1))
  )
})

# arm name mapping for readability
arm_map <- c(
  model_A = "baseline",
  model_B = "hardneg",
  model_C = "baseline_eqstep",
  model_D = "hardneg_sched"
)
df <- df %>%
  mutate(
    arm = factor(recode(arm, !!!arm_map), levels = c("baseline", "baseline_eqstep", "hardneg", "hardneg_sched"), ordered = TRUE),
    recall = round(recall, 2)
  )

# ---- output source data for reproducibility ----
write.csv(df, file = file.path(out_dir, "detr_seed11_fixed_recall_metrics.csv"), row.names = FALSE)

# ---- plotting theme ----
theme_set(
  theme_classic(base_size = 6.5, base_family = "Arial") +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks = element_line(linewidth = 0.35, colour = "black"),
      legend.title = element_text(size = 6.2),
      legend.text = element_text(size = 5.8),
      strip.text = element_text(size = 6.2, face = "bold"),
      plot.title = element_text(size = 7, face = "bold"),
      panel.grid = element_blank(),
      plot.margin = margin(5, 5, 5, 5)
    )
)

pal <- c(
  "baseline" = "#5B7C99",
  "baseline_eqstep" = "#D39C45",
  "hardneg" = "#6F8F6A",
  "hardneg_sched" = "#9F6A83"
)

p_fpr <- ggplot(df, aes(x = recall, y = FPR, color = arm, shape = arm)) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 1.8) +
  scale_color_manual(values = pal, name = "Training arm") +
  scale_shape_manual(values = c(16, 17, 15, 18), name = "Training arm") +
  scale_x_continuous(breaks = c(0.80, 0.85, 0.90, 0.95), labels = c("0.80", "0.85", "0.90", "0.95")) +
  coord_cartesian(ylim = c(0, 0.35)) +
  labs(
    x = "Recall target",
    y = "False positive rate (FPR)",
    title = "DETR seed11: fixed-recall evaluation"
  ) +
  guides(color = guide_legend(ncol = 2), shape = guide_legend(ncol = 2))

p_fppi <- ggplot(df, aes(x = recall, y = FPPI, color = arm, shape = arm)) +
  geom_line(linewidth = 0.7, linetype = "solid") +
  geom_point(size = 1.8) +
  scale_color_manual(values = pal, name = "Training arm") +
  scale_shape_manual(values = c(16, 17, 15, 18), name = "Training arm") +
  scale_x_continuous(breaks = c(0.80, 0.85, 0.90, 0.95), labels = c("0.80", "0.85", "0.90", "0.95")) +
  coord_cartesian(ylim = c(0, 0.40)) +
  labs(
    x = "Recall target",
    y = "False positives per image (FPPI)",
    title = "DETR seed11: fixed-recall evaluation"
  )

plot_out <- p_fpr / p_fppi +
  plot_layout(guides = "collect") +
  plot_annotation(tag_levels = "A")

# ---------- save in publication sizes ----------
save_pub_r <- function(plot, filename, width_mm = 183, height_mm = 120, dpi = 600) {
  w <- width_mm / 25.4
  h <- height_mm / 25.4
  svglite::svglite(paste0(filename, ".svg"), width = w, height = h)
  print(plot)
  dev.off()

  grDevices::cairo_pdf(paste0(filename, ".pdf"), width = w, height = h, family = "Arial")
  print(plot)
  dev.off()

  ragg::agg_tiff(paste0(filename, ".tiff"), width = w, height = h, units = "in", res = dpi)
  print(plot)
  dev.off()
}

save_pub_r(plot_out, out_prefix, width_mm = 183, height_mm = 130, dpi = 600)

cat("saved_prefix=", out_prefix, "\n")
cat("source_data=", file.path(out_dir, "detr_seed11_fixed_recall_metrics.csv"), "\n")
