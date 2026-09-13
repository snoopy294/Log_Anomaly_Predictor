#!/usr/bin/env Rscript

# -----------------------------------------------------------------------------
# ENTITY-KEY DIVERGENCE WARNING (as of the feat/resume-detection-overhaul branch)
#
# This script keys `entity_id` on Source IP (see the `entity_id = ...` line
# below). `cicids_to_clean.py`, the Python equivalent of this converter, was
# changed on this branch to key `entity_id` on Destination IP instead, for
# CICIDS-specific detection-quality reasons: source IPs for attacks like
# DoS/DDoS/PortScan have almost no genuine benign history, which makes
# per-entity anomaly baselines built on Source IP meaningless for those
# attack types.
#
# The two converters now disagree on what an "entity" is, and they were NOT
# reconciled: backend.py's web-upload path for CICIDS files calls THIS R
# script, not cicids_to_clean.py.
#
# Consequence: a model trained on Python-converted (destination-centric)
# CICIDS data must NOT be used to score data produced by this R script (or by
# backend.py's upload path) without first reconciling the entity-key
# semantics. Doing so will silently produce meaningless scores -- there is no
# version/compatibility check anywhere in the pipeline that catches an
# entity-key mismatch between a model and its scoring input (only vocab_size
# is checked). If/when a destination-centric model is wired into
# backend.py's live scoring path, this divergence must be resolved first.
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(optparse)
  library(readr)
  library(dplyr)
  library(stringr)
  library(lubridate)
})

event_type_from_proto_dport <- function(proto, dport) {
  proto_s <- toupper(str_trim(as.character(proto)))
  
  base <- if (proto_s %in% c("6", "TCP")) {
    "TCP"
  } else if (proto_s %in% c("17", "UDP")) {
    "UDP"
  } else {
    return("P0_WELL_KNOWN")  # force exactly your 7 types
  }
  
  p <- suppressWarnings(as.integer(dport))
  if (is.na(p)) return(paste0(base, "_REGISTERED"))
  
  if (p >= 0 && p <= 1023) {
    paste0(base, "_WELL_KNOWN")
  } else if (p <= 49151) {
    paste0(base, "_REGISTERED")
  } else {
    paste0(base, "_EPHEMERAL")
  }
}


parse_timestamp_utc <- function(x) {
  # CICIDS often looks like "4/7/2017 8:54" or "4/7/2017 8:54:00"
  # Try common orders robustly, assume UTC if no tz.
  t <- suppressWarnings(parse_date_time(
    x,
    orders = c("mdy HM", "mdy HMS", "dmy HM", "dmy HMS", "Ymd HMS", "Ymd HM"),
    tz = "UTC"
  ))
  # Ensure POSIXct in UTC
  force_tz(as.POSIXct(t), tzone = "UTC")
}

option_list <- list(
  make_option(c("-i", "--in_csv"), type = "character", help = "Input CICIDS/ISCX CSV (e.g., Tuesday-WorkingHours...)"),
  make_option(c("-o", "--out_csv"), type = "character", help = "Output CSV in model schema"),
  make_option(c("--verbose"), action = "store_true", default = FALSE, help = "Print some diagnostics")
)

opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$in_csv) || is.null(opt$out_csv)) {
  cat("ERROR: You must pass --in_csv and --out_csv\n", file = stderr())
  quit(status = 2)
}

# Read CSV (CICIDS files can be big; readr is faster)
df <- read_csv(opt$in_csv, show_col_types = FALSE, progress = FALSE)

# Trim column names (your file has leading spaces like ' Source IP')
names(df) <- str_trim(names(df))

required <- c("Timestamp", "Source IP", "Destination IP", "Destination Port", "Protocol")
missing <- setdiff(required, names(df))
if (length(missing) > 0) {
  cat("ERROR: Missing required columns after trimming names:\n", paste(missing, collapse = ", "), "\n", file = stderr())
  quit(status = 2)
}

# Trim key string cols
df <- df %>%
  mutate(
    `Source IP` = str_trim(as.character(`Source IP`)),
    `Destination IP` = str_trim(as.character(`Destination IP`)),
    Label = if ("Label" %in% names(df)) str_trim(as.character(Label)) else ""
  )

# Compute bytes using best available approximation
bytes <- NULL
if (all(c("Total Length of Fwd Packets", "Total Length of Bwd Packets") %in% names(df))) {
  bytes <- suppressWarnings(as.numeric(df$`Total Length of Fwd Packets`)) +
    suppressWarnings(as.numeric(df$`Total Length of Bwd Packets`))
} else if (all(c("Subflow Fwd Bytes", "Subflow Bwd Bytes") %in% names(df))) {
  bytes <- suppressWarnings(as.numeric(df$`Subflow Fwd Bytes`)) +
    suppressWarnings(as.numeric(df$`Subflow Bwd Bytes`))
} else {
  bytes <- rep(0, nrow(df))
}
bytes[is.na(bytes)] <- 0

out <- tibble(
  timestamp = parse_timestamp_utc(df$Timestamp),
  # entity_id keyed on Source IP -- diverges from cicids_to_clean.py, which
  # keys on Destination IP as of this branch. See warning block at top of file.
  entity_id = as.character(df$`Source IP`),
  event_type = mapply(event_type_from_proto_dport, df$Protocol, df$`Destination Port`),
  dst_id = as.character(df$`Destination IP`),
  bytes = as.numeric(bytes),
  Label = if ("Label" %in% names(df)) as.character(df$Label) else ""
) %>%
  filter(!is.na(timestamp), !is.na(entity_id), !is.na(event_type))

# Write as ISO-8601 UTC for your Python parser
out <- out %>% mutate(timestamp = format(timestamp, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"))

write_csv(out, opt$out_csv)

if (opt$verbose) {
  cat("Wrote:", opt$out_csv, "\n")
  cat("Rows:", nrow(out), "Cols:", ncol(out), "\n")
  cat("Unique entities:", n_distinct(out$entity_id), "\n")
  cat("Event types preview:\n")
  print(head(sort(unique(out$event_type)), 10))
  cat("Label counts (top):\n")
  print(head(sort(table(out$Label), decreasing = TRUE), 10))
}
