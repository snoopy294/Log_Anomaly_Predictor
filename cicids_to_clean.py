#!/usr/bin/env python3
"""
cicids_to_clean.py
-------------------
Pure-Python fallback for `cicids_into_clean.R` (Rscript is not on PATH on this
machine). Converts a CICIDS/ISCX flow dataset (CSV or Parquet) into the clean
schema new.py expects:

    timestamp, entity_id, event_type, dst_id, bytes, Label

Field semantics are translated line-for-line from cicids_into_clean.R:

- event_type_from_proto_dport (R lines 11-32): protocol -> {TCP,UDP,P0_WELL_KNOWN}
  base, then a destination-port bucket suffix (_WELL_KNOWN/_REGISTERED/_EPHEMERAL).
  NaN port -> "_REGISTERED" (R line 23: `if (is.na(p)) return(paste0(base,
  "_REGISTERED"))`).
- bytes (R lines 82-92): "Total Length of Fwd Packets" + "Total Length of Bwd
  Packets" if both present, else "Subflow Fwd Bytes" + "Subflow Bwd Bytes",
  else 0. NaNs coerced to 0 (R line 92).
- entity_id = Destination IP, dst_id = Source IP, Label = Label column
  (empty string if missing). Destination is the entity (not R lines
  94-101's original Source-IP mapping) so that per-entity anomaly
  baselines are built from a host's own traffic history, which has a
  genuine benign period, rather than from a single attacker machine's
  history, which does not. See
  docs/superpowers/specs/2026-09-10-cicids-destination-centric-entity-design.md.
- timestamp parsed to UTC ISO-8601 "%Y-%m-%dT%H:%M:%SZ" (R lines 35-45, 105).
- Column names are whitespace-trimmed first (R line 64: raw CICIDS CSVs have
  leading spaces like " Source IP").
- Rows with unparseable timestamp/entity_id/event_type are dropped (R line 102).

Usage:
    python cicids_to_clean.py --in_path data/raw_cicids/CICIDS_Flow.parquet \
        --out_csv data/cicids_clean.csv
"""
import argparse
import sys
import numpy as np
import pandas as pd

# Column-name aliases: this specific CICIDS_Flow.parquet ships snake_case /
# lower-case names (already whitespace-clean, since it's Parquet not raw CSV
# export) instead of the " Source IP" / "Destination Port" style R expects.
# We look up whichever of these exists after trimming.
COLUMN_ALIASES = {
    "Timestamp": ["Timestamp", "timestamp"],
    "Source IP": ["Source IP", "source_ip", "Src IP"],
    "Destination IP": ["Destination IP", "destination_ip", "Dst IP"],
    "Destination Port": ["Destination Port", "destination_port", "Dst Port"],
    "Protocol": ["Protocol", "protocol"],
    "Label": ["Label", "attack_label", "label"],
    "Total Length of Fwd Packets": ["Total Length of Fwd Packets", "total_length_of_fwd_packets"],
    "Total Length of Bwd Packets": ["Total Length of Bwd Packets", "total_length_of_bwd_packets"],
    "Subflow Fwd Bytes": ["Subflow Fwd Bytes", "subflow_fwd_bytes"],
    "Subflow Bwd Bytes": ["Subflow Bwd Bytes", "subflow_bwd_bytes"],
}


def resolve_column(df: pd.DataFrame, canonical: str):
    for cand in COLUMN_ALIASES.get(canonical, [canonical]):
        if cand in df.columns:
            return cand
    return None


def event_type_from_proto_dport(proto: pd.Series, dport: pd.Series) -> pd.Series:
    """Vectorized translation of R's event_type_from_proto_dport (lines 11-32).

    base: TCP if proto in {6,"TCP"}; UDP if proto in {17,"UDP"}; else the whole
    function returns "P0_WELL_KNOWN" immediately (R line 19), i.e. non-TCP/UDP
    protocols never get a port suffix.
    Suffix (only for TCP/UDP): NaN port -> "_REGISTERED" (R line 23);
    0<=port<=1023 -> "_WELL_KNOWN"; port<=49151 -> "_REGISTERED"; else "_EPHEMERAL".
    """
    proto_s = proto.astype(str).str.strip().str.upper()
    is_tcp = proto_s.isin(["6", "TCP"])
    is_udp = proto_s.isin(["17", "UDP"])

    p = pd.to_numeric(dport, errors="coerce")
    suffix = pd.Series(
        np.select(
            [p.isna(), (p >= 0) & (p <= 1023), (p <= 49151)],
            ["_REGISTERED", "_WELL_KNOWN", "_REGISTERED"],
            default="_EPHEMERAL",
        ),
        index=proto.index,
    )

    result = pd.Series("P0_WELL_KNOWN", index=proto.index)
    result.loc[is_tcp] = "TCP" + suffix.loc[is_tcp]
    result.loc[is_udp] = "UDP" + suffix.loc[is_udp]
    return result


def parse_timestamp_utc(s: pd.Series) -> pd.Series:
    """Translation of R's parse_timestamp_utc (lines 35-45).

    R tries several orders (mdy/dmy, HM/HMS) via lubridate::parse_date_time.
    This CICIDS_Flow.parquet's Timestamp column is "DD/MM/YYYY HH:MM:SS"
    (confirmed: values like "03/07/2017 08:55:58" correspond to Monday
    3 July 2017, the first day of the CICIDS2017 capture window) so we parse
    with dayfirst=True, matching R's "dmy HMS"/"dmy HM" branches.
    """
    s = s.astype(str).str.strip()
    # format="mixed": this file mixes "DD/MM/YYYY HH:MM:SS" and unpadded
    # "D/M/YYYY H:MM" (no seconds) rows; a single inferred format silently
    # NaT's out whichever variant doesn't match, so let pandas parse row-wise.
    return pd.to_datetime(s, dayfirst=True, errors="coerce", utc=True, format="mixed")


def compute_bytes(df: pd.DataFrame) -> pd.Series:
    """Translation of R lines 82-92."""
    fwd_col = resolve_column(df, "Total Length of Fwd Packets")
    bwd_col = resolve_column(df, "Total Length of Bwd Packets")
    sub_fwd_col = resolve_column(df, "Subflow Fwd Bytes")
    sub_bwd_col = resolve_column(df, "Subflow Bwd Bytes")

    if fwd_col and bwd_col:
        b = pd.to_numeric(df[fwd_col], errors="coerce") + pd.to_numeric(df[bwd_col], errors="coerce")
    elif sub_fwd_col and sub_bwd_col:
        b = pd.to_numeric(df[sub_fwd_col], errors="coerce") + pd.to_numeric(df[sub_bwd_col], errors="coerce")
    else:
        b = pd.Series(np.zeros(len(df)), index=df.index)

    return b.fillna(0.0).astype(float)


def convert(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    ts_col = resolve_column(df, "Timestamp")
    src_col = resolve_column(df, "Source IP")
    dst_col = resolve_column(df, "Destination IP")
    dport_col = resolve_column(df, "Destination Port")
    proto_col = resolve_column(df, "Protocol")
    label_col = resolve_column(df, "Label")

    required = {"Timestamp": ts_col, "Source IP": src_col, "Destination IP": dst_col,
                "Destination Port": dport_col, "Protocol": proto_col}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"Missing required columns (after alias resolution): {missing}. "
                          f"Available columns: {df.columns.tolist()}")

    out = pd.DataFrame({
        "timestamp": parse_timestamp_utc(df[ts_col]),
        "entity_id": df[dst_col].astype(str).str.strip(),
        "event_type": event_type_from_proto_dport(df[proto_col], df[dport_col]),
        "dst_id": df[src_col].astype(str).str.strip(),
        "bytes": compute_bytes(df),
        "Label": df[label_col].astype(str).str.strip() if label_col else "",
    })

    before = len(out)
    out = out.dropna(subset=["timestamp", "entity_id", "event_type"]).copy()
    dropped = before - len(out)

    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    if verbose:
        print(f"Rows in: {before}, rows out: {len(out)}, dropped: {dropped}", file=sys.stderr)
        print("Unique entities:", out["entity_id"].nunique(), file=sys.stderr)
        print("Event types:", sorted(out["event_type"].unique())[:10], file=sys.stderr)
        print("Label counts (top 15):", file=sys.stderr)
        print(out["Label"].value_counts().head(15), file=sys.stderr)

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_path", required=True, help="Input CICIDS/ISCX file (.csv or .parquet)")
    ap.add_argument("--out_csv", required=True, help="Output CSV in clean model schema")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.in_path.lower().endswith(".parquet"):
        df = pd.read_parquet(args.in_path)
    else:
        df = pd.read_csv(args.in_path)

    out = convert(df, verbose=args.verbose)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}: {len(out)} rows, {out['entity_id'].nunique()} unique entities")


if __name__ == "__main__":
    main()
