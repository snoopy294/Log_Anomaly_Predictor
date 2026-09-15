.PHONY: train train-cicids results test serve demo

# Quick run on the tiny synthetic sample (no attack labels, so no detection metrics)
train:
	python new.py --train_csv data/train_data.csv --plot

# Reference CICIDS2017 run behind the README results (GPU recommended)
train-cicids:
	python new.py --train_csv data/cicids_clean.csv --train_format clean \
	  --seq_len 16 --step 1 --min_events_per_entity 10 \
	  --split_mode cicids_days \
	  --dst_top_n 200 --bytes_num_buckets 8 --epochs 20 \
	  --train_only_label BENIGN --alert_score combo_score --target_fpr 0.01 --plot

# Regenerate README results tables from the consolidated benchmark summary
results:
	python scripts/render_results.py --summary outputs/benchmark_summary.json

test:
	python -m pytest tests -q

serve:
	python backend.py

demo: serve
