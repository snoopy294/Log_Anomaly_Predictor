.PHONY: train serve demo

train:
	python new.py --train_csv data/train_data.csv --plot

serve:
	python backend.py

demo: serve
