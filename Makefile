# Makefile for got-death-prediction
# Run the full pipeline with: make all

DATA_DIR  = data
OUT_DIR   = reports
MODEL_DIR = models
FIG_DIR   = reports/figures

.PHONY: all data models visualize clean

# Run the full pipeline
all: data models visualize

# Build features from raw JSON
data:
	python3 -m src.data --data_dir $(DATA_DIR) --out_dir $(OUT_DIR)

# Train and evaluate all models
models:
	python3 -m src.models --data_dir $(OUT_DIR) --out_dir $(OUT_DIR) --model_dir $(MODEL_DIR)

# Generate all visualizations
visualize:
	python3 -m src.visualize --data_dir $(OUT_DIR) --model_dir $(MODEL_DIR) --out_dir $(FIG_DIR)

# Remove all generated artifacts
clean:
	rm -f $(OUT_DIR)/*.parquet
	rm -f $(OUT_DIR)/metrics.json
	rm -f $(MODEL_DIR)/*.joblib
	rm -f $(FIG_DIR)/*.png
