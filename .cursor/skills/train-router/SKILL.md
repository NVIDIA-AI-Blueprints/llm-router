# Train a Custom Router

Trigger: "train router", "customize router", "collect training data", "model-router train"

## Data Collection
1. Prepare questions.txt (one question per line, 500+ recommended)
2. Run: model-router collect --config configs/your-config.yaml --questions questions.txt --output data/train.csv --judge vote
3. This calls every model in the pool for each question and judges correctness via majority vote

## Training
1. Run: model-router train --config configs/your-config.yaml --data data/train.csv --output-dir checkpoints/custom/
2. For kmeans: embeds questions, fits KMeans clusters, Platt calibration -> saves .pkl
3. For prefill: extracts hidden states, sweeps layers, trains MLP -> saves .pt

## Evaluation
1. Split your data: 80% train, 20% test
2. Run: model-router evaluate --config configs/your-config.yaml --checkpoint checkpoints/custom/router.pkl --data data/test.csv
3. Look for: per-model AUC > 0.75, cost savings > 40%

## Deploy
1. Update routing.checkpoint in your config to point to the new checkpoint
2. Restart the server
