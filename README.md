# Gemma-2-9B QLoRA fine-tune on single RTX 3090

Fine-tune Google's Gemma-2-9B model using QLoRA on Alpaca dataset subset for efficient instruction tuning.

## Hardware & Runtime
- **GPU**: Single RTX 3090 (24GB VRAM)
- **Expected runtime**: ~3 hours
- **Dataset**: ~5k Alpaca examples

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Login to Hugging Face
huggingface-cli login

# Train the model
python train.py
```

## Target Performance
- **Final eval loss**: <1.6

## Inference After Training
Run the merge and inference script to merge LoRA weights and launch a Gradio chat demo:
```bash
python merge_and_inference.py
```