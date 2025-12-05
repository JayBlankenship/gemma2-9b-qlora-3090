import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
import mlflow
import mlflow.pytorch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set environment variables for reproducibility
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Model and dataset configuration
MODEL_NAME = "google/gemma-2-9b"
DATASET_NAME = "tatsu-lab/alpaca"
MAX_SEQ_LENGTH = 512
NUM_TRAIN_EXAMPLES = 5000
NUM_EVAL_EXAMPLES = 500

# QLoRA configuration
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training configuration
OUTPUT_DIR = "./gemma2-9b-qlora"
BATCH_SIZE = 1  # Per device, with gradient accumulation
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
WARMUP_RATIO = 0.1
SAVE_STEPS = 500
EVAL_STEPS = 500
LOGGING_STEPS = 10

class VRAMCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            vram_used = torch.cuda.memory_allocated() / 1024**3
            print(f"Step {state.global_step}: VRAM used: {vram_used:.2f} GB")

def load_model_and_tokenizer():
    # 4-bit quantization with NF4 and double quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    return model, tokenizer

def setup_lora_config():
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )

def load_and_prepare_dataset(tokenizer):
    dataset = load_dataset(DATASET_NAME, split="train")

    # Subset to first 5000 examples
    dataset = dataset.select(range(min(NUM_TRAIN_EXAMPLES + NUM_EVAL_EXAMPLES, len(dataset))))

    # Split into train and eval
    train_dataset = dataset.select(range(NUM_TRAIN_EXAMPLES))
    eval_dataset = dataset.select(range(NUM_TRAIN_EXAMPLES, len(dataset)))

    def tokenize_function(examples):
        # Format as instruction-response
        texts = []
        for instruction, input_text, output in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            if input_text:
                text = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output}"
            else:
                text = f"Instruction: {instruction}\nResponse: {output}"
            texts.append(text)

        return tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_SEQ_LENGTH)

    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    return train_dataset, eval_dataset

def main():
    # Set Hugging Face token from .env
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    # Start MLflow run
    mlflow.start_run(run_name="gemma2-9b-qlora-finetune")

    model, tokenizer = load_model_and_tokenizer()
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)

    train_dataset, eval_dataset = load_and_prepare_dataset(tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        logging_steps=LOGGING_STEPS,
        save_total_limit=2,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        tf32=True,
        dataloader_pin_memory=False,
        report_to=["mlflow"],
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # VRAM callback
    vram_callback = VRAMCallback()

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        peft_config=lora_config,
        callbacks=[vram_callback],
    )

    # Compile model for faster training
    trainer.model = torch.compile(trainer.model, mode="reduce-overhead")

    # Train
    trainer.train()

    # Save the adapter (LoRA weights only)
    trainer.save_model()

    # Log final metrics
    final_metrics = trainer.evaluate()
    mlflow.log_metrics(final_metrics)

    mlflow.end_run()

    print("Training completed!")
    print(f"Final eval loss: {final_metrics['eval_loss']}")

if __name__ == "__main__":
    main()