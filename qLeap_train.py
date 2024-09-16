import torch
torch.cuda.empty_cache()

import os
import re
import torch
import nltk
import spacy
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
import logging
import argparse
import wandb  # Added for Weights & Biases integration

# ----------------------------- #
# Part 1: Install and Setup Libraries
# ----------------------------- #

# Ensure NLTK's punkt tokenizer is available
nltk.download('punkt')

# Initialize spaCy English model
try:
    nlp = spacy.load('en_core_web_trf')
except OSError:
    print("SpaCy English model not found. Downloading...")
    os.system('python -m spacy download en_core_web_trf')
    nlp = spacy.load('en_core_web_trf')

# ----------------------------- #
# Part 2: Load and Clean the Text Data
# ----------------------------- #

def load_and_clean_text(file_path):
    """
    Loads text from a file and removes Project Gutenberg's license and headers/footers.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Remove Project Gutenberg's license text and headers/footers
    start_pattern = r'\*\*\* START OF THIS PROJECT GUTENBERG EBOOK.*\*\*\*'
    end_pattern = r'\*\*\* END OF THIS PROJECT GUTENBERG EBOOK.*\*\*\*'

    text = re.sub(f'.*{start_pattern}', '', text, flags=re.DOTALL)
    text = re.sub(f'{end_pattern}.*', '', text, flags=re.DOTALL)
    return text.strip()

# Replace 'psychology_of_unconscious.txt' with your actual file path
file_path = 'psychology_of_unconscious.txt'
clean_text = load_and_clean_text(file_path)

# ----------------------------- #
# Part 3: Parse Text into Discourse Units
# ----------------------------- #

def parse_discourse_units(text):
    """
    Parses text into discourse units using spaCy.
    Currently splits text into sentences.
    """
    paragraphs = text.split('\n\n')
    paragraphs = [para.strip() for para in paragraphs if para.strip()]
    
    discourse_units = []
    for para in paragraphs:
        doc = nlp(para)
        sentences = [sent.text for sent in doc.sents]
        discourse_units.extend(sentences)
    return discourse_units

discourse_units = parse_discourse_units(clean_text)

# ----------------------------- #
# Part 4: Load the Tokenizer and Model
# ----------------------------- #

model_name = "unsloth/Meta-Llama-3.1-8B"
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model with half-precision
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cuda",
)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()
model.config.use_cache = False  # Disable caching for training

# ----------------------------- #
# Part 5: Create Chunks
# ----------------------------- #

def create_chunks(discourse_units, tokenizer, max_length=2048, overlap_size=100):
    """
    Creates chunks from discourse units using a sliding window with overlapping chunks.
    """
    chunks = []
    current_chunk = ''
    current_length = 0

    for unit in discourse_units:
        unit_tokens = tokenizer.encode(unit, add_special_tokens=False)
        unit_length = len(unit_tokens)

        if current_length + unit_length <= max_length:
            current_chunk += unit + ' '
            current_length += unit_length
        else:
            chunks.append(current_chunk.strip())
            overlap_tokens = tokenizer.encode(current_chunk, add_special_tokens=False)[-overlap_size:]
            overlap_text = tokenizer.decode(overlap_tokens, skip_special_tokens=True)
            current_chunk = overlap_text + ' ' + unit + ' '
            current_length = len(tokenizer.encode(current_chunk, add_special_tokens=False))
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

chunks = create_chunks(discourse_units, tokenizer)

# ----------------------------- #
# Part 6: Create and Tokenize Dataset
# ----------------------------- #

dataset = Dataset.from_dict({'text': chunks})

def tokenize_function(examples):
    result = tokenizer(
        examples['text'],
        max_length=2048,
        padding='max_length',
        truncation=True,
        return_overflowing_tokens=False,
    )
    
    # Create labels by shifting the input_ids
    result["labels"] = result["input_ids"].copy()
    
    # Shift the labels to align with the next token prediction
    for i, label in enumerate(result["labels"]):
        result["labels"][i] = [-100] + label[:-1]
    
    return result

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=20,  # Adjust based on available CPU cores
    remove_columns=['text'],
)

# Split the dataset into training and validation sets
split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = split['train']
eval_dataset = split['test']

# ----------------------------- #
# Part 7: Configure Training Arguments
# ----------------------------- #

from transformers import TrainingArguments

# Initialize Weights & Biases
wandb.init(
    project="quantum-leap-training",
    config={
        "model_name": model_name,
        "epochs": 3,
        "batch_size": 2,
        "learning_rate": 2e-5,
    },
    sync_tensorboard=True,
)

training_args = TrainingArguments(
    output_dir='./meta-llama-3.1-8b-finetuned',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Increased batch size
    gradient_accumulation_steps=4,  # Reduced accumulation steps
    learning_rate=2e-5,
    warmup_steps=500,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=2,
    fp16=True,  # Enable mixed precision training
    bf16=False,
    optim='adamw_torch_fused',
    save_strategy='steps',  # Changed from 'epoch' to 'steps'
    save_steps=500,  # Save every 500 steps
    evaluation_strategy='steps',
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model='perplexity',
    report_to="wandb",  # Changed to report to Weights & Biases
    max_grad_norm=1.0,
    gradient_checkpointing=True,
    dataloader_num_workers=4,  # Utilize multiple CPU cores for data loading
)

# ----------------------------- #
# Part 8: Define Compute Metrics Function
# ----------------------------- #

def compute_metrics(eval_pred):
    """
    Computes perplexity based on model predictions and labels.
    """
    logits, labels = eval_pred
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    perplexity = torch.exp(loss).item()
    return {"perplexity": perplexity}

# ----------------------------- #
# Part 9: Initialize and Start Training
# ----------------------------- #

def main():
    parser = argparse.ArgumentParser(description='Train Quantum Leap model')
    # Removed the --device argument as device mapping is handled automatically
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # Initialize Trainer without moving the model to a specific device
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        compute_metrics=compute_metrics,
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model('./meta-llama-3.1-8b-finetuned')

    print("Training completed and model saved!")

if __name__ == '__main__':
    main()