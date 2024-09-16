# !pip install -U -q jupyter ipywidgets nbformat pandas numpy bitsandbytes wandb torch transformers datasets tokenizers accelerate spacy nltk peft deepspeed xformers
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
# # Check if CUDA is installed
# nvcc --version

# # If CUDA is not installed, download and install CUDA Toolkit (example for CUDA 12.1)
# wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
# sudo sh cuda_12.1.0_530.30.02_linux.run

# # Set CUDA_HOME environment variable
# export CUDA_HOME=/usr/local/cuda-12.1
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# # Verify CUDA_HOME is set correctly
# echo $CUDA_HOME

# # Retry installing DeepSpeed
# pip install deepspeed
# login to hugging face
!huggingface-cli login --token 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 --add-to-git-credential
!lscpu && free -h && df -h && nvidia-smi
# ChatGPT optimized script
import os
import re
import torch
import nltk
import spacy
import logging
import argparse
import subprocess
import sys
import gc
import weakref
import wandb  # Weights & Biases

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset


# ----------------------------- #
# Part 1: Install and Setup Libraries
# ----------------------------- #

# Set environment variable once
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Clear CUDA cache
torch.cuda.empty_cache()

# Ensure NLTK's punkt tokenizer is available
nltk.download('punkt')

# Initialize spaCy English model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("SpaCy English model not found. Downloading...")
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)
    nlp = spacy.load('en_core_web_sm')

# ----------------------------- #
# Part 2: Load and Clean the Text Data
# ----------------------------- #

def load_and_clean_text(file_path):
    """
    Loads text from a file and removes Project Gutenberg's license and headers/footers.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text.strip()

# Parse command-line arguments for flexibility
parser = argparse.ArgumentParser(description='Fine-tune LLaMA-3.1-8B')
parser.add_argument('--file_path', type=str, required=True, help='/home/ubuntu/quantumLeap/psychology_of_unconscious.txt')
parser.add_argument('--monitor_gpu', action='store_true', help='Enable GPU monitoring')
args = parser.parse_args()

if args.monitor_gpu:
    def display_nvidia_smi():
        try:
            result = subprocess.check_output(['nvidia-smi'], universal_newlines=True)
            print(result)
        except Exception as e:
            print(f"Error running nvidia-smi: {e}")

    def list_gpu_processes():
        try:
            result = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'])
            processes = result.decode('utf-8').strip().split('\n')
            processes = sorted([p.split(', ') for p in processes], key=lambda x: int(x[2]), reverse=True)
            
            print("Top GPU Processes:")
            for pid, name, mem in processes[:10]:
                print(f"PID: {pid}, Process Name: {name}, GPU Memory Usage: {mem} MiB")
        except Exception as e:
            print(f"Error listing GPU processes: {e}")

    def list_variables():
        current_module = sys.modules[__name__]
        variables = [(name, type(value).__name__, sys.getsizeof(value))
                     for name, value in vars(current_module).items()
                     if not name.startswith('_')]
        
        sorted_vars = sorted(variables, key=lambda x: x[2], reverse=True)
        
        print("Variables in current session:")
        for name, type_name, size in sorted_vars:
            print(f"{name}: Type = {type_name}, Size = {size} bytes")

    def list_top_gpu_variables(top_n=10):
        tensor_list = []
        for obj in gc.get_objects():
            try:
                if isinstance(obj, torch.Tensor) and obj.is_cuda:
                    tensor_list.append(weakref.ref(obj))
            except:
                continue
        
        tensor_sizes = []
        for tensor_ref in tensor_list:
            tensor = tensor_ref()
            if tensor is not None:
                try:
                    size = tensor.element_size() * tensor.nelement()
                    tensor_sizes.append((tensor, size))
                except:
                    continue
        
        tensor_sizes.sort(key=lambda x: x[1], reverse=True)
        
        print("Top GPU Variables:")
        for idx, (tensor, size) in enumerate(tensor_sizes[:top_n], 1):
            print(f"{idx}. Tensor Shape: {tensor.shape}, Size: {size / 1e6:.2f} MB")

    display_nvidia_smi()
    list_gpu_processes()
    list_variables()
    list_top_gpu_variables()

# Load and clean text
clean_text = load_and_clean_text(r"/home/ubuntu/quantumLeap/psychology_of_unconscious.txt")

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

# if not os.path.exists('discourse_units.txt'):
#     discourse_units = parse_discourse_units(clean_text)
#     with open('discourse_units.txt', 'w') as f:
#         for unit in discourse_units:
#             f.write(unit + '\n')
# else:
#     with open('discourse_units.txt', 'r') as f:
#         discourse_units = f.read().splitlines()


discourse_units = parse_discourse_units(clean_text)
with open('discourse_units.txt', 'w') as f:
    for unit in discourse_units:
        f.write(unit + '\n')

discourse_units

# ----------------------------- #
# Part 4: Create Chunks - Model Load
# ----------------------------- #

model_name = "/home/ubuntu/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/48d6d0fc4e02fb1269b36940650a1b7233035cbb"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for mixed precision
    device_map="cuda:0",         # Explicitly map to GPU 0
)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# add padding to the tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# ----------------------------- #
# Part 4: Create Chunks - Model Load
# ----------------------------- #

# **Set pad_token to eos_token here**
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
def create_chunks(discourse_units, tokenizer, max_length=1024, overlap_size=100):
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


# if not os.path.exists('chunks.txt'):
#     # Load tokenizer before creating chunks
#     tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
#     chunks = create_chunks(discourse_units, tokenizer)
#     with open('chunks.txt', 'w') as f:
#         for chunk in chunks:
#             f.write(chunk + '\n')
# else:
#     with open('chunks.txt', 'r') as f:
#         chunks = f.read().splitlines()

chunks = create_chunks(discourse_units, tokenizer)
with open('chunks.txt', 'w') as f:
    for chunk in chunks:
        f.write(chunk + '\n')

chunks
# ----------------------------- #
# Part 5: GPU Monitoring and Cleanup (Optional)
# ----------------------------- #

# Already handled above based on --monitor_gpu flag

# ----------------------------- #
# Part 6: Create and Tokenize Dataset
# ----------------------------- #

# if the clean_text variable is not defined, then we need to load the clean_text from the file
if not hasattr(globals(), 'clean_text'):
    with open('psychology_of_unconscious.txt', 'r') as f:
        clean_text = f.read()
        

# if the discourse_units variable is not defined, then we need to load the discourse_units from the file
if not hasattr(globals(), 'discourse_units'):
    with open('discourse_units.txt', 'r') as f:
        discourse_units = f.read().splitlines()
        

# if the chunks variable is not defined, then we need to load the chunks from the file
if not hasattr(globals(), 'chunks'):
    with open('chunks.txt', 'r') as f:
        chunks = f.read().splitlines()

dataset = Dataset.from_dict({'text': chunks})

def tokenize_function(examples):
    result = tokenizer(
        examples['text'],
        max_length=1024,
        padding='max_length',  # This requires pad_token to be set
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
    num_proc=16,  # Adjusted based on CPU cores
    remove_columns=['text'],
)

# Split the dataset into training and validation sets
split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split['train']
eval_dataset = split['test']

# Clear unnecessary variables to free up memory
del clean_text, discourse_units, chunks
torch.cuda.empty_cache()
gc.collect()

# ----------------------------- #
# Part 7: Configure Training Arguments
# ----------------------------- #

# Create DeepSpeed configuration file programmatically if needed
deepspeed_config = {
    "train_batch_size": 32,
    "gradient_accumulation_steps": 2,
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "initial_scale_power": 16
    },
    "zero_optimization": {
        "stage": 1,
        "offload_optimizer": {
            "device": "none",
            "pin_memory": False
        },
        "overlap_comm": True,
        "contiguous_gradients": True
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 2e-5,
            "warmup_num_steps": 500
        }
    },
    "steps_per_print": 2000,
    "wall_clock_breakdown": False
}

# Save DeepSpeed config to a file
with open('deepspeed_config.json', 'w') as f:
    import json
    json.dump(deepspeed_config, f, indent=2)

# Initialize TrainingArguments
training_args = TrainingArguments(
    output_dir='./meta-llama-3.1-8b-finetuned',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Increased batch size
    gradient_accumulation_steps=2,  # Reduced accumulation steps
    learning_rate=2e-5,
    warmup_steps=500,
    logging_dir='./logs',
    logging_steps=50,  # Reduced logging frequency
    save_total_limit=2,
    fp16=True,  # Disabled FP16 as using BF16
    bf16=False,    # Enabled BF16
    optim='adamw_hf',  # Changed to a more compatible optimizer
    save_strategy='steps',
    save_steps=500,  # Save every 500 steps
    evaluation_strategy='steps',
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model='perplexity',
    report_to="wandb",  # Report to Weights & Biases
    max_grad_norm=1.0,
    gradient_checkpointing=True,
    dataloader_num_workers=16,  # Optimized number of workers
    deepspeed="deepspeed_config.json",  # Integrate DeepSpeed
)
# login to wandb
wandb.login(key='0123456789abcdef0123456789abcdef')

# Initialize Weights & Biases after TrainingArguments
wandb.init(
    project="quantum-leap-training",
    config=training_args.to_dict(),
    sync_tensorboard=True,
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
    # Set up logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        compute_metrics=compute_metrics,
    )

    # Start training with error handling
    try:
        trainer.train()
        trainer.save_model('./meta-llama-3.1-8b-finetuned')
        print("Training completed and model saved!")
    except Exception as e:
        logger.error(f"Training failed: {e}")

if __name__ == '__main__':
    main()


# ----------------------------- #
# Part 10: Inference Section
# ----------------------------- #

# Inference Time
def inference():
    def display_nvidia_smi():
        try:
            result = subprocess.check_output(['nvidia-smi'], universal_newlines=True)
            print(result)
        except Exception as e:
            print(f"Error running nvidia-smi: {e}")

    display_nvidia_smi()

    # Load the tokenizer and model with bf16
    tokenizer = AutoTokenizer.from_pretrained('./meta-llama-3.1-8b-finetuned')
    model = AutoModelForCausalLM.from_pretrained(
        './meta-llama-3.1-8b-finetuned',
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Define a sample prompt
    prompt = "How are you doing?"
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    
    # Generate outputs
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100)
    
    # Decode and print the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)

# Uncomment the following line to run inference after training
# inference()

