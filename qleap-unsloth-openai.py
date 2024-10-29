# ----------------------------- #
# Part 1: Import Libraries
# ----------------------------- #

import json
import ast
import logging
import csv
import os
import torch
from typing import List, Dict, Any
from datasets import Dataset
from transformers import TextStreamer
from unsloth import (
    FastLanguageModel,
    UnslothTrainer,
    UnslothTrainingArguments,
    is_bfloat16_supported,
)

# Configure logging
logging.basicConfig(
    filename='transformation_errors.log',
    filemode='w',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define paths
INPUT_CSV_PATH = '/root/quantumLeap/data/psychologoy-of-unconscious-mind/concept_examples.csv'
OUTPUT_JSON_PATH = '/root/qLeap-fft/data/input/Instruction_Data/transformed_data.json'

# ----------------------------- #
# Part 2: Load and Clean the Text Data
# ----------------------------- #

def read_csv_data(input_csv_path: str) -> List[Dict[str, str]]:
    """Read and validate the input CSV file."""
    try:
        with open(input_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        raise

def transform_data(original_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Transform the original data by expanding example scenarios."""
    new_data = []

    for idx, entry in enumerate(original_data, start=1):
        concept_name = entry.get('concept_name', '').strip()
        detailed_explanation = entry.get('detailed_explanation', '').strip()
        example_scenario_str = entry.get('example_scenario', '').strip()

        if not all([concept_name, detailed_explanation, example_scenario_str]):
            logging.error(f"Entry {idx} is missing required fields. Skipping.")
            continue

        try:
            example_scenarios = json.loads(example_scenario_str)
        except json.JSONDecodeError:
            try:
                example_scenarios = ast.literal_eval(example_scenario_str)
            except (ValueError, SyntaxError) as e:
                logging.error(f"Entry {idx} ('{concept_name}') has invalid example_scenario: {e}")
                continue

        if not isinstance(example_scenarios, list):
            logging.error(f"Entry {idx} ('{concept_name}'): example_scenario is not a list")
            continue

        for scenario_idx, scenario in enumerate(example_scenarios, start=1):
            if not isinstance(scenario, str):
                logging.error(f"Entry {idx} ('{concept_name}'): non-string scenario at position {scenario_idx}")
                continue

            new_data.append({
                'concept_name': concept_name,
                'detailed_explanation': detailed_explanation,
                'example_scenario': scenario.strip()
            })

    return new_data

# Process and save the data
original_data = read_csv_data(INPUT_CSV_PATH)
transformed_data = transform_data(original_data)

# Save transformed data
os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(transformed_data, f, ensure_ascii=False, indent=4)

print(f"Processed {len(transformed_data)} examples")

# ----------------------------- #
# Part 3: Create Instruction Prompt Template and Process Data
# ----------------------------- #

# Import Jinja2 for template rendering
from jinja2 import Template

# Define the instruction template
instruction_template = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{assistant_response}<|eot_id|>"""

def create_instruction_dataset(transformed_data: List[Dict[str, str]]) -> Dataset:
    """Create an instruction dataset from transformed data."""
    template = Template(instruction_template)

    def instruction_prompt_func(examples):
        prompts = []
        for cn, de, es in zip(
            examples["concept_name"],
            examples["detailed_explanation"],
            examples["example_scenario"]
        ):
            # Prepare the user message
            user_message = f"Explain the concept of {cn} and provide an example."

            # Prepare the assistant's response
            assistant_response = f"{de}\n\nExample:\n{es}"

            # Render the prompt using the template
            rendered_prompt = template.render(
                user_message=user_message,
                assistant_response=assistant_response
            )
            prompts.append(rendered_prompt)
        return {"text": prompts}

    dataset = Dataset.from_list(transformed_data)
    return dataset.map(instruction_prompt_func, batched=True)

# Create the dataset
instruction_dataset = create_instruction_dataset(transformed_data)

# Print a sample to verify
print("\nSample processed example:")
print(instruction_dataset[0]["text"])

# ----------------------------- #
# Part 4: Load the Tokenizer and Model
# ----------------------------- #

# Model initialization parameters
base_model_slug = "unsloth/Llama-3.2-1B-bnb-4bit"
model_name = "lora_model_pum"
max_seq_length = 1024  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

# Initialize model and tokenizer
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_slug,  # Base model slug
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Define special tokens
special_tokens = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "user",
    "assistant"
]
# we will first check if any of the special_tokens are not there in the tokenizer, if not then we will add, otherwise we will not do anything
tokens_not_in_vocab = []
for i in special_tokens:
    if i not in tokenizer.get_vocab():
        tokens_not_in_vocab.append(i)
    else:
        pass
        
# Add special tokens to the tokenizer
if len(tokens_not_in_vocab) > 0:
    special_tokens_dict = {'additional_special_tokens': tokens_not_in_vocab}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens to the tokenizer.")
else:
    num_added_toks = 0

# Resize model embeddings if new tokens were added
if num_added_toks > 0:
    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized model embeddings to {len(tokenizer)} tokens.")

# Set eos_token_id and pad_token_id
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
tokenizer.pad_token_id = tokenizer.eos_token_id  # Use eos_token as pad_token

# Update model configuration
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# Configure model
model.config.torch_dtype = torch.bfloat16

# Prepare the model for training
model = FastLanguageModel.get_peft_model(
    model,
    r=128,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "embed_tokens", "lm_head",
    ],  # Add for continual pretraining
    lora_alpha=32,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=True,   # We support rank stabilized LoRA
    loftq_config=None, # And LoftQ
)

# Prepare the model for inference
FastLanguageModel.for_inference(model)

# Test the tokenization
test_prompt = instruction_template.format(
    user_message="Explain the concept of Semiotics and provide an example.",
    assistant_response="Semiotics is the study of signs and symbols and their use or interpretation.\n\nExample:\nA red traffic light signifies 'stop' to drivers."
)

inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
print("Tokenized input IDs:", inputs["input_ids"])

# Generate a test output
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=True
    )

# Decode and print the output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
print("Generated text:\n", generated_text)

# ----------------------------- #
# Part 5: Configure Training Arguments
# ----------------------------- #

def setup_training(model, tokenizer, dataset,
                   batch_size=2, gradient_accumulation=8, max_steps=10):
    """Setup the training configuration."""
    from datetime import datetime
    import pytz
    import wandb

    # Define your parameters
    batchSize = batch_size
    ga = gradient_accumulation
    maxSteps = max_steps
    lRate = 5e-5
    embLRate = 1e-5
    optim = "adamw_8bit"
    lrSchedule = "linear"

    # Get the current date and time in Indian Standard Time (IST)
    ist = pytz.timezone('Asia/Kolkata')
    current_datetime = datetime.now(ist)

    # Format the datetime string
    # Example format: 20240428_153045 (YYYYMMDD_HHMMSS)
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

    # Create the run name with the current date and time
    run_name = f"""Unsloth-CPT-Instruct-{formatted_datetime}-{base_model_slug}-{max_seq_length}_max_seq_length-{batchSize}_batchSize-{ga}_ga-{maxSteps}_maxSteps-{lRate}_lRate-{embLRate}_embLRate-{optim}_optim-{lrSchedule}_lrSchedule"""

    # Initialize Weights & Biases
    # Set your W&B API key as an environment variable for security.
    # Example: export WANDB_API_KEY="your_api_key"
    wandb.login()  # Assumes API key is set in the environment variable
    wandb.init(project="Unsloth-CPT", name=run_name)

    training_args = UnslothTrainingArguments(
        per_device_train_batch_size=batchSize,
        gradient_accumulation_steps=ga,
        max_steps=maxSteps,
        warmup_steps=10,
        learning_rate=lRate,
        embedding_learning_rate=embLRate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim=optim,
        weight_decay=0.01,
        lr_scheduler_type=lrSchedule,
        seed=3407,
        output_dir="outputs",
        report_to=["tensorboard", "wandb"],
        logging_dir="./trel-fft-logs"
    )

    return UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        args=training_args
    )

# Setup trainer
trainer = setup_training(model, tokenizer, instruction_dataset)

# Start training
trainer.train()

# ----------------------------- #
# Part 6: Save the Instruction Fine-Tuned Model
# ----------------------------- #

model.save_pretrained("lora_model_pum_instruct")  # Local saving
tokenizer.save_pretrained("lora_model_pum_instruct")

# Hugging Face authentication token should be set via environment variable or login
# Example: export HUGGINGFACE_TOKEN="your_hf_token"

# Uncomment and set to True if you wish to push to Hugging Face Hub
if False:
    model.push_to_hub("your-username/your-model-name", token=os.getenv("HUGGINGFACE_TOKEN"))
    tokenizer.push_to_hub("your-username/your-model-name", token=os.getenv("HUGGINGFACE_TOKEN"))
    model.push_to_hub_gguf("your-username/your-model-name", tokenizer, quantization_method="q4_k_m", token=os.getenv("HUGGINGFACE_TOKEN"))


# ----------------------------- #
# Part 7: Generate Inference from Instruction Fine-Tuned Model
# ----------------------------- #

import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer
import warnings
warnings.filterwarnings('ignore')

# Model initialization parameters
max_seq_length = 1024
dtype = None
load_in_4bit = True

# Load the fine-tuned model
if False:
    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lora_model_pum_instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Prepare the model for inference
FastLanguageModel.for_inference(model)

# Set model dtype
model.config.torch_dtype = torch.bfloat16

# Instruction prompt matching the fine-tuning template
instruction_template = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

# Example usage
concept_name = "Semiotics"

# Prepare the user message
user_message = f"Explain the concept of {concept_name} and provide an example."

# Format input
prompt = instruction_template.format(user_message=user_message)

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Initialize text streamer
text_streamer = TextStreamer(tokenizer)

# Generate output with modified parameters
outputs = model.generate(
    **inputs,
    streamer=text_streamer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    min_length=50,
    early_stopping=True
)

# Optional: Print the full response
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nGenerated Response:")
print(generated_text)
