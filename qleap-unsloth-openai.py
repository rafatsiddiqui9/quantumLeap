# # ----------------------------- #
# # Part 1.1: Install and Setup Libraries
# # ----------------------------- #

# # run below in terminal only. This code works only for Ola Krutrim Cloud Instance. Restart once you have installed the following
# # pip install uv #install this in the virtual environment where you want to execute the notebook.
# # uv venv virtualenvironment # if you are not in an externally managed environment, then you can run this
# # source virtualenvironment/bin/activate # if you were able to run above code, then activate. DO NOT use --system flag in subsequent lines if you are able to do thi
# !uv pip install unsloth --system
# !uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --system
# !uv pip install wandb bitsandbytes ipywidgets nltk spacy huggingface_hub datasets --system
# # restart once you have installed all of the above

# !nvidia-smi

# !nvcc --version

# import torch
# print(torch.__version__)          # Should reflect 2.5.0+cu124
# print(torch.version.cuda)         # Should output 12.4
# print(torch.cuda.is_available())  # Should return True

# ----------------------------- #
# Part 1.2: Import Libraries
# ----------------------------- #

import os
import re
import torch
import nltk
import spacy
import xformers
import bitsandbytes
import datasets
import huggingface_hub
import wandb
import ipywidgets
import unsloth
import json
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

# Ensure NLTK's punkt tokenizer is available
import nltk
try:
    nltk.data.find('tokenizers/punkt')
    print('punkt was already available.')
except LookupError:
    nltk.download('punkt')
    print('punkt was not available. It has been downloaded')

# Initialize spaCy English model
try:
    nlp = spacy.load('en_core_web_sm')
    print('en_core_web_sm was already available.')
except OSError:
    print("SpaCy English model not found. Downloading...")
    os.system('python -m spacy download en_core_web_sm')
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
    # # Remove Project Gutenberg's license text and headers/footers
    # start_pattern = r'\*\*\* START OF THIS PROJECT GUTENBERG EBOOK.*\*\*\*'
    # end_pattern = r'\*\*\* END OF THIS PROJECT GUTENBERG EBOOK.*\*\*\*'

    # text = re.sub(f'.*{start_pattern}', '', text, flags=re.DOTALL)
    # text = re.sub(f'{end_pattern}.*', '', text, flags=re.DOTALL)
    return text.strip()

# Replace 'psychology_of_unconscious.txt' with your actual file path
file_path = '/root/quantumLeap/data/psychologoy-of-unconscious-mind/psychology_of_unconscious.txt'
clean_text = load_and_clean_text(file_path)

# ----------------------------- #
# Part 3: Parse Text into Discourse Units
# ----------------------------- #

# def parse_discourse_units(text):
#     """
#     Parses text into discourse units using spaCy.
#     Currently splits text into sentences.
#     """
#     paragraphs = text.split('\n\n')
#     paragraphs = [para.strip() for para in paragraphs if para.strip()]
    
#     discourse_units = []
#     for para in paragraphs:
#         doc = nlp(para)
#         sentences = [sent.text for sent in doc.sents]
#         discourse_units.extend(sentences)
#     return discourse_units

# discourse_units = parse_discourse_units(clean_text)

# # Save discourse_units to a JSON file
# with open('/root/quantumLeap/data/psychologoy-of-unconscious-mind/discourse_units_final.json', 'w', encoding='utf-8') as f:
#     json.dump(discourse_units, f, ensure_ascii=False, indent=4)
    
# Load discourse_units from the JSON file
with open('/root/quantumLeap/data/psychologoy-of-unconscious-mind/discourse_units_final.json', 'r', encoding='utf-8') as f:
    discourse_units = json.load(f)

len(discourse_units)

# ----------------------------- #
# Part 4: Create Chunks Using Hybrid Strategy
# ----------------------------- #

def create_chunks(discourse_units, tokenizer, max_length=512, overlap_size=0):
    """
    Creates chunks from discourse units using a sliding window with overlapping chunks.
    """
    chunks = []
    current_chunk = []
    current_length = 0

    for unit in discourse_units:
        unit_tokens = tokenizer.encode(unit, add_special_tokens=False)
        unit_length = len(unit_tokens)

        if current_length + unit_length <= max_length:
            current_chunk.append(unit)
            current_length += unit_length
        else:
            # Append the current chunk
            chunks.append(' '.join(current_chunk))
            # Create overlap
            overlap_text = ' '.join(current_chunk)[-overlap_size:]
            overlap_tokens = tokenizer.encode(overlap_text, add_special_tokens=False)
            overlap_text = tokenizer.decode(overlap_tokens, skip_special_tokens=True)
            # Start new chunk with overlap and current unit
            current_chunk = [overlap_text, unit]
            current_length = len(tokenizer.encode(overlap_text, add_special_tokens=False)) + unit_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

from unsloth import FastLanguageModel
import torch
max_seq_length = 1024 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
base_model_slug = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = base_model_slug, # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = "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", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# ----------------------------- #
# Part 5: : Load the Tokenizer and Model
# ----------------------------- #
model = FastLanguageModel.get_peft_model(
    model,
    r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",

                      "embed_tokens", "lm_head",], # Add for continual pretraining
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = True,   # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# ----------------------------- #
# Part 6: Create Chunks (After Tokenizer is Loaded)
# ----------------------------- #

chunks_max_length = max_seq_length
overlap_size = 1
chunks = create_chunks(discourse_units, tokenizer, max_length=chunks_max_length, overlap_size=overlap_size)

# Save chunks to a JSON file (Optional)
with open(f'/root/quantumLeap/data/psychologoy-of-unconscious-mind/chunks_final_{chunks_max_length}_{overlap_size}.json', 'w', encoding='utf-8') as f:
    json.dump(chunks, f, ensure_ascii=False, indent=4)

# # If you need to reload from JSON (Optional)
# with open('/root/quantumLeap/data/psychologoy-of-unconscious-mind/chunks_final.json', 'r', encoding='utf-8') as f:
#     chunks = json.load(f)
    
print(len(chunks))

# ----------------------------- #
# Part 7: Create and Tokenize Dataset
# ----------------------------- #

# Create a Dataset object from chunks

book_title = 'Psychology of the Unconscious by C. G. Jung'
wikipedia_prompt = """
### Title: {}

### Article: {}
"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    titles = book_title
    texts  = examples["text"]
    outputs = []
    for title, text in zip([book_title]*len(chunks), texts):
        text = wikipedia_prompt.format(title, text) + EOS_TOKEN
        outputs.append(text)
    return { "text" : outputs, }
pass

# convert chunks variable to huggingface dataset

dataset = Dataset.from_dict({"text": chunks})

# dataset = dataset.train_test_split(test_size = 0.1)["train"]

dataset = dataset.map(formatting_prompts_func, batched = True,)

len(dataset)

# Find the maximum length of the text field in the entire dataset
max_length = max(len(text) for text in dataset['text'])
print(f"The maximum length of the text field in the dataset is: {max_length} characters")

# ----------------------------- #
# Part 8: Configure Training Arguments
# ----------------------------- #

from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments

from datetime import datetime
import pytz
import wandb

# Define your parameters
batchSize = 2
ga = 8
maxSteps = 10
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
run_name = f"""Unsloth-CPT-Base-{formatted_datetime}-{base_model_slug}-{max_seq_length}_max_seq_length-{batchSize}_batchSize-{ga}_ga-{maxSteps}_maxSteps-{lRate}_lRate-{embLRate}_embLRate-{optim}_optim-{lrSchedule}_lrSchedule"""

# Initialize Weights & Biases
# It's recommended to set your W&B API key as an environment variable for security.
# Example: export WANDB_API_KEY="your_api_key"
wandb.login(key="1ca3c5e9222c2504acbc07cf7f88267006ae68c4")  # Consider using environment variables for security
wandb.init(project="Unsloth-CPT", name=run_name)

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,

    args = UnslothTrainingArguments(
        per_device_train_batch_size = batchSize,
        gradient_accumulation_steps = ga,

        # Use warmup_ratio and num_train_epochs for longer runs!
        max_steps = maxSteps,
        warmup_steps = 10,
        # warmup_ratio = 0.1,
        # num_train_epochs = 1,

        # Select a 2 to 10x smaller learning rate for the embedding matrices!
        learning_rate =lRate,
        embedding_learning_rate = embLRate,

        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = optim,
        weight_decay = 0.01,
        lr_scheduler_type = lrSchedule,
        seed = 3407,
        output_dir = "outputs",
        
        
        report_to=["tensorboard", "wandb"],
        logging_dir=f"./trel-fft-logs/{run_name}",

    ),
)

# ----------------------------- #
# Part 9: Define Compute Metrics Function
# ----------------------------- #

def compute_metrics(eval_pred):
    """
    Computes perplexity based on model predictions and labels.
    """
    logits, labels = eval_pred
    # Convert to torch tensors
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    
    # Ensure shapes match
    if logits.shape[:2] != labels.shape:
        raise ValueError(f"Logits shape {logits.shape} does not match labels shape {labels.shape}")
    
    # Shift logits and labels
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Check label values
    if shift_labels.max() >= model.config.vocab_size:
        raise ValueError(f"Label value {shift_labels.max()} exceeds vocab size {model.config.vocab_size}")
    
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    perplexity = torch.exp(loss).item()
    return {"perplexity": perplexity}

# ----------------------------- #
# Part 10: Initialize logging
# ----------------------------- #

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    level=logging.INFO,  # Set to DEBUG for more detailed logs
)
logger = logging.getLogger(__name__)
    
# ----------------------------- #
# Part 11: Start Training
# ----------------------------- #

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# ----------------------------- #
# Part 12: Save the Bsae Fine-Tuned Model
# ----------------------------- #

model.save_pretrained("lora_model_pum") # Local saving
tokenizer.save_pretrained("lora_model_pum")

# huggingface-cli login --token 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 --add-to-git-credential # uncomment and add "!" if not using python terminal
if False:
    model.push_to_hub("olabs-ai/qLeap_base_v01", token = "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") # Online saving
    tokenizer.push_to_hub("olabs-ai/qLeap_base_v01", token = "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") # Online saving
    model.push_to_hub_gguf("olabs-ai/qLeap_base_v01", tokenizer, quantization_method = "q4_k_m", token = "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")
    
    
# ----------------------------- #
# Part 13: Generate Inference from Base Fine-Tuned Model for testing purpose
# ----------------------------- #
    
import torch
max_seq_length = 1024 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model_pum", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    
instruction_prompt = """Below is an instruction that describes a concept in the field of psychology, sociology, anthropology, ethnography, or qualitative research or cultural studies. Write a response that appropriately completes the request.

### Instruction: 
concept_name: {}
detailed_explanation: {}
Given the concept in concept_name variable and its detailed explanation in detailed_explanation variable, provide an example scenario that illustrates the concept.
### Response:
{}"""

FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    instruction_prompt.format(
        "Hero Archetype", # concept_name
        "The hero archetype is a common motif in literature and folklore, representing a protagonist who embodies bravery, resilience, and a quest for a greater purpose.", # detailed_explanation
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")
model.config.torch_dtype = torch.bfloat16 
from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 512) # using repetition_penalty of 0.1 leads to repetition of text and high values lead to wierd grammer issues


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
base_model_slug = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
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
