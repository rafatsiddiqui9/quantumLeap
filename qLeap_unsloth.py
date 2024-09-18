# pip install unsloth
# Also get the latest nightly Unsloth!
# pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# pip install -q -U xformers torch--no-cache-dir


from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",      # New Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",           # Llama-3 15 trillion tokens model 2x faster!
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",        # Phi-3 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",             # Gemma 2.2x faster!
] # More models at https://huggingface.co/unsloth

# if the model is already downloaded, then don't download it again; otherwise download it
import os

model_name = "unsloth/Meta-Llama-3.1-8B"
models_dir = os.path.join(os.path.dirname(os.getcwd()), "models")
model_path = os.path.join(models_dir, model_name)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if os.path.exists(model_path):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
else:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        token="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",
    )
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    
    
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
# import re

# def load_and_clean_text(file_path):
#     """
#     Loads text from a file and removes Project Gutenberg's license and headers/footers.
#     """
#     with open(file_path, 'r', encoding='utf-8') as f:
#         text = f.read()

#     # Remove Project Gutenberg's license text and headers/footers
#     start_pattern = r'\*\*\* START OF THIS PROJECT GUTENBERG EBOOK.*\*\*\*'
#     end_pattern = r'\*\*\* END OF THIS PROJECT GUTENBERG EBOOK.*\*\*\*'

#     text = re.sub(f'.*{start_pattern}', '', text, flags=re.DOTALL)
#     text = re.sub(f'{end_pattern}.*', '', text, flags=re.DOTALL)
#     return text.strip()

# # Replace 'psychology_of_unconscious.txt' with your actual file path
# file_path = '/content/psychology_of_unconscious.txt'
# clean_text = load_and_clean_text(file_path)


# Load files if you do not want to generate fresh data

with open(r"psychology_of_unconscious.txt", 'r', encoding='utf-8') as f:
    clean_text = f.read()
type(clean_text)
# If you need to reload from file (Optional)
with open(r"discourse_units copy.txt", 'r') as f:
    discourse_units = f.read().splitlines()

discourse_units[0:2]
# If you need to reload from file (Optional)
with open(r"chunks copy.txt", 'r') as f:
    chunks = f.read().splitlines()

chunks[0]

book_title = 'Psychology of the Unconscious by C. G. Jung'
wikipedia_prompt = """
Psychology Book

### Title: {}

### Article: {}
"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    titles = book_title
    texts  = examples["text"]
    outputs = []
    for title, text in zip([book_title]*len(chunks), texts):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = wikipedia_prompt.format(title, text) + EOS_TOKEN
        outputs.append(text)
    return { "text" : outputs, }
pass

# convert chunks variable to huggingface dataset

from datasets import Dataset

dataset = Dataset.from_dict({"text": chunks})

dataset = dataset.train_test_split(train_size = 0.90)["train"]

dataset = dataset.map(formatting_prompts_func, batched = True,)

dataset

# Wikipedia provides a title and an article text.
# Use https://translate.google.com!
wikipedia_prompt = """
### Text:
{}"""
EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    titles = examples["title"]
    texts  = examples["text"]
    outputs = []
    for title, text in zip(titles, texts):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = wikipedia_prompt.format(title, text) + EOS_TOKEN
        outputs.append(text)
    return { "text" : outputs, }
pass

from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,

    args = UnslothTrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,

        # Use warmup_ratio and num_train_epochs for longer runs!
        max_steps = 120,
        warmup_steps = 10,
        # warmup_ratio = 0.1,
        # num_train_epochs = 1,

        # Select a 2 to 10x smaller learning rate for the embedding matrices!
        learning_rate = 5e-5,
        embedding_learning_rate = 1e-5,

        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)
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

# Instruction FineTune - Create an instruction_pompt based on the concept_examples.csv file

import json
import ast
import logging


# read the file /Users/rafatsiddiqui/Downloads/oLabs/code/oFlow/Code/Synthetic-Data/concept_examples.csv as a dictionary

import csv

with open('/Users/rafatsiddiqui/Downloads/oLabs/code/oFlow/Code/Synthetic-Data/concept_examples.csv', 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)
    
type(data)


# Configure logging
logging.basicConfig(
    filename='transformation_errors.log',
    filemode='w',
    level=logging.ERROR,
    format='%(levelname)s:%(message)s'
)

# Sample original data
original_data = data

def transform_data(original_data):
    """
    Transforms the original data by expanding 'example_scenario' into separate dictionaries.

    Parameters:
        original_data (list): List of dictionaries with 'concept_name', 'detailed_explanation', and 'example_scenario'.

    Returns:
        new_data (list): Transformed list with one 'example_scenario' per dictionary.
    """
    new_data = []

    for idx, entry in enumerate(original_data, start=1):
        concept_name = entry.get('concept_name', '').strip()
        detailed_explanation = entry.get('detailed_explanation', '').strip()
        example_scenario_str = entry.get('example_scenario', '').strip()

        if not concept_name or not detailed_explanation or not example_scenario_str:
            logging.error(f"Entry {idx} is missing required fields. Skipping.")
            continue

        # Attempt to parse with json.loads
        try:
            example_scenarios = json.loads(example_scenario_str)
            if not isinstance(example_scenarios, list):
                raise ValueError("Parsed 'example_scenario' is not a list.")
        except json.JSONDecodeError:
            # Fallback to ast.literal_eval
            try:
                example_scenarios = ast.literal_eval(example_scenario_str)
                if not isinstance(example_scenarios, list):
                    raise ValueError("Parsed 'example_scenario' is not a list.")
            except (ValueError, SyntaxError) as e:
                logging.error(f"Entry {idx} ('{concept_name}') has invalid 'example_scenario': {e}")
                continue

        # Iterate through each scenario and create a new entry
        for scenario_idx, scenario in enumerate(example_scenarios, start=1):
            if not isinstance(scenario, str):
                logging.error(f"Entry {idx} ('{concept_name}') has non-string scenario at position {scenario_idx}. Skipping this scenario.")
                continue

            new_entry = {
                'concept_name': concept_name,
                'detailed_explanation': detailed_explanation,
                'example_scenario': scenario.strip()
            }
            new_data.append(new_entry)

    return new_data

# Transform the data
transformed_data = transform_data(original_data)

# Optional: Save the transformed data to a JSON file
with open('transformed_data.json', 'w', encoding='utf-8') as f:
    json.dump(transformed_data, f, ensure_ascii=False, indent=4)

print(f"Transformation complete. {len(transformed_data)} entries created.")
print("Check 'transformation_errors.log' for any errors encountered during transformation.")

print(len(transformed_data))


instruction_prompt = """Below is an instruction that describes a concept in the field of psychology, sociology, anthropology, ethnography, or qualitative research or cultural studies. Write a response that appropriately completes the request.

### Instruction: Given the concept and its detailed explanation, provide an example scenario that illustrates the concept.
concept_name: {}
detailed_explanation: {}

### Response:
{}"""


EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

def instruction_prompt_func(examples):
    concept_name = examples["concept_name"]
    detailed_explanation = examples["detailed_explanation"]
    example_scenario = examples["example_scenario"]
    return { "text" : instruction_prompt.format(concept_name, detailed_explanation, example_scenario), }
pass


# convert transformed_data to a huggingface dataset
instruction_dataset = Dataset.from_dict(transformed_data)
instruction_dataset = instruction_dataset.map(instruction_prompt_func, batched = True,)

from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = instruction_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 8,

    args = UnslothTrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,

        # Use num_train_epochs and warmup_ratio for longer runs!
        max_steps = 120,
        warmup_steps = 10,
        # warmup_ratio = 0.1,
        # num_train_epochs = 1,

        # Select a 2 to 10x smaller learning rate for the embedding matrices!
        learning_rate = 5e-5,
        embedding_learning_rate = 1e-5,

        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.00,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)
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


# Inference

instruction_prompt = """Below is an instruction that describes a concept in the field of psychology, sociology, anthropology, ethnography, or qualitative research or cultural studies. Write a response that appropriately completes the request.

### Instruction: Given the concept and its detailed explanation, provide an example scenario that illustrates the concept.
concept_name: {}
detailed_explanation: {}

### Response:
{}"""

FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    instruction_prompt.format(
        "Give an example scenario that illustrates the concept of Hero archetype as described by Jungian psychology.", # instruction
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)

# Text Streaming

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)

# Save the model
model.save_pretrained("qLeap_model_v0") # Local saving
tokenizer.save_pretrained("qLeap_model_v0")
model.push_to_hub("olabs-ai/qLeap_model_v0", token = "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") # Online saving
tokenizer.push_to_hub("olabs-ai/qLeap_model_v0", token = "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") # Online saving


if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# alpaca_prompt = You MUST copy from above!

inputs = tokenizer(
[
    instruction_prompt.format(
        "When trying to understand how nature plays a role in the development of a child's personality, which concept should be considered?",
        "", # output - leave this blank for generation!
    ),
], return_tensors = "pt").to("cuda")


from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                   repetition_penalty = 0.1)
# Merge to 16bit
if False: model.save_pretrained_merged("qLeap_model_v0_16bit", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("olabs-ai/qLeap_model_v0_16bit", tokenizer, save_method = "merged_16bit", token = "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")

# Merge to 4bit
if False: model.save_pretrained_merged("qLeap_model_v0_4bit", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("olabs-ai/qLeap_model_v0_4bit", tokenizer, save_method = "merged_4bit", token = "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")

# Just LoRA adapters
if False: model.save_pretrained_merged("qLeap_model_v0_LoRA", tokenizer, save_method = "lora",)
if False: model.push_to_hub_merged("olabs-ai/qLeap_model_LoRA", tokenizer, save_method = "lora", token = "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")
# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("qLeap_model_v0_8bit_Q8_0", tokenizer,)
if False: model.push_to_hub_gguf("olabs-ai/qLeap_model_v0_8bit_Q8_0", tokenizer, token = "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")

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("qLeap_model_v0_16bit_GGUF", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("olabs-ai/qLeap_model_v0_16bit_GGUF", tokenizer, quantization_method = "f16", token = "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")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("qLeap_model_v0_q4_k_m_16bit", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("olabs-ai/qLeap_model_v0_q4_k_m_16bit", tokenizer, quantization_method = "q4_k_m", token = "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")
if False: model.push_to_hub_gguf("olabs-ai/qLeap_model_v0_q5_k_m_16bit", tokenizer, quantization_method = "q5_k_m", token = "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")
