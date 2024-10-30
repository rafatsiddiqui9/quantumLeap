# Continued Pretraining

# %%capture
# !pip install unsloth
# # Also get the latest nightly Unsloth!
# !pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
import os


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

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name =  "unsloth/Llama-3.2-1B-Instruct-bnb-4bit", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token =os.getenv("HUGGINGFACE_TOKEN"), # use one if using gated models like meta-llama/Llama-2-7b-hf
)

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

# Wikipedia provides a title and an article text.
# Use https://translate.google.com!
_wikipedia_prompt = """Wikipedia Article
### Title: {}

### Article:
{}"""
# becomes:
wikipedia_prompt = """विकिपीडिया लेख
### शीर्षक: {}

### आलेख:{}
# """

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

from datasets import load_dataset

dataset = load_dataset("wikimedia/wikipedia", "20231101.hi", split = "train",)

# We select 1% of the data to make training faster!
dataset = dataset.train_test_split(train_size = 0.01)["train"]

dataset = dataset.map(formatting_prompts_func, batched = True,)

# Filter out rows longer than 10000 characters
dataset = dataset.filter(lambda x: len(x['text']) <= 1000)
# Show first example
dataset['text'][0]

# Find the maximum length of the text field in the entire dataset
max_length = max(len(text) for text in dataset['text'])
print(f"The maximum length of the text field in the dataset is: {max_length} characters")

from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
import wandb
# Create the run name with the current date and time
from datetime import datetime
import pytz
import wandb

# Get the current date and time in Indian Standard Time (IST)
ist = pytz.timezone('Asia/Kolkata')
current_datetime = datetime.now(ist)

# Format the datetime string
# Example format: 20240428_153045 (YYYYMMDD_HHMMSS)
formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")
run_name = f"""Unsloth-CPT-Base-Alpaca-Hindi-{formatted_datetime}"""

# Initialize Weights & Biases
# It's recommended to set your W&B API key as an environment variable for security.
# Example: export WANDB_API_KEY="your_api_key"
wandb.login(key=os.getenv("WANDB_API_KEY"))  # Consider using environment variables for security
wandb.init(project="Unsloth-CPT-Hindi", name=run_name)
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
        report_to=["tensorboard", "wandb"],
        logging_dir=f"./trel-fft-logs/{run_name}",
    ),
)

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

#Instruction Finetuning

from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
import os


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name =  "lora_model", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token =os.getenv("HUGGINGFACE_TOKEN"), # use one if using gated models like meta-llama/Llama-2-7b-hf
)

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

from datasets import load_dataset
alpaca_dataset = load_dataset("BhabhaAI/alpaca-gpt4-hindi-trans", split = "train")
print(alpaca_dataset[0])
alpaca_dataset

alpaca_prompt = """निम्नलिखित एक आदेश है जो कार्य का वर्णन करता है। एक प्रतिक्रिया लिखें जो अनुरोध को उचित रूप से पूरा करती है।

### दिशानिर्देश:
{}

### प्रतिक्रिया:
{}"""

def formatting_prompts_func(examples):
    """
    Format the examples according to the prompt template.
    This function handles batched processing of multiple examples.
    """
    texts = []
    for instruction, input_text, output in zip(examples['instruction'], examples['input'], examples['output']):
        # Combine instruction and input if input is not empty
        full_instruction = f"{instruction}\n{input_text}" if input_text else instruction
        
        # Format the text according to the template and add EOS token
        text = alpaca_prompt.format(full_instruction, output) + tokenizer.eos_token
        texts.append(text)
    
    return {"text": texts}

# Apply the formatting function to the dataset
formatted_dataset = alpaca_dataset.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=alpaca_dataset.column_names  # Remove original columns
)

# Filter out rows longer than 10000 characters (before formatting)
def get_combined_length(example):
    # Combine instruction, input, and output fields
    instruction_length = len(example['instruction'])
    input_length = len(example['input']) if example['input'] else 0
    output_length = len(example['output'])
    return instruction_length + input_length + output_length

# First, let's see the distribution of lengths before filtering
original_max_length = max(get_combined_length(example) for example in alpaca_dataset)
print(f"Original maximum combined length: {original_max_length} characters")

# Filter the dataset
filtered_alpaca_dataset = alpaca_dataset.filter(lambda x: get_combined_length(x) <= 1000)
print(f"Original dataset size: {len(alpaca_dataset)}")
print(f"Filtered dataset size: {len(filtered_alpaca_dataset)}")

# Now format the filtered dataset
formatted_dataset = filtered_alpaca_dataset.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=filtered_alpaca_dataset.column_names
)

# Show first example
print("\nFirst formatted example:")
print(formatted_dataset['text'][0])

# Find the maximum length of the formatted text field
max_formatted_length = max(len(text) for text in formatted_dataset['text'])
print(f"\nThe maximum length of the formatted text field in the dataset is: {max_formatted_length} characters")

from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = filtered_alpaca_dataset,
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

model.save_pretrained("lora_model_instruct") # Local saving
tokenizer.save_pretrained("lora_model_instruct")
if True:
    model.push_to_hub("olabs-ai/unsloth-cpt-hindi-v01", token = os.getenv("HUGGINGFACE_TOKEN")) # Online saving
    tokenizer.push_to_hub("olabs-ai/unsloth-cpt-hindi-v01", token = os.getenv("HUGGINGFACE_TOKEN")) # Online saving
    model.push_to_hub_gguf("olabs-ai/unsloth-cpt-hindi-v01", tokenizer, quantization_method = "q4_k_m", token = os.getenv("HUGGINGFACE_TOKEN"))
    
    
# alpaca_prompt = Copied from above
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    alpaca_prompt.format(
        # "Continue the fibonacci sequence: 1, 1, 2, 3, 5, 8,", # instruction
            "फाइबोनैचि अनुक्रम जारी रखें: 1, 1, 2, 3, 5, 8,", # निर्देश        
            "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)

# alpaca_prompt = Copied from above
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    alpaca_prompt.format(
        # "What is Korean music like?"
        "कोरियाई संगीत कैसा है?", # निर्देश
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 2048)

if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model_instruct", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# alpaca_prompt = You MUST copy from above!

inputs = tokenizer(
[
    alpaca_prompt.format(
        # "Describe the planet Earth extensively.", # instruction
        "पृथ्वी का विस्तृत वर्णन करें।",
        "", # output - leave this blank for generation!
    ),
], return_tensors = "pt").to("cuda")


from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                   repetition_penalty = 0.1)