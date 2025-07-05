# Step 4: Import the librairies
from enum import Enum
from functools import partial
import pandas as pd
import torch
import json
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType

def print_model_named_parameters_device(model):
    for i in model.named_parameters():
        print(f"named_parameter: {i[0]} -> {i[1].device}")

def preprocess(sample):
      messages = sample["messages"]
      first_message = messages[0]

      # Instead of adding a system message, we merge the content into the first user message
      if first_message["role"] == "system":
          system_message_content = first_message["content"]
          # Merge system content with the first user message
          messages[1]["content"] = (
                system_message_content +
                "Also, before making a call to a function take the time to plan the function to take. "
                "Make that thinking process between <think>{your thoughts}</think>\n\n"
                 + messages[1]["content"]
          )
          # Remove the system message from the conversation
          messages.pop(0)

      return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

seed = 42
set_seed(seed)

chat_template = ( #Jinja2 template
    "{{ bos_token }}"
    "{% if messages[0]['role'] == 'system' %}"
        "{{ raise_exception('System role not supported') }}"
    "{% endif %}"
    "{% for message in messages %}"
        "{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] | trim + '<end_of_turn><eos>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
        "{{'<start_of_turn>model\n'}}"
    "{% endif %}"
)

# Step 5: Processing the dataset into inputs
model_name = "google/gemma-2-2b-it"
dataset_name = "Jofthomas/hermes-function-calling-thinking-V1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.chat_template = chat_template

dataset = load_dataset(dataset_name)
dataset = dataset.rename_column("conversations", "messages")
# Step 6: A Dedicated Dataset for This Unit
dataset = dataset.map(preprocess, remove_columns="messages")
dataset = dataset["train"].train_test_split(0.1)
print(dataset)
# Step 7: Checking the inputs
print(dataset["train"][8]["text"])
# Sanity check
print(tokenizer.pad_token)
print(tokenizer.eos_token)

# Step 8: Let's Modify the Tokenizer
class ChatmlSpecialTokens(str, Enum):
    tools = "<tools>"
    eotools = "</tools>"
    think = "<think>"
    eothink = "</think>"
    tool_call="<tool_call>"
    eotool_call="</tool_call>"
    tool_response="<tool_reponse>"
    eotool_response="</tool_reponse>"
    pad_token = "<pad>"
    eos_token = "<eos>"
    @classmethod
    def list(cls):
        return [c.value for c in cls]

tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        pad_token=ChatmlSpecialTokens.pad_token.value,
        additional_special_tokens=ChatmlSpecialTokens.list()
    )
tokenizer.chat_template = chat_template

model = AutoModelForCausalLM.from_pretrained(
    model_name
    , attn_implementation='eager'
    , device_map="cuda:0"
)
model.config.use_cache = False # Fixes "HybridCache object is not iterable" error
model.resize_token_embeddings(len(tokenizer))
model.to(torch.bfloat16)
print('============ original model layers placement =============')
print_model_named_parameters_device(model)

# Step 9: Let's configure the LoRA
# TODO: Configure LoRA parameters
# r: rank dimension for LoRA update matrices (smaller = more compression)
rank_dimension = 16
# lora_alpha: scaling factor for LoRA layers (higher = stronger adaptation)
lora_alpha = 64
# lora_dropout: dropout probability for LoRA layers (helps prevent overfitting)
lora_dropout = 0.05

peft_config = LoraConfig(
    r=rank_dimension,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    # wich layer in the transformers do we target ?
    target_modules=["gate_proj","q_proj","lm_head","o_proj","k_proj","embed_tokens","down_proj","up_proj","v_proj"],
    task_type=TaskType.CAUSAL_LM)

# Step 10: Let's define the Trainer and the Fine-Tuning hyperparameters
username="uvv001"# REPLCAE with your Hugging Face username
output_dir = "gemma-2-2B-it-thinking-function_calling-V0" # The directory where the trained model checkpoints, logs, and other artifacts will be saved. It will also be the default name of the model when pushed to the hub if not redefined later.
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
gradient_accumulation_steps = 4
logging_steps = 5
learning_rate = 1e-4 # The initial learning rate for the optimizer.

max_grad_norm = 1.0
num_train_epochs=1
warmup_ratio = 0.1
lr_scheduler_type = "cosine"
max_seq_length = 1500

training_arguments = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    save_strategy="no",
    eval_strategy="epoch",
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    max_grad_norm=max_grad_norm,
    weight_decay=0.1,
    warmup_ratio=warmup_ratio,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
    bf16=True,
    hub_private_repo=False,
    push_to_hub=False,
    num_train_epochs=num_train_epochs,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    packing=True,
    max_seq_length=max_seq_length,
)

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset["train"], #.select(range(10)),
    eval_dataset=dataset["test"], #.select(range(2)),
    processing_class=tokenizer,
    peft_config=peft_config,
)

trainer.train()
trainer.save_model()
print("Model has been saved successfuly")

# Step 11: Let's push the Model and the Tokenizer to the Hub
trainer.push_to_hub(f"{username}/{output_dir}")

# Since we also modified the chat_template Which is contained in the tokenizer, let's also push the tokenizer with the model.
tokenizer.eos_token = "<eos>"
# push the tokenizer to hub ( replace with your username and your previously specified
tokenizer.push_to_hub(f"{username}/{output_dir}", token=True)

