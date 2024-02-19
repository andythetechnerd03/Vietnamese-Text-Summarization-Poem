import yaml

new_model = "pphuc25/poem-vistral"
yaml_string = """
base_model: Viet-Mistral/Vistral-7B-Chat
model_type: MistralForCausalLM
tokenizer_type: AutoTokenizer
is_mistral_derived_model: true

load_in_8bit: false
load_in_4bit: true
strict: false

datasets:
  - path: pphuc25/poem-5-words-vietnamese
    type:
      system_prompt: "Bạn là một nhà thơ chuyên nghiệp, nhiệm vụ của bạn là chuyển bài văn này thành 1 bài thơ 5 chữ từ khoảng 1 đến 3 khổ"
      field_system: system
      field_instruction: input
      field_output: output
      format: "[INST] <<SYS>>\n{instruction}\n<</SYS>>\n\n{input} [/INST] "

dataset_prepared_path:
val_set_size: 0.05
output_dir: ./qlora-out

adapter: qlora
lora_model_dir:

sequence_len: 1096
sample_packing: true
pad_to_sequence_len: true

lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules:
lora_target_linear: true
lora_fan_in_fan_out:

wandb_project:
wandb_entity:
wandb_watch:
wandb_name:
wandb_log_model:

mlflow_experiment_name: colab-example

gradient_accumulation_steps: 4
micro_batch_size: 16
num_epochs: 5
max_steps: 20
optimizer: paged_adamw_32bit
lr_scheduler: cosine
learning_rate: 0.0002

train_on_inputs: false
group_by_length: false
bf16: true
fp16: false
tf32: false

gradient_checkpointing: true
early_stopping_patience:
resume_from_checkpoint:
local_rank:
logging_steps: 1
xformers_attention:
flash_attention: false

warmup_steps: 10
evals_per_epoch:
saves_per_epoch:
debug:
deepspeed:
weight_decay: 0.0
fsdp:
fsdp_config:
special_tokens:

"""

# Convert the YAML string to a Python dictionary
yaml_dict = yaml.safe_load(yaml_string)

# Specify your file path
yaml_file = 'config.yaml'

# Write the YAML file
with open(yaml_file, 'w') as file:
    yaml.dump(yaml_dict, file)