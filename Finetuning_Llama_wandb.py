import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
    GenerationConfig,
    TrainingArguments, 
    Trainer,
    pipeline,
    utils
)
import sys
utils.logging.set_verbosity_error

#wandb setup
import wandb
wandb.login()

# sweep_config = {
#     'method': 'random'
#     }

# metric = {
#     'name': 'loss',
#     'goal': 'minimize'   
#     }

# sweep_config['metric'] = metric

# parameters_dict = {
#     'learning_rate': {
#         'distribution': 'uniform',
#         'min': 1e-5,
#         'max': 1e-2
#       },
#     'lora_dropout': {
#           'values': [0.05, 0.1, 0.15]
#         },
#     }
# sweep_config['parameters'] = parameters_dict

sweep_config = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        "lora_dropout": {"values": [0.05, 0.075, 0.1, 0.125, 0.15]},
        "learning_rate": {"max": 0.01, "min": 0.00001},
        "lora_rank": {"values": [2, 4, 8, 16, 32]},
        "lora_alpha": {"values": [16, 32]}
    },
}
sweep_id = wandb.sweep(sweep_config, project="textmetrics")

device = "cuda:0"
model_name = "/home/leahheil/TextMetrics/llama/weights"
peft_model_path="/home/leahheil/TextMetrics/peft_output/finetuned_model"

def load_data():
    df_train = pd.read_csv("/home/leahheil/TextMetrics/dataset/dataset-annotated-train.csv")
    df_val = pd.read_csv("/home/leahheil/TextMetrics/dataset/dataset-annotated-val.csv")
    df_test = pd.read_csv("/home/leahheil/TextMetrics/dataset/dataset-annotated-test.csv")

    # drop indices and turn into dataset format
    data_train = df_train.reset_index(drop=True)
    data_test = df_test.reset_index(drop=True)
    data_val = df_val.reset_index(drop=True)
    data_train = Dataset.from_pandas(data_train)
    data_test = Dataset.from_pandas(data_test)
    data_val = Dataset.from_pandas(data_val)

    # create the datasetdict object
    data = DatasetDict()
    data['train'] = data_train
    data['test'] = data_test
    data['val'] = data_val

    return data


def tokenize_function(example, tokenizer):
    prompt = [p for p in example["prompt"]]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt", max_length=2048).input_ids
    example['labels'] = tokenizer(example["text"], padding="max_length", truncation=True, return_tensors="pt", max_length=2048).input_ids
    return example


def generate(prompt, tokenizer, model):
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.float16, device_map="auto")
    job_ad = pipe(prompt, 
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=4096,)
    return job_ad


def finetune_model(model, dataset, config):
    peft_training_args = TrainingArguments(
        report_to = 'wandb',                     # enable logging to W&B
        run_name = 'textmetrics',            # name of the W&B run
        output_dir="/home/leahheil/TextMetrics/peft_output",
        evaluation_strategy="steps",
        save_strategy="epoch",
        learning_rate=config.learning_rate, # Higher learning rate than full fine-tuning.
        # num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        group_by_length=True,
        fp16=False,
        bf16=True,
        eval_accumulation_steps=1,
        gradient_accumulation_steps=1,
        max_grad_norm=0.3,
        eval_steps=50,
        logging_steps=50,
        max_steps=800,
        warmup_steps=30,
    )

    # Set up peft trainer
    peft_trainer = Trainer(
        model=model,
        args=peft_training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
    )

    peft_trainer.train()
    peft_trainer.model.save_pretrained(peft_model_path)

def hyper_param_search():
    run = wandb.init()
    # wandbconfig = wandb.config

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 

    print("Loading data")
    data = load_data()
    dataset = data.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    dataset = dataset.remove_columns(['prompt', 'text'])

    # Load LLaMA model
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = False

    # Load LoRA configuration
    lora_config = LoraConfig(
        r = wandb.config.lora_rank, # Rank
        lora_alpha = wandb.config.lora_alpha,
        lora_dropout = wandb.config.lora_dropout,
        bias = "none",
        task_type = "CAUSAL_LM",
        target_modules = ["q_proj","v_proj"]
    )

    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=2)
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    finetune_model(peft_model, dataset, wandb.config)

    trained_peft_model = PeftModel.from_pretrained(peft_model, peft_model_path)
    original_job_ads = []
    trained_job_ads = []

    for i in range(3):
        original_job_ad = generate(data["val"][i]["prompt"], tokenizer, model)
        original_job_ads.append(original_job_ad)
        trained_job_ad = generate(data["val"][i]["prompt"], tokenizer, trained_peft_model)
        trained_job_ads.append(trained_job_ad)
    
    baseline_job_ads = data["val"][0:3]["text"]
    zipped_summaries = list(zip(baseline_job_ads, original_job_ads, trained_job_ads))
    df = pd.DataFrame(zipped_summaries, columns = ['human_baseline', 'original_model', 'peft_model'])
    # df.to_csv("/home/leahheil/TextMetrics/output_job_ads/job_ads_run_"+str(wandb.config.learning_rate)+"_"+str(wandb.config.lora_dropout))
    
    eval_examples = wandb.Table(dataframe=df)
    run.log({"table_key": eval_examples})

    # print(original_job_ads)
    # print(trained_job_ads)


def main():
    wandb.agent(sweep_id, function=hyper_param_search, count=20)
    wandb.finish()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Stopping the program")
        sys.exit(0)