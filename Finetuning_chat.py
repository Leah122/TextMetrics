import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
import argparse

from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    GenerationConfig,
    TrainingArguments, 
    Trainer,
    pipeline,
    utils
)
import sys
import evaluate
utils.logging.set_verbosity_error
from tqdm import tqdm
from sklearn.model_selection import train_test_split

device = "cuda:0"
model_name = "NousResearch/Llama-2-7b-chat-hf"
peft_model_path="/home/leahheil/TextMetrics/chat_output/finetuned_model"

def add_token(prompt):
    return f"<s>[INST] {prompt} [/INST]"

def load_data():
    df_train = pd.read_csv("/home/leahheil/TextMetrics/dataset/dataset_annotated_train_en.csv")
    df_val = pd.read_csv("/home/leahheil/TextMetrics/dataset/dataset_annotated_val_en.csv")
    df_test = pd.read_csv("/home/leahheil/TextMetrics/dataset/dataset_annotated_test_en.csv")

    # remove dutch prompts and texts
    df_train = df_train.drop(columns=['prompt', 'text'])
    df_test = df_test.drop(columns=['prompt', 'text'])
    df_val = df_val.drop(columns=['prompt', 'text'])
    df_train.rename(columns={'prompt_en': 'prompt', 'text_en': 'text'}, inplace=True)
    df_test.rename(columns={'prompt_en': 'prompt', 'text_en': 'text'}, inplace=True)
    df_val.rename(columns={'prompt_en': 'prompt', 'text_en': 'text'}, inplace=True)

    df_train['prompt'] = df_train['prompt'].apply(add_token)
    df_val['prompt'] = df_val['prompt'].apply(add_token)
    df_test['prompt'] = df_test['prompt'].apply(add_token)

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


def load_new_data():
    df = pd.read_csv("/home/leahheil/TextMetrics/dataset/dataset_extra_data.csv")
    df['prompt'] = df['prompt'].apply(add_token)
    df = df.reset_index(drop=True)

    df_train, df_test = train_test_split(test_size=0.3, random_state=2)
    df_test, df_val = train_test_split(test_size=0.5, random_state=2)

    data_train = Dataset.from_pandas(df_train)
    data_test = Dataset.from_pandas(df_test)
    data_val = Dataset.from_pandas(df_val)

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


#TODO: generation in 8 or maybe even 4 bits, will probably make it a lot faster with minimal accuracy loss.
def generate(prompt, tokenizer, model, length):
    input_ids = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt", max_length=2048).input_ids.to(device)
    generation_config = GenerationConfig(max_new_tokens=length, do_sample=True, temperature=1, top_k=8, pad_token_id=tokenizer.eos_token_id)
    response_token_ids = model.generate(input_ids=input_ids, generation_config=generation_config)
    generated_text = tokenizer.decode(response_token_ids[0], skip_special_tokens=True)
    # pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.float16, device_map="auto")
    # job_ad = pipe(prompt,
    #     do_sample=True,
    #     top_k=10,
    #     num_return_sequences=1,
    #     pad_token_id=0,
    #     max_length=2048,)
    return generated_text


def finetune_model(model, tokenizer, dataset, steps):
    # generation_config = GenerationConfig(max_new_tokens=2048, do_sample=True, top_k=5, pad_token_id=0)

    peft_training_args = TrainingArguments(
        output_dir="/home/leahheil/TextMetrics/chat_output",
        overwrite_output_dir=True,
        load_best_model_at_end=True,
        greater_is_better=False,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=100,
        learning_rate=1.5e-5, # Sources are really divided on optimal learning rate, some say around 3e-4, but others advice not higher than 1e-5.
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        group_by_length=True,
        fp16=False, # TODO: True gives no errors, just changed to false to check
        bf16=False,
        eval_accumulation_steps=1,
        gradient_accumulation_steps=1,
        max_grad_norm=0.3,
        eval_steps=50,
        logging_steps=50,
        max_steps=steps,
        warmup_steps=0,
        logging_first_step=True,
        # generation_config=generation_config,
    )

    # Set up peft trainer
    peft_trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=peft_training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
    )

    peft_trainer.train()

    peft_trainer.model.save_pretrained(peft_model_path)


def calculate_rouge(data, tokenizer, base_model, trained_model, nr_samples, len_samples=2000):
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    baseline_job_ads = data['val'][:nr_samples]['text']
    original_job_ads = []
    trained_job_ads = []

    for i in tqdm(range(len(baseline_job_ads))):
        original_job_ad = generate(data["val"][i]["prompt"], tokenizer, base_model, len_samples)
        original_job_ads.append(original_job_ad)
        trained_job_ad = generate(data["val"][i]["prompt"], tokenizer, trained_model, len_samples)
        trained_job_ads.append(trained_job_ad)

    print(original_job_ads)
    print(trained_job_ads)

    original_model_results_rouge = rouge.compute(
        predictions=original_job_ads,
        references=baseline_job_ads,
        use_aggregator=True,
        use_stemmer=True,
    )

    peft_model_results_rouge = rouge.compute(
        predictions=trained_job_ads,
        references=baseline_job_ads,
        use_aggregator=True,
        use_stemmer=True,
    )

    original_model_results_bleu = bleu.compute(predictions=original_job_ads, references=baseline_job_ads)
    peft_model_results_bleu = bleu.compute(predictions=trained_job_ads, references=baseline_job_ads)

    print("GENERATED TEXTS:")
    print(original_job_ads)
    print(trained_job_ads)

    print('ORIGINAL MODEL rouge/bleu:')
    print(original_model_results_rouge, " / ", original_model_results_bleu)
    print('TRAINED MODEL rouge/bleu:')
    print(peft_model_results_rouge, " / ", peft_model_results_bleu)

    improvement_rouge = (np.array(list(peft_model_results_rouge.values())) - np.array(list(original_model_results_rouge.values())))
    for key, value in zip(peft_model_results_rouge.keys(), improvement_rouge):
        print(f'{key}: {value*100:.2f}%')


def main(samples=False, nr_samples=1, len_samples=2000, steps=500):
    print("Loading tokenizer")
    # Load LLaMA tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    print("Loading model")
    # Load LLaMA model
    model = LlamaForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
    model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    print("Loading data")
    data = load_data()
    dataset = data.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    dataset = dataset.remove_columns(['prompt', 'text'])
    print("Dataset shapes: ")
    print(dataset)

    # Load LoRA configuration
    lora_config = LoraConfig(
        r=4, # Rank
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ["q_proj","v_proj"]
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    print("Start training")
    finetune_model(peft_model, tokenizer, dataset, steps)

    # load trained model and generate some job ads
    trained_peft_model = PeftModel.from_pretrained(peft_model, peft_model_path)

    if samples:
        print("Calculating Rouge with ", nr_samples, " samples of length ", len_samples)
        calculate_rouge(data, tokenizer, model, trained_peft_model, nr_samples, len_samples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--samples', help='if it should generate samples and calculate scores', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-n', '--number_samples', type=int, help='number of samples to generate', default=2)
    parser.add_argument('-l', '--len_samples', type=int, help='length of the generated samples', default=2000)
    parser.add_argument('-i', '--num_iterations', type=int, help='number of steps for training', default=800)
    args = parser.parse_args()
    print(args)

    try:
        main(args.samples, args.number_samples, args.len_samples, args.num_iterations)
    except KeyboardInterrupt:
        print("Stopping the program")
        sys.exit(0)