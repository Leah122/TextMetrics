import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict

import argparse
import torch
from peft import PeftModel, PeftConfig
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    GenerationConfig,
    utils,
    pipeline,
)
import sys
import evaluate
utils.logging.set_verbosity_error
from tqdm import tqdm
from datetime import datetime


device = "cuda:0"
model_name = "NousResearch/Llama-2-7b-chat-hf"
peft_model_path="/home/leahheil/TextMetrics/chat_output/finetuned_model"

def add_token(prompt):
    return f"<s>[INST] {prompt} [/INST]"

def load_data():
    # df_train = pd.read_csv("/home/leahheil/TextMetrics/dataset/dataset_annotated_train_en.csv")
    df_val = pd.read_csv("/home/leahheil/TextMetrics/dataset/dataset_annotated_val_en.csv")
    df_test = pd.read_csv("/home/leahheil/TextMetrics/dataset/dataset_annotated_test_en.csv")

    # remove dutch prompts and texts
    # df_train = df_train.drop(columns=['prompt', 'text'])
    df_test = df_test.drop(columns=['prompt', 'text'])
    df_val = df_val.drop(columns=['prompt', 'text'])
    # df_train.rename(columns={'prompt_en': 'prompt', 'text_en': 'text'}, inplace=True)
    df_test.rename(columns={'prompt_en': 'prompt', 'text_en': 'text'}, inplace=True)
    df_val.rename(columns={'prompt_en': 'prompt', 'text_en': 'text'}, inplace=True)

    # df_train['prompt'] = df_train['prompt'].apply(add_token)
    df_val['prompt'] = df_val['prompt'].apply(add_token)
    df_test['prompt'] = df_test['prompt'].apply(add_token)

    # drop indices and turn into dataset format
    # data_train = df_train.reset_index(drop=True)
    data_test = df_test.reset_index(drop=True)
    data_val = df_val.reset_index(drop=True)
    # data_train = Dataset.from_pandas(data_train)
    data_test = Dataset.from_pandas(data_test)
    data_val = Dataset.from_pandas(data_val)

    # create the datasetdict object
    data = DatasetDict()
    # data['train'] = data_train
    data['test'] = data_test
    data['val'] = data_val

    return data


def tokenize_function(example, tokenizer):
    prompt = [p for p in example["prompt"]]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt", max_length=2048).input_ids
    example['labels'] = tokenizer(example["text"], padding="max_length", truncation=True, return_tensors="pt", max_length=2048).input_ids
    return example


#TODO: generation in 8 or maybe even 4 bits, will probably make it a lot faster with minimal accuracy loss.
# generation with pipeline did not work for some reason, so I switched to tokenizing, generating and decoding by hand.
# this does mean that I cannot do it in batches, meaning that not a lot of GPU is used.
# the bigger top_k is the longer it takes, 5 with max_length=2000 takes around 2 minutes per generated text if loaded in 8 bit.
def generate(prompt, tokenizer, model, length):
    prompt = f"<s>[INST] {prompt} [/INST]"
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=2000)
    generated_text = pipe(prompt)
    del pipe
    return generated_text


def calculate_rouge(data, tokenizer, base_model, num_samples, len_samples, file_name):
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    prompts = data['val'][:num_samples]['prompt']
    baseline_job_ads = data['val'][:num_samples]['text']
    generated_job_ads = []

    # generate texts
    for i in tqdm(range(len(baseline_job_ads))):
        generated_job_ad = generate(prompts[i], tokenizer, base_model, len_samples)
        generated_job_ads.append(generated_job_ad[0]['generated_text'])

    # calculate scores
    rouge_score = rouge.compute(
        predictions=generated_job_ads,
        references=baseline_job_ads,
        use_aggregator=True,
        use_stemmer=True,
    )
    bleu_score = bleu.compute(predictions=generated_job_ads, references=baseline_job_ads)

    print('rouge/bleu:')
    print(rouge_score, " / ", bleu_score)

    # save job ads and scores to a file
    if file_name is not None:
        df = pd.DataFrame(prompts, columns=['prompt'])
        df['generated_text'] = generated_job_ads
        df.to_csv("/home/leahheil/TextMetrics/generated_texts/" + str(file_name))


def main(nr_samples, len_samples, file_name, ft_model):
    config = PeftConfig.from_pretrained(peft_model_path)
    
    # Load LLaMA tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_name, device_map="auto")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    # Load LLaMA model
    model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True) #quantization_config=bnb_config
    # model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    print("Loading data")
    data = load_data()
    dataset = data.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    dataset = dataset.remove_columns(['prompt', 'text'])

    # trained_peft_model = PeftModel.from_pretrained(model, peft_model_path)

    # print("Generating Job Ad with original model")
    # prompt = "Write a job description for a DevOps Engineer. Include the following aspects:\n- Job description: As a DevOps engineer, you will contribute to the realization and further development of innovative solutions for public transportation in the Netherlands. You will ensure that requirements are translated into a working product and contribute to the automated deployment of the system. Additionally, you will be responsible for monitoring applications and related tooling. You will work both at CGI Rotterdam and at client locations in Utrecht.\n- Required skills: HBO or WO degree with an IT component, minimum of 3 years of experience in a DevOps role, knowledge and experience with Linux, GIT, Docker, and scripting programming languages, knowledge of Check MK and Apache Nifi is a plus, Agile mindset, strong communication skills in Dutch, ability to prioritize and meet deadlines.\n- Job benefits: Permanent contract, possibility of part-time work, ownership by becoming a shareholder, opportunity for personal development through CGI Academy, work in a close-knit team of professionals.\n- Target audience: IT professionals with experience in DevOps roles.\n- Pull factors: Working on innovative solutions for public transportation, direct impact on society, possibility of ownership, flexibility in working hours, opportunity for personal development."
    # pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=2000)
    # result = pipe(f"<s>[INST] {prompt} [/INST]")
    # print(result[0]['generated_text'])

    # trained_model = PeftModel.from_pretrained(model, peft_model_path)

    # print("Generating Job Ad with finetuned model")
    # prompt = "Write a job description for a DevOps Engineer. Include the following aspects:\n- Job description: As a DevOps engineer, you will contribute to the realization and further development of innovative solutions for public transportation in the Netherlands. You will ensure that requirements are translated into a working product and contribute to the automated deployment of the system. Additionally, you will be responsible for monitoring applications and related tooling. You will work both at CGI Rotterdam and at client locations in Utrecht.\n- Required skills: HBO or WO degree with an IT component, minimum of 3 years of experience in a DevOps role, knowledge and experience with Linux, GIT, Docker, and scripting programming languages, knowledge of Check MK and Apache Nifi is a plus, Agile mindset, strong communication skills in Dutch, ability to prioritize and meet deadlines.\n- Job benefits: Permanent contract, possibility of part-time work, ownership by becoming a shareholder, opportunity for personal development through CGI Academy, work in a close-knit team of professionals.\n- Target audience: IT professionals with experience in DevOps roles.\n- Pull factors: Working on innovative solutions for public transportation, direct impact on society, possibility of ownership, flexibility in working hours, opportunity for personal development."
    # pipe = pipeline(task="text-generation", model=trained_model, tokenizer=tokenizer, max_length=2000)
    # result = pipe(f"<s>[INST] {prompt} [/INST]")
    # print(result[0]['generated_text'])

    if ft_model:
        trained_peft_model = PeftModel.from_pretrained(model, peft_model_path)
        calculate_rouge(data, tokenizer, trained_peft_model, nr_samples, len_samples, file_name)
    else:
        calculate_rouge(data, tokenizer, model, nr_samples, len_samples, file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('-n', '--number_samples', type=int, help='number of samples to generate', default=50)
    parser.add_argument('-l', '--len_samples', type=int, help='length of the generated samples', default=2000)
    parser.add_argument('-f', '--file_name', help='name of the output file for the generated texts')
    parser.add_argument('--use_finetuned_model', help="wether or not to use the finetuned model, defaults to False", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    print(args)

    try:
        main(args.number_samples, args.len_samples, args.file_name, args.use_finetuned_model)
    except KeyboardInterrupt:
        print("Stopping the program")
        sys.exit(0)


