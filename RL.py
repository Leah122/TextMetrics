# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, GenerationConfig
from datasets import load_dataset
from peft import PeftModel, PeftConfig, LoraConfig, TaskType, get_peft_model

import sys
# trl: Transformer Reinforcement Learning library
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from trl import create_reference_model
from trl.core import LengthSampler

import torch
# import evaluate
import requests
import argparse

import numpy as np
import pandas as pd

# from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_dataset

# tqdm library makes the loops show a smart progress meter.
from tqdm import tqdm
tqdm.pandas()
device = 0 if torch.cuda.is_available() else "cpu"

from transformers import (
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer
)

from tqdm import tqdm

model_name = "/home/leahheil/TextMetrics/llama/weights"
peft_model_path = "/home/leahheil/TextMetrics/peft_output/finetuned_model"

def value_to_cefr_level(x):
    if x == 0:
        return "A1"
    elif x == 1:
        return "A2"
    elif x == 2: 
        return "B1"
    elif x == 3: 
        return "B2"
    elif x == 4: 
        return "C1"
    elif x == 5: 
        return "C2"
    elif x == 6: 
        return "unknown"
        

def cefr_level_to_value(x):
    if x == "A1" or x == "A1+":
        return 0
    elif x == "A2" or x == "A2+":
        return 1
    elif x == "B1" or x == "B1+":
        return 2
    elif x == "B2" or x == "B2+":
        return 3
    elif x == "C1" or x == "C1+":
        return 4
    elif x == "C2" or x == "C2+":
        return 5
    else:
        return 6

def get_textmetrics_analysis(content, language):
    base_url = "https://api.textmetrics.com/"
    key = "Bearer ZQcKRxAyxmSx8jySKrLAS5ir-WJsQTEUIooqbqn5yy2kSCTV47c6Gbo4Vb6aqdGWMNE4kep8kq0L1jau2c1KS3-cs-Xv5QGAtLZBttyU3nQmzf8waJfs_tY61nTuaX6KePa4UyEyDDiM_gqqO9fZy70dmjVTpTFEv86me-9CbWBbM3RxYY9Yp-j_VAPB3On4FLmYGcaQdekj-gL71i7NVY3yDYexieVCrVMUGj-W_4qX1azA-i9U-NlHBXitQ6_GJBP94OBZJOeCqsc99bbGUA"

    display_language = 'en'
    request_entries = 3

    url = base_url + "contentquality/suggestions/"

    qualityLevels = {"ReadingLevel": 2, "DifficultWordsLevel": 2, "LongSentencesLevel": 2, "AdjectivesLevel": 1,
                     "WhitespacesLevel": 1, "BulletPointsLevel": 1, "ImagesLevel": 1, "GenderLevel": "n",
                     "SentimentLevel": "positive", "GenderList": 1, "JargonList": 1, "SingularPluralLevel": 1,
                     "RepetitionLevel": 1, "TextLengthRuleLevel": 1}

    # defining a params dict for the parameters to be sent to the API
    PARAMS = {"Content": content, "LanguageCode": display_language, "RuleSet": 183,
              "ContentLanguageCode": language, "QualityLevels": qualityLevels}

    i = 0
    r = requests.post(url=url, json=PARAMS, headers={"Authorization": key})
    while i < request_entries and r.status_code != 200:
        r = requests.post(url=url, json=PARAMS, headers={"Authorization": key})
        i = i + 1

    # extracting data in json format
    data = r.json()

    if data['Details']['ReadingLevel'] is None:
        return "unkown"
    else:
        return data['Details']['ReadingLevel']
    

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


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


#TODO: generation in 8 or maybe even 4 bits, will probably make it a lot faster with minimal accuracy loss.
def generate(prompt, tokenizer, model, length):
    input_ids = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt", max_length=2048).input_ids.to(device)
    generation_config = GenerationConfig(max_new_tokens=length, do_sample=True, temperature=1, top_k=8, pad_token_id=tokenizer.eos_token_id)
    response_token_ids = model.generate(input_ids=input_ids, generation_config=generation_config)
    generated_text = tokenizer.decode(response_token_ids[0], skip_special_tokens=True)
    return generated_text


def calculate_score(text):
    Y_true = cefr_level_to_value("B1")
    Y_pred = cefr_level_to_value(get_textmetrics_analysis(text, "en"))

    if Y_pred == 6: # happens if cefr level is unknown
        return -1.0

    score = 1/(abs(Y_pred - Y_true) + 1)

    return score

def evaluate_cefr(model, tokenizer, dataset, num_samples):
    cefr_scores = []

    for i, sample in tqdm(enumerate(dataset)):
        prompt = sample["prompt"] + "\n"

        if i > num_samples:
            break

        generated_text = generate(prompt, tokenizer, model, 1000)

        # print(generated_text)
        # print('\n')
        cefr_scores.append(calculate_score(generated_text))
        
    print("CEFR scores: ", cefr_scores)
    mean = np.mean(cefr_scores)
    std = np.std(cefr_scores)
    return mean, std

def collator(data): 
    return dict((key, [d[key] for d in data]) for key in data[0])


def RLHF(ppo_model, ref_model, dataset, data, tokenizer, r=1.42e-5, epochs=2, max_steps=1):
    learning_rate=1.42e-5
    max_ppo_epochs=3
    mini_batch_size=4
    batch_size=16

    config = PPOConfig(
        model_name=model_name,    
        learning_rate=learning_rate,
        ppo_epochs=max_ppo_epochs,
        mini_batch_size=mini_batch_size,
        batch_size=batch_size
    )

    ppo_trainer = PPOTrainer(config=config, 
                            model=ppo_model, 
                            ref_model=ref_model, 
                            tokenizer=tokenizer, 
                            dataset=dataset["train"], 
                            data_collator=collator)
    
    output_min_length = 400
    output_max_length = 600
    output_length_sampler = LengthSampler(output_min_length, output_max_length)
    # max_new_tokens = 30
    generation_kwargs = {
        "top_k": 0,
        "do_sample": True,
        "min_length": -1, 
        "max_new_tokens": 500,
        "pad_token_id": tokenizer.eos_token_id
    }

    reward_kwargs = {
        "top_k": None, # Return all scores.
        "function_to_apply": "none", # You want the raw logits without softmax.
        "batch_size": 16
    }

    max_ppo_steps = max_steps

    for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        # Break when you reach max_steps.
        if step >= max_ppo_steps:
            break

        prompt_tensors = batch["input_ids"]
        prompt_tensors = [torch.tensor(prompt).to(device) for prompt in prompt_tensors]
        
        # Get response
        job_ad_tensors = []

        # for prompt_tensor in prompt_tensors:
        job_ad_tensors = ppo_trainer.generate(prompt_tensors, **generation_kwargs)
        #, return_prompt=True, length_sampler=output_length_sampler
            # summary_tensors.append(summary.squeeze()[-max_new_tokens:])
        # print(job_ad_tensors)
        # This needs to be called "response".
        # batch["response"] = tokenizer.batch_decode(summary_tensors, skip_special_tokens=True)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in job_ad_tensors]
        # print(batch["response"])

        # Compute reward outputs.
        rewards = []
        for i in batch["response"]: 
            rewards.append(calculate_score(i))

        # rewards = calculate_scores(batch["response"]))
        reward_tensors = [torch.tensor(reward) for reward in rewards]

        # Run PPO step. 
        stats = ppo_trainer.step(prompt_tensors, job_ad_tensors, reward_tensors)
        ppo_trainer.log_stats(stats, batch, reward_tensors)

        print(f'objective/kl: {stats["objective/kl"]}')
        print(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
        print(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')
        print('-'.join('' for x in range(100)))

    mean_after, std_after = evaluate_cefr(model=ppo_model, 
                                    tokenizer=tokenizer, 
                                    dataset=data["val"], 
                                    num_samples=1)
    print(f'toxicity [mean, std] after detox: [{mean_after}, {std_after}]')
    return mean_after, std_after



def main():
    config = PeftConfig.from_pretrained(peft_model_path)

    # Load LLaMA tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(config.base_model_name_or_path, device_map="auto")
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left" 

    # Load LLaMA model
    model = LlamaForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto", load_in_8bit=True) #quantization_config=bnb_config
    model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    print("Loading data")
    data = load_data()
    dataset = data.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    dataset = dataset.remove_columns(['prompt', 'text'])

    # load trained model and generate some job ads
    trained_peft_model = PeftModel.from_pretrained(model, peft_model_path)
    trained_peft_model.print_trainable_parameters()

    # create ppo model and reference model
    ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(trained_peft_model, #peft_model
                                                               torch_dtype=torch.bfloat16,
                                                               device_map="auto",
                                                               is_trainable=True)
    ref_model = create_reference_model(ppo_model)

    # print trainable parameters for sanity check
    print(f'PPO model parameters to be updated:\n{print_number_of_trainable_model_parameters(ppo_model)}\n')
    print(ppo_model.v_head)
    print(f'Reference model parameters to be updated:\n{print_number_of_trainable_model_parameters(ref_model)}\n')

    # calculate mean and std scores before training
    mean_before, std_before = evaluate_cefr(model=ref_model, tokenizer=tokenizer, dataset=data["val"], num_samples=1)
    print(f'cefr score [mean, std] before detox: [{mean_before}, {std_before}]')

    mean_after, std_after = RLHF(ppo_model, ref_model, dataset, data, tokenizer)

    mean_improvement = (mean_before - mean_after) / mean_before
    std_improvement = (std_before - std_after) / std_before

    print(f'Percentage improvement of cefr score after finetuning:')
    print(f'mean: {mean_improvement*100:.2f}%')
    print(f'std: {std_improvement*100:.2f}%')



if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Arguments')
    # parser.add_argument('-s', '--samples', help='if it should generate samples and calculate scores')
    # parser.add_argument('-n', '--number_samples', type=int, help='number of samples to generate', default=2)
    # parser.add_argument('-l', '--len_samples', type=int, help='length of the generated samples')
    # parser.add_argument('-i', '--num_iterations', type=int, help='number of steps for training')
    # args = parser.parse_args()
    # print(args)

    try:
        main()
    except KeyboardInterrupt:
        print("Stopping the program")
        sys.exit(0)