import torch
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from peft import PeftConfig, PeftModel
from generate_response_ import generate_response_


model_id = "sdhan/SD_SOLAR_10.7B_v1.0"
finetuned_model = "QA_qlora_model"
peft_config = PeftConfig.from_pretrained(finetuned_model)


quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )


model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    quantization_config=quant_config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)


tokenizer = AutoTokenizer.from_pretrained(
    peft_config.base_model_name_or_path
)
tokenizer.use_default_system_prompt = False


# QLoRA 모델 로드
peft_model = PeftModel.from_pretrained(model, finetuned_model, torch_dtype=torch.float16)

# QLoRA 가중치를 베이스 모델에 병합
merged_model = peft_model.merge_and_unload()


test_data = load_dataset('csv', data_files={'test': "test.csv"}, split="test")

# Model inference
submission_dict = {}


for row in tqdm(test_data):
    try:
        context = row['context']
        question = row['question']
        id = row['id']

        if context is not None and question is not None:
            question_prompt = f"너는 주어진 Context를 토대로 Question에 답하는 챗봇이야. \
                                Question에 대한 답변만 가급적 한 단어로 최대한 간결하게 답변하도록 해. \
                                Context: {context} Question: {question}\n Answer:"

            answer = generate_response_(question_prompt, model=merged_model, tokenizer=tokenizer)
            submission_dict[id] = answer
        else:
            submission_dict[id] = "Invalid question or context"

    except Exception as e:
        print(f"Error processing question {e}")


# Submission
df = pd.DataFrame(list(submission_dict.items()),
                  columns=['id', 'answer'])
df.to_csv('QA_finetune.csv', index=False)
