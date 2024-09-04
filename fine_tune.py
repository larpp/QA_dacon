import os
import torch
from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from f1_score import evaluate


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )


    model_id = "sdhan/SD_SOLAR_10.7B_v1.0"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        low_cpu_mem_usage=True
    )
    model.config.use_cache = False
    
    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        task_type="CAUSAL_LM",
    )

    # model = prepare_model_for_kbit_training(model) # GPU memory 부족하면 사용
    model = get_peft_model(model, peft_params)

    train_dataset = load_dataset('csv', data_files='train.csv', split='train')


    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.use_default_system_prompt = False


    question_prompt = "너는 주어진 Question에 답하는 챗봇이야. Question에 대한 답변만 가급적 한 단어로 최대한 간결하게 답변하도록 해.\nQuestion: {question}\nAnswer:"


    def preprocess_function(examples):
        questions = [question_prompt.format(question=q.strip()) for q in examples["question"]]
        context = examples['context']
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=512,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answer"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = context[i].find(answer)
            end_char = start_char + len(answer)
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            # 이 부분에서 계속 오류가 뜸. context가 길어서 중간에 짤리는 (max_length=512) 바람에 while문에서 idx를 계속 증가시킴
            # while sequence_ids[idx] == 1:
            #     idx += 1
            # context_end = idx - 1
            # 수정 코드
            while idx < 512:
                if sequence_ids[idx] == 1:
                    idx += 1
                else:
                    break
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions (문장의 위치가 아닌 token 위치)
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs


    train_ds = train_dataset.map(preprocess_function,
                                batched=True,
                                remove_columns=train_dataset.column_names,
                                num_proc=os.cpu_count()) # cpu 병렬 처리를 통해 더 빠르게 mapping 가능
    
    # eval_ds = eval_dataset.map(preprocess_function,
    #                             batched=True,
    #                             remove_columns=eval_dataset.column_names,
    #                             num_proc=os.cpu_count()) # cpu 병렬 처리를 통해 더 빠르게 mapping 가능


    training_params = SFTConfig(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        optim="paged_adamw_32bit",
        save_steps=903,
        # eval_strategy="steps",  # "epoch"로 설정하면 epoch 끝날 때마다 평가
        eval_strategy="no",
        # eval_steps=1,  # "save_steps"와 동일한 값을 권장
        eval_steps=None,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant_with_warmup",
        max_seq_length=512,
        # dataset_text_field="text",
        # load_best_model_at_end=True,  # 최상의 모델을 자동으로 로드하도록 설정
        # metric_for_best_model="f1",  # 최상의 모델을 선택할 기준 메트릭
        # greater_is_better=True,  # f1-score가 높을수록 좋기 때문에 True
        # save_total_limit=1  # Only keep the best model
    )


    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


    # Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_params,
        peft_config=peft_params,
        train_dataset=train_ds,
        # eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        # compute_metrics=evaluate
        compute_metrics=None
    )

    trainer.train()

    finetuned_model = "QA_qlora_model"
    trainer.model.save_pretrained(finetuned_model)


if __name__=='__main__':
    main()
    