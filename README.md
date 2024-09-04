# Dacon - 2024 인하 인공지능 챌린지

[대회 사이트](https://dacon.io/competitions/official/236291/overview/description)

경제 기사 (context)가 주어졌을 때, 사용자의 질문에 대한 답변을 생성하는 QA task

## Train
### LLM model
한국어로 학습시킨 LLM model을 사용했다. 아래의 사이트에서 확인활 수 있다.

[Open-Ko-LLM Leaderboard](https://huggingface.co/spaces/upstage/open-ko-llm-leaderboard)

### QLoRA

LLM 모델의 용량이 크기 때문에 효율적인 학습을 위해 QLoRA 방식을 사용했다. 먼저 LLM 모델을 4 bit quantization 하고 특정 module에 r=8 값으로 LoRA를 적용시켰다.

### Instruction tuning

Question prompt를 통해 instruction tuningdmf 적용시켰다.
