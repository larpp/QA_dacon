# 모델로 추론 후, 전처리 수행 후, 완성된 정답 반환
def generate_response(question_prompt, model, tokenizer):
    # 생성할 최대 token 수, 답변 생성 수, padding token의 idx 지정해 모델 pipeline 설정 후 답변 생성
    # response = qa_pipeline(
    #     question_prompt,
    #     max_new_tokens=50,
    #     num_return_sequences=1,
    #     pad_token_id=tokenizer.eos_token_id
    # )[0]['generated_text']
    inputs = tokenizer(question_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in response:
        # Answer 이후에 생성된 token들만을 답변으로 사용
        response = response.split("Answer:", 1)[1][:20]

        # Token 반족 생성 및 noise token 관련 처리
        if "Que" in response:
            response = response.split("Que", 1)[0]
        if "⊙" in response:
            response = response.split("⊙", 1)[0]
        if "Con" in response:
            response = response.split("Con", 1)[0]
    return response