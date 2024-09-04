import re
import string
from collections import Counter


# Define F1 score
def normalize_answer(s):
    def remove_(text):
        '''불필요한 기호 제거'''
        text = re.sub("'", " ", text)
        text = re.sub('"', " ", text)
        text = re.sub('《', " ", text)
        text = re.sub('》', " ", text)
        text = re.sub('<', " ", text)
        text = re.sub('>', " ", text)
        text = re.sub('〈', " ", text)
        text = re.sub('〉', " ", text)
        text = re.sub("\(", " ", text)
        text = re.sub("\)", " ", text)
        text = re.sub("‘", " ", text)
        text = re.sub("’", " ", text)
        return text

    def white_space_fix(text):
        '''연속된 공백일 경우 하나의 공백으로 대체'''
        return ' '.join(text.split())

    def remove_punc(text):
        '''구두점 제거'''
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        '''소문자 변환'''
        return text.lower()

    return white_space_fix(remove_punc(lower(remove_(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    # 문자 단위로 f1-score를 계산
    prediction_Char = []
    for tok in prediction_tokens:
        now = [a for a in tok]
        prediction_Char.extend(now)

    ground_truth_Char = []
    for tok in ground_truth_tokens:
        now = [a for a in tok]
        ground_truth_Char.extend(now)

    common = Counter(prediction_Char) & Counter(ground_truth_Char)
    num_same = sum(common.values())
    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_Char)
    recall = 1.0 * num_same / len(ground_truth_Char)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1

def evaluate(ground_truth_df, predictions_df):
    predictions = dict(zip(predictions_df['question'], predictions_df['answer']))
    f1 = exact_match = total = 0

    for index, row in ground_truth_df.iterrows():
        question_text = row['question']
        answer_text = row['answer']
        total += 1
        if question_text not in predictions:
            continue
        predictions = predictions[question_text]
        f1 = f1 + f1_score(predictions, answer_text)
    
    f1 = 100.0 * f1 / total

    return {'f1': f1}