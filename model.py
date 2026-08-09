import pandas as pd
import torch
from transformers import (
    AutoModelForQuestionAnswering, AutoTokenizer, get_linear_schedule_with_warmup
)

data = pd.read_json("train-v2.0.json")

parse_dict = {"context": [], "question": [], "answers": [], "answer_start": [], "is_impossible": []}

all_contexts = []
cnt = 0
for block in data["data"]:
    for paragraph in block["paragraphs"]:
        context = paragraph["context"]
        qas = paragraph["qas"]
        all_contexts.append(context)
        for qa in qas:
            parse_dict["is_impossible"].append(qa["is_impossible"])
            flag = "plausible_answers" if qa["is_impossible"] else "answers"
            parse_dict["context"].append(len(context))
            parse_dict["question"].append(qa["question"])
            parse_dict["answers"].append(qa[flag][0]["text"])
            parse_dict["answer_start"].append(qa[flag][0]["answer_start"])

df = pd.DataFrame(parse_dict)

device = torch.device("mps" if torch.mps.is_available() else "cpu")

MODEL_NAME = "prajjwal1/bert-mini"
MAX_LEN = 256

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
