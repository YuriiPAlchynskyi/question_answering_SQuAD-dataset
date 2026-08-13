import pandas as pd
import torch
from transformers import AutoTokenizer
import json
import argparse

# parser = argparse.ArgumentParser(description="Preprocess QA data for training.")
# parser.add_argument(
#     "--input", "-i",
#     type=str,
#     required=True,
#     help="Path to the raw data file to preprocess"
# )
# parser.add_argument(
#     "--output", "-o",
#     type=str,
#     required=True,
#     help="Path to write the preprocessed output file"
# )
#
# args = parser.parse_args()
#
# print(f"Reading from: {args.input}")
# print(f"Writing to: {args.output}")

data = pd.read_json("dev-v2.0.json")

parse_dict = {"context": [], "question": [], "answers": [], "answer_start": [], "is_impossible": []}

all_contexts = []
for block in data["data"]:
    for paragraph in block["paragraphs"]:
        context = paragraph["context"]
        qas = paragraph["qas"]
        all_contexts.append(context)
        for qa in qas:
            flag = "plausible_answers" if qa["is_impossible"] else "answers"
            if qa[flag]:
                parse_dict["answers"].append(qa[flag][0]["text"])
                parse_dict["answer_start"].append(qa[flag][0]["answer_start"])
            else:
                continue
            parse_dict["is_impossible"].append(qa["is_impossible"])
            parse_dict["context"].append(len(all_contexts) - 1)
            parse_dict["question"].append(qa["question"])

df = pd.DataFrame(parse_dict)
df["answer_end"] = df["answer_start"] + df["answers"].str.len() - 1

device = torch.device("mps" if torch.mps.is_available() else "cpu")

MODEL_NAME = "prajjwal1/bert-tiny"
MAX_LEN = 300
BATCH_SIZE = 64

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def find_start_end_token(seq_ids, context_s_idx, context_e_idx, off_set_mapping):
    num_of_items = len(off_set_mapping)
    seq_len = len(off_set_mapping[0])
    start_tokens_idx, end_tokens_idx = [], []

    for i in range(num_of_items):
        sequence_ids = seq_ids(i)
        s, e = context_s_idx[i], context_e_idx[i]
        pairs = off_set_mapping[i]

        context_start, context_end = sequence_ids.index(1), seq_len - 1

        start_token = end_token = None

        l, r = context_start, context_end
        while l <= r:
            mid = (l + r) // 2
            cur_s, cur_e = pairs[mid]
            if cur_s <= s <= cur_e:
                start_token = mid
                break
            elif cur_s > s:
                r = mid - 1
            else:
                l = mid + 1

        l, r = context_start, context_end
        while l <= r:
            mid = (l + r) // 2
            cur_s, cur_e = pairs[mid]
            if cur_s <= e <= cur_e:
                end_token = mid
                break
            elif cur_e > e:
                r = mid - 1
            else:
                l = mid + 1

        if start_token is None or end_token is None:
            start_tokens_idx.append(0)
            end_tokens_idx.append(0)
        else:
            start_tokens_idx.append(start_token)
            end_tokens_idx.append(end_token)

    return start_tokens_idx, end_tokens_idx


def make_batch(X_df):
    context_batch = [all_contexts[idx] for idx in X_df["context"]]
    questions = X_df["question"].to_list()
    answers_start_idx = X_df["answer_start"].to_list()
    answers_end_idx = X_df["answer_end"].to_list()

    encodings = tokenizer(
        questions,
        context_batch,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding="max_length",
        truncation="only_second",
        return_tensors="pt",
        return_offsets_mapping=True
    )

    sequence_ids = encodings.sequence_ids

    input_ids = encodings["input_ids"].tolist()
    attention_mask = encodings["attention_mask"].tolist()
    offset_mapping = encodings["offset_mapping"].tolist()
    token_type_ids = encodings["token_type_ids"].tolist()

    start_token_idx, end_token_idx = find_start_end_token(sequence_ids, answers_start_idx, answers_end_idx,
                                                          offset_mapping)

    return input_ids, attention_mask, token_type_ids, start_token_idx, end_token_idx


batches = []

for j in range(0, len(df), BATCH_SIZE):
    X_batch = df.iloc[j:j + BATCH_SIZE]

    input_ids, attention_mask, token_type_ids, start_token_idx, end_token_idx = make_batch(X_batch)
    question, context, answer, answer_start, answer_end = X_batch["question"].to_list(), X_batch["context"].to_list(), \
        X_batch["answers"].to_list(), \
        X_batch["answer_start"].to_list(), X_batch["answer_end"].to_list()

    for i in range(len(X_batch)):
        batches.append(
            {"input_ids": input_ids[i], "attention_mask": attention_mask[i], "token_type_ids": token_type_ids[i],
             "start_token_idx": start_token_idx[i], "end_token_idx": end_token_idx[i], "question": question[i],
             "context": context[i], "answer": answer[i], "answer_start": answer_start[i], "answer_end": answer_end[i]})

    print(f"Batch:{j // 64}")

with open("preprocessed_batches_val.json", "w") as f:
    json.dump(batches, f)
