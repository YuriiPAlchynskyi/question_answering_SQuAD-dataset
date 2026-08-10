import pandas as pd
import torch
from transformers import (
    AutoModelForQuestionAnswering, AutoTokenizer, get_linear_schedule_with_warmup
)

data = pd.read_json("train-v2.0.json")

parse_dict = {"context": [], "question": [], "answers": [], "answer_start": [], "is_impossible": []}

all_contexts = []
for block in data["data"]:
    for paragraph in block["paragraphs"]:
        context = paragraph["context"]
        qas = paragraph["qas"]
        all_contexts.append(context)
        for qa in qas:
            parse_dict["is_impossible"].append(qa["is_impossible"])
            flag = "plausible_answers" if qa["is_impossible"] else "answers"
            parse_dict["context"].append(len(all_contexts) - 1)
            parse_dict["question"].append(qa["question"])
            parse_dict["answers"].append(qa[flag][0]["text"])
            parse_dict["answer_start"].append(qa[flag][0]["answer_start"])

df = pd.DataFrame(parse_dict)

device = torch.device("mps" if torch.mps.is_available() else "cpu")

MODEL_NAME = "prajjwal1/bert-mini"
MAX_LEN = 300

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME, attn_implementation="eager").to(device)


def find_start_end_token(seq_ids, context_s_idx, context_e_idx, off_set_mapping):
    num_of_items = off_set_mapping.size(0)
    seq_len = off_set_mapping.size(1)
    start_token_idx, end_token_idx = [], []

    for i in range(num_of_items):
        sequence_ids = seq_ids(i)
        s, e = context_s_idx[i], context_e_idx[i]
        pairs = off_set_mapping[i]

        context_start, context_end = sequence_ids.index(1), seq_len - 1
        start_was_added, end_was_added = False, False
        for j in range(context_start, context_end):
            cur_s, cur_e = pairs[j]
            if cur_s <= s <= cur_e and not start_was_added:
                start_token_idx.append(j)
                start_was_added = True
            if cur_s <= e <= cur_e and not end_was_added:
                end_was_added = True
                end_token_idx.append(j)

        if not end_was_added or not start_was_added:
            if not start_was_added:
                start_token_idx.append(0)
            start_token_idx[-1] = 0
            end_token_idx.append(0)

    return start_token_idx, end_token_idx


def make_batch(X_df):
    context_batch = [all_contexts[idx] for idx in X_df["context"]]
    questions = X_df["question"].to_list()
    answers_start_idx = X_df["answer_start"].to_list()
    answers_end_idx = (X_df["answer_start"] + X_df["answers"].str.len() - 1).to_list()

    encodings = tokenizer(
        questions,
        context_batch,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding=True,
        truncation="only_second",
        return_tensors="pt",
        return_offsets_mapping=True
    )

    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    offset_mapping = encodings["offset_mapping"].to(device)
    token_type_ids = encodings["token_type_ids"].to(device)

    sequence_ids = encodings.sequence_ids

    start_token_idx, end_token_idx = find_start_end_token(sequence_ids, answers_start_idx, answers_end_idx,
                                                          offset_mapping)
    start_token_idx = torch.tensor(start_token_idx, dtype=torch.long).to(device)
    end_token_idx = torch.tensor(end_token_idx, dtype=torch.long).to(device)

    return input_ids, attention_mask, token_type_ids, start_token_idx, end_token_idx


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
NUM_EPOCHS = 1
BATCH_SIZE = 64

num_batches_per_epoch = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
total_steps = num_batches_per_epoch * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)

model.train()
for epoch in range(NUM_EPOCHS):
    for i in range(0, len(df), BATCH_SIZE):
        X_batch = df.iloc[i:i + BATCH_SIZE]
        input_ids, attention_mask, token_type_ids, start_token_idx, end_token_idx = make_batch(X_batch)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_token_idx,
            end_positions=end_token_idx
        )

        optimizer.zero_grad()
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        print(loss.item())
