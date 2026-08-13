import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import re
import string
import collections

CHECKPOINT_PATH = f"./model_train{1}"
MAX_LEN = 300
BATCH_SIZE = 64

df = pd.read_json("preprocessed_batches_val.json").sample(frac=1).reset_index(drop=True)

device = torch.device("mps" if torch.mps.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
model = AutoModelForQuestionAnswering.from_pretrained(CHECKPOINT_PATH).to(device)

model.eval()


def normalize_text(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def compute_em(pred, gold):
    return int(normalize_text(pred) == normalize_text(gold))


def compute_precision_recall_f1(pred, gold):
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        # both empty -> perfect match, one empty -> zero
        equal = int(pred_tokens == gold_tokens)
        return float(equal), float(equal), float(equal)

    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


total_start_avg = total_end_avg = 0.0
em_scores, precisions, recalls, f1_scores = [], [], [], []

with torch.no_grad():
    for i in range(0, len(df), BATCH_SIZE):
        cur_batch = df[i:i+BATCH_SIZE]
        input_ids, attention_mask, token_type_ids, start_token_idx, end_token_idx = cur_batch["input_ids"].tolist(), cur_batch[
            "attention_mask"].tolist(), cur_batch["token_type_ids"].tolist(), cur_batch["start_token_idx"].tolist(), cur_batch["end_token_idx"].tolist()

        start_positions = torch.tensor(start_token_idx, dtype=torch.long).to(device)
        end_positions = torch.tensor(end_token_idx, dtype=torch.long).to(device)

        outputs = model(
            input_ids=torch.tensor(input_ids, dtype=torch.long).to(device),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long).to(device),
            token_type_ids=torch.tensor(token_type_ids, dtype=torch.long).to(device),
        )

        pred_start = outputs.start_logits.argmax(dim=-1)
        pred_end = outputs.end_logits.argmax(dim=-1)

        total_start_avg += (pred_start - start_positions).float().abs().mean()
        total_end_avg += (pred_end - end_positions).float().abs().mean()

        for j in range(len(cur_batch)):
            ids = input_ids[j]
            ttypes = token_type_ids[j]
            amask = attention_mask[j]

            context_indices = [
                idx for idx, (tt, am) in enumerate(zip(ttypes, amask))
                if tt == 1 and am == 1
            ]
            if not context_indices:
                continue

            context_start = context_indices[0]
            context_end = context_indices[-1]

            p_start = pred_start[j].item()
            p_end = pred_end[j].item()

            p_start = max(context_start, min(p_start, context_end))
            p_end = max(context_start, min(p_end, context_end))

            if p_end < p_start:
                pred_text = ""
            else:
                pred_text = tokenizer.decode(ids[p_start:p_end + 1], skip_special_tokens=True)

            gold_start = start_token_idx[j]
            gold_end = end_token_idx[j]
            gold_text = tokenizer.decode(ids[gold_start:gold_end + 1], skip_special_tokens=True)

            em_scores.append(compute_em(pred_text, gold_text))
            precision, recall, f1 = compute_precision_recall_f1(pred_text, gold_text)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)


num_of_batches = len(df) // BATCH_SIZE
print(f"avg-start-dist: {(total_start_avg / num_of_batches).item()}")
print(f"avg-end-dist: {(total_end_avg / num_of_batches).item()}")
print(f"EM: {sum(em_scores) / len(em_scores):.4f}")
print(f"Precision: {sum(precisions) / len(precisions):.4f}")
print(f"Recall: {sum(recalls) / len(recalls):.4f}")
print(f"F1: {sum(f1_scores) / len(f1_scores):.4f}")