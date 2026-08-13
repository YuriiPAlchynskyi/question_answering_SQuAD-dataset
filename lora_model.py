from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd
import torch
from transformers import (
    AutoModelForQuestionAnswering, AutoTokenizer, get_linear_schedule_with_warmup
)

df = pd.read_json("preprocessed_batches.json").sample(frac=1).reset_index(drop=True)

device = torch.device("mps" if torch.mps.is_available() else "cpu")

MODEL_NAME = "prajjwal1/bert-medium"
MAX_LEN = 300

model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME, attn_implementation="eager").to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


lora_config = LoraConfig(
    task_type=TaskType.QUESTION_ANS,
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    bias="none",
    modules_to_save=["qa_outputs"]
)

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
NUM_EPOCHS = 1
BATCH_SIZE = 8

steps_per_epoch = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
total_steps = steps_per_epoch * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)

model.train()
for epoch in range(NUM_EPOCHS):
    for i in range(0, len(df), BATCH_SIZE):
        cur_batch = df[i:i + BATCH_SIZE]
        input_ids, attention_mask, token_type_ids, start_token_idx, end_token_idx = cur_batch["input_ids"].tolist(), \
        cur_batch[
            "attention_mask"].tolist(), cur_batch["token_type_ids"].tolist(), cur_batch["start_token_idx"].tolist(), \
        cur_batch["end_token_idx"].tolist()

        start_positions = torch.tensor(start_token_idx, dtype=torch.long).to(device)
        end_positions = torch.tensor(end_token_idx, dtype=torch.long).to(device)
        outputs = model(
            input_ids=torch.tensor(input_ids, dtype=torch.long).to(device),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long).to(device),
            token_type_ids=torch.tensor(token_type_ids, dtype=torch.long).to(device),
            start_positions=start_positions,
            end_positions=end_positions
        )

        optimizer.zero_grad()
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        start_pred = outputs.start_logits.argmax(dim=-1)
        end_pred = outputs.end_logits.argmax(dim=-1)
        print(
            f"loss: {loss.item()} batch:{i // BATCH_SIZE}/{len(df) // BATCH_SIZE} avg-start-dist: {(start_positions - start_pred).abs().float().mean()}, avg-end-dist: {(end_pred - end_positions).abs().float().mean()}")

model.save_pretrained(f"./model_train{2}")
tokenizer.save_pretrained(f"./model_train{2}")
