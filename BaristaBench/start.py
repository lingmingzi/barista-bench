import argparse
import json
import os
import re
from typing import Dict, List

import pandas as pd
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def get_train_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================
if "__file__" in globals():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    BASE_DIR = os.getcwd()

LOCAL_TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
LOCAL_TEST_PATH = os.path.join(BASE_DIR, "test.csv")
LOCAL_MENU_PATH = os.path.join(BASE_DIR, "menu.md")

KAGGLE_TRAIN_PATH = "/kaggle/input/barista-bench/train.csv"
KAGGLE_TEST_PATH = "/kaggle/input/barista-bench/test.csv"
KAGGLE_MENU_PATH = "/kaggle/input/barista-bench/menu.md"

TRAIN_PATH = LOCAL_TRAIN_PATH if os.path.exists(LOCAL_TRAIN_PATH) else KAGGLE_TRAIN_PATH
TEST_PATH = LOCAL_TEST_PATH if os.path.exists(LOCAL_TEST_PATH) else KAGGLE_TEST_PATH
MENU_PATH = LOCAL_MENU_PATH if os.path.exists(LOCAL_MENU_PATH) else KAGGLE_MENU_PATH

OUTPUT_DIR = os.path.join(BASE_DIR, "qwen_lora")
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_menu() -> str:
    if os.path.exists(MENU_PATH):
        with open(MENU_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def load_df(path: str, fallback: Dict[str, List[str]]):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(fallback)


MENU_CONTEXT = load_menu()
TRAIN_DF = load_df(TRAIN_PATH, {"id": [0], "order": ["Latte"], "expected_json": ["{}"]})
TEST_DF = load_df(TEST_PATH, {"id": [1001], "order": ["Latte"]})

# ==========================================
# 2. PROMPT TEMPLATE
# ==========================================
SYSTEM_PROMPT = (
    "You are a Barista POS agent. Parse the order into a JSON object that EXACTLY "
    "matches this schema: {\"drink\": string, \"size\": string or null, "
    "\"final_milk\": string or null, \"modifiers\": list of strings, "
    "\"food_items\": list of strings, \"total_price\": number}. "
    "Use the menu and pricing rules. Output ONLY a JSON object, no extra text.\n"
    f"MENU:\n{MENU_CONTEXT}\n"
)


def build_prompt(order_text: str) -> str:
    return (
        SYSTEM_PROMPT
        + "ORDER:\n"
        + order_text.strip()
        + "\n"
        + "JSON:\n"
    )


def extract_json(text: str) -> str:
    try:
        json.loads(text)
        return text
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return "{}"


# ==========================================
# 3. DATASET
# ==========================================
class BaristaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 1024):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        order_text = str(row["order"])
        target_json = str(row["expected_json"]).strip()

        prompt = build_prompt(order_text)
        full_text = prompt + target_json

        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        prompt_ids = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )["input_ids"][0]

        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]

        labels = input_ids.clone()
        prompt_len = min(prompt_ids.numel(), self.max_length)
        labels[:prompt_len] = -100
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ==========================================
# 4. TRAINING
# ==========================================
def train_model():
    set_seed(42)
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dtype = get_train_dtype()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=train_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    eval_ratio = 0.05
    if len(TRAIN_DF) > 1 and 0.0 < eval_ratio < 1.0:
        eval_df = TRAIN_DF.sample(frac=eval_ratio, random_state=42)
        train_df = TRAIN_DF.drop(eval_df.index)
    else:
        train_df = TRAIN_DF
        eval_df = TRAIN_DF

    train_dataset = BaristaDataset(train_df, tokenizer)
    eval_dataset = BaristaDataset(eval_df, tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available() and train_dtype == torch.float16,
        bf16=torch.cuda.is_available() and train_dtype == torch.bfloat16,
        logging_steps=50,
        save_strategy="epoch",
        report_to=[],
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        dataloader_pin_memory=torch.cuda.is_available(),
        group_by_length=True,
        gradient_checkpointing=True,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.0)],
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


# ==========================================
# 5. INFERENCE
# ==========================================
def load_model_for_inference():
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR if os.path.exists(OUTPUT_DIR) else MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    infer_dtype = get_train_dtype()

    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=infer_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if os.path.exists(OUTPUT_DIR):
        model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    else:
        model = base_model

    model.eval()
    model.config.use_cache = True
    try:
        if torch.cuda.is_available() and hasattr(torch, "compile"):
            model = torch.compile(model, mode="reduce-overhead")
    except Exception:
        pass
    return model, tokenizer


def generate_prediction_batch(model, tokenizer, order_texts: List[str]) -> List[str]:
    prompts = [build_prompt(t) for t in order_texts]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    results = []
    for prompt, text in zip(prompts, texts):
        text = text[len(prompt):].strip()
        results.append(extract_json(text))
    return results


# ==========================================
# 6. SAVE SUBMISSION
# ==========================================
def build_submission():
    id_columns = [col for col in TEST_DF.columns if "id" in col.lower()]
    if id_columns:
        id_col = id_columns[0]
    else:
        id_col = "id"
        TEST_DF[id_col] = TEST_DF.index

    model, tokenizer = load_model_for_inference()

    predictions = []
    print(f"Processing {len(TEST_DF)} rows...")
    batch_size = 8 if torch.cuda.is_available() else 2
    orders = [row["order"] if "order" in row else row.get("text", "") for _, row in TEST_DF.iterrows()]
    for i in tqdm(range(0, len(orders), batch_size), total=(len(orders) + batch_size - 1) // batch_size):
        batch_orders = orders[i:i + batch_size]
        preds = generate_prediction_batch(model, tokenizer, batch_orders)
        predictions.extend(preds)

    submission = pd.DataFrame({
        id_col: TEST_DF[id_col],
        "prediction": predictions,
    })

    submission.to_csv("submission.csv", index=False)
    print("\nsubmission.csv created successfully!")
    print(submission.head())


# ==========================================
# 7. ENTRY
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train LoRA adapter")
    parser.add_argument("--predict", action="store_true", help="Generate submission.csv")
    try:
        args, _ = parser.parse_known_args()
    except SystemExit:
        args = parser.parse_args([])

    if not args.train and not args.predict:
        args.train = True
        args.predict = True

    if args.train:
        train_model()

    if args.predict:
        build_submission()


if __name__ == "__main__":
    main()