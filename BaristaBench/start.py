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
    Trainer,
    TrainingArguments,
)

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
            padding="max_length",
            return_tensors="pt",
        )["input_ids"][0]

        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]

        labels = input_ids.clone()
        labels[prompt_ids != self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ==========================================
# 4. TRAINING
# ==========================================
def train_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
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

    dataset = BaristaDataset(TRAIN_DF, tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
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

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if os.path.exists(OUTPUT_DIR):
        model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    else:
        model = base_model

    model.eval()
    return model, tokenizer


def generate_prediction(model, tokenizer, order_text: str) -> str:
    prompt = build_prompt(order_text)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )[0]

    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    text = text[len(prompt):].strip()
    return extract_json(text)


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
    for _, row in tqdm(TEST_DF.iterrows(), total=len(TEST_DF)):
        order_text = row["order"] if "order" in row else row.get("text", "")
        pred = generate_prediction(model, tokenizer, order_text)
        predictions.append(pred)

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