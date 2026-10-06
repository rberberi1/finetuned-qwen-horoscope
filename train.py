from datasets import load_dataset
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

login()

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="float32") # load the pretrained weights in float32; fp16=True needs float32 weights (Qwen3 is stored in bf16, which fp16 training can't handle)

dataset = load_dataset("karthiksagarn/astro_horoscope", split="train")
dataset = dataset.filter(lambda row: row["category"] != "birthday")  # the app offers general, love, career and wellness


# Format each row as the same chat the app sends: sign + topic in, horoscope out
def to_chat(row):
    messages = [
        {"role": "user", "content": f"My sign is {row['sign'].title()}. Give me my {row['category']} horoscope for today."},
        {"role": "assistant", "content": row["horoscope"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
    return tokenizer(text, truncation=True, max_length=512)


dataset = dataset.map(to_chat, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=0.1)

training_args = TrainingArguments(
    output_dir="qwen3-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    fp16=True,
    learning_rate=2e-5,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
trainer.save_model("qwen3-finetuned/final")
trainer.push_to_hub()
