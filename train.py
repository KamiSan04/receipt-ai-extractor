from transformers import LayoutLMv3ForTokenClassification, TrainingArguments, Trainer
from datasets import load_from_disk, load_dataset
import torch

ds_info = load_dataset("mp-02/sroie")
label_list = ds_info["train"].features["ner_tags"].feature.names

processed_train = load_from_disk("processed_train")
processed_test = load_from_disk("processed_test")

processed_train.set_format("torch")
processed_test.set_format("torch")

model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=len(label_list),
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    fp16=True,
    learning_rate=5e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_train,
    eval_dataset=processed_test,
)

trainer.train()

trainer.save_model("./final_model")