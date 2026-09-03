from transformers import LayoutLMv3ForTokenClassification, Trainer, TrainingArguments
from datasets import load_from_disk, load_dataset
from seqeval.metrics import classification_report
import numpy as np

ds_info = load_dataset("mp-02/sroie")
label_list = ds_info["train"].features["ner_tags"].feature.names

processed_test = load_from_disk("processed_test")
processed_test.set_format("torch")

model = LayoutLMv3ForTokenClassification.from_pretrained("./final_model")

args = TrainingArguments(output_dir="./eval_tmp", per_device_eval_batch_size=2, report_to="none")
trainer = Trainer(model=model, args=args)

predictions, labels, _ = trainer.predict(processed_test)
predictions = np.argmax(predictions, axis=2)

true_predictions = []
true_labels = []

for pred, label in zip(predictions, labels):
    pred_labels = []
    true_lbls = []
    for p, l in zip(pred, label):
        if l != -100:
            pred_labels.append(label_list[p])
            true_lbls.append(label_list[l])
    true_predictions.append(pred_labels)
    true_labels.append(true_lbls)

print(classification_report(true_labels, true_predictions))