from transformers import LayoutLMv3Processor
from datasets import load_dataset

processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

ds = load_dataset("mp-02/sroie")

label_list = ds["train"].features["ner_tags"].feature.names

def normalize_box(box, width, height):
    return [
        int(1000 * box[0] / width),
        int(1000 * box[1] / height),
        int(1000 * box[2] / width),
        int(1000 * box[3] / height),
    ]

def process_example(example):
    image = example["image"].convert("RGB")
    width, height = image.size
    boxes = [normalize_box(box, width, height) for box in example["bboxes"]]

    encoding = processor(
        image,
        example["words"],
        boxes=boxes,
        word_labels=example["ner_tags"],
        truncation=True,
        padding="max_length",
    )
    return encoding

processed_train = ds["train"].map(process_example, remove_columns=ds["train"].column_names)
processed_test = ds["test"].map(process_example, remove_columns=ds["test"].column_names)

print(processed_train)
print(processed_train[0].keys())