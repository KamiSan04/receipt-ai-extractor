from transformers import LayoutLMv3Processor
from datasets import load_dataset

processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

ds = load_dataset("mp-02/sroie")

label_list = ds["train"].features["ner_tags"].feature.names
print("Labels:", label_list)

example = ds["train"][0]
image = example["image"].convert("RGB")
width, height = image.size

def normalize_box(box, width, height):
    return [
        int(1000 * box[0] / width),
        int(1000 * box[1] / height),
        int(1000 * box[2] / width),
        int(1000 * box[3] / height),
    ]

boxes = [normalize_box(box, width, height) for box in example["bboxes"]]

encoding = processor(
    image,
    example["words"],
    boxes=boxes,
    word_labels=example["ner_tags"],
    truncation=True,
    padding="max_length",
    return_tensors="pt",
)

print(encoding.keys())
print(encoding["input_ids"].shape)
print(encoding["bbox"].shape)
print(encoding["labels"].shape)
print(encoding["labels"][0][:20])