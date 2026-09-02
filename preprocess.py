from transformers import LayoutLMv3Processor
from datasets import load_dataset
from PIL import Image

processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

ds = load_dataset("darentang/sroie", revision="refs/convert/parquet")

label_list = ds["train"].features["ner_tags"].feature.names

example = ds["train"][0]
print(type(example["image_path"]))
print(example["image_path"])