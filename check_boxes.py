from datasets import load_dataset

ds = load_dataset("mp-02/sroie")
example = ds["train"][0]
image = example["image"]

print("Image size:", image.size)
print("First 5 boxes:", example["bboxes"][:5])

all_x = [coord for box in example["bboxes"] for coord in [box[0], box[2]]]
all_y = [coord for box in example["bboxes"] for coord in [box[1], box[3]]]
print("Box x range:", min(all_x), max(all_x))
print("Box y range:", min(all_y), max(all_y))