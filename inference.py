import pytesseract
import re
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

label_list = ['S-COMPANY', 'S-DATE', 'S-ADDRESS', 'S-TOTAL', 'O']

processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained("./final_model")

def run_ocr(image):
    width, height = image.size
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    words = []
    boxes = []
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        if word == "":
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        box = [x, y, x + w, y + h]
        norm_box = [
            int(1000 * box[0] / width),
            int(1000 * box[1] / height),
            int(1000 * box[2] / width),
            int(1000 * box[3] / height),
        ]
        words.append(word)
        boxes.append(norm_box)

    return words, boxes

def find_total_fallback(words):
    for i, word in enumerate(words):
        if word.lower() == "total":
            if i > 0 and words[i-1].lower() == "sub":
                continue
            for j in range(i+1, min(i+4, len(words))):
                match = re.match(r'\$?(\d+\.\d{2})', words[j])
                if match:
                    return match.group(1)
    return None

def extract_fields(image_path):
    image = Image.open(image_path).convert("RGB")
    words, boxes = run_ocr(image)
    print(words)

    encoding = processor(
        image,
        words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**encoding)

    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    word_ids = encoding.word_ids(batch_index=0)

    results = {}
    seen_words = set()
    for idx, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen_words:
            continue
        seen_words.add(word_id)
        label = label_list[predictions[idx]]
        if label == "O":
            continue
        field = label.replace("S-", "").lower()
        results.setdefault(field, []).append(words[word_id])

    final = {field: " ".join(vals) for field, vals in results.items()}

    if "total" not in final:
        fallback = find_total_fallback(words)
        if fallback:
            final["total"] = fallback

    return final

if __name__ == "__main__":
    result = extract_fields("test_receipt.jpg")
    print(result)

