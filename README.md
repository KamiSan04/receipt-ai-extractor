# Receipt AI Extractor

A fine-tuned LayoutLMv3 model that reads receipt/invoice images and pulls out the important fields — company name, date, address, total amount — automatically, instead of typing them in by hand.

I built this as a resume/learning project to actually understand document AI end to end — environment setup, data quirks, preprocessing, training on a laptop GPU, evaluation, inference, and (eventually) a small API around it. It's not polished, and I'm not pretending otherwise. Below is what's done, how it works, and where it's genuinely weak.

## What it does

Give it a receipt image → OCR reads the text and where each word sits → that goes into LayoutLMv3 (a model that understands text, position, *and* the image together) → it labels which words belong to which field → out comes structured data like:

```json
{"company": "Main Street Restaurant", "address": "6332 Business Drive Suite 528 Palo Alto California 94301", "date": "Fri 04/07/2017", "total": "29.01"}
```

It only extracts what's already there. No generation, no summarizing. Smart auto-fill, basically.

## Why LayoutLMv3

Plain text models ignore position. On a receipt, position is half the signal — company name is usually top, total is usually near a line that literally says "TOTAL." LayoutLMv3 uses text + bounding boxes + the raw image together, so it can actually use that.

## Hardware

Trained entirely locally — no Colab, no cloud GPU:
- RTX 4050 Laptop GPU, 6GB VRAM
- Windows, VS Code
- Python venv (`docai-env`) to keep dependencies isolated
- PyTorch built for CUDA 13.2 (matched to the driver, not guessed)

6GB is tight for a transformer + vision encoder combo, so training uses a small batch size (2), gradient accumulation (8 steps, so effective batch ~16), and fp16 mixed precision to cut memory roughly in half.

## What's been done so far

**Environment setup** — venv, PyTorch (CUDA-matched build), transformers, datasets, seqeval, pytesseract, pillow, accelerate. Verified CUDA was actually detected before writing a single line of model code.

**Dataset** — started with `darentang/sroie`, which turned out to be broken on a modern Python version (its loading script depends on an old caching library that doesn't work anymore). Switched to `mp-02/sroie` instead, which bundles the actual images directly rather than a path string pointing at nothing.

**Preprocessing** — tokenized text, fed words + bounding boxes + image into LayoutLMv3's processor, aligned labels to tokens. Two real bugs hit here:
- Assumed the bounding boxes needed manual normalization to LayoutLMv3's 0–1000 scale. They didn't — this dataset already ships pre-normalized. Doing it anyway pushed coordinates out of valid range and crashed training with a cryptic CUDA assertion.
- The processor adds an extra batch dimension to image tensors even for a single image, which quietly breaks once you batch more than one example together. Had to manually strip that dimension.

**Training** — fine-tuned LayoutLMv3-base for 3 epochs, batch size 2, gradient accumulation 8, fp16. About 19 minutes on the laptop GPU. Loss dropped from ~8.0 to ~0.2, eval loss ended at ~0.05.

**Evaluation** — per-field precision/recall/F1 using seqeval:

| Field | Precision | Recall | F1 |
|---|---|---|---|
| Address | 0.98 | 0.98 | 0.98 |
| Company | 0.93 | 0.99 | 0.96 |
| Date | 0.93 | 0.97 | 0.95 |
| Total | 0.73 | 0.56 | 0.63 |

Address, company, and date are genuinely strong. Total is the clear weak point.

**Inference** — a script that takes any new receipt image (not from the training set), runs it through Tesseract OCR to get words + positions, feeds that into the fine-tuned model, and returns clean JSON. Tested on an actual photographed receipt (not a clean scan) — company, address, and date came back correct. Total came back empty at first, confirming the eval numbers weren't a fluke.

**Total fallback heuristic** — since the model alone misses close to half of all totals, I added a simple rule-based backup: if the model finds no TOTAL field, scan the raw OCR words for the literal word "total" (excluding "sub total") and grab the number that follows it. This is a deliberate patch, not a real fix — it only works if OCR actually read the word "total" correctly in the first place.

And that's exactly where it broke on my test image: Tesseract misread the actual "Total" line as the word `"sale"` due to print quality/lighting in the photo, so the real total value never even made it into the extracted word list. Neither the model nor the fallback rule could recover it — the information was already lost at the OCR step. I'm leaving this exact failure in as documentation rather than cherry-picking a cleaner test image, because it's a genuinely common failure mode for OCR-based pipelines, not a bug in my code: if the OCR step misreads a critical word, nothing downstream can un-misread it.

## What's not done yet

- FastAPI wrapper around inference
- No line-item extraction (itemized list on a receipt) — only 4 flat fields right now, which is the bigger real-world gap
- Not deployed anywhere yet (Hugging Face Hub + Spaces planned)

## Known limitations, stated plainly

- SROIE is a small (973 total receipts), clean, English-only academic dataset. Real-world receipts — creased, faded thermal paper, non-English, phone-camera angles — will likely break this model in ways SROIE never tests.
- TOTAL field extraction is mediocre (0.63 F1). If this were a real product, this is the field that would need the most work, and it's also arguably the field people care about most.
- OCR errors cascade. If Tesseract misreads a word — which it did, in my own test — no amount of downstream logic (model prediction or rule-based fallback) can recover information that was never correctly read in the first place.
- The fallback heuristic for TOTAL is a patch, not a solution. It only helps when OCR gets the word "total" right; it does nothing when OCR itself fails, which is exactly what happened in testing.
- Trained on a small effective batch size due to VRAM limits. A bigger GPU would probably squeeze out better numbers, especially on the weaker TOTAL field.
- No line-item table understanding — real invoices have structured item lists (qty, description, price per row), and this project currently treats everything as a flat sequence of labeled words, losing that row/column structure entirely.

## Stack

- PyTorch (CUDA)
- Hugging Face `transformers` (LayoutLMv3-base)
- Hugging Face `datasets` (`mp-02/sroie`)
- `seqeval` for evaluation
- Tesseract OCR (`pytesseract`) for inference on new images
- FastAPI (planned) for a minimal demo endpoint
