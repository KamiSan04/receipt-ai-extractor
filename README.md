# Receipt AI Extractor

A fine-tuned LayoutLMv3 model that reads receipt/invoice images and pulls out the important fields — company name, date, address, total amount — automatically, instead of typing them in by hand.

I built this as a resume/learning project to actually understand document AI end to end — environment setup, data quirks, preprocessing, training on a laptop GPU, evaluation, and (eventually) a small API around it. It's not polished, and I'm not pretending otherwise. Below is what's done, how it works, and where it's genuinely weak.

## What it does

Give it a receipt image → OCR reads the text and where each word sits → that goes into LayoutLMv3 (a model that understands text, position, *and* the image together) → it labels which words belong to which field → out comes structured data.

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

**Environment setup** — venv, PyTorch (CUDA-matched build), transformers, datasets, seqeval, pytesseract, pillow, accelerate. Verified CUDA was actually detected before writing a single line of model code — no point building on top of a CPU fallback and not realizing it.

**Dataset** — started with `darentang/sroie`, which turned out to be broken on a modern Python version (its loading script depends on an old caching library that doesn't work anymore). Switched to `mp-02/sroie` instead, which bundles the actual images directly rather than a path string pointing at nothing. Small annoyance, real lesson: not every dataset on the Hub still works out of the box.

**Preprocessing** — tokenized text, fed words + bounding boxes + image into LayoutLMv3's processor, aligned labels to tokens (sub-word splits mean one word can become several tokens, so only the first one keeps the real label, the rest get ignored during loss calculation). Hit two real bugs here worth mentioning honestly:
- Assumed the bounding boxes needed manual normalization to LayoutLMv3's 0–1000 scale. They didn't — this dataset already ships pre-normalized. Doing it anyway pushed some coordinates out of valid range and crashed training with a cryptic CUDA assertion. Fix was to just... not normalize them again.
- The processor adds an extra batch dimension to image tensors even for a single image, which quietly breaks once you batch more than one example together. Had to manually strip that dimension.

Both were the kind of bug where the error message points at CUDA internals but the actual problem is upstream data shape — a good reminder that GPU errors aren't always GPU problems.

**Training** — fine-tuned LayoutLMv3-base for 3 epochs, batch size 2, gradient accumulation 8, fp16. Took about 19 minutes on the laptop GPU. Loss dropped from ~8.0 to ~0.2, eval loss ended at ~0.05. No OOM crashes with these settings, though there isn't a huge cushion of headroom left on 6GB.

**Evaluation** — per-field precision/recall/F1 using seqeval. Results:

| Field     | Precision | Recall | F1   |
|---        |---        |---     |---   |
| Address   | 0.98      | 0.98   | 0.98 |
| Company   | 0.93      | 0.99   | 0.96 |
| Date      | 0.93      | 0.97   | 0.95 |
| Total     | 0.73      | 0.56   | 0.63 |

Address, company, and date are genuinely strong. Total is the weak point — the model misses close to half of actual totals, and isn't always right when it does guess. My read on why: receipts often have several numbers that could plausibly be "the total" (subtotal, tax, amount due, change), and the dataset just has fewer TOTAL examples than the other fields to learn from. This isn't hidden anywhere — it's the most honest number in this whole project.

## What's not done yet

- Inference script (feed it a brand new receipt photo, get JSON back) — in progress
- FastAPI wrapper around inference
- No line-item extraction (itemized list on a receipt) — only 4 flat fields right now, which is the bigger real-world gap
- Not deployed anywhere yet (Hugging Face Hub + Spaces planned)

## Known limitations, stated plainly

- SROIE is a small (973 total receipts), clean, English-only academic dataset. Real-world receipts — creased, faded thermal paper, non-English, phone-camera angles — will likely break this model in ways SROIE never tests.
- TOTAL field extraction is mediocre, as shown above. If this were a real product, this is the field that would need the most work, and it's also arguably the field people care about most.
- OCR (for new images, at inference time) is a separate step from the model — if OCR misreads something, the model has no way to correct it.
- Trained on a small effective batch size due to VRAM limits. A bigger GPU would probably squeeze out better numbers, especially on the weaker TOTAL field.
- No line-item table understanding — real invoices have structured item lists (qty, description, price per row), and this project currently treats everything as a flat sequence of labeled words, losing that row/column structure entirely.

## Stack

- PyTorch (CUDA)
- Hugging Face `transformers` (LayoutLMv3-base)
- Hugging Face `datasets` (`mp-02/sroie`)
- `seqeval` for evaluation
- Tesseract OCR (`pytesseract`) for inference on new images
- FastAPI (planned) for a minimal demo endpoint
