# Receipt AI Extractor

A fine-tuned LayoutLMv3 model that reads receipt/invoice images and pulls out the important stuff — vendor name, date, address, total amount — automatically, instead of you typing it in by hand.

This is a learning project. I built it to actually understand how document AI works under the hood, not just call an API and call it a day. So expect some rough edges — see the "What's not great" section below, I'm not going to pretend it's production-ready.

## What it actually does

You give it a receipt image → it runs OCR to read the text and find where each word sits on the page → feeds that into LayoutLMv3 (a transformer model that understands text *and* layout *and* the image itself, together) → outputs which words belong to which field.

It's not generating anything new. It's just labeling what's already there. Think smart auto-fill for a form, powered by a model that's seen enough receipts to guess the pattern.

## Why LayoutLMv3

Normal NLP models only look at text. But on a receipt, position matters a lot — the total is usually bottom-right-ish, near a line that says "TOTAL." A model that only reads text left-to-right misses that. LayoutLMv3 uses the words, their bounding boxes (position), and the raw image together, so it can pick up on those layout cues.

## Dataset

Trained on SROIE — 626 training receipts, 347 test receipts, labeled with company name, date, address, and total. It's a small, well-worn academic dataset, not real-world messy receipts. That's a limitation, not a secret — more on that below.

## Hardware / why this is tuned the way it is

Trained locally on an RTX 4050 laptop GPU — 6GB VRAM. That's tight for a transformer + vision encoder combo, so:
- Small batch size (2-4)
- Gradient accumulation to simulate a bigger batch without the memory cost
- Mixed precision (fp16) to cut memory usage roughly in half

No cloud GPU, no Colab crutch. If you're also GPU-poor, this repo should actually work for you without modification.

## What's not great (being honest here)

- **SROIE is a clean, small, English-only dataset.** Real receipts are creased, blurry, in different languages, and printed on thermal paper that fades. This model has not seen that. It will probably choke on a genuinely messy real-world receipt.
- **Only 4 fields.** No line-item extraction yet (that's the hard, actually-useful part for real invoicing use cases — todo).
- **OCR is a separate, imperfect step.** If OCR misreads a word, the model never gets a chance to fix that — garbage in, garbage out.
- **Small training set.** 626 examples is not a lot for a transformer. It works because we're fine-tuning a model that's already pretrained on tons of data, not training from scratch, but it's still a small dataset and probably overfits somewhat to SROIE's specific formatting quirks.
- **No line-item table structure** — an invoice usually has a table of items; this project treats everything as flat labeled words, which loses the row/column relationships in an itemized list.
- This was trained on a laptop GPU with small batches. It works, but a properly resourced training run would likely do noticeably better.

## Status

Work in progress — built step by step (environment setup → data loading → preprocessing → training → evaluation → inference → API), documented as I actually learned each piece rather than written after the fact.

## Stack

- PyTorch (CUDA)
- Hugging Face `transformers` (LayoutLMv3)
- Hugging Face `datasets` (SROIE)
- `seqeval` for evaluation
- Tesseract OCR (`pytesseract`) for inference on new images
- FastAPI for the demo endpoint
