# Receipt AI Extractor

A fine-tuned LayoutLMv3 model that reads receipt/invoice images and pulls out the important fields — company name, date, address, total amount — automatically, instead of typing them in by hand.

I built this as a resume/learning project to actually understand document AI end to end — environment setup, data quirks, preprocessing, training on a laptop GPU, evaluation, inference, and an API around it. It's not polished, and I'm not pretending otherwise. Below is what's done, how it works, and where it's genuinely weak.

## What it does

Give it a receipt image → OCR reads the text and where each word sits → that goes into LayoutLMv3 (a model that understands text, position, *and* the image together) → it labels which words belong to which field → out comes structured data like:

```json
{"company": "Main Street Restaurant", "address": "6332 Business Drive Suite 528 Palo Alto California 94301", "date": "Fri 04/07/2017", "total": "29.01"}
```

It only extracts what's already there. No generation, no summarizing. Smart auto-fill, basically.

## Try it yourself (run locally)

The fine-tuned model is hosted on Hugging Face Hub: [`KamiSan04/receipt-ai-extractor`](https://huggingface.co/KamiSan04/receipt-ai-extractor).

I looked into hosting a live demo on Hugging Face Spaces, but Spaces recently restricted the Gradio SDK (the thing that would actually let a browser talk to a Python model) to paid accounts only for new Spaces. The remaining free "static" templates all run client-side in the browser and can't load a ~500MB multimodal transformer model without a much bigger separate effort (converting to ONNX, browser memory limits, etc.) — a different project in its own right, not a quick add-on. So for now, this runs locally instead of on a hosted demo page, which is a completely normal way to share a project like this:

```bash
git clone https://github.com/YOUR_USERNAME/receipt-ai-extractor.git
cd receipt-ai-extractor

python -m venv docai-env
docai-env\Scripts\activate   # Windows

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt

uvicorn api:app --reload
```

Then open `http://127.0.0.1:8000/docs`, try the `/extract` endpoint, and upload a receipt image.

(The `cu132` PyTorch index URL is what matched my GPU/driver — check https://pytorch.org/get-started/locally/ for the right one for your machine.)

## Why LayoutLMv3

Plain text models ignore position. On a receipt, position is half the signal — company name is usually top, total is usually near a line that literally says "TOTAL." LayoutLMv3 uses text + bounding boxes + the raw image together, so it can actually use that.

## Hardware

Trained entirely locally — no Colab, no cloud GPU:
- RTX 4050 Laptop GPU, 6GB VRAM
- Windows, VS Code
- Python venv (`docai-env`) to keep dependencies isolated
- PyTorch built for CUDA 13.2 (matched to the driver, not guessed)

6GB is tight for a transformer + vision encoder combo, so training uses a small batch size (2), gradient accumulation (8 steps, so effective batch ~16), and fp16 mixed precision to cut memory roughly in half.

## How it was built

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

**Inference** — a script that takes any new receipt image, runs it through Tesseract OCR to get words + positions, feeds that into the fine-tuned model, and returns clean JSON. Tested on an actual photographed receipt (not a clean scan) — company, address, and date came back correct.

**Total fallback heuristic** — since the model alone misses close to half of all totals, I added a rule-based backup: if the model finds no TOTAL field, scan the raw OCR words for the literal word "total" (excluding "sub total") and grab the number that follows it. This is a patch, not a fix — it only works if OCR read the word "total" correctly in the first place. On my own test photo, it didn't: Tesseract misread the "Total" line as the word "sale" due to lighting/print quality, so the real value never made it into the OCR output at all. Neither the model nor the fallback could recover it — the information was lost before either of them ever saw it. I left this exact result in rather than swap in a cleaner test image, because it's an honest, common failure mode for OCR pipelines, not a bug in the code.

**API** — wrapped the inference function in a FastAPI app with a single `POST /extract` endpoint: upload an image, get JSON back. Tested through FastAPI's auto-generated `/docs` page.

**Model hosting** — pushed the fine-tuned model and processor config to Hugging Face Hub at `KamiSan04/receipt-ai-extractor`, so it's downloadable without needing my local files.

## What's not done

- Live hosted demo (see the Spaces note above — currently run-locally only)
- No line-item extraction (itemized list on a receipt) — only 4 flat fields right now, which is the bigger real-world gap

## Known limitations, stated plainly

- SROIE is a small (973 total receipts), clean, English-only academic dataset. Real-world receipts — creased, faded thermal paper, non-English, phone-camera angles — will likely break this model in ways SROIE never tests.
- TOTAL field extraction is mediocre (0.63 F1), the weakest part of the whole pipeline, and arguably the field people care about most.
- OCR errors cascade. If Tesseract misreads a word — which it did, in my own test — no downstream logic (model prediction or rule-based fallback) can recover information that was never correctly read in the first place.
- The fallback heuristic for TOTAL only helps when OCR gets the word "total" right; it does nothing when OCR itself fails.
- Trained on a small effective batch size due to VRAM limits. A bigger GPU would probably squeeze out better numbers, especially on the weaker TOTAL field.
- No line-item table understanding — real invoices have structured item lists (qty, description, price per row), and this project currently treats everything as a flat sequence of labeled words, losing that row/column structure entirely.
- No hosted demo — currently requires cloning the repo and running it locally (see above for why).

## Stack

- PyTorch (CUDA)
- Hugging Face `transformers` (LayoutLMv3-base)
- Hugging Face `datasets` (`mp-02/sroie`)
- `seqeval` for evaluation
- Tesseract OCR (`pytesseract`) for inference on new images
- FastAPI + Uvicorn for the API
- Hugging Face Hub for model hosting
