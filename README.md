# Hugging Face Transformers Learning Notebooks

A collection of Python notebooks created while working through the Coursera
Hugging Face Transformers tutorial. The notebooks cover tokenization, common
Transformers pipelines, model discovery on the Hugging Face Hub, and batched
chat-template tokenization with padding and attention masks.

## Notebooks

| Notebook | Topics | Run in Colab |
| --- | --- | --- |
| `HuggingFace_Tokenizer.ipynb` | Loading an `AutoTokenizer`, inspecting tokenizer metadata, encoding text, examining subword and special tokens, and decoding token IDs | [Open in Colab](https://colab.research.google.com/github/abhsrivastava/hugging_face_transformers/blob/main/HuggingFace_Tokenizer.ipynb) |
| `HuggingFace_Transformers.ipynb` | Searching the Hub by pipeline task, sentiment classification, zero-shot image classification with CLIP, transparent-image preprocessing, and Whisper speech recognition | [Open in Colab](https://colab.research.google.com/github/abhsrivastava/hugging_face_transformers/blob/main/HuggingFace_Transformers.ipynb) |
| `Padding_Truncation_AttentionMasks.ipynb` | Applying a model's chat template to a batch, left padding, truncation, PyTorch tensors, and attention masks | [Open in Colab](https://colab.research.google.com/github/abhsrivastava/hugging_face_transformers/blob/main/Padding_Truncation_AttentionMasks.ipynb) |

## What the notebooks demonstrate

- Loading pretrained tokenizers and models with the Transformers API
- Understanding input IDs, special tokens, subword tokens, and decoding
- Discovering popular models with `huggingface_hub.list_models`
- Running text, image, and automatic speech-recognition pipelines
- Using CLIP candidate labels for zero-shot image classification
- Handling PNG transparency before passing an image to a model
- Applying chat templates to multiple conversations in one batch
- Using dynamic padding and attention masks for variable-length inputs

## Run in Google Colab

Use an **Open in Colab** link in the table above. Each notebook installs its own
Python dependencies, so you can run the cells from top to bottom in a fresh
Colab runtime.

The Transformers pipeline notebook reads a Hugging Face access token from
Colab Secrets. Add a secret named `HF_TOKEN`, enable notebook access to it, and
never place the token directly in a notebook cell.

## Run locally

Python 3.10 or newer is recommended. Create a virtual environment and install
the packages used by the notebooks:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install jupyter transformers torch torchvision torchaudio huggingface_hub Pillow requests
jupyter notebook
```

Model downloads may require several gigabytes of disk space. Runtime and memory
requirements vary by model.

## Models used

- `distilbert-base-uncased`
- `distilbert-base-uncased-finetuned-sst-2-english`
- `openai/clip-vit-base-patch32`
- `openai/whisper-tiny`
- `HuggingFaceTB/SmolLM2-1.7B-Instruct`

## Notes

- ImageNet-style classification chooses from a fixed label set. The image
  example instead uses CLIP for zero-shot classification with candidate labels.
- Attention masks tell a model which padded positions to ignore; padding still
  occupies space in a rectangular batch tensor.
- The notebooks are educational examples rather than production applications.
