# Hugging Face Transformers Learning Notebooks

A collection of Python notebooks created while working through the Coursera
Hugging Face Transformers tutorial. The notebooks cover tokenization,
task-specific model classes, inference, common Transformers pipelines, model
discovery on the Hugging Face Hub, batched chat-template tokenization,
multilingual sentiment classification, and end-to-end model fine-tuning.

## Notebooks

| Notebook | Topics | Run in Colab |
| --- | --- | --- |
| `HuggingFace_Tokenizer.ipynb` | Loading an `AutoTokenizer`, inspecting tokenizer metadata, encoding text, examining subword and special tokens, and decoding token IDs | [Open in Colab](https://colab.research.google.com/github/abhsrivastava/hugging_face_transformers/blob/main/HuggingFace_Tokenizer.ipynb) |
| `HuggingFace_Transformers.ipynb` | Searching the Hub by pipeline task, sentiment classification, zero-shot image classification with CLIP, transparent-image preprocessing, and Whisper speech recognition | [Open in Colab](https://colab.research.google.com/github/abhsrivastava/hugging_face_transformers/blob/main/HuggingFace_Transformers.ipynb) |
| `Padding_Truncation_AttentionMasks.ipynb` | Applying a model's chat template to a batch, left padding, truncation, PyTorch tensors, and attention masks | [Open in Colab](https://colab.research.google.com/github/abhsrivastava/hugging_face_transformers/blob/main/Padding_Truncation_AttentionMasks.ipynb) |
| `AutoModelsForTask.ipynb` | Choosing a task-specific AutoModel class, inspecting model configuration and label mappings, running sentiment inference, interpreting logits/softmax/argmax, and understanding the classification head | [Open in Colab](https://colab.research.google.com/github/abhsrivastava/hugging_face_transformers/blob/main/AutoModelsForTask.ipynb) |
| `Full_End_to_End_Classification_Example.ipynb` | Building a multilingual sentiment workflow with batched tokenization, padding and attention masks, manual inference, confidence thresholds, business-routing rules, and pipeline comparisons | [Open in Colab](https://colab.research.google.com/github/abhsrivastava/hugging_face_transformers/blob/main/Full_End_to_End_Classification_Example.ipynb) |
| `Fine_Tuning_Example_End_to_End.ipynb` | Fine-tuning ModernBERT on Bugzilla severity labels, using stratified data splits and dynamic padding, evaluating with macro F1 and a confusion matrix, and saving, reloading, and running the fine-tuned model | [Open in Colab](https://colab.research.google.com/github/abhsrivastava/hugging_face_transformers/blob/main/Fine_Tuning_Example_End_to_End.ipynb) |

## What the notebooks demonstrate

- Loading pretrained tokenizers and models with the Transformers API
- Understanding input IDs, special tokens, subword tokens, and decoding
- Discovering popular models with `huggingface_hub.list_models`
- Running text, image, and automatic speech-recognition pipelines
- Using CLIP candidate labels for zero-shot image classification
- Handling PNG transparency before passing an image to a model
- Applying chat templates to multiple conversations in one batch
- Using dynamic padding and attention masks for variable-length inputs
- Choosing `AutoModelForSequenceClassification` for classification tasks
- Inspecting model architecture, configuration, and label mappings
- Running inference with `model.eval()` and `torch.no_grad()`
- Converting logits to probabilities with softmax and selecting a label with argmax
- Comparing a task-specific model with its base `AutoModel`
- Running multilingual sequence classification on a batch of inputs
- Applying confidence thresholds and simple business-routing rules
- Preparing labeled data with cleaning, numeric label encoding, and stratified splits
- Fine-tuning a sequence-classification model with the Hugging Face `Trainer`
- Evaluating a classifier with accuracy, macro F1, a classification report, and a confusion matrix
- Saving a fine-tuned model, reloading it from disk, and using it for batch inference

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
python -m pip install jupyter transformers datasets evaluate accelerate torch torchvision torchaudio huggingface_hub Pillow requests pandas scikit-learn matplotlib
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
- `tabularisai/multilingual-sentiment-analysis`
- `answerdotai/ModernBERT-base`

## Dataset used for fine-tuning

- [`AliArshad/Bugzilla_Eclipse_Bug_Reports_Dataset`](https://huggingface.co/datasets/AliArshad/Bugzilla_Eclipse_Bug_Reports_Dataset) — historical Bugzilla reports labeled as `blocker`, `critical`, `major`, `normal`, `minor`, or `trivial`

## Notes

- Use a task-specific model class when you need a prediction head. The base
  `AutoModel` returns hidden representations but cannot directly classify text.
- ImageNet-style classification chooses from a fixed label set. The image
  example instead uses CLIP for zero-shot classification with candidate labels.
- Attention masks tell a model which padded positions to ignore; padding still
  occupies space in a rectangular batch tensor.
- Sequence classifiers consume ordinary text and do not require chat templates.
- The multilingual sentiment model uses the `CC-BY-NC-4.0` license; review the
  model card and license before commercial use.
- Bug-severity labels are imbalanced and may be noisy. Macro F1 and per-class
  results provide more information than accuracy alone, and high-impact
  predictions should remain subject to human review.
- The notebooks are educational examples rather than production applications.
