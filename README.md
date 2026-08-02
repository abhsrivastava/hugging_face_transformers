# Hugging Face Transformers Learning Notebooks

A collection of Python notebooks created while working through the Coursera
Hugging Face Transformers tutorial. The notebooks cover tokenization,
task-specific model classes, inference, common Transformers pipelines, model
discovery on the Hugging Face Hub, batched chat-template tokenization,
multilingual sentiment classification, and end-to-end model fine-tuning.

## Featured examples

### 1. End-to-end manual inference

[`Full_End_to_End_Classification_Example.ipynb`](Full_End_to_End_Classification_Example.ipynb)
builds a complete multilingual customer-feedback classifier without hiding the
important steps behind a pipeline. It covers batched tokenization, dynamic
padding, attention masks, device placement, `torch.no_grad()`, logits, softmax,
argmax, confidence thresholds, human review, and business-routing rules.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abhsrivastava/hugging_face_transformers/blob/main/Full_End_to_End_Classification_Example.ipynb)

### 2. End-to-end fine-tuning and inference

[`Fine_Tuning_Example_End_to_End.ipynb`](Fine_Tuning_Example_End_to_End.ipynb)
fine-tunes ModernBERT to classify Bugzilla reports into six severity levels. It
covers data cleaning, numeric label encoding, stratified train/validation/test
splits, dynamic padding, Hugging Face `Trainer`, macro F1, classification
reports, a confusion matrix, model persistence, reloading from disk, and batch
inference with the saved model.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abhsrivastava/hugging_face_transformers/blob/main/Fine_Tuning_Example_End_to_End.ipynb)

## More learning notebooks

| Notebook | Topics | Run in Colab |
| --- | --- | --- |
| `HuggingFace_Tokenizer.ipynb` | Loading an `AutoTokenizer`, inspecting tokenizer metadata, encoding text, examining subword and special tokens, and decoding token IDs | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abhsrivastava/hugging_face_transformers/blob/main/HuggingFace_Tokenizer.ipynb) |
| `HuggingFace_Transformers.ipynb` | Searching the Hub by pipeline task, sentiment classification, zero-shot image classification with CLIP, transparent-image preprocessing, and Whisper speech recognition | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abhsrivastava/hugging_face_transformers/blob/main/HuggingFace_Transformers.ipynb) |
| `Padding_Truncation_AttentionMasks.ipynb` | Applying a model's chat template to a batch, left padding, truncation, PyTorch tensors, and attention masks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abhsrivastava/hugging_face_transformers/blob/main/Padding_Truncation_AttentionMasks.ipynb) |
| `AutoModelsForTask.ipynb` | Choosing a task-specific AutoModel class, inspecting model configuration and label mappings, running sentiment inference, interpreting logits/softmax/argmax, and understanding the classification head | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abhsrivastava/hugging_face_transformers/blob/main/AutoModelsForTask.ipynb) |

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

## Choose where to run the notebooks

| Experience | Editor | Compute |
| --- | --- | --- |
| Google Colab in a browser | Colab browser editor | Remote Colab CPU/GPU |
| Google Colab in VS Code | VS Code | Remote Colab CPU/GPU |
| Fully local VS Code | VS Code | Your computer's CPU/GPU |

### Option 1: Google Colab in a browser

Select an **Open in Colab** badge or link in this README. Each notebook installs
its own Python packages, so it can be run from top to bottom in a fresh Colab
runtime. For fine-tuning, select **Runtime > Change runtime type** and choose a
GPU before running the notebook.

The Transformers pipeline notebook reads a Hugging Face access token from
Colab Secrets. Add a secret named `HF_TOKEN`, enable notebook access to it, and
never place the token directly in a notebook cell.

### Option 2: Google Colab inside VS Code

This option provides the VS Code editor while the notebook executes on a remote
Google Colab runtime. Install these two VS Code extensions:

1. [Google Colab](https://marketplace.visualstudio.com/items?itemName=google.colab)
2. [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)

The Colab extension is built on the VS Code Jupyter extension and installs its
required notebook support. Then:

1. Clone this repository and open the folder in VS Code.
2. Open one of the `.ipynb` files.
3. Select **Select Kernel > Colab > Auto Connect**.
4. Sign in to Google when prompted.
5. Run the cells normally from VS Code.

The packages, models, and training outputs in this mode live on the temporary
Colab runtime, not on your local computer. Save anything important to GitHub,
Google Drive, or your computer before disconnecting.

### Option 3: Run fully locally in VS Code

Install [VS Code](https://code.visualstudio.com/), Python 3.10 or newer, and the
same two extensions listed above. Clone and open the repository:

```bash
git clone https://github.com/abhsrivastava/hugging_face_transformers.git
cd hugging_face_transformers
code .
```

In the VS Code terminal, create a project-specific virtual environment and
install the notebook dependencies:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Or activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Then install the packages:

```bash
python -m pip install --upgrade pip
python -m pip install jupyter transformers datasets evaluate accelerate torch torchvision torchaudio huggingface_hub Pillow requests pandas scikit-learn matplotlib
```

Open a notebook, choose **Select Kernel > Python Environments > .venv**, and run
its cells. This mode uses your computer's memory, storage, CPU, and any locally
configured GPU. Model downloads may require several gigabytes of disk space,
and ModernBERT fine-tuning is much faster with a compatible GPU.

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
