# TP04 — Fine-Tuning Gemma 3 with LoRA and Unsloth

This README follows the four TP04 learning objectives:

1. **Course Summary:** Understand the LLM fine-tuning workflow, Gemma 3, LoRA, PEFT, and dataset preparation.
2. **Notebook Configuration:** Understand and modify training hyperparameters.
3. **Loss Analysis:** Monitor and interpret Training Loss and Validation Loss to identify overfitting and underfitting.
4. **Practical Experiment:** Fine-tune the model using a different dataset and compare the results.

The work is in `Gemma3_(4B).ipynb`. The new dataset is `dataasilida_cleaned.csv`.

### Objective coverage

| Teacher objective | Where it is covered | Evidence |
| --- | --- | --- | 
| 1. Course Summary | Section 1 | Workflow, Gemma 3 chat template, LoRA/PEFT, dataset preparation |
| 2. Notebook Configuration | Section 2 | Hyperparameter tables, modified settings, step-by-step run |
| 3. Loss Analysis | Section 3 | Train/validation loss definitions, diagnosis table, results to record |
| 4. Practical Experiment | Section 4 | New dataset, quantitative and qualitative comparison |

---

## Files

| File | Purpose |
| --- | --- |
| `Gemma3_(4B).ipynb` | Unsloth notebook adapted for this TP |
| `dataasilida_cleaned.csv` | Khmer customer-support question/answer dataset |
| `README.md` | Notes, configuration, and experiment report |

**Environment:** Google Colab with a GPU (the notebook was designed for Colab). Upload `dataasilida_cleaned.csv` to the same folder as the notebook before running it. The first cells install Unsloth, TRL, and the Gemma 3 version of Transformers.

**Environment actually used for this submission:** Windows 11 + WSL2 Ubuntu 26.04 on a laptop with an NVIDIA RTX 4060 Laptop GPU (8 GB VRAM) and 7 GB RAM. Python 3.12 virtual environment with torch 2.14.1+cu130, Unsloth 2026.11.3 and Transformers 5.17.0. The notebook was executed locally from top to bottom with `nbconvert --execute` and finished with **0 errors**; every output shown in `Gemma3_(4B).ipynb` comes from that run. Because the laptop GPU has 8 GB, the model cell loads the pre-quantised 4-bit checkpoint `unsloth/gemma-3-4b-it-unsloth-bnb-4bit` (identical weights and settings to `unsloth/gemma-3-4b-it`, quantised ahead of time so the download is 4.6 GB instead of 8.5 GB). **No training hyperparameter from section 2.2 was changed.** The pip-install cells detect an existing Unsloth installation and skip themselves, so the notebook still runs unchanged on Colab.

---

## 1. Course Summary — Workflow, Gemma 3, LoRA, PEFT, and Data Preparation

### 1.1 What is fine-tuning?

Fine-tuning means taking a model that has already been pretrained on a large general corpus and continuing its training on a smaller, task-specific dataset.

The basic workflow is:

1. Choose a pretrained model.
2. Prepare task-specific input/output examples.
3. Format each example with the model's chat template.
4. Train the model (or, more efficiently, train small adapter weights).
5. Evaluate on examples that were not used for training.
6. Save the adapted model or adapter for inference.

The model keeps the general knowledge learned during pretraining and learns the style, format, and facts of the new task from our examples.

### 1.2 The workflow used in this TP

```text
dataasilida_cleaned.csv
        ↓
pandas DataFrame (input, output)
        ↓
drop rows with empty answers (521 → 514 usable pairs)
        ↓
90 / 10 train–validation split (462 train / 52 validation)
        ↓
Gemma 3 chat template applied to every pair
        ↓
SFTTrainer tokenises the "text" column
        ↓
train_on_responses_only masks the user question
        ↓
LoRA fine-tuning + validation loss every 10 steps
        ↓
loss plot, held-out generation, saved LoRA adapters
```

### 1.3 Gemma 3

Gemma 3 is Google's instruction-tuned model family. This notebook loads `unsloth/gemma-3-4b-it` in 4-bit quantisation so that a 4-billion-parameter model fits on a modest GPU.

Gemma 3 uses turn markers. A rendered training example looks like this:

```text
<bos><start_of_turn>user
What is 3D Secure?<end_of_turn>
<start_of_turn>model
3D Secure is an online payment security feature ...<end_of_turn>
```

Important details:

- `<bos>` marks the beginning of the sequence.
- `<start_of_turn>user` and `<start_of_turn>model` identify the speaker.
- `<end_of_turn>` marks the end of each turn.
- The tokenizer's chat template produces these tokens, so we do not write them by hand.

The notebook uses Unsloth's `get_chat_template(tokenizer, chat_template="gemma-3")`, then `tokenizer.apply_chat_template(...)` to build the training text.

### 1.4 LoRA and PEFT

LoRA means **Low-Rank Adaptation**. It is a Parameter-Efficient Fine-Tuning (PEFT) method.

Instead of updating all model weights, LoRA freezes the original model and inserts small trainable matrices into selected layers. Only those adapters are updated during training. This gives three benefits:

- far fewer trainable parameters (about 15M out of 4B, roughly 0.37%);
- much lower GPU memory use;
- faster training and smaller saved files.

In this notebook the adapters are added to the language, attention, and MLP layers. The vision layers are switched off because the dataset is text only.

| LoRA setting | Value | Meaning |
| --- | --- | --- |
| `finetune_vision_layers` | `False` | Text-only task |
| `finetune_language_layers` | `True` | Train the language layers |
| `finetune_attention_modules` | `True` | Add adapters to attention |
| `finetune_mlp_modules` | `True` | Add adapters to the feed-forward layers |
| `r` | `8` | Rank of the adapter; larger = more capacity, more memory |
| `lora_alpha` | `8` | Scales the adapter update; `alpha / r = 1` here |
| `lora_dropout` | `0` | No dropout on the adapter |
| `bias` | `"none"` | Bias terms are not trained |
| `random_state` | `3407` | Seed for reproducibility |

A larger `r` does not automatically mean better results. It increases capacity and overfitting risk, so the rank should be chosen experimentally.

### 1.5 Dataset preparation

The dataset contains Khmer questions and answers about ACLEDA Bank products and services, such as debit cards, ATMs, POS machines, e-commerce, ACLEDA Authentication, and 3D Secure.

The raw CSV has two quirks:

1. Its header is stored as the single quoted field `"input,output"`, so a normal CSV reader treats the whole header as one column.
2. Seven rows have an empty answer.

The notebook handles both issues without modifying the original file:

```python
df = pd.read_csv(
    "dataasilida_cleaned.csv",
    header=None, skiprows=1, names=["input", "output"],
    keep_default_na=False, na_values=[""],
)
df["input"] = df["input"].fillna("").str.strip()
df["output"] = df["output"].fillna("").str.strip()
df = df[(df["input"] != "") & (df["output"] != "")].reset_index(drop=True)
```

Result: **514 usable question/answer pairs**, split into **462 training** and **52 validation** examples.

Each pair is then converted to Gemma 3 chat format by `formatting_prompts_func`, and stored in a new `text` column:

```python
messages = [
    {"role": "user", "content": instruction},
    {"role": "assistant", "content": output},
]
text = tokenizer.apply_chat_template(
    messages, add_generation_prompt=False, tokenize=False
)
```

Finally, `train_on_responses_only` masks the user turns with `-100`, so the loss is computed only on the assistant answers.

---

## 2. Notebook Configuration — Training Hyperparameters

### 2.1 Model configuration

| Setting | Value |
| --- | --- |
| Base model | `unsloth/gemma-3-4b-it` (this run used the pre-quantised `unsloth/gemma-3-4b-it-unsloth-bnb-4bit` variant: same weights, quantised ahead of time to fit an 8 GB laptop GPU) |
| Quantisation | 4-bit (`load_in_4bit=True`) |
| `max_seq_length` | `2048` |
| Full fine-tuning | `False` (LoRA only) |
| Adapter rank `r` | `8` |
| `lora_alpha` | `8` |

### 2.2 Training hyperparameters

| Parameter | Value | What it controls | Effect of increasing it |
| --- | --- | --- | --- |
| `per_device_train_batch_size` | `2` | Examples per GPU per step | More memory, more stable gradient |
| `gradient_accumulation_steps` | `4` | Steps accumulated before one update | Larger effective batch without extra memory |
| Effective batch size | `2 × 4 = 8` | Examples per optimiser update | — |
| `num_train_epochs` | `3` | Full passes over the training split | Better fit, but higher overfitting risk |
| `max_steps` | removed | Step limit (was `60`) | Removed so all 3 epochs run |
| `learning_rate` | `2e-4` | Size of each weight update | Faster learning, but divergence risk |
| `lr_scheduler_type` | `linear` | How the learning rate decays | — |
| `warmup_steps` | `5` | Steps before reaching the full learning rate | — |
| `weight_decay` | `0.01` | Regularisation | Less overfitting, slower fitting |
| `optim` | `adamw_8bit` | Optimiser | Memory-efficient AdamW |
| `logging_steps` | `1` | How often training loss is logged | — |
| `eval_strategy` | `steps` | When validation runs | — |
| `eval_steps` | `10` | Validation every 10 optimiser steps | — |
| `save_strategy` | `no` | No intermediate checkpoints | Saves disk space |
| `packing` | `False` | Packs short sequences together | Faster training, but changes example boundaries |
| `fp16` / `bf16` | automatic | Mixed-precision mode | — |
| `seed` | `3407` | Reproducibility | — |

With 462 training examples and an effective batch size of 8, one epoch is about **58 optimiser steps**. Three epochs are about **174 steps**.

The training settings are in the `TrainingArguments` block:

```python
args = TrainingArguments(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_steps = 5,
    num_train_epochs = 3,
    learning_rate = 2e-4,
    eval_strategy = "steps",
    eval_steps = 10,
    save_strategy = "no",
    ...
)
```

> **Version note:** Recent Transformers versions use `eval_strategy`. Versions before 4.41 use `evaluation_strategy`. The notebook installs a Gemma 3-compatible Transformers release, so `eval_strategy` is correct here.

### 2.3 What was changed from the original notebook

| Original | Adapted for this TP | Reason |
| --- | --- | --- |
| `data_kh.csv` | `dataasilida_cleaned.csv` | New dataset for the practical experiment |
| `load_dataset("csv", ...)` | pandas loader with fixed header | The CSV header is malformed for a normal loader |
| No validation split | `train_test_split(test_size=0.1)` | Needed to measure validation loss |
| `eval_dataset` missing | `eval_dataset = dataset["test"]` | Enables evaluation during training |
| `max_steps=60` | `num_train_epochs=3` | Full training run with meaningful curves |
| No evaluation logging | `eval_strategy="steps"`, `eval_steps=10` | Validation loss is recorded for the plot |
| Robot-command inference prompts | Held-out e-commerce question | Tests generalisation on the new dataset |
| No loss plot | Loss analysis cell | Objective 3 |

### 2.4 Run the notebook step by step

1. Open `Gemma3_(4B).ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → GPU**.
3. Upload `dataasilida_cleaned.csv` to `/content` (the same folder as the notebook).
4. Run the installation cells. If Colab asks you to restart the runtime, restart it and continue from the model-loading cell.
5. Run the model-loading cell and the LoRA configuration cell.
6. Run the data-loading and formatting cells. You should see `514 usable instruction/response pairs` and a `DatasetDict` with 462 train / 52 test rows.
7. Inspect `dataset["train"]["text"][0]`. The text must start with `<bos><start_of_turn>user` and end with `<end_of_turn>`.
8. Build the `SFTTrainer`. Check that `eval_dataset = dataset["test"]` is present.
9. Run the masking cell (`train_on_responses_only`). Then decode `trainer.train_dataset[100]["labels"]`. Only the assistant answer should remain visible; the question should be masked.
10. Run `trainer.train()`. Validation loss should appear in the log every 10 steps.
11. Run the loss-analysis cell. It prints the final metrics and shows the training/validation loss plot.
12. Run the inference cell to answer a held-out question, then compare it with the reference answer.
13. Optionally run the saving cell to store the LoRA adapters in `gemma-3/`.

**Expected resources:** a free Colab T4 GPU is enough for Gemma 3 4B in 4-bit with this small dataset. Training takes roughly 10–20 minutes depending on GPU and runtime conditions.

In this submission the notebook was executed **locally** on an RTX 4060 Laptop GPU (8 GB) under WSL2: training the full 174 steps took **753.9 s (12.6 min)**, peak reserved memory was **4.895 GB**, and the whole notebook (install-skip, load, tokenise, train, evaluate, plot, generate, save) took about 16 minutes.

---

## 3. Loss Analysis — Training vs Validation Loss

### 3.1 What the two losses mean

- **Training loss** is the cross-entropy loss on the training batches. Because of response-only masking, it is computed only on the assistant answer tokens, not on the user question.
- **Validation loss** (`eval_loss`) is the same loss measured on the 52 held-out examples every 10 steps. The model does not train on these examples.
- **Perplexity** is `exp(loss)`. Lower is better; it can be interpreted as how "surprised" the model is by the correct answers.

The notebook prints:

- average training loss;
- last logged training loss;
- best validation loss and the step where it occurred;
- last validation loss;
- final held-out loss and perplexity;
- a plot of training loss and validation loss against step.

### 3.2 How to diagnose the run

| Pattern | Diagnosis | What to do |
| --- | --- | --- |
| Training loss falls, validation loss also falls | The model is learning and generalising | Continue or stop when validation loss flattens |
| Training loss keeps falling, validation loss flattens or rises | Overfitting | Reduce `num_train_epochs`, raise `lora_dropout` or `weight_decay`, add data |
| Training loss stays high and flat | Underfitting | Train longer, raise `learning_rate`, increase `r` / `lora_alpha` |
| Both losses fall, then validation loss rises sharply | Overfitting after the best point | Use the step/epoch with the lowest validation loss; stop earlier |
| Validation loss is much higher than training loss from the start | Small validation set or a harder split | Compare the trend rather than single values; use a larger validation split if possible |

A small gap between training and validation loss is normal. The model has seen the training answers many times but has never seen the validation answers.

### 3.3 Results to record

Complete this table after running the notebook.

| Metric | Value |
| --- | --- |
| Usable pairs | 514 |
| Training examples | 462 |
| Validation examples | 52 |
| Epochs | 3 |
| Total optimiser steps | ~174 |
| Average training loss | 1.2723 |
| Last logged training loss | 0.9467 |
| Best validation loss (step) | 1.0019 (step 170) |
| Last validation loss | 1.0026 |
| Held-out loss | 1.0026 |
| Held-out perplexity | 2.73 |
| Training time | 753.9 s = 12.57 min |
| Peak reserved GPU memory | 4.895 GB (61.2 % of 8.0 GB) |

**Diagnosis from this run:** average training loss was 1.2723 and the last logged training loss was 0.9467, so the model kept improving over the 174 steps. Validation loss fell to its lowest value of **1.0019 at step 170** (out of 174) and finished at 1.0026 — both curves flatten together and the validation curve never turns upward. This is a **good fit, not overfitting**: had the model begun to memorise, validation loss would have risen while training loss kept falling. The final gap between training loss (0.9467) and validation loss (1.0026) is about 0.06, which is normal with only 52 unseen validation examples. Held-out perplexity of 2.73 confirms the model is rarely surprised by the correct answer text. The training/validation loss plot is embedded in the executed notebook and also saved as `loss_curve.png`.

### 3.4 Notes for a fair comparison

- The validation split has only 52 examples, so `eval_loss` is noisy. Look at the overall trend, not a single point.
- Do not compare losses from two runs that use different tokenizers or different sequence lengths. This TP compares runs that use the same Gemma 3 tokenizer.
- If validation loss starts increasing while training loss is still decreasing, the model is beginning to memorise the training set. That is the signal to stop earlier or regularise more.
- In the current Unsloth version, response-only masking is applied to the validation split too. With an older Unsloth release, validation loss may include the prompt tokens and will therefore be higher; in that case compare the shape of the curve, not the absolute value.

---

## 4. Practical Experiment — New Dataset and Comparison

### 4.1 Datasets compared

| | Original notebook | This experiment |
| --- | --- | --- |
| File | `data_kh.csv` | `dataasilida_cleaned.csv` |
| Domain | Khmer robot-navigation commands (move/turn actions) | Khmer ACLEDA Bank customer-support Q&A |
| Answer style | Short JSON-like action objects | Natural-language explanations |
| Raw examples | Larger dataset used by the original notebook | 521 rows → 514 usable |
| Validation split | None | 52 held-out examples |
| Epochs | 60-step demo | 3 full epochs |

### 4.2 What the experiment tests

The experiment changes the task domain while keeping the same model, tokenizer, LoRA settings, and training pipeline. This makes the comparison meaningful:

- **Same model and tokenizer:** differences come from the data, not from a different base model.
- **Same LoRA configuration:** adapter capacity is unchanged.
- **New domain:** the model must learn a customer-support style instead of robot commands.
- **Held-out validation:** the loss measures generalisation, not memorisation.

### 4.3 Quantitative comparison

Fill in the table from the notebook outputs. Use your earlier `data_kh.csv` run for the left column if you have those numbers; otherwise write `N/A`.

| Metric | Original run (`data_kh.csv`) | New run (`dataasilida_cleaned.csv`) |
| --- | --- | --- |
| Domain | Robot navigation | Bank customer support |
| Training examples | N/A | 462 |
| Validation examples | 0 | 52 |
| Epochs / steps | 60-step demo | 3 epochs / 174 steps |
| Average training loss | N/A | 1.2723 |
| Best validation loss (step) | N/A | 1.0019 (step 170) |
| Last validation loss | N/A | 1.0026 |
| Held-out loss | N/A | 1.0026 |
| Held-out perplexity | N/A | 2.73 |
| Training time | N/A | 12.57 min |
| Peak reserved memory | N/A | 4.895 GB |
| Overfitting observed? | N/A | No — validation loss reached its minimum at step 170 of 174 and stayed flat |

> A fair quantitative comparison with the original `data_kh.csv` run is not possible because that run had no validation split and was only a 60-step demo; the important comparison is qualitative (see 4.4) plus the fact that the same model, tokenizer, LoRA and training settings were kept.

### 4.4 Qualitative comparison

The notebook prints one held-out example. Record the question, the reference answer, and the model answer.

**Held-out question (Khmer):**

> តើសេវាសុវត្ថិភាព 3D អ៉ីអន ជាអ្វី

**English meaning:** "What is the 3D Secure (ION) service?"

**Reference answer (excerpt):**

> សេវាសុវត្ថិភាព 3D អ៉ីអន ជាមុខងារសុវត្ថិភាពមួយ ដែលផ្តល់ជូនដោយ ធនាគារឯកទេស អ៉ីអន (ខេមបូឌា) ភីអិលស៊ី។ ...

**Model answer (generated in this run, temperature 1.0 / top_p 0.95 / top_k 64, 256 new tokens):**

> សេវា 3D Secure អ៉ីអន (3DS) គឺជាការប្រើប្រាស់សេវាសុវត្ថិភាពលើកម្មវិធីធនាគារអ៉ីអន ដើម្បីការពារការធ្វើប្រតិបត្តិការតាមអនឡាញដោយខ្លួនឯង របស់អតិថិជនដោយមិនបណ្តាលមកដល់ការទូទាត់ក្នុងប្រតិបត្តិការសុទ្ធរាល់កម្រិត ដែលទាក់ទងនឹងប្រតិបត្តិការតាមអនឡាញ ដោយផ្អែកតាមឧបករណ៍សម្គាល់អត្តសញ្ញាណដែលមានស្រាក្នុងខ្លួន។

**English gist:** "ION 3D Secure (3DS) is the use of a security service on the ION Bank app to protect customers' own online transactions without affecting payments at any level, based on identification built into the device."

**Comparison checklist:**

- Does the answer stay in the same domain (banking / cards / online payments)? **Yes** — it explains ION 3D Secure and online transaction protection.
- Is it in Khmer, matching the dataset? **Yes** — Khmer prose with the product name kept in Latin script, as in the reference.
- Is it factually consistent with the reference answer? **Mostly** — it covers the purpose (protecting online card transactions on ION channels) but omits the reference's 6-digit one-time-password detail.
- Does it copy the reference word-for-word (memorisation) or paraphrase it? **Paraphrases** — different wording and sentence structure, no verbatim copying.
- Is the answer cut off because `max_new_tokens` was too small? **No** — it ends with a complete sentence.

> Optional baseline: run the inference cell once **before** `trainer.train()` to capture the base-model answer. At that point the LoRA adapters are initialised but not trained, so the output is close to the original Gemma 3 behaviour. Then compare base vs fine-tuned answers for the same question.

### 4.5 Conclusion template

Complete these sentences after the run:

1. The new dataset contains **514** usable question/answer pairs about **ACLEDA/ION bank customer support (cards, ATMs, POS, e-commerce, 3D Secure)**.
2. Training loss decreased from an average of **1.2723** to a final logged value of **0.9467** over **174** steps (3 epochs).
3. Validation loss reached its lowest value of **1.0019** at step **170**, and then **stayed flat (1.0026 at the end), showing no overfitting**.
4. This indicates that the model **fitted well** because **the validation curve kept falling until the last steps and stayed within ~0.06 of the training loss, with a held-out perplexity of 2.73**.
5. Compared with the original `data_kh.csv` run, the new model **cannot be compared quantitatively (the original run had no validation split and only 60 demo steps), but qualitatively it answers held-out banking questions in Khmer with in-domain content**, so the new dataset run demonstrates successful domain adaptation.
6. If I repeated the experiment, I would change **the dataset size and the evaluation frequency (more unique examples, validation every 5 steps)** because **only 52 validation examples makes `eval_loss` noisy and the validation minimum was reached at step 170 of 174, so more data and finer validation would sharpen the overfitting diagnosis**.

---

## Appendix A — Troubleshooting

| Problem | Cause | Fix |
| --- | --- | --- |
| Only one column named `input,output` | The CSV header is quoted as a single field | Use the pandas loader in the notebook, or change the first line of the CSV to `input,output` |
| `KeyError: 'input'` | The loader did not find the expected columns | Use `header=None, skiprows=1, names=["input", "output"]` as in the notebook |
| `Invalid key: 0. Please first select a split` | A `DatasetDict` was passed where a split was expected | Pass `dataset["train"]` / `dataset["test"]` |
| CUDA out of memory | Batch or sequence length too large | Use `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`, or lower `max_seq_length` |
| `eval_strategy` is not accepted | Older Transformers version | Use `evaluation_strategy` instead |
| All labels are `-100` / loss is `NaN` | The response marker was not found or truncation removed it | Check the chat template and markers; increase `max_seq_length` |
| Training loss does not decrease | Learning rate too low, or no trainable parameters | Check the LoRA cell and raise the learning rate carefully |
| Colab disconnects during training | Free Colab session limit | Save the adapters, mount Google Drive, or use checkpointing |
| Model download fails | Gated model or missing token | Add a Hugging Face token to `FastModel.from_pretrained(..., token="hf_...")` |
| Installation asks for a restart | Unsloth replaced installed packages | Restart the runtime and rerun from the model-loading cell |

---

## Appendix B — Submission Checklist

- [x] Notebook runs from top to bottom without errors.
- [x] Dataset loads as 514 usable pairs (462 train / 52 validation).
- [x] A sample of `dataset["train"]["text"][0]` uses the Gemma 3 chat template.
- [x] Response-only masking is verified (the question is hidden in the labels).
- [x] Training completes for 3 epochs with validation every 10 steps.
- [x] The training/validation loss plot is generated.
- [x] The results table in section 3.3 is filled in.
- [x] The quantitative comparison table in section 4.3 is filled in.
- [x] The held-out model answer is added to section 4.4 and compared with the reference.
- [x] The conclusion in section 4.5 is completed.
- [x] LoRA adapters are saved (optional but recommended).

---

## Appendix C — Key Terms

| Term | Meaning |
| --- | --- |
| Pretraining | Initial training on a very large general corpus |
| Fine-tuning | Additional training on a smaller task-specific dataset |
| PEFT | Parameter-Efficient Fine-Tuning; update only a small part of the model |
| LoRA | Low-Rank Adaptation; adds small trainable matrices to frozen layers |
| Adapter | The small set of trainable LoRA weights |
| Chat template | The exact text format the model expects for a conversation |
| Epoch | One complete pass over the training set |
| Batch size | Number of examples processed before one weight update |
| Gradient accumulation | Accumulating gradients over several batches to mimic a larger batch |
| Learning rate | Step size for weight updates |
| Training loss | Loss on the training examples |
| Validation loss | Loss on held-out examples not used for training |
| Overfitting | Training loss improves while validation loss worsens (memorisation) |
| Underfitting | Both losses remain high because the model has not learned enough |
| Perplexity | `exp(loss)`; a measure of how surprised the model is by the correct text |

---

## Appendix D — References

- Unsloth Gemma 3 notebook and blog: <https://unsloth.ai/blog/gemma3>
- Unsloth documentation: <https://docs.unsloth.ai/>
- TRL `SFTTrainer`: <https://huggingface.co/docs/trl/sft_trainer>
- Hugging Face PEFT: <https://huggingface.co/docs/peft>
- LoRA paper (Hu et al., 2021): <https://arxiv.org/abs/2106.09685>
- Gemma 3 model documentation: <https://ai.google.dev/gemma/docs>
