# LLM Fine-Tuning with Unsloth — Lesson Summary

These notes summarize the lesson **LLM Model Fine-Tuning**, presented by **Vann Visal**, using the uploaded classroom slide photos. The photos cover fine-tuning, Unsloth, installation, chat templates, and a LoRA configuration example. Some slides are repeated, so the ideas are grouped by topic here.

## 1. What is fine-tuning?

Fine-tuning means taking a model that has already been trained and continuing its training with data for a particular task. We start with the knowledge the model already has, then help it learn the kind of answers we need.

For example, a general language model can answer many kinds of questions. If we train it further using good examples of customer questions and suitable replies, it may become better at answering questions in that business's style.

The lesson describes three main stages:

1. A model learns general patterns from a large collection of data during pretraining.
2. The pretrained model receives additional training on a smaller, task-specific dataset.
3. The adapted model is evaluated before being used in an application.

The important idea is that we build on an existing model instead of starting its training from zero. Fine-tuning aims to improve a specific behavior, but the result still depends on the data and training choices.

## 2. What is Unsloth?

Unsloth is a Python library used to make model fine-tuning more efficient. The lesson mentions model families such as **Llama 3, Mistral, Phi-4, and Gemma**.

The slides emphasize two practical benefits: faster training and lower memory use. This can make experiments easier when GPU memory is limited.

The introduction slide reports a memory reduction of around **70%**. Treat this as a claim from the lesson rather than a guaranteed result for every model or device. Actual savings depend on the model, hardware, and configuration.

## 3. Preparing the environment

Before installing a training library, we need to check the Python environment and available hardware.

| Item | What the lesson says | What to remember |
| --- | --- | --- |
| Environment | Mentions Linux, Windows, macOS, Google Colab, and Kaggle | This is the slide's overview; it does not establish that every setup supports the same training features. |
| Python | Lists Python 3.9 or later | The required version depends on the package release being installed. |
| GPU | Highlights NVIDIA CUDA GPUs | GPU compatibility and available memory matter for practical training. |
| pip | Recommends updating pip | Install packages into the same Python environment used by the notebook or script. |

The slide also describes CPU use as possible but slower. This should not be taken as confirmation that the demonstrated Unsloth training setup runs on CPU. The photos do not provide a tested compatibility guide.

## 4. Installation commands shown in class

The lesson gives this basic installation sequence:

```bash
pip install --upgrade pip
pip install unsloth
```

It then checks the installation by importing the package and printing its version:

```python
import unsloth

print(unsloth.__version__)
```

These are the commands presented in class, not a complete installation procedure for every machine. The photos do not specify a tested combination of Python, PyTorch, CUDA, and Unsloth versions.

### If Python cannot find the package

The installation slide shows this error:

```text
ModuleNotFoundError: No module named 'unsloth'
```

This means the Python interpreter running the code cannot find the package. The slide suggests checking the Python path and reinstalling:

```bash
pip uninstall unsloth
pip install unsloth
```

For notebook work, also check which Python environment the kernel uses. Installing a package in one environment will not make it available automatically in another.

## 5. Why chat templates matter

A chat template turns a conversation into the text format a particular model expects. It identifies who is speaking and where each message starts and ends.

The lesson uses three roles:

| Role | Purpose | Example |
| --- | --- | --- |
| `system` | Gives general instructions for the conversation | “Explain concepts in simple English.” |
| `user` | Contains the user's question or request | “What is fine-tuning?” |
| `assistant` | Contains the answer the model should learn to produce | “Fine-tuning adapts a pretrained model using task-specific examples.” |

In the slide's example, `{SYSTEM}`, `{INPUT}`, and `{OUTPUT}` are placeholders. We replace them with the actual instruction, question, and answer from a training example.

### Llama 3 template from the lesson

```python
chat_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>

{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{OUTPUT}<|eot_id|>"""
```

The special tokens provide structure:

| Token | Meaning in this example |
| --- | --- |
| `<|begin_of_text|>` | Marks the beginning of the sequence. |
| `<|start_header_id|>` | Starts a message's role header. |
| `<|end_header_id|>` | Ends the role header before the message content. |
| `<|eot_id|>` | Marks the end of a turn. |

The main lesson is to use a template that matches the model. Different model families may expect different tokens and formatting. The example above illustrates formatting; it does not run training by itself.

## 6. LoRA and the configuration example

The benefits slide includes a call to `FastVisionModel.get_peft_model(...)`. This is a **vision-language model** example, with options for both visual and language components.

LoRA stands for **Low-Rank Adaptation**. It is a form of parameter-efficient fine-tuning: small trainable adapter matrices are added to selected parts of the model while the original model weights generally remain frozen. This reduces the number of parameters that need training compared with updating the whole model.

An easy way to think about it is to keep the pretrained model's existing knowledge and train a smaller set of adjustments for the new task.

### Which components are selected?

The example enables all four of these options:

| Option | What it selects for adaptation |
| --- | --- |
| `finetune_vision_layers=True` | Visual processing layers. |
| `finetune_language_layers=True` | Language processing layers. |
| `finetune_attention_modules=True` | Attention modules. |
| `finetune_mlp_modules=True` | Feed-forward, or MLP, modules. |

Here, `True` means the component is selected in the example. It does not mean that every task needs every component enabled, or that all original weights are updated.

### Settings visible in the slide

| Setting | Example value | Simple explanation |
| --- | --- | --- |
| `r` | `16` | LoRA rank: controls the size of the adapter's low-rank update. A larger rank adds capacity and trainable parameters. |
| `lora_alpha` | `16` | Controls the scaling of the LoRA update together with the rank and scaling method. |
| `lora_dropout` | `0` | No LoRA dropout is applied in this configuration. |
| `bias` | `"none"` | Bias parameters are not trained through this adapter setting. |
| `random_state` | `3407` | Sets a random seed to help make experiments repeatable. |
| `use_rslora` | `False` | Rank-stabilized LoRA is disabled in this example. |
| `loftq_config` | `None` | No LoftQ configuration is supplied. |

These values are examples from the lesson. A larger rank does not automatically give better results, and the settings still need to be evaluated for the chosen dataset.

## 7. Main benefits discussed

The lesson highlights three expected benefits:

- **Task adaptation:** further training can help a model respond more appropriately in a specialized area.
- **Resource efficiency:** optimized training and adapter methods can reduce memory and computation needs.
- **More relevant output:** suitable training examples can help the model follow the desired style and task requirements.

These are goals of fine-tuning, not automatic outcomes. Poor examples or too much training can make the model memorize training data rather than perform well on new questions.

## 8. How the ideas fit together

A practical study workflow is:

1. Decide what behavior the model should learn.
2. Choose a suitable pretrained model.
3. Prepare clear examples of inputs and desired outputs.
4. Format those examples using the model's chat template.
5. Configure the components and LoRA adapters to train.
6. Run training in a compatible environment.
7. Compare the adapted model with the original model using examples kept out of training.
8. Save the useful result for later inference.

This workflow connects the concepts in the photos. The uploaded slides do not contain a complete runnable training script, dataset, evaluation, or saving procedure.

## 9. Quick revision

| Question | Short answer |
| --- | --- |
| What is fine-tuning? | Continuing training of a pretrained model for a specific task. |
| What does Unsloth help with? | Making supported model training more efficient. |
| What is a chat template? | The format used to represent conversation roles, messages, and boundaries. |
| What is LoRA? | A method that trains small adapter updates rather than all original model weights. |
| Does installing Unsloth train a model? | No. Training also needs a model, data, configuration, and a training run. |
| How do we know fine-tuning helped? | Evaluate on unseen examples and compare with the original model. |

## Source and scope

Based on the eight uploaded classroom photos of **LLM Model Fine-Tuning — Vann Visal**. Two photos show the same chat-template slide. The Git introduction PDF and *The Art of Readable Code* PDF in the project folder cover different topics and are not used for this fine-tuning summary.

The small examples and connecting explanations in these notes are added to make the lesson easier to understand. No installation or training was executed while preparing this summary.
