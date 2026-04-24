# Gemma 2B Fine-Tuning on Dolly 15k

By default, the `google/gemini-2b` model, released in 2024, is incoherent and produces non human-aligned or turn aligned responses. This notebook is a demonstration of how to instruction-tune a base model for question-answering.

Fine tuning large models is a delicate process, because there is risk of catastropic forgetting, the model overwriting and forgetting pretraining knowledge. The goal of this repository is to demonstrate a successful fine-tune, evaluated using state-of-the-art models as LLM-as-a-Judge to judge increases in instruction following, helpfulness, and fluency (general grammar and coherence).

[Full online report (Google Documents)](https://docs.google.com/document/d/1gmUemWx8zt6N7PIbGn-L2yHQA1rUb76D09YVTAsJqsE/edit?usp=sharing).

## Results

| Model      | Distribution   | Judge                  |   Instruction Following |   Helpfulness |   Fluency |
|:-----------|:---------------|:-----------------------|------------------------:|--------------:|----------:|
| base       | in             | deepseek/deepseek-v3.2 |                    1.21 |          1.16 |      1.5  |
| base       | in             | openai/gpt-5.2         |                    1.22 |          1.12 |      1.52 |
| base       | out            | deepseek/deepseek-v3.2 |                    1.16 |          1.1  |      1.46 |
| base       | out            | openai/gpt-5.2         |                    1.14 |          1.08 |      1.62 |
| fine-tuned | in             | deepseek/deepseek-v3.2 |                    3.64 |          3.24 |      4.49 |
| fine-tuned | in             | openai/gpt-5.2         |                    3.56 |          2.94 |      4.49 |
| fine-tuned | out            | deepseek/deepseek-v3.2 |                    2.82 |          2.74 |      4.14 |
| fine-tuned | out            | openai/gpt-5.2         |                    2.84 |          2.6  |      4.2  |

The fine tuned model is located at `bdanko/fine-tuned-gemma-2b-dolly` for evaluation. Inferenced test sets are located at https://huggingface.co/bdanko/fine-tuned-gemma-2b-dolly/tree/main/eval_results

## Dataset Databricks Dolly 15K

(https://huggingface.co/datasets/databricks/databricks-dolly-15k)

Training 14,911 samples and In-distribution test at least 100 samples.

([tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)) 

Minimum 50 samples (No retraining on Alpaca)

## Model

Evaluating and training Gemma 2B (https://huggingface.co/google/gemma-2b) using `torch.bfloat16` precision, the default export.

The model has 2.6 billion parameters. At this precision, each parameter takes 2 bytes. Static memory is $2.6 \text{ billion} \times 2 \text{ bytes} \approx 5.2 \text{ GB}$. KV cache calls for ~6 GB to 7 GB of VRAM total when running inference.

When running fine tuning we need to store gradients and optimizer states. So the weights are 2 bytes/param (5.2 GB), the gradients are 2 bytes/param (5.2 GB), and for AdamW, we require 12 bytes/param (31.2 GB).

To comfortably perform full fine tuning, it's necessary to use a 40GB A100. This notebook caps out at ~39GB VRAM.

## Training

- Use the Hugging Face Transformers library.
- Response-only masking.
- Train for 3 epochs.
- Use AdamW optimization.
- A learning rate of $2 \times 10^{-5}$. A lower rate prevents catastrophic forgetting from the base model.

Used a consistent seed (`SEED=15179996`) for reproducibility.

Full fine tuning takes ~1 hour on an A100, and caps 37.9 VRAM memory. Evaluation takes ~20 minutes with a batch size of 1. With a batch size 16, 100 Dolly evaluations were complete in ~4 minutes.

## Evaluations

Gemma has specific control tokens (`<start_of_turn>`, `<end_of_turn>`) we must use to align the context correctly during prompting and fine tuning.

### In-Distribution (ID)

Evaluate on at least 100 held-out Dolly samples.

### Out-of-Distribution (OOD)

Use Alpaca dataset, 50 samples.

### LLM As a Judge

We'll compare GPT-5.2 (openai/gpt-5.2 on OpenRouter) and DeepSeek (deepseek/deepseek-v3.2 on OpenRouter) as LLM judges.

Judges will grade based on Instruction Following (Degree of adherence to task), Helpfulness (Practical usefulness), Fluency (Grammar and coherence) graded on a scale of 1-5. 

In order to make the grading fair, we'll use a standardized rubric:

| Metric | Description | Scale |
| --- | --- | --- |
| Instruction Following | Degree of adherence to task | 1-5 |
| Helpfulness | Practical usefulness | 1-5 |
| Fluency | Grammar and coherence | 1-5 |

> You are an impartial judge evaluating the quality of an AI response. You will be provided with an instruction, a context (if applicable), and the AI's response.
> Evaluation Process:
> 1. Step-by-Step Analysis: Critique the response based on the rubric.
> 2. Score Assignment: Provide a score for each metric (1-5).
> 3. Final Verdict: Justify why the scores were given.
> Rubric:
> Instruction Following: Did it do exactly what was asked? (5 = Perfect, 1 = Irrelevant)
> Helpfulness: Is the information useful and correct? (5 = Insightful, 1 = Harmful/Useless)
> Fluency: Is the text natural and grammatically sound? (5 = Native-level, 1 = Incoherent)"
> Please conclude your evaluation with a Markdown table and a brief 'Verdict' section using the *exact* headers below:
> ```
> ## Final Scores
> | Metric	| Score |
> | --- | --- |
> | Instruction Following |	[1-5] |
> | Helpfulness	| [1-5] |
> | Fluency | [1-5] |
> ## Verdict
> [One sentence summarizing the reasoning for the above scores]
> ```