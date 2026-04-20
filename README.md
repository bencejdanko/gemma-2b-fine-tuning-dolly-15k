# Gemma 2B Fine-Tuning on Dolly 15k

Online report: https://docs.google.com/document/d/1gmUemWx8zt6N7PIbGn-L2yHQA1rUb76D09YVTAsJqsE/edit?usp=sharing

## Dataset Databricks Dolly 15K

(https://huggingface.co/datasets/databricks/databricks-dolly-15k)

Training 3,000 samples and In-distribution test at least 100 samples.

([tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)) 

Minimum 50 samples (No retraining on Alpaca)

## Model

Evaluating and training Gemma 2B (https://huggingface.co/google/gemma-2b-it) using `torch.bfloat16` precision, the default export.

The model has 2.6 billion parameters. At this precision, each parameter takes 2 bytes. Static memory is $2.6 \text{ billion} \times 2 \text{ bytes} \approx 5.2 \text{ GB}$. KV cache calls for ~6 GB to 7 GB of VRAM total when running inference.

When running fine tuning we need to store gradients and optimizer states. So the weights are 2 bytes/param (5.2 GB), the gradients are 2 bytes/param (5.2 GB), and for AdamW, we require 12 bytes/param (31.2 GB).

To comfortably perform full fine tuning, we need something like 80GB+ memory from an A100.

## Training

- Use the Hugging Face Transformers library.
- Response-only masking.
- Train for 3 epochs.
- Use AdamW optimization.
- A learning rate of $2 \times 10^{-5}$. A lower rate prevents catastrophic forgetting from the base model.

Use a consistent seed (`SEED=15179996`) for reproducibility.

## Evaluations

### System prompt for model:

> You are a helpful, accurate, and concise AI assistant. Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

Gemma has specific control tokens we must use to align the context correctly during prompting and fine tuning.

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

## Ablations:

| Model (base/fine-tuned) | Distribution (in/out) | Judge* | Instruction Following | Helpfulness | Fluency |
| --- | --- | --- | --- | --- | --- |
| base | in | openai/gpt-5.2 | | |
| base | in | deepseek/deepseek-v3.2 | | |
| fine-tuned | in | openai/gpt-5.2 | | |
| fine-tuned | in | deepseek/deepseek-v3.2 | | |
| base | out | openai/gpt-5.2 | | |
| base | out | deepseek/deepseek-v3.2 | | |
| fine-tuned | out | openai/gpt-5.2 | | |
| fine-tuned | out | deepseek/deepseek-v3.2 | | |

[*] Judges evaluated on the same outputs.

The fine tuned model is located at `bdanko/fine-tuned-gemma-2b-dolly` for evaluation. Inferenced test sets are located at https://huggingface.co/bdanko/fine-tuned-gemma-2b-dolly/tree/main/eval_results


### Base deliverables

- `bdanko/base-gemma-2b-dolly-evaluation-gpt-5.2-judgement`: Reasoned judgement evaluations on `bdanko/base-gemma-2b-dolly-evaluations`
- `bdanko/base-gemma-2b-dolly-evaluation-deepseek-v3.2-judgement`: Reasoned judgement evaluations on`bdanko/base-gemma-2b-dolly-evaluations`
- `bdanko/base-gemma-2b-alpaca-evaluation-gpt-5.2-judgement`: Reasoned judgement evaluations on `bdanko/base-gemma-2b-alpaca-evaluations`
- `bdanko/base-gemma-2b-alpaca-evaluation-deepseek-v3.2-judgement`: Reasoned judgement evaluations on`bdanko/base-gemma-2b-alpaca-evaluations`


### Fine-tuning deliverables

After training, we'll save model `bdanko/fine-tuned-gemma-2b-dolly` for evaluation.

- `bdanko/fine-tuned-gemma-2b-dolly-evaluation-gpt-5.2-judgement`: Reasoned judgement evaluations on `bdanko/fine-tuned-gemma-2b-dolly-evaluations`
- `bdanko/fine-tuned-gemma-2b-dolly-evaluation-deepseek-v3.2-judgement`: Reasoned judgement evaluations on`bdanko/fine-tuned-gemma-2b-dolly-evaluations`
- `bdanko/fine-tuned-gemma-2b-alpaca-evaluation-gpt-5.2-judgement`: Reasoned judgement evaluations on `bdanko/fine-tuned-gemma-2b-alpaca-evaluations`
- `bdanko/fine-tuned-gemma-2b-alpaca-evaluation-deepseek-v3.2-judgement`: Reasoned judgement evaluations on`bdanko/fine-tuned-gemma-2b-alpaca-evaluations`

## Qualitative Assesments

- 3 example improvement (from the base model) cases with explanation
- 1 failure case with root cause