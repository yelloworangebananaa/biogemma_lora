# Gemma 4 E2B Biology LoRA Experiment (BioGemma)

An experimental biology-focused LoRA fine-tune of Gemma 4 E2B using a subset of `trillionlabs/TheBioCollection`.

This project investigated whether lightweight domain adaptation on biological and biomedical text could improve a small language model’s biology knowledge and reasoning.

## Project Status

This is an **experimental research project**, not a production model.

The fine-tuned model:

- Reduced held-out biology next-token loss substantially
- Improved high-school biology accuracy by 2 percentage points
- Did not improve overall biology benchmark accuracy
- Decreased anatomy accuracy by 2 percentage points

The current results demonstrate successful adaptation to biological text, but they do **not** demonstrate that the fine-tuned model is generally better at biology than the base model.

## Research Question

Can LoRA fine-tuning on biological and biomedical text improve Gemma 4 E2B’s performance on held-out biology text and biology-focused multiple-choice benchmarks?

## Main Findings

The experiment produced two different results:

1. The model became substantially better at predicting held-out biology text.
2. That improvement did not translate into higher overall multiple-choice biology accuracy.

Held-out loss decreased from a base-model value of approximately **9.55** to approximately **7.35** after 500 optimizer steps.

However, overall biology benchmark accuracy remained unchanged at **56.5%**.

## Benchmark Results

| Subject | Base Gemma 4 E2B | Biology LoRA | Percentage-Point Change |
|---|---:|---:|---:|
| **Overall** | **56.5%** | **56.5%** | **0.0** |
| Anatomy | 56.0% | 54.0% | -2.0 |
| College Biology | 56.0% | 56.0% | 0.0 |
| High School Biology | 64.0% | 66.0% | +2.0 |
| Medical Genetics | 50.0% | 50.0% | 0.0 |

Overall, some tradeoffs seemed to happen where it got better at High School Biology, but then lost its expertise in Anatomy.

## Training Dynamics

### Held-Out Biology Loss

![Held-out biology loss across training checkpoints](<training_loss.png>)

Held-out biology loss decreased consistently throughout training.

| Checkpoint | Approximate Held-Out Loss |
|---:|---:|
| Base model | 9.55 |
| Step 100 | 9.05 |
| Step 200 | 8.28 |
| Step 300 | 7.72 |
| Step 400 | 7.46 |
| Step 500 | 7.35 |

Because held-out loss was still decreasing at the final checkpoint, the model had probably not fully converged when training stopped.

This makes insufficient training one plausible explanation for the limited benchmark improvement. However, this experiment does not prove that additional training would improve multiple-choice accuracy.

### Training Loss

![Gemma 4 E2B biology training loss](<held_out_biology_loss.png>)

The sampled training loss was noisy because individual batches contained examples with different lengths, formats, topics, and levels of difficulty.

Despite the variation between steps, the overall trend decreased during training. This indicates that the LoRA adapters learned statistical patterns from the biology dataset rather than remaining unchanged.

## Interpretation

The results are not necessarily contradictory.

Next-token loss and multiple-choice accuracy measure different capabilities.

### Held-Out Loss

Held-out next-token loss measures how well the model predicts tokens in unseen biological and biomedical text.

A lower held-out loss suggests that the model became better adapted to the language, terminology, and statistical patterns of the biology corpus.

### Multiple-Choice Accuracy

The benchmark measures whether the model selects the correct answer from a set of choices.

This may require:

- Factual recall
- Multi-step reasoning
- Understanding the question format
- Comparing similar answer choices
- Rejecting distractors
- Following answer-format instructions
- Producing a reliably parseable response

A model can therefore improve at predicting biological text without improving at multiple-choice reasoning.

## Training Configuration

| Parameter | Value |
|---|---|
| Base architecture | Gemma 4 E2B |
| Fine-tuning method | LoRA |
| Dataset | `trillionlabs/TheBioCollection` |
| Free-text samples | Approximately 6,400 |
| Instruction-style samples | Approximately 3,600 |
| Maximum sequence length | 128 tokens |
| Optimizer steps | 500 |
| Effective batch size | 8 |
| Maximum token positions processed | Approximately 512,000 |
| Learning rate | `2.5e-6` |
| Warmup steps | 20 |
| Learning-rate schedule | Cosine decay |
| LoRA rank | 8 |
| LoRA alpha | 8 |
| LoRA dropout | 0 |
| Weight decay | 0.01 |
| Training environment | Google Colab |
| Training GPU | NVIDIA T4 |

The maximum token count assumes that every sequence reached the full 128-token limit. The actual number of non-padding training tokens may have been lower.

## Dataset

The experiment used a sampled subset of:

```text
trillionlabs/TheBioCollection
```

The source dataset contains biological and biomedical material collected from multiple formats and subject areas.

The training subset combined:

- Free-text biological material
- Instruction-style examples
- Biomedical explanations
- Domain-specific scientific language

The training data was not designed exclusively for multiple-choice question answering.

This creates a training-evaluation mismatch: the model was optimized for next-token prediction on biology text but evaluated partly through multiple-choice reasoning tasks.

## Method

The experiment followed this general pipeline:

1. Stream and sample examples from TheBioCollection.
2. Convert examples into a consistent text format.
3. Tokenize examples with a maximum sequence length of 128.
4. Fine-tune Gemma 4 E2B using rank-8 LoRA adapters.
5. Save adapter checkpoints during training.
6. Evaluate held-out biology loss at multiple checkpoints.
7. Compare the base and fine-tuned models on biology benchmark subsets.
8. Merge the LoRA adapters into the base model.
9. Export the merged model for local inference.

## Repository Structure

```text
gemma4-e2b-biology-lora-experiment/
├── README.md
├── image(54).png
├── image(55).png
├── notebooks/
│   ├── training.ipynb
│   ├── evaluation.ipynb
│   └── merge_and_export.ipynb
├── evaluation/
│   ├── benchmark_results.csv
│   ├── held_out_loss.csv
│   └── predictions/
├── adapter/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── Modelfile
├── requirements.txt
└── LICENSE
```

Large merged model files and GGUF files should generally be uploaded to Hugging Face rather than committed directly to the GitHub repository.

## Installation

Clone the repository:

```bash
git clone https://github.com/yelloworangebanananaa/gemma4-e2b-biology-lora-experiment.git
cd gemma4-e2b-biology-lora-experiment
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Typical dependencies include:

```text
torch
transformers
datasets
peft
accelerate
safetensors
sentencepiece
numpy
pandas
matplotlib
```

The exact package versions used for the original experiment should be recorded in `requirements.txt`.

## Loading the LoRA Adapter

The adapter can be loaded using Hugging Face Transformers and PEFT.

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = "REPLACE_WITH_EXACT_BASE_MODEL_ID"
ADAPTER_PATH = "./adapter"

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_ID,
    trust_remote_code=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)

model.eval()

prompt = """
Explain the difference between competitive inhibition and
noncompetitive inhibition in enzyme kinetics.
""".strip()

inputs = tokenizer(
    prompt,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.2,
        do_sample=True,
        top_p=0.95,
    )

print(
    tokenizer.decode(
        output[0],
        skip_special_tokens=True,
    )
)
```

Replace `BASE_MODEL_ID` with the exact Hugging Face model identifier used during training.

## Using the GGUF Model

After the merged model has been converted successfully, it can be loaded in Ollama, LM Studio, or llama.cpp.

The expected quantized filename is:

```text
gio-biology-gemma4-e2b-Q4_K_M.gguf
```

### Ollama

Place the GGUF and `Modelfile` in the same folder:

```text
gio-biology/
├── gio-biology-gemma4-e2b-Q4_K_M.gguf
└── Modelfile
```

Example `Modelfile`:

```text
FROM ./gio-biology-gemma4-e2b-Q4_K_M.gguf

PARAMETER temperature 0.2
PARAMETER top_p 0.95
PARAMETER top_k 64
PARAMETER repeat_penalty 1.05
PARAMETER num_ctx 4096

SYSTEM """
You are a biology-focused scientific assistant.
Give accurate, clear, and evidence-aware answers.
Clearly distinguish established evidence from uncertainty.
"""
```

Create the Ollama model:

```bash
ollama create gio-biology -f Modelfile
```

Run it:

```bash
ollama run gio-biology
```

### LM Studio

1. Download the completed `.gguf` file.
2. Import it into LM Studio.
3. Load the model in the Chat interface.
4. Begin with a context length of 4,096 tokens.
5. Increase GPU offloading until the model approaches the available VRAM limit.
6. Use a low temperature for factual biology questions.

A completed GGUF file is required. A partial BF16 file left behind after a killed conversion process should not be used.

## Example Prompts

### Molecular Biology

```text
Explain how transcriptional attenuation regulates the trp operon.
Separate the mechanism into initiation, translation, and termination stages.
```

### Cell Biology

```text
Compare receptor-mediated endocytosis, phagocytosis, and pinocytosis.
Include the major proteins and cellular structures involved.
```

### Genetics

```text
A heterozygous individual is crossed with a homozygous recessive individual.
Explain the expected genotype and phenotype ratios.
```

### Microbiology

```text
Explain the roles of LasR, RhlR, and PqsE in Pseudomonas aeruginosa
quorum sensing. Distinguish direct regulation from indirect effects.
```

### Scientific Uncertainty

```text
Evaluate the following biological claim. Identify what is established,
what is uncertain, and what experiment would best test it:

[INSERT CLAIM]
```

## Limitations

### Limited Training Duration

Training ended after 500 optimizer steps.

Held-out loss was still decreasing, so the final checkpoint had probably not fully converged. However, continued loss reduction does not guarantee improved performance on external benchmarks.

### Short Sequence Length

The maximum sequence length was only 128 tokens.

This may have truncated:

- Long explanations
- Multi-step biological mechanisms
- Complete scientific arguments
- Instruction-response pairs
- Context surrounding technical terms

Longer sequences may be more appropriate for scientific material.

### Training-Objective Mismatch

The model was trained using next-token prediction but evaluated partly through multiple-choice questions.

The training data did not directly optimize:

- Answer-choice comparison
- Multiple-choice reasoning
- Output formatting
- Distractor rejection
- Concise answer selection

### Broad Dataset Scope

TheBioCollection covers broad biological and biomedical material.

It was not specifically optimized for:

- High-school biology
- Anatomy
- Medical genetics
- College biology examinations
- MMLU-style question answering

### Limited LoRA Capacity

The experiment used rank-8 LoRA adapters.

This may have been sufficient for modest domain adaptation but insufficient for producing major changes in reasoning or factual performance.

### Single Experimental Run

Only one primary training configuration was tested.

The experiment did not systematically compare:

- Multiple random seeds
- Different learning rates
- Different LoRA ranks
- Different dataset mixtures
- Different sequence lengths
- Different numbers of training steps
- Different checkpoint-selection criteria

### Benchmark Uncertainty

Small benchmark differences may result from only a few changed answers.

The reported two-percentage-point changes were not tested for statistical significance.

### Possible Data Contamination

TheBioCollection is a large aggregated corpus.

Although evaluation questions were not intentionally added to the training subset, complete absence of overlap has not been independently verified.

## Future Work

The next version should focus on controlled experiments rather than simply increasing every training parameter.

### 1. Evaluate More Checkpoints

Evaluate the model at regular checkpoints:

```text
500 steps
1,000 steps
1,500 steps
2,000 steps
```

This would show whether lower held-out loss eventually produces benchmark improvements or whether benchmark accuracy remains flat.

### 2. Increase Sequence Length

Test sequence lengths such as:

```text
512 tokens
1,024 tokens
```

Longer sequences would preserve more complete biological explanations and reasoning chains.

### 3. Improve the Dataset Mixture

A stronger mixture could include:

- Biology textbook explanations
- Biology question-answer pairs
- Mechanistic reasoning examples
- Multiple-choice questions with explanations
- Experimental interpretation questions
- Genetics problems
- Biochemistry calculations
- Microbiology case analysis

### 4. Increase LoRA Capacity

Compare:

```text
Rank 8
Rank 16
Rank 32
```

This would test whether the adapter lacked sufficient capacity.

### 5. Use Multiple Random Seeds

Repeat each configuration with multiple random seeds.

This would help determine whether small score changes are reproducible or caused by random variation.

### 6. Save Per-Question Predictions

For every benchmark question, record:

- Question
- Correct answer
- Base-model answer
- Fine-tuned-model answer
- Parsed response
- Whether the answer changed
- Whether the change was wrong-to-right or right-to-wrong

This provides more useful evidence than aggregate accuracy alone.

### 7. Perform Paired Statistical Analysis

Because the same questions are answered by both models, paired methods should be used.

Possible approaches include:

- McNemar’s test
- Bootstrap confidence intervals
- Paired accuracy differences
- Error-category analysis

## Reproducibility

For stronger reproducibility, future releases should record:

- Exact base-model revision
- Exact dataset revision
- Random seed
- Sample-selection method
- Complete package versions
- Prompt template
- Chat template
- Decoding parameters
- Evaluation parser
- Checkpoint hashes
- GPU type
- Total training time
- Number of non-padding tokens processed

Without these details, small benchmark differences can be difficult to reproduce.

## Intended Use

This project is intended for:

- Machine-learning experimentation
- Biology education experiments
- LoRA fine-tuning research
- Local language-model deployment
- Benchmark and evaluation research

## Out-of-Scope Use

This model should not be relied upon for:

- Medical diagnosis
- Treatment recommendations
- Medication decisions
- Laboratory safety decisions
- Clinical interpretation
- Patient-specific advice
- High-stakes scientific conclusions

The model may generate inaccurate, incomplete, outdated, or fabricated information.

Important scientific claims should be verified using reliable primary literature or expert guidance.

## Conclusion

The experiment successfully adapted Gemma 4 E2B to the statistical patterns of a held-out biology corpus.

Held-out biology loss decreased from approximately 9.55 to 7.35, showing that the LoRA adapters learned meaningful domain-specific information.

However, this adaptation did not produce a measurable improvement in overall biology multiple-choice accuracy. Overall accuracy remained at 56.5%, with a two-point gain in high-school biology offset by a two-point decline in anatomy.

The continuing decline in held-out loss suggests that training may have stopped before convergence. Insufficient training is therefore a plausible explanation for the limited benchmark improvement, but it has not been established as the cause.

The project succeeded at domain-language adaptation but did not yet demonstrate reliable improvement in biology question answering.

## Model Release Status

- [x] LoRA training completed
- [x] Held-out loss evaluated
- [x] Base and fine-tuned benchmarks compared
- [x] Adapter checkpoint saved
- [x] Merged Hugging Face model created
- [x] GGUF conversion completed
- [ ] Quantized model uploaded
- [ ] Multi-seed evaluation completed
- [ ] Statistical significance analysis completed
- [ ] Longer training run completed

## License

Use of this repository must comply with:

- The license and usage terms of the base Gemma model
- The license and usage terms of TheBioCollection
- The licenses of all included software dependencies

Model weights, adapters, code, and dataset-derived artifacts may be subject to different licenses. Review each license before redistribution or commercial use.

## Acknowledgments

This project used:

- Gemma
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT
- PyTorch
- llama.cpp
- Google Colab
- TheBioCollection

## Citation

```bibtex
@software{kim_gemma4_biology_lora,
  author = {Gio Kim},
  title = {Gemma 4 E2B Biology LoRA Experiment},
  year = {2026},
  url = {https://github.com/yelloworangebanananaa/biogemma_lora}
}
```
