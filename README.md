# Ableism and LLMs

*Repository for M2 internship project on ableist biases in LLMs*

As Large Language Models (LLMs) are increasingly adopted in sensitive domains such as healthcare and justice, ensuring they are free from harmful biases is critical. While AI ethics research has documented the presence of various stereotypes in these models, ableist biases — prejudice and discrimination against people with disabilities — remain largely understudied.

This project investigates to what extent LLMs reproduce ableist biases, and proposes methods and tools to evaluate them.

## Experimental setup

Three instruction-tuned models (Llama 3.1 8B, Mistral 7B v0.3, Qwen3 8B) are prompted in French to write fictional biographies, under two conditions: with no mention of disability in the prompt (`nodis`) and with the person explicitly described as disabled (`withdis`). Each model × condition is run 50 times with two inference techniques — full-precision Transformers (`torch`) and 4-bit AWQ quantization via vLLM (`awq`) — yielding the 600 biographies in `generated_biographies/`. A small-scale baseline experiment (54 biographies: 3 models × 3 short prompt versions × both conditions × 3 runs, MLX quantization, `mlx`) is stored in `biography_generation/outputs/baseline/`.

Generated files follow the naming convention parsed by the analysis scripts (`resources/helper_functions.py:parse_doc_id`):

```
{model}_{prompt_version}_{variant}_{run}_{technique}.txt
e.g.  mistral_v1_long_withdis_38_awq.txt
```

The biographies are then analysed along several axes: representation (inferred gender, disability mentions and categories, named entities), the *malgré* ("despite") narrative trope, and lexico-syntactic complexity.

## Project Structure

```
Ableism-and-LLMs/
├── biography_generation/           # Biography generation with LLMs
│   ├── bio_generation.py           # Main batch script (torch + AWQ, all models, 50 runs each)
│   ├── config.py                   # Sampling parameters and HF model ids (awq / mlx / torch)
│   ├── prompts.py                  # French prompt sets (v1–v3, short/long, nodis/withdis)
│   ├── utils/
│   │   ├── common.py               # Chat formatting, JSON parsing, result saving
│   │   ├── torch_utils.py          # Transformers loading/generation
│   │   ├── awq_utils.py            # vLLM AWQ loading/generation
│   │   └── mlx_utils.py            # MLX loading/generation (Apple silicon, used by notebooks)
│   ├── llama3_1_8B.ipynb           # Per-model baseline experiment notebooks
│   ├── mistral7B.ipynb             #   (MLX or Transformers variants)
│   ├── qwen3_8B.ipynb
│   ├── outputs/                    # Raw generation results (results_{model}.json)
│   │   └── baseline/               # Baseline experiment results (JSON + txt/)
│   └── logs/                       # Cluster job logs and energy consumption log
├── biography_analysis/             # Analysis scripts, notebooks and resources
│   ├── representation_analysis.py  # Gender + disability + NER detection over all biographies
│   ├── trope_detection.py          # "Malgré" trope detection (spaCy dependency parsing)
│   ├── complexity_statistics.py    # ALSI complexity features + statistical comparisons
│   ├── biography_analyses.ipynb    # Statistics tables, plots and statistical analyses
|   |                               #   from the detection results
│   ├── resources/
│   │   ├── helper_functions.py     # doc_id parsing, Cohen's kappa, lists parsing
│   │   ├── gender/                 # 3rd-person gender detection (adapted from Ducel et al., 2024)
│   │   │   │                       #   + lexicons
│   │   │   └── gender_detection_evaluation/
│   │   │       ├── gender_detection_benchmark.py     # Runs detection on the benchmark corpus
│   │   │       ├── gender_detection_evaluation.ipynb # Detectors vs manual annotations
│   │   │       ├── baseline_gender_observations.ipynb # Gender distribution in the baseline outputs
│   │   │       ├── biographies_gender_benchmark/     # Held-out benchmark corpus (108 bios)
│   │   │       └── *.csv                             # Manually annotated evaluation sets
│   │   ├── disabilities/           # Lexicon-based disability detection and categorisation
│   │   ├── named_entities/         # NER (Babelscape/wikineural-multilingual-ner)
│   │   │                           #   + person name cleaning and first/last name splitting
│   │   └── complexity_tool/
│   │       ├── ALSI-main/          # ALSI tool (Loignon, 2021) — see Third-party tools below
│   │       └── conversion.R        # Converts ALSI .Rds output to .xlsx
│   └── outputs/                    # Analysis outputs (CSV/XLSX results, plots/)
├── generated_biographies/          # The 600 generated biographies (.txt)
├── requirements/
│   ├── generation.txt
│   └── analysis.txt
└── README.md
```

## Requirements

Python version used: **Python 3.13.5**

The complexity analysis step additionally requires **R** with the following packages: `udpipe`, `tidyverse`, `data.table`, `utf8`, `zoo`, `future`, `future.apply`, `writexl`.

Notes:

- `vllm` (in `requirements/generation.txt`) is only needed for the AWQ generation path and requires Linux/CUDA; it is left unpinned.
- Gender detection and trope detection use the spaCy model `fr_dep_news_trf` (`python -m spacy download fr_dep_news_trf`).

## Installation

Clone the repository:

```bash
git clone https://github.com/Lucandica/Ableism-and-LLMs.git
cd Ableism-and-LLMs
```

Install Python dependencies for the relevant section:

```bash
# For biography generation
pip install -r requirements/generation.txt

# For biography analysis
pip install -r requirements/analysis.txt
```

## Usage

The project is split into two main steps:

### 1. Biography generation

The main batch script generates biographies for all three models with both techniques (Transformers and AWQ/vLLM). It requires a Hugging Face token in the `HF_TOKEN` environment variable (gated Llama weights):

```bash
cd biography_generation
python bio_generation.py
```

Prompt versions to run, models and sampling parameters are configured at the top of `bio_generation.py` (see `prompts.py` and `config.py`). Results are appended to `biography_generation/outputs/results_{model}.json`, with prompt metadata and one entry per generation. Mistral has no system role, so the system prompt is prepended to the user prompt for that model.

The per-model notebooks (`llama3_1_8B.ipynb`, `mistral7B.ipynb`, `qwen3_8B.ipynb`) run the baseline experiment (short prompts v1–v3, both conditions, 3 runs each), with MLX (Apple silicon) and Transformers variants. Results are saved to `biography_generation/outputs/baseline/` (one JSON per model, plus individual `.txt` files in `txt/`).

All generated biographies are also available as individual `.txt` files in `generated_biographies/` (see naming convention above).

### 2. Biography analysis

**Representation analysis** — detects, for each biography: the inferred gender of the character (morpho-syntactic marker detection adapted to third person from Ducel et al., 2024), disability terms and their categories (lexicon based on the French Ministry of Agriculture official disability typologies), and named entities (person names — cleaned and split into first and last names — organisations, locations). Results are merged into `biography_analysis/outputs/gender_dis_ner_detection.csv`.

```bash
cd biography_analysis
python representation_analysis.py
```

**Trope detection** — detects every occurrence of _malgré_ ("despite") across all biographies using spaCy dependency parsing (`fr_dep_news_trf`). For each match, it extracts the object phrase governed by _malgré_ (head token, lemma, POS and a short readable phrase), then retrieves the preceding sentence and checks it for disability keyword hits. Results are saved to `biography_analysis/outputs/trope_detection.csv`.

```bash
cd biography_analysis
python trope_detection.py
```

**Complexity statistics** — runs the ALSI tool (R) to extract lexico-syntactic complexity features, then compares them across categories (model, quantization technique, disability in prompt) using Wilcoxon tests with Benjamini–Hochberg correction and Cohen's d. Results are saved to `biography_analysis/outputs/biographies_stats_complexity.xlsx` (one sheet per grouping variable).

```bash
cd biography_analysis
python complexity_statistics.py          # runs ALSI then computes statistics
python complexity_statistics.py --csv path/to/features.csv  # skip ALSI, use existing CSV
```

This requires R and the ALSI dependencies (see Requirements above). Generated biographies must be placed in `biography_analysis/resources/complexity_tool/ALSI-main/corpus/biographies/` before running.

**Results notebook** — `biography_analyses.ipynb` gathers all the detection results and produces the statistics tables, figures and statistical analyses, broken down by prompt condition, inferred gender, model and quantization technique:

- *Representation analyses* (from `gender_dis_ner_detection.csv`): disability category, gender, location and name representation (first and last names by inferred gender), with chi-squared tests of independence and per-category linear probability models estimating the effects of inferred gender, model and quantization on the probability that each disability category is mentioned.
- *Tropes analysis* (from `trope_detection.csv`): distribution of the objects governed by _malgré_ and their relation to disability mentions in the preceding context.
- *Complexity analyses* (from `biographies_stats_complexity.xlsx` and the ALSI features): overview of significant features per comparison, strongest effects, and distributions of selected features (word count, sentence length, MATTR, dependency depth, clausal density).

Plots are saved to `biography_analysis/outputs/plots/`.

**Gender detection evaluation** — in `resources/gender/gender_detection_evaluation/`. `gender_detection_benchmark.py` applies the third-person gender detector to a held-out benchmark of 108 newly generated biographies (`biographies_gender_benchmark/`: 3 models × 3 prompt versions × short/long × nodis/withdis × 3 runs), which were not used while developing the detector. `gender_detection_evaluation.ipynb` then compares the detectors against manual annotations (classification reports and confusion matrices): the original first-person detector from Ducel et al. (2024) vs the third-person adaptation used here, evaluated both on the 600-biography corpus (used during development) and on the held-out benchmark. `baseline_gender_observations.ipynb` observes the gender distribution of the protagonists in the 54 baseline biographies, depending on the prompt used.

## Third-party tools

### ALSI — Analyseur Lexico-Syntaxique Intégré

The complexity analysis uses [ALSI](https://github.com/gloignon/ALSI), an automated lexico-syntactic analyzer for French developed by Guillaume Loignon. The full tool source is included in `biography_analysis/resources/complexity_tool/ALSI-main/` for reproducibility (version downloaded on April 21, 2025).

ALSI is licensed for **personal and research use only** (non-commercial). See `biography_analysis/resources/complexity_tool/ALSI-main/LICENSE.md` for terms.

**If you use this repository for research, please cite the ALSI paper:**

> Loignon, G. (2021). ILSA: an automated language complexity analysis tool for French. *Mesure et évaluation en éducation, 44*, 61–88. https://doi.org/10.7202/1095682ar

ALSI also bundles several lexical databases; please cite the relevant papers if you use features derived from them (see `biography_analysis/resources/complexity_tool/ALSI-main/README.md` for the full bibliography).

## References

- Ducel, F., Névéol, A., & Fort, K. (2024). "You'll be a nurse, my son!" Automatically assessing gender biases in autoregressive language models in French and Italian. *Language Resources and Evaluation, 59*(2), 1495–1523. https://doi.org/10.1007/s10579-024-09780-6

- Loignon, G. (2021). ILSA: an automated language complexity analysis tool for French. *Mesure et évaluation en éducation, 44*, 61–88. https://doi.org/10.7202/1095682ar
