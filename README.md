# Ableism and LLMs

*Repository for M2 internship project on ableist biases in LLMs*

As Large Language Models (LLMs) are increasingly adopted in sensitive domains such as healthcare and justice, ensuring they are free from harmful biases is critical. While AI ethics research has documented the presence of various stereotypes in these models, ableist biases — prejudice and discrimination against people with disabilities — remain largely understudied.

This project investigates to what extent LLMs reproduce ableist biases, and proposes methods and tools to evaluate them.

## Project Structure

```
Ableism-and-LLMs/
├── biography_generation/       # Scripts to generate biographies using LLMs
├── biography_analysis/         # Analysis notebooks and resources
│   ├── resources/
│   │   ├── gender/             # Gender detection lexical resources (based on Ducel et al., 2024)
│   │   ├── disabilities/       # Disability detection module
│   │   ├── name_entities/      # Named entity recognition module
│   │   └── complexity_tool/    # Lexico-syntactic complexity analysis
│   │       └── ALSI-main/      # ALSI tool (Loignon, 2021) — see Third-party tools below
│   ├── outputs/                # Analysis outputs (CSV/XLSX results)
│   ├── complexity_statistics.py
│   ├── representation_analysis.py
│   └── biographies_analysis.ipynb
├── generated_biographies/      # LLM-generated biography files (output of generation step)
├── requirements/
│   ├── generation.txt
│   └── analysis.txt
└── README.md
```

## Requirements

Python version used: **Python 3.13.5**

The complexity analysis step additionally requires **R** with the following packages: `udpipe`, `tidyverse`, `data.table`, `utf8`, `zoo`, `future`, `future.apply`, `writexl`.

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

1. **Biography generation**

   Run the scripts in `biography_generation/` to generate biographies using LLMs.
   Both MLX and Transformers variations of the code are available.
   The output files will be saved in `biography_generation/outputs/`.
   All generated biographies are also available in `.txt` format in `generated_biographies/`.

2. **Biography analysis**

   Open and run `biography_analysis/biographies_analysis.ipynb`.
   The notebook covers:

   - Gender marker detection for gender inferences
   - Disability detection and categorisation
   - Named entity recognition (names, organisations, locations)
   - Lexico-syntactic complexity analysis (via ALSI)

   Outputs (CSV/XLSX files) are saved in `biography_analysis/outputs/`.

   **Complexity analysis** can also be run standalone:

   ```bash
   cd biography_analysis
   python complexity_statistics.py          # runs ALSI then computes statistics
   python complexity_statistics.py --csv path/to/features.csv  # skip ALSI, use existing CSV
   ```

   This requires R and the ALSI dependencies (see Requirements above). Generated biographies must be placed in `biography_analysis/resources/complexity_tool/ALSI-main/corpus/biographies/` before running.

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
