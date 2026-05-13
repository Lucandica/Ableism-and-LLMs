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
│   │   └── name_entities/      # Named entity recognition module
│   ├── outputs/                # Analysis outputs (CSV results)
│   ├── biographies_parsed.csv  # Parsed biographies dataset
│   └── biographies_analysis.ipynb
├── generated_biographies/      # LLM-generated biography files (output of generation step)
├── requirements/
│   ├── generation.txt
│   └── analysis.txt
└── README.md
```

## Requirements

Python version used: **Python 3.13.5**

## Installation

Clone the repository:

```bash
git clone https://github.com/Lucandica/Ableism-and-LLMs.git
cd Ableism-and-LLMs
```

Install dependencies for the relevant section:

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

   Outputs (CSV files) are saved in `biography_analysis/outputs/`.

## References

- Ducel, F., Névéol, A., & Fort, K. (2024). "You'll be a nurse, my son!" Automatically assessing gender biases in autoregressive language models in French and Italian. *Language Resources and Evaluation, 59*(2), 1495–1523. https://doi.org/10.1007/s10579-024-09780-6
