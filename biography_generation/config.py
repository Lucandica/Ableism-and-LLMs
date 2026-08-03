"""
Parameters used for the generation of the biographies in our experiences, and model importation links.
To add a model or a link, please follow the same pattern.
If a model is added, please verify that the parameters exist or add required parameters, and follow recommendation form model's developers.
"""

# Models parameters, based on recommended parameters of Llama and Qwen 
BASE_PARAMS = {
    "temp": 0.7,
    "top_p": 0.90,
    "repetition_penalty":1.0,
}

LLAMA3_PARAMS = BASE_PARAMS

MISTRAL_PARAMS = BASE_PARAMS

# Qwen has additional parameters, also based on recommendations
QWEN3_PARAMS = {
    **BASE_PARAMS,
    "top_k": 20,
}

# Models import depending on technique used (mlx, awq or torch)
LLAMA3 = {
    "awq": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "mlx": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "torch": "meta-llama/Llama-3.1-8B-Instruct"
}

MISTRAL = {
    "awq": "solidrust/Mistral-7B-Instruct-v0.3-AWQ",
    "mlx": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "torch": "mistralai/Mistral-7B-Instruct-v0.3"
}

QWEN3 = {
    "awq": "Qwen/Qwen3-8B-AWQ",
    "mlx": "mlx-community/Qwen3-8B-4bit",
    "torch": "Qwen/Qwen3-8B",
}
