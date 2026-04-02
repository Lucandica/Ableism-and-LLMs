# Models parameters
BASE_PARAMS = {
    "temp": 0.7,
    "top_p": 0.90,
    "repetition_penalty":1.0,
}

LLAMA3_PARAMS = BASE_PARAMS

MISTRAL_PARAMS = BASE_PARAMS

QWEN3_PARAMS = {
    **BASE_PARAMS,
    "top_k": 20,
}

# Models import depending on technique used (mlx or torch)

LLAMA3 = {
    "mlx": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "torch": "meta-llama/Llama-3.1-8B-Instruct"
}

MISTRAL = {
    "mlx": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "torch": "mistralai/Mistral-7B-Instruct-v0.3"
}

QWEN3 = {
    "mlx": "mlx-community/Qwen3-8B-4bit",
    "torch": "Qwen/Qwen3-8B",
}
