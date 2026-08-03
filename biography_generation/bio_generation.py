"""
Generation pipeline for higher-scale experiment, used to generate biographies.
This generations are made using AWQ and Transformers on Llama3.1, Mistral and Qwen3, 
but additional techniques and models can be added by modifying config.py, and files in utils/ folders.
"""

import os
import torch

from huggingface_hub import login

# Quantize/Full model specific imports
from utils.torch_utils import load_model_torch, generate_biography_torch # Import for transformers models
from utils.awq_utils import load_model_awq, generate_biography_awq # Import for AWQ models

# Models specific imports
from config import LLAMA3, LLAMA3_PARAMS
from config import MISTRAL, MISTRAL_PARAMS
from config import QWEN3, QWEN3_PARAMS

# Common imports and prompts
from utils.common import save_response
from prompts import PROMPTS, build_mistral_prompt

if __name__ == '__main__': # Generate by calling the file's name

    # As approval is required to use Llama3.1 a HF token is mandatory to run this code with this model. 
    TOKEN = os.environ["HF_TOKEN"]
    login(TOKEN)

    # Choose needed prompt, adding them in the list will loop through all called prompt.
    prompt_versions = [
                    "v1_long"
                    #, "v2_long"
                    #, "v3_long"
                    ]

    # Change for needed models, code will loop through all called models
    models_and_parameters = {
        "llama":   {"model": LLAMA3,   "params": LLAMA3_PARAMS},
        "mistral": {"model": MISTRAL,  "params": MISTRAL_PARAMS},
        "qwen":    {"model": QWEN3,    "params": QWEN3_PARAMS},
    }

    for PROMPT_VERSION in prompt_versions: # Prompts loop
        prompt_set = PROMPTS[PROMPT_VERSION]
        system_prompt = prompt_set["system"]
        user_prompt_nodis = prompt_set["no_dis"]
        user_prompt_withdis = prompt_set["with_dis"]

        for MODEL in models_and_parameters: # Models loop
            params = models_and_parameters[MODEL]["params"]

            # Torch generation loop, can be commented out if not used
            model_torch, tokenizer_torch = load_model_torch(models_and_parameters[MODEL]["model"])

            for iteration_number in range(50): # Iterations loop, can be changed depending on preferences
                if MODEL == "mistral":
                    biography_no_dis_torch = generate_biography_torch(
                        model=model_torch,
                        tokenizer=tokenizer_torch,
                        user_prompt=build_mistral_prompt(system_prompt=system_prompt, user_prompt=user_prompt_nodis),
                        params=params
                    )
                    biography_with_dis_torch = generate_biography_torch(
                        model=model_torch,
                        tokenizer=tokenizer_torch,
                        user_prompt=build_mistral_prompt(system_prompt=system_prompt, user_prompt=user_prompt_withdis),
                        params=params
                    )
                else:
                    biography_no_dis_torch = generate_biography_torch(
                        model=model_torch,
                        tokenizer=tokenizer_torch,
                        user_prompt=user_prompt_nodis,
                        system_prompt=system_prompt,
                        params=params,
                        enable_thinking=False if MODEL == "qwen" else None
                    )
                    biography_with_dis_torch = generate_biography_torch(
                        model=model_torch,
                        tokenizer=tokenizer_torch,
                        user_prompt=user_prompt_withdis,
                        system_prompt=system_prompt,
                        params=params,
                        enable_thinking=False if MODEL == "qwen" else None
                    )

                save_response(data=biography_no_dis_torch,
                        filepath=f"outputs/results_{MODEL}.json",
                        prompt_set=PROMPT_VERSION,
                        variant="nodis",
                        technique="torch",
                        system_prompt=system_prompt,
                        user_prompt=user_prompt_nodis,
                        iteration_number=iteration_number)
                save_response(data=biography_with_dis_torch,
                        filepath=f"outputs/results_{MODEL}.json",
                        prompt_set=PROMPT_VERSION,
                        variant="withdis",
                        technique="torch",
                        system_prompt=system_prompt,
                        user_prompt=user_prompt_withdis,
                        iteration_number=iteration_number)

            del model_torch, tokenizer_torch
            torch.cuda.empty_cache() # Delete model from memory to save space.

            # AWQ generation loop, can be commented out if not used
            model_awq = load_model_awq(models_and_parameters[MODEL]["model"])

            for iteration_number in range(50): # Iterations loop, can be changed depending on preferences
                biography_no_dis_awq = generate_biography_awq(
                    model=model_awq,
                    user_prompt=user_prompt_nodis,
                    system_prompt=system_prompt,
                    params=params,
                    enable_thinking=False if MODEL == "qwen" else None,
                    use_system_role=False if MODEL == "mistral" else True
                )
                biography_with_dis_awq = generate_biography_awq(
                    model=model_awq,
                    user_prompt=user_prompt_withdis,
                    system_prompt=system_prompt,
                    params=params,
                    enable_thinking=False if MODEL == "qwen" else None,
                    use_system_role=False if MODEL == "mistral" else True
                )
                save_response(data=biography_no_dis_awq,
                        filepath=f"outputs/results_{MODEL}.json",
                        prompt_set=PROMPT_VERSION,
                        variant="nodis",
                        technique="awq",
                        system_prompt=system_prompt,
                        user_prompt=user_prompt_nodis,
                        iteration_number=iteration_number)
                save_response(data=biography_with_dis_awq,
                        filepath=f"outputs/results_{MODEL}.json",
                        prompt_set=PROMPT_VERSION,
                        variant="withdis",
                        technique="awq",
                        system_prompt=system_prompt,
                        user_prompt=user_prompt_withdis,
                        iteration_number=iteration_number)

            del model_awq
