import os
import torch

from huggingface_hub import login

from utils.torch_utils import load_model_torch, generate_biography_torch
from utils.awq_utils import load_model_awq, generate_biography_awq

from config import LLAMA3, LLAMA3_PARAMS
from config import MISTRAL, MISTRAL_PARAMS
from config import QWEN3, QWEN3_PARAMS
from utils.common import save_response
from prompts import PROMPTS, build_mistral_prompt

if __name__ == '__main__':

    TOKEN = os.environ["HF_TOKEN"]
    login(TOKEN)

    prompt_versions = [
                    "v1_long"
                    #, "v2_long"
                    #, "v3_long"
                    ]

    models_and_parameters = {
        "llama":   {"model": LLAMA3,   "params": LLAMA3_PARAMS},
        "mistral": {"model": MISTRAL,  "params": MISTRAL_PARAMS},
        "qwen":    {"model": QWEN3,    "params": QWEN3_PARAMS},
    }

    for PROMPT_VERSION in prompt_versions:
        prompt_set = PROMPTS[PROMPT_VERSION]
        system_prompt = prompt_set["system"]
        user_prompt_nodis = prompt_set["no_dis"]
        user_prompt_withdis = prompt_set["with_dis"]

        for MODEL in models_and_parameters:
            params = models_and_parameters[MODEL]["params"]

            # Torch
            model_torch, tokenizer_torch = load_model_torch(models_and_parameters[MODEL]["model"])

            for iteration_number in range(50):
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
            torch.cuda.empty_cache()

            # AWQ
            model_awq = load_model_awq(models_and_parameters[MODEL]["model"])

            for iteration_number in range(50):
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
