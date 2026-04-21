import os
from huggingface_hub import login
from utils.torch_utils import load_model, generate_biography

from config import LLAMA3, LLAMA3_PARAMS
from config import MISTRAL, MISTRAL_PARAMS
from config import QWEN3, QWEN3_PARAMS
from utils.common import save_response
from prompts import PROMPTS, build_mistral_prompt

TOKEN = os.environ["HF_TOKEN"]
login(TOKEN)

prompt_versions = ["v1_long", "v2_long", "v3_long"]
models_and_parameters = {"llama": {
                            "model":LLAMA3,
                            "params":LLAMA3_PARAMS
                            },

                         "mistral":{
                            "model":MISTRAL,
                            "params":MISTRAL_PARAMS
                            },
                          "qwen": {
                            "model":QWEN3,
                            "params":QWEN3_PARAMS
                            }
                        }

for PROMPT_VERSION in prompt_versions:
    prompt_set = PROMPTS[PROMPT_VERSION]
    system_prompt = prompt_set["system"]
    user_prompt_nodis = prompt_set["no_dis"]
    user_prompt_withdis = prompt_set["with_dis"]


    for MODEL in models_and_parameters:
        model, tokenizer = load_model(models_and_parameters[MODEL]["model"])
        params = models_and_parameters[MODEL]["params"]

        if MODEL == "mistral":
            for iteration_number in range(3):
            # Generate no-disability prompt biography
                biography_no_dis = generate_biography(
                    model = model,
                    tokenizer = tokenizer,
                    user_prompt = build_mistral_prompt(system_prompt=system_prompt, user_prompt=user_prompt_nodis),
                    params=params
                )
                biography_with_dis = generate_biography(
                    model = model,
                    tokenizer = tokenizer,
                    user_prompt = build_mistral_prompt(system_prompt=system_prompt, user_prompt=user_prompt_withdis),
                    params=params
                )

                save_response(data=biography_no_dis,
                        filepath=f"outputs/results_{MODEL}.json",
                        prompt_set=PROMPT_VERSION,
                        variant="nodis",
                        system_prompt=system_prompt,
                        user_prompt=user_prompt_nodis,
                        iteration_number=iteration_number)
                save_response(data=biography_with_dis,
                        filepath=f"outputs/results_{MODEL}.json",
                        prompt_set=PROMPT_VERSION,
                        variant="withdis",
                        system_prompt=system_prompt,
                        user_prompt=user_prompt_withdis,
                        iteration_number=iteration_number)
        else:
            for iteration_number in range(3):
            # Generate no-disability prompt biography
                biography_no_dis = generate_biography(
                    model = model,
                    tokenizer = tokenizer,
                    user_prompt = user_prompt_nodis,
                    system_prompt = system_prompt,
                    params = params,
                    enable_thinking = False if MODEL == "qwen" else None
                )
            # Generate with-disability prompt biography
                biography_with_dis = generate_biography(
                    model = model,
                    tokenizer = tokenizer,
                    user_prompt = user_prompt_withdis,
                    system_prompt = system_prompt,
                    params = params,
                    enable_thinking = False if MODEL == "qwen" else None
                )
            # Save results
                save_response(data=biography_no_dis,
                        filepath=f"outputs/results_{MODEL}.json",
                        prompt_set=PROMPT_VERSION,
                        variant="nodis",
                        system_prompt=system_prompt,
                        user_prompt=user_prompt_nodis,
                        iteration_number=iteration_number)

                save_response(data=biography_with_dis,
                        filepath=f"outputs/results_{MODEL}.json",
                        prompt_set=PROMPT_VERSION,
                        variant="withdis",
                        system_prompt=system_prompt,
                        user_prompt=user_prompt_withdis,
                        iteration_number=iteration_number)