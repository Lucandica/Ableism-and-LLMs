from vllm import LLM, SamplingParams
from utils.common import parse_json_response

def load_model_awq(model_cfg: dict):
    return LLM(model=model_cfg["awq"], quantization="awq")

def generate_biography_awq(model, user_prompt, params, system_prompt=None, max_tokens=2048, enable_thinking=None, use_system_role=True):
    if system_prompt:
        if use_system_role:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ]
        else:
            messages = [
                {"role": "user", "content": f"{system_prompt}\n{user_prompt}"}
            ]
    else:
        messages = [
            {"role": "user", "content": user_prompt}
        ]
    sp_kwargs = dict(
        temperature=params["temp"],
        top_p=params["top_p"],
        repetition_penalty=params["repetition_penalty"],
        max_tokens=max_tokens,
    )
    if "top_k" in params:
        sp_kwargs["top_k"] = params["top_k"]

    chat_kwargs = {}
    if enable_thinking is False:
        chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

    outputs = model.chat(messages, SamplingParams(**sp_kwargs), **chat_kwargs)
    response = outputs[0].outputs[0].text
    return parse_json_response(response)