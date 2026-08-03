"""All functions used for generation with AWQ quantized models using vLLm"""

from vllm import LLM, SamplingParams
from utils.common import parse_json_response

def load_model_awq(model_cfg: dict):
    """
    Load model.
    VLLM doesn't require a tokenizer.

    Args:
        model_cfg: dictionary of the model and tokenizer references.

    Returns:
        Loaded model
    """

    return LLM(model=model_cfg["awq"], quantization="awq")

def generate_biography_awq(model, user_prompt, params, system_prompt=None, max_tokens=2048, enable_thinking=None, use_system_role=True):
    """
    Get needed configuration, parameters and prompt to generate biographies.

    Args:
        model: Model loaded using transformers.
        user_prompt: User prompt use for generation by the model.
        params: parameters associated to the models.
        system_prompt: (None by default: Mistral doesn't take a system prompt) system prompt use to direct model's behavior.
        max_token: (2048 by default) Stop the generation once reached.
        enable_thinking: (None by default) True to activate Qwen's thinking mode, caution: uses more tokens.
        use_system_role: (True by default) True to be able to specify certain role as system and user.

    Returns:
        response: json of text corresponding to the model output in the format:
            {"story": [story content]} if the output follows a correct a JSON format,
            {"story": [full model output], "parsing_error": true} if the output doesn't follow a correct JSON format.
    """
    if system_prompt: # Define role and content of the message (system prompt, user prompt)
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