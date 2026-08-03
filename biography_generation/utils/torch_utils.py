"""All functions used for generation with full models using transformers"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.common import get_inputs, parse_json_response

def load_model_torch(model_cfg: str):
    """
    Load model and tokenizer.
    
    Args:
        model_cfg: dictionary of the model and tokenizer references.

    Returns:
        model: loaded model
        tokenizer: loaded tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["torch"])
    model = AutoModelForCausalLM.from_pretrained(model_cfg["torch"], dtype=torch.float16, device_map="auto")
    return model, tokenizer


def generate_biography_torch(model, tokenizer, user_prompt, params, system_prompt=None, max_tokens=2048, enable_thinking = None):
    """
    Get needed configuration, parameters and prompt to generate biographies.

    Args:
        model: Model loaded using transformers.
        tokenizer: Tokenizer associated to the model, loaded using transformers.
        user_prompt: User prompt use for generation by the model.
        params: parameters associated to the models.
        system_prompt: (None by default: Mistral doesn't take a system prompt) system prompt use to direct model's behavior.
        max_token: (2048 by default) Stop the generation once reached.
        enable_thinking: (None by default) True to activate Qwen's thinking mode, caution: uses more tokens.

    Returns:
        response: json of text corresponding to the model output in the format:
            {"story": [story content]} if the output follows a correct a JSON format,
            {"story": [full model output], "parsing_error": true} if the output doesn't follow a correct JSON format.
    """

    # Initiate tokenizer
    inputs = tokenizer(
        get_inputs(tokenizer=tokenizer, user_prompt=user_prompt, system_prompt=system_prompt, enable_thinking=enable_thinking,),
        return_tensors="pt"
    ).to(model.device)

    # Generation
    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=params.get("temp", 0.7),
        top_p=params.get("top_p", 0.9),
        top_k=params.get("top_k", 0),
        repetition_penalty=params.get("repetition_penalty", 1.0),
        do_sample=True,
    )

    response = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens = True) # Filter to only have outputs
    return parse_json_response(response)