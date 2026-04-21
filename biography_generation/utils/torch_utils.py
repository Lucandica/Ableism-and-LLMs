import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.common import get_inputs, parse_json_response

def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id["torch"])
    model = AutoModelForCausalLM.from_pretrained(model_id["torch"], torch_dtype=torch.float16)
    return model, tokenizer


def generate_biography(model, tokenizer, user_prompt, params, system_prompt=None, max_tokens=2048, enable_thinking = None):
    inputs = tokenizer(
        get_inputs(tokenizer=tokenizer, user_prompt=user_prompt, system_prompt=system_prompt, enable_thinking=enable_thinking,),
        return_tensors="pt"
    ).to(model.device)
    
    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=params.get("temp", 0.7),
        top_p=params.get("top_p", 0.9),
        top_k=params.get("top_k", 0),
        repetition_penalty=params.get("repetition_penalty", 1.0),
        do_sample=True,
    )

    response = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens = True)
    return parse_json_response(response)