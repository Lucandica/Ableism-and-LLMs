from utils.common import get_inputs, parse_json_response
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm import load, generate

def load_model(model_id: str):
        return load(model_id["mlx"])

def define_parameters(params: dict):
    sampler = make_sampler(
        temp=params.get("temp", 0.0),
        top_p=params.get("top_p", 0.0),
        top_k=params.get("top_k", 0),
    )

    logits_processors = make_logits_processors(
        repetition_penalty=params.get("repetition_penalty")
    )

    return sampler, logits_processors

def generate_biography(model, tokenizer, user_prompt: str, params: dict, max_tokens=640, system_prompt=None, enable_thinking=None, verbose=True):
    sampler, logits_processors = define_parameters(params=params)
    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=get_inputs(tokenizer=tokenizer, user_prompt=user_prompt, system_prompt=system_prompt, enable_thinking=enable_thinking,),
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        
        verbose=verbose
    )
    return parse_json_response(response)
