"""All functions used for generation with quantized models using MLX"""

from utils.common import get_inputs, parse_json_response
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm import load, generate

def load_model_mlx(model_cfg: str):
    """
    Load model and tokenizer.

    Args:
        model_cfg: dictionary of the model and tokenizer references.
    
    Returns:
        (model, tokenizer): model and tokenizer loaded.
    """
    return load(model_cfg["mlx"])

def define_parameters(params: dict):
    """
    Initiate parameters used for generation.
    
    Args: 
        params: dictionary of parameters used with associated values.

    Returns:
        sampler, logits_processors: parameters initiated for generation
    """
    sampler = make_sampler(
        temp=params.get("temp", 0.0),
        top_p=params.get("top_p", 0.0),
        top_k=params.get("top_k", 0),
    )

    logits_processors = make_logits_processors(
        repetition_penalty=params.get("repetition_penalty")
    )

    return sampler, logits_processors

def generate_biography_mlx(model, tokenizer, user_prompt: str, params: dict, max_tokens=640, system_prompt=None, enable_thinking=None, verbose=True):
    """
    Get needed configuration, parameters and prompt to generate biographies.

    Args:
        model: Model loaded using transformers.
        tokenizer: Tokenizer associated to the model, loaded using transformers.
        user_prompt: User prompt use for generation by the model.
        params: parameters associated to the models.
        max_token: (640 by default) Stop the generation once reached.
        system_prompt: (None by default: Mistral doesn't take a system prompt) system prompt use to direct model's behavior.
        enable_thinking: (None by default) True to activate Qwen's thinking mode, caution: uses more tokens.
        verbose: (True by default) if True, show generation in real time.

    Returns:
        response: json of text corresponding to the model output in the format:
            {"story": [story content]} if the output follows a correct a JSON format,
            {"story": [full model output], "parsing_error": true} if the output doesn't follow a correct JSON format.
    """

    # Initiate parameters used in generation
    sampler, logits_processors = define_parameters(params=params)

    # Generation
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
