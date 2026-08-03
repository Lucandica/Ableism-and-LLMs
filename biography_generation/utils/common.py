import json
import re
import os


def get_inputs(tokenizer, user_prompt: str, system_prompt: str = None, enable_thinking: bool = None):
    """
    Format inputs messages into chat template.
    Only used for MLX and transformer models.

    Args:
        tokenizer: Initiated tokenizer.
        user_prompt: String corresponding to user prompt.
        system_prompt: (None by default because Mistral models don't take system prompts) String corresponding to system prompt.
        enable_thinking: (None by default) True to activate Qwen's thinking mode, caution: uses more tokens.

    Returns:
        tokenizer.apply_chat_template(): tokenized input for model
    
    """

    messages = []

    # Add system prompt if there is one
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Add user prompt
    messages.append({"role": "user", "content": user_prompt})

    kwargs = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking

    # Apply tokenization
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        **kwargs
    )

def parse_json_response(response: str) -> dict:
    """
    Function that observe JSON outputs of the model.
    It flags incorrect outputs and try to correct them.
    
    Args:
        responses: Raw text output from a model.

    Returns:
        {"story": [story content]} if the output follows a correct a JSON format or manage to be corrected by unwrapping,
        {"story": [full model output], "parsing_error": true} if the output doesn't follow a correct JSON format and couldn't be corrected.
    
    """

    # Strip markdown code fences the model may have wrapped the JSON in
    response = re.sub(r"```json|```", "", response).strip()
    
    def unwrap(data: dict) -> dict:
        """Recursively unwrap double-encoded JSON in story field"""
        story = data.get("story")
        if isinstance(story, str):
            try:
                inner = json.loads(story)
                if isinstance(inner, dict):
                    return unwrap(inner)  # recurse in case of triple encoding
            except json.JSONDecodeError:
                pass
        return data

    # First, try to unwrap json file
    try:
        return unwrap(json.loads(response))
    except json.JSONDecodeError:
        pass

    # Fall back to extracting the outermost {...} block, in case the model added prose before or after the JSON
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            return unwrap(json.loads(match.group()))
        except json.JSONDecodeError:
            pass

    # If no option works, flag parsing error, and resolve it during JSON to txt conversion.
    return {"story": response, "parsing_error": True}

def save_response(data: dict, filepath: str, prompt_set: str, variant: str, system_prompt: str, user_prompt: str, iteration_number: int, technique: str):
    """Append a response with its prompt metadata to a JSON file"""
    responses = []

    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            existing = json.load(f)
            responses = existing if isinstance(existing, list) else [existing]

    entry = {
        "prompt_set": prompt_set,
        "variant": variant,
        "technique": technique,
        "system_prompt": system_prompt.strip(),
        "user_prompt": user_prompt.strip(),
        "iteration_number": iteration_number,
        **data
    }

    responses.append(entry)

    # Write current output to json file with its metadata
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)
