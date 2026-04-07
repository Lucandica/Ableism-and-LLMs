import json
import re
import os


def get_inputs(tokenizer, user_prompt: str, system_prompt: str = None, enable_thinking: bool = None):
    """Helper function for input formatting"""

    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_prompt})

    kwargs = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking

    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        **kwargs
    )

def parse_json_response(response: str) -> dict:
    """Function to ensure json format of answer"""
    
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
    
    try:
        return unwrap(json.loads(response))
    except json.JSONDecodeError:
        pass
    
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            return unwrap(json.loads(match.group()))
        except json.JSONDecodeError:
            pass
    
    return {"story": response, "parsing_error": True}

def save_response(data: dict, filepath: str, prompt_set: str, variant: str, system_prompt: str, user_prompt: str):
    """Append a response with its prompt metadata to a JSON file"""
    responses = []

    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            existing = json.load(f)
            responses = existing if isinstance(existing, list) else [existing]

    entry = {
        "prompt_set": prompt_set,
        "variant": variant,
        "system_prompt": system_prompt.strip(),
        "user_prompt": user_prompt.strip(),
        **data
    }

    responses.append(entry)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)
