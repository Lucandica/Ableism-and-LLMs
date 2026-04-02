import json
import re


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
    
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    return {"story": response, "parsing_error": True}

def save_response(data: dict, filepath: str):
    """Save JSON response to file"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)