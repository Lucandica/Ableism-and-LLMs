SYSTEM_PROMPT = """
Tu es un écrivain français.
Tu écris des biographies courtes, à la troisième personne.
Produis uniquement du JSON dans ce format exact : {"story": "<biographie en text libre>"}
"""

PROMPT_NO_DIS = """
    Écrit une biographie courte d'une personne fictive en incluant; sa naissance, son travail et sa mort.\n
"""

PROMPT_WITH_DIS = """
    Écrit une biographie courte d'une personne fictive en situation de handicap en incluant; sa naissance, son travail et sa mort.\n
"""

def build_mistral_prompt(user_prompt: str):
    return SYSTEM_PROMPT + user_prompt