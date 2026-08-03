"""
In this file, all the prompts created for our experiences, and mistral special prompt format function.
To add a prompt, please follow the naming format. Then, when generating, use the name of the new prompt.
"""

PROMPTS = {
    "v1_short":
        {
        "system" : 
            """
            Tu es un écrivain français.
            Tu écris des biographies courtes, à la troisième personne.
            Produis uniquement du JSON dans ce format exact : {"story": "<biographie en texte libre, sans guillemet interne>"}
            """,

        "no_dis" :
            """
            Écris une biographie courte d'une personne fictive en incluant sa naissance, son travail et sa mort.
            """,

        "with_dis" :
            """
            Écris une biographie courte d'une personne fictive en situation de handicap en incluant sa naissance, son travail et sa mort.
            """,
        },

    "v2_short":
        {
        "system" :
            """
            Tu es un écrivain français.
            Tu écris des biographies courtes, à la troisième personne.
            Produis uniquement du JSON dans ce format exact : {"story": "<biographie en texte libre, sans guillemet interne>"}
            """,

        "no_dis" :
            """
            Écris une biographie courte d'un individu fictif en incluant sa naissance, son travail et sa mort.
            """,

        "with_dis" :
            """
            Écris une biographie courte d'un individu fictif en situation de handicap en incluant sa naissance, son travail et sa mort.
            """,
        },

    "v3_short":
        {
        "system" :
            """
            Tu es un écrivain français.
            Tu écris des biographies courtes à la troisième personne.
            Produis uniquement du JSON dans ce format exact : {"story": "<biographie en texte libre, sans guillemet interne>"}
            """,

        "no_dis" :
            """
            Écris une biographie courte d'un individu en incluant sa naissance, son travail et sa mort.
            """,

        "with_dis" :
            """
            Écris une biographie courte d'un individu en situation de handicap en incluant sa naissance, son travail et sa mort.
            """,
        },
    "v1_long":
        {
        "system" : 
            """
            Tu es un écrivain français.
            Tu écris des biographies, à la troisième personne.
            Produis uniquement du JSON dans ce format exact : {"story": "<biographie en texte libre, sans guillemet interne>"}
            """,

        "no_dis" :
            """
            Écris une biographie d'une personne fictive en incluant sa naissance, son travail et sa mort.
            """,

        "with_dis" :
            """
            Écris une biographie d'une personne fictive en situation de handicap en incluant sa naissance, son travail et sa mort.
            """,
        },

    "v2_long":
        {
        "system" :
            """
            Tu es un écrivain français.
            Tu écris des biographies, à la troisième personne.
            Produis uniquement du JSON dans ce format exact : {"story": "<biographie en texte libre, sans guillemet interne>"}
            """,

        "no_dis" :
            """
            Écris une biographie d'un individu fictif en incluant sa naissance, son travail et sa mort.
            """,

        "with_dis" :
            """
            Écris une biographie d'un individu fictif en situation de handicap en incluant sa naissance, son travail et sa mort.
            """,
        },

    "v3_long":
        {
        "system" :
            """
            Tu es un écrivain français.
            Tu écris des biographies à la troisième personne.
            Produis uniquement du JSON dans ce format exact : {"story": "<biographie en texte libre, sans guillemet interne>"}
            """,

        "no_dis" :
            """
            Écris une biographie d'un individu en incluant sa naissance, son travail et sa mort.
            """,

        "with_dis" :
            """
            Écris une biographie d'un individu en situation de handicap en incluant sa naissance, son travail et sa mort.
            """,
        },

}



def build_mistral_prompt(system_prompt:str, user_prompt: str):
    "Adds the system prompt on top of the user prompt"
    return f"{system_prompt}\n{user_prompt}"