import re
import os
import pandas as pd

DISABILITIES = {
# Categories from: https://handicap.agriculture.gouv.fr/les-grandes-familles-ou-typologies-de-handicap-a231.html
 
"handicaps moteurs":
    [
        # https://handicap.agriculture.gouv.fr/les-grandes-familles-ou-typologies-de-handicap-a231.html
        # Dictionnaire du handicap, 7ème édition, Zribi, G. & Poupée-Fontaine D., 2014
        # https://fondationordredemalte.org/liste-des-handicaps-reconnus-comprendre-accompagner-inclure/
        # https://www.education.gouv.fr/handicap-tous-concernes-99935
        # https://bnau.fr/les-acronymes-du-handicap/
        "handicap moteur", "déficience motrice", "membre inférieur", "membre supérieur", "membres inférieurs", "membres supérieurs", "rhumatisme", "arthrose", "hémiplégie", "hémiplégique", "paraplégie",
        "paraplégique", "tétraplégie", "tétraplégique", "quadriplégique", "quadriplégie", "lombalgie", "TMS", "tms", "trouble musculo-squelettique", "malformation", "paralysie", "paralysé",
        "AVC", "accident vasculaire cérébral", "amputation", "amputé", "infirmité motrice cérébrale", "IMC", "diplégie", "diplégique", "hernie discale",
    ],
 
"handicaps sensoriels":
    [
        # https://handicap.agriculture.gouv.fr/les-grandes-familles-ou-typologies-de-handicap-a231.html
        # https://www.agefiph.fr/sites/default/files/import_destination/quest-ce-le-handicap_personnalisable.pdf
        # Dictionnaire du handicap, 7ème édition, Zribi, G. & Poupée-Fontaine D., 2014
        # https://www.nidcd.nih.gov/health/taste-disorders
        # https://www.education.gouv.fr/handicap-tous-concernes-99935
        # https://fondationordredemalte.org/liste-des-handicaps-reconnus-comprendre-accompagner-inclure/
        # https://www.apollohospitals.com/fr/diseases-and-conditions/hypogeusia
        # https://www.apollohospitals.com/fr/diseases-and-conditions/hyposmia
        "handicap sensoriel",
        # Déficience visuelle
        "handicap visuel", "déficience visuelle", "aveugle", "non-voyant", "malvoyant", "daltonisme", "daltonien", "daltonienne",
        "cécité", "malvoyance", "rétinopathie diabétique", "glaucome", "dégénérescence maculaire", "DMLA", "cataracte", "myopie", "myope", "presbytie", "presbyte",
        "strabisme", "rétinite",
        # Déficience auditive
        "handicap auditif", "déficience auditive", "surdité", "perte auditive", "sourd", "malentendant", "agnosie", "amblyopie", "sourd-aveugle",
        # Déficience de goût
        "agueusie", "hypogueusie", "dysgueusie",
        # Déficience de toucher
        "dysesthésie",
        # Déficience olfactive
        "hyposmie", "anosmie",
        # Autre
        "muet",

    ],
 
"handicaps psychiques":
    [
        # https://handicap.agriculture.gouv.fr/les-grandes-familles-ou-typologies-de-handicap-a231.html
        # Dictionnaire du handicap, 7ème édition, Zribi, G. & Poupée-Fontaine D., 2014
        "handicap psychique", "trouble psychique", "maladie psychique", "phobie", "anxiété généralisée", "peur panique", "agoraphobie", "agora-phobie",
        "trouble obsessionnel compulsif", "TOC", "bipolarité", "bipolaire", "troubles psychosomatiques", "addiction", "névrose", "dépression", "schizophrénie", "trouble anxieux",
        "psychoses", "abandonnisme", "syndrome d'abandon", "névrose d'abandon", "anorexie", "anorexique", "apragmatisme", "boulimie", "boulimique", "trouble alimentaire",
        "pica", "hallucination", "hystérie", "psychopathie", "psychopathe", "stress post-traumatique", "PTSD", "trouble de la personnalité", "borderline",
    ],
 
"handicaps cognitifs":
    [
        # https://handicap.agriculture.gouv.fr/les-grandes-familles-ou-typologies-de-handicap-a231.html
        # https://handicap.agriculture.gouv.fr/journee-nationale-des-dys-a350.html
        # https://www.agefiph.fr/sites/default/files/import_destination/quest-ce-le-handicap_personnalisable.pdf
        # Dictionnaire du handicap, 7ème édition, Zribi, G. & Poupée-Fontaine D., 2014
        # https://www.autisme-france.fr/terminologie-de-lautisme
        # https://handicap.gouv.fr/la-strategie-nationale-autisme-et-troubles-du-neurodeveloppement-2018-2022
        "handicap cognitif", "trouble cognitif", "trouble du neurodéveloppement", "trouble du déficit de l'attention", "hyperactivité", "TDA", "TDAH", "trouble du langage",
        "trouble de l'apprentissage", "trouble de la communication", "dysphasie", "trouble de la parole", "trouble articulatoire", "trouble de la fluence", "dyscalculie",
        "dyslexie", "dysgraphie", "trouble développemental de la coordination", "dyspraxie", "TSA", "autisme", "autiste", "autistique", "asperger", "asperger", "rett",
        "aphasie", "trouble de la mémoire", "trouble des fonctions exécutives", "haut potentiel intellectuel", "HPI",
    ],
 
"handicaps mentaux":
    [
        # https://handicap.agriculture.gouv.fr/les-grandes-familles-ou-typologies-de-handicap-a231.html
        # https://fr.wikipedia.org/wiki/Handicap_mental
        "handicap mental", "déficience intellectuelle", "trisomie", "trisomie 21", "down", "digeorge", "vélocardiofacial", "phelan-macdermid", "prader-willi", "phénylcétonurie",
        "x-fragile", "neurofibromatose", "hypothyroïdie", "mowat-wilson", "ciliopathie"
    ],
 
"maladies chroniques évolutives ou invalidantes":
    [
        # https://handicap.agriculture.gouv.fr/les-grandes-familles-ou-typologies-de-handicap-a231.html
        # https://fr.wikipedia.org/wiki/Maladie_neuro-%C3%A9volutive#Les_maladies_neurod%C3%A9g%C3%A9n%C3%A9ratives
        # https://fr.wikipedia.org/wiki/Maladie_neuro-%C3%A9volutive
        # https://www.inserm.fr/dossier/mucoviscidose/
        # https://www.logiadapt.fr/blog/maladies-neurologiques
        "maladie chronique", "diabète", "hémophilie", "hémophile", "sida", "VIH", "cancer", "hyperthyroïdie", "hypertension artérielle", "insuffisance cardiaque",
        "eczéma", "épilepsie", "insuffisance rénale", "crohn", "polyarthrite", "arthrite", "hépatite", "fibromyalgie", "endométriose", "sclérose", "neuromyélite", "narcolepsie",
        "drépanocytose", "alzheimer", "parkinson", "athétose", "chorée",
        "mucoviscidose",
        # Maladies neurodégénératives
        "myopathie", "huntington", "benson", "acp", "ataxie", "atrophie", "corps de lewy", "creutzfeldt-jakob", "pick", "myofasciite",
        # Maladies rares et syndromes
        "wolfram", "charcot-marie", "ehlers-danlos", "fibrodysplasie", "dystrophie", "alpers", "mueller-weiss", "encéphalopathie", "tay-sachs",
        # Maladies respiratoires chroniques
        "BPCO", "bronchopneumopathie chronique obstructive", "insuffisance respiratoire",
    ],
 
"maladies génétiques":
    [
        # https://fr.wikipedia.org/wiki/Liste_des_maladies_g%C3%A9n%C3%A9tiques_%C3%A0_g%C3%A8ne_identifi%C3%A9
        # https://fr.wikipedia.org/wiki/Dr%C3%A9panocytose
        # https://www.sciencedirect.com/science/article/abs/pii/S0003448721003954
        # https://www.inserm.fr/dossier/hemophilie/
        # https://www.inserm.fr/dossier/mucoviscidose/
        # https://www.logiadapt.fr/blog/maladies-neurologiques
        "maladie génétique", "x fragile", "klinefelter", "triple x", "turner", "trisomie", "coronarienne", "spina bifida", "duchenne", "hypercholestérolémie",
        "hémochromatose", "neurofibromatose", "drépanocytose", "amylose", "adrénoleucodystrophie", "mitochondrial", "usher", "cri du chat", "maladie de dercum", "duane",
        "hémophilie", "hémophile", "phénylcétonurie", "vélocardiofacial", "mucoviscidose", "fibrose", "prader-willi", "hypothyroïdie", "williams-beuren", "swb", "mowat-wilson",
        "ciliopathie", "huntington", "myopathie",
    ],

"aides":
    [
        # https://handicap.agriculture.gouv.fr/les-grandes-familles-ou-typologies-de-handicap-a231.html
        "aide à la mobilité", "chien guide", "appareil auditif", "implant cochléaire", "canne blanche", "braille", "prothèse auditive", "aide auditive", "fauteuil roulant",
        "fauteuil électrique", "béquille", "déambulateur", "verticalisateur", "prothèse", "orthèse", "mobilité réduite", "PMR",  "langue des signes", "LSF",
        "lecture labiale",
    ],

"formulations":
    [
        "syndrome de", "maladie de",
    ],

"autres termes":
    [
        # https://handicap.agriculture.gouv.fr/les-grandes-familles-ou-typologies-de-handicap-a231.html
        "handicap", "handicapé", "polyhandicap", "polyhandicapé", "plurihandicap", "plurihandicapé", "surhandicap", "surhandicapé",
        "maladie rare", "maladie neurologique", "maladie",
    ],
}

PATTERN = re.compile(r'(?i)(?<!\w)(?:' + '|'.join(rf"{re.escape(word)}e?s?" for words in DISABILITIES.values() for word in words) + r')(?!\w)')

SPECIAL_PATTERN = re.compile(r'(?i)(?<!\w)(?:syndrome\s+de|maladie\s+de)\s+\w+')

REVERSE_LOOKUP = {word.lower(): cat for cat, words in DISABILITIES.items() for word in words}

IRREGULAR = {
    "muette": "muet",
    "muettes": "muet",
}


def normalize(word: str, reverse_lookup: dict = None) -> str:
    """Strip French feminine/plural suffixes, preferring candidates found in the lookup."""
    if word in IRREGULAR:
        return IRREGULAR[word]
    for suffix in ('ées', 'ée', 'es', 'e', 's'):
        if word.endswith(suffix):
            candidate = word[:-len(suffix)]
            # If we have a lookup, only accept the candidate if it's a known key
            if reverse_lookup is None or candidate in reverse_lookup:
                return candidate
    return word


def category_of(hit, reverse_lookup=REVERSE_LOOKUP):
    """Resolve the disability category of a single detected term.
    Exact match -> normalized (strips feminine/plural suffix) -> "syndrome de ..."/"maladie de ..." extended forms.
    The hit is lowercased so it matches the lowercase REVERSE_LOOKUP keys (handles acronyms such as TSA, LSF).
    Returns None for genuinely unknown terms.
    """
    hit = hit.lower()
    cat = reverse_lookup.get(hit) or reverse_lookup.get(normalize(hit, reverse_lookup))
    if cat is None:
        if re.match(r'syndrome de\s+\w+', hit):
            cat = reverse_lookup.get('syndrome de')
        elif re.match(r'maladie de\s+\w+', hit):
            cat = reverse_lookup.get('maladie de')
    return cat


def get_categories(hits, reverse_lookup=REVERSE_LOOKUP):
    seen = []
    has_autres_termes = False

    for hit in hits:
        cat = category_of(hit, reverse_lookup)
        if cat is None:
            continue
        if cat == "autres termes":
            has_autres_termes = True
        elif cat not in seen:
            seen.append(cat)

    if not seen and has_autres_termes:
        seen.append("autres termes")

    return seen


def collect_hits(pattern, text):
    if pd.isna(text):
        return []
    hits = [hit.lower() for hit in pattern.findall(text)]

    for extended_hit in [h.lower() for h in SPECIAL_PATTERN.findall(text)]:
        base_form = 'syndrome de' if extended_hit.startswith('syndrome') else 'maladie de'
        if base_form in hits:
            hits.remove(base_form)
        if extended_hit not in hits:
            hits.append(extended_hit)


    seen: set[str] = set()
    normalized_hits: list[str] = []
    for raw in hits:
        key = normalize(raw, REVERSE_LOOKUP)
        if key not in seen:
            seen.add(key)
            normalized_hits.append(key)
    return normalized_hits

def apply_disability_detection(folder):
    rows = []
    for file in (f for f in os.scandir(folder) if f.name.endswith('.txt')):
        with open(file.path, 'r', encoding='utf-8') as f:
            text = f.read()
            rows.append({"doc_id": file.name, "disabilities": collect_hits(PATTERN, text)})

    df_dis = pd.DataFrame(rows)
    df_dis["categories"] = df_dis["disabilities"].apply(lambda hits: get_categories(hits, REVERSE_LOOKUP))
    
    return df_dis