import re
import os
import pandas as pd

DISABILITIES = {
# Categories from: https://handicap.agriculture.gouv.fr/les-grandes-familles-ou-typologies-de-handicap-a231.html

"handicaps moteurs":
    [
        # https://handicap.agriculture.gouv.fr/les-grandes-familles-ou-typologies-de-handicap-a231.html
        # Dictionnaire du handicap, 7eme edition, Zribi, G. & Poupee-Fontaine D., 2014
        # https://fondationordredemalte.org/liste-des-handicaps-reconnus-comprendre-accompagner-inclure/
        # https://www.education.gouv.fr/handicap-tous-concernes-99935
        # https://bnau.fr/les-acronymes-du-handicap/
        "handicap moteur", "déficience motrice", "membre inférieur", "membre supérieur", "membres inférieurs", "membres supérieurs", "rhumatisme", "arthrose", "hémiplégie", "hémiplégique", "paraplégie",
        "paraplégique", "tétraplégie", "tétraplégique", "quadriplégique", "quadriplégie", "lombalgie", "TMS", "trouble musculo-squelettique", "paralysie", "paralysé",
        "AVC", "accident vasculaire cérébral", "amputation", "amputé", "infirmité motrice cérébrale", "diplégie", "diplégique", "hernie discale",
    ],

"handicaps sensoriels":
    [
        # https://handicap.agriculture.gouv.fr/les-grandes-familles-ou-typologies-de-handicap-a231.html
        # https://www.agefiph.fr/sites/default/files/import_destination/quest-ce-le-handicap_personnalisable.pdf
        # Dictionnaire du handicap, 7eme edition, Zribi, G. & Poupee-Fontaine D., 2014
        # https://www.nidcd.nih.gov/health/taste-disorders
        # https://www.education.gouv.fr/handicap-tous-concernes-99935
        # https://fondationordredemalte.org/liste-des-handicaps-reconnus-comprendre-accompagner-inclure/
        # https://www.apollohospitals.com/fr/diseases-and-conditions/hypogeusia
        # https://www.apollohospitals.com/fr/diseases-and-conditions/hyposmia
        "handicap sensoriel",
        # Deficience visuelle
        "handicap visuel", "déficience visuelle", "aveugle", "non-voyant", "malvoyant", "daltonisme", "daltonien", "daltonienne",
        "cécité", "malvoyance", "rétinopathie diabétique", "glaucome", "dégénérescence maculaire", "DMLA", "cataracte",
        "rétinite",
        # Deficience auditive
        "handicap auditif", "déficience auditive", "surdité", "perte auditive", "sourd", "malentendant", "agnosie", "amblyopie", "sourd-aveugle",
        # Deficience de gout
        "agueusie", "hypogueusie", "dysgueusie",
        # Deficience de toucher
        "dysesthésie",
        # Deficience olfactive
        "hyposmie", "anosmie",
        # Autre
        "muet",
    ],

"handicaps psychiques":
    [
        # https://handicap.agriculture.gouv.fr/les-grandes-familles-ou-typologies-de-handicap-a231.html
        # https://fr.wikipedia.org/wiki/Trouble_psychique
        # Dictionnaire du handicap, 7eme edition, Zribi, G. & Poupee-Fontaine D., 2014
        "handicap psychique", "trouble psychique", "maladie psychique", "phobie", "phobie sociale", "anxiété", "anxiété généralisée", "anxieux", "anxieuse",
        "peur panique", "trouble panique", "agoraphobie", "agora-phobie",
        "trouble obsessionnel compulsif", "TOC", "bipolarité", "bipolaire", "trouble bipolaire",
        "trouble psychosomatique", "troubles psychosomatiques", "addiction", "névrose",
        "dépression", "dépressif", "dépressive", "schizophrénie", "schizophrène", "trouble anxieux",
        "psychose", "abandonnisme", "syndrome d'abandon", "névrose d'abandon", "anorexie", "anorexique", "apragmatisme", "boulimie", "boulimique",
        "trouble alimentaire", "TCA",
        "pica", "hallucination", "hystérie", "psychopathie", "psychopathe", "stress post-traumatique", "PTSD", "trouble de la personnalité", "borderline",
    ],

"handicaps cognitifs":
    [
        # https://handicap.agriculture.gouv.fr/les-grandes-familles-ou-typologies-de-handicap-a231.html
        # https://handicap.agriculture.gouv.fr/journee-nationale-des-dys-a350.html
        # https://www.agefiph.fr/sites/default/files/import_destination/quest-ce-le-handicap_personnalisable.pdf
        # Dictionnaire du handicap, 7eme edition, Zribi, G. & Poupee-Fontaine D., 2014
        # https://www.autisme-france.fr/terminologie-de-lautisme
        # https://handicap.gouv.fr/la-strategie-nationale-autisme-et-troubles-du-neurodeveloppement-2018-2022
        "handicap cognitif", "trouble cognitif", "trouble du neurodéveloppement", "trouble du déficit de l'attention", "hyperactivité", "TDA", "TDAH", "trouble du langage",
        "trouble de l'apprentissage", "trouble de la communication", "dysphasie", "trouble de la parole", "trouble articulatoire", "trouble de la fluence", "dyscalculie",
        "dyslexie", "dysgraphie", "trouble développemental de la coordination", "dyspraxie", "TSA", "autisme", "autiste", "autistique", "asperger", "rett",
        "aphasie", "trouble de la mémoire", "trouble des fonctions exécutives",
    ],

"handicaps mentaux":
    [
        # https://handicap.agriculture.gouv.fr/les-grandes-familles-ou-typologies-de-handicap-a231.html
        # https://fr.wikipedia.org/wiki/Handicap_mental
        "handicap mental", "déficience intellectuelle", "trisomie", "trisomie 21", "down", "digeorge", "vélocardiofacial", "phelan-macdermid", "prader-willi", "phénylcétonurie",
        "x-fragile", "neurofibromatose", "mowat-wilson", "ciliopathie"
    ],

"maladies chroniques évolutives ou invalidantes":
    [
        # https://handicap.agriculture.gouv.fr/les-grandes-familles-ou-typologies-de-handicap-a231.html
        # https://fr.wikipedia.org/wiki/Maladie_neuro-%C3%A9volutive
        # https://www.inserm.fr/dossier/mucoviscidose/
        # https://www.logiadapt.fr/blog/maladies-neurologiques
        "maladie chronique", "diabète", "diabétique", "hémophilie", "hémophile", "sida", "VIH", "cancer", "hyperthyroïdie", "hypothyroïdie",
        "hypertension artérielle", "insuffisance cardiaque",
        "eczéma", "épilepsie", "épileptique", "asthme", "asthmatique", "insuffisance rénale", "crohn", "polyarthrite", "arthrite", "hépatite", "fibromyalgie", "endométriose",
        "sclérose", "neuromyélite", "narcolepsie",
        "drépanocytose", "alzheimer", "parkinson", "athétose", "chorée",
        "mucoviscidose", "AVC", "accident vasculaire cérébral",
        # Maladies neurodegeneratives
        "myopathie", "huntington", "benson", "ataxie", "atrophie musculaire", "corps de lewy", "creutzfeldt-jakob", "pick", "myofasciite",
        # Maladies rares et syndromes
        "wolfram", "charcot-marie", "ehlers-danlos", "fibrodysplasie", "dystrophie", "alpers", "mueller-weiss", "encéphalopathie", "tay-sachs",
        # Maladies respiratoires chroniques
        "BPCO", "bronchopneumopathie chronique obstructive", "insuffisance respiratoire",
    ],

"maladies génétiques":
    [
        # https://fr.wikipedia.org/wiki/Liste_des_maladies_g%C3%A9n%C3%A9tiques_%C3%A0_g%C3%A8ne_identifi%C3%A9
        # https://fr.wikipedia.org/wiki/Dr%C3%A9panocytose
        # https://www.inserm.fr/dossier/hemophilie/
        # https://www.inserm.fr/dossier/mucoviscidose/
        # https://www.logiadapt.fr/blog/maladies-neurologiques
        "maladie génétique", "x fragile", "klinefelter", "triple x", "turner", "trisomie", "coronarienne", "spina bifida", "duchenne", "hypercholestérolémie",
        "hémochromatose", "neurofibromatose", "drépanocytose", "amylose", "adrénoleucodystrophie", "mitochondrial", "usher", "cri du chat", "maladie de dercum", "duane",
        "hémophilie", "hémophile", "phénylcétonurie", "vélocardiofacial", "mucoviscidose", "prader-willi", "williams-beuren", "swb", "mowat-wilson",
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
        "maladie rare", "maladie neurologique", "maladie", "malformation",
    ],
}

_ALL_WORDS = sorted({w.lower() for words in DISABILITIES.values() for w in words},
                    key=lambda w: (-len(w), w))

PATTERN = re.compile(r'(?i)(?<!\w)(?:'
                     + '|'.join(rf"{re.escape(word)}e?s?" for word in _ALL_WORDS)
                     + r')(?!\w)')

SPECIAL_PATTERN = re.compile(r'(?i)(?<!\w)(?:syndrome\s+de|maladie\s+de)\s+\w+')

BASE_FORMULATIONS = ('syndrome de', 'maladie de')

REVERSE_LOOKUP: dict[str, list[str]] = {}
for _cat, _words in DISABILITIES.items():
    for _word in _words:
        _key = _word.lower()
        if _cat not in REVERSE_LOOKUP.setdefault(_key, []):
            REVERSE_LOOKUP[_key].append(_cat)

IRREGULAR = {
    "muette": "muet", "muettes": "muet",
    "dépressive": "dépressif", "dépressives": "dépressif",
    "anxieuse": "anxieux", "anxieuses": "anxieux",
}


def normalize(word: str, reverse_lookup: dict = None) -> str:
    """Strip French feminine/plural suffixes, preferring candidates found in the lookup."""
    if word in IRREGULAR:
        return IRREGULAR[word]
    for suffix in ('ées', 'ée', 'es', 'e', 's'):
        if word.endswith(suffix):
            candidate = word[:-len(suffix)]
            if reverse_lookup is None or candidate in reverse_lookup:
                return candidate
    return word


def categories_of(hit, reverse_lookup=REVERSE_LOOKUP):
    """All disability categories a detected term belongs to."""
    hit = hit.lower()
    cats = reverse_lookup.get(hit) or reverse_lookup.get(normalize(hit, reverse_lookup))
    if cats is None:
        if re.match(r'syndrome de\s+\w+', hit):
            cats = reverse_lookup.get('syndrome de')
        elif re.match(r'maladie de\s+\w+', hit):
            cats = reverse_lookup.get('maladie de')
    return list(cats) if cats else []


def get_categories(hits, reverse_lookup=REVERSE_LOOKUP):
    seen = []
    has_autres_termes = False
    for hit in hits:
        for cat in categories_of(hit, reverse_lookup):
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
    extended = [h.lower() for h in SPECIAL_PATTERN.findall(text)]
    if extended:
        hits = [h for h in hits if h not in BASE_FORMULATIONS]
        for extended_hit in extended:
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