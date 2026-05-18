import re
import os
import pandas as pd

DISABILITIES_FR = {
    "handicaps moteurs":
    # https://accessibe.com/glossary/motor-impairment
    # https://www.nhs.uk/conditions/social-care-and-support-guide/care-services-equipment-and-care-homes/household-gadgets-and-equipment-to-make-life-easier/
    ["dystrophie musculaire", "dystrophie", "myasthénie grave", "parkinson", "maladie de parkinson",
    "sclérose en plaques", "sclérose", "sep", "paralysie cérébrale", "paralysie", "pc",
    "trouble développemental de la coordination", "trouble de la coordination", "trouble du développement", "tdc",
    "dyspraxie", "accident vasculaire cérébral", "arthrite", "ostéoporose", "tremblement", "rigidité", "spasticité",
    "lésion de la moelle épinière", "lésions de la moelle épinière", "moelle épinière", "blessure", "blessures",
    "perte de membre", "amputation", "blessures musculo-squelettiques", "musculo-squelettique",
    "kinésithérapie", "thérapie", "ergothérapie", "médicament", "traitement médical", "dopamine",
    "intervention chirurgicale", "chirurgie", "chirurgies",
    "handicap moteur", "déficience motrice", "déficience", "fonction musculaire", "trouble moteur",
    "mobilité", "adaptation", "adaptabilité",
    "canne", "déambulateur", "cadre de marche", "rollateurs", "fauteuil roulant",
    "scooter de mobilité", "scooter", "équipement pour personnes handicapées", "équipement"],

    "handicaps sensoriels":
    # https://kines.rutgers.edu/dshw/disabilities/sensory/1061-sensory-disabilities
    # https://maplecommunity.com.au/types-of-sensory-disabilities-and-impairment/
    # https://www.washington.edu/doit/working-together-computers-and-people-sensory-impairments
    ["trouble déficit de l'attention avec hyperactivité", "tdah", "trouble déficit de l'attention", "tda",
     "trouble du spectre autistique", "spectre autistique", "autisme", "autiste", "tsa",
    "cécité", "cécité partielle", "déficience visuelle", "aveugle", "basse vision", "handicap visuel",
    "vision", "acuité visuelle limitée", "daltonisme", "daltonien",
    "cataracte", "dégénérescence maculaire", "glaucome", "trachome",
    "surdité", "surdité partielle", "sourd", "perte auditive", "perte de l'ouïe",
    "perte auditive de transmission", "perte auditive neurosensorielle", "oreille interne", "nerf auditif",
    "perte auditive mixte", "trouble du spectre de la neuropathie auditive", "infection", "rubéole maternelle",
    "syphilis", "trouble du traitement sensoriel", "tts", "dysfonction de l'intégration sensorielle",
    "condition", "dépassé", "sur-stimulation", "surstimulation",
    "déficience tactile", "déficience gustative", "déficience olfactive",
    "régression développementale", "retard de langage",
    "aides techniques", "apprentissage des compétences tactiles", "entraînement sensoriel",
    "services de réadaptation", "modification de l'environnement",
    "aide à la mobilité", "chien guide", "appareil auditif", "implant cochléaire",
    "appareils prothétiques", "prothèse", "lecteurs d'écran", "appareils braille", "lunettes", "loupe d'écran"],

    "handicap psychique et handicap mental":
    # https://www.psychiatry.org/patients-families/eating-disorders/what-are-eating-disorders
    # https://www.who.int/news-room/fact-sheets/detail/mental-disorders
    # https://www.epa.gov/americaschildrenenvironment/health-neurodevelopmental-disorders
    # https://www.unafam.org/troubles-et-handicap-psy/troubles-psychiques/les-principaux-troubles
    ["trouble anxieux", "anxiété généralisée", "panique", "attaque de panique", "anxiété sociale",
    "anxiété de séparation", "dépression", "culpabilité excessive", "faible estime de soi", "désespoir",
    "pensée suicidaire", "suicidaire", "suicide", "trouble bipolaire", "symptômes maniaques",
    "euphorie", "irritabilité", "syndrome de stress post-traumatique",
    "sspt", "traumatisme", "schizophrénie", "délire", "hallucination",
    "trouble alimentaire", "anorexie", "boulimie", "hyperphagie boulimique", "hyperphagie",
    "trouble d'évitement ou de restriction de l'alimentation", "arfid", "pica",
    "comportement perturbateur", "trouble dissocial", "trouble neurodéveloppemental",
    "déficience intellectuelle", "déficit de l'attention", "hyperactivité", "troubles d'apprentissage",
    "retard mental", "trouble des conduites", "trouble obsessionnel compulsif", "toc",
    "trouble de la personnalité borderline",
    "psychoéducation", "traitement psychologique", "réadaptation psychosociale", "réadaptation"],

    "maladies chroniques évolutives":
    # https://medlineplus.gov/degenerativenervediseases.html
    # https://en.wikipedia.org/wiki/Degenerative_disease
    ["maladie d'alzheimer", "alzheimer", "sclérose latérale amyotrophique", "sla",
    "cancers", "cancer",
    "maladie de charcot-marie-tooth", "charcot-marie-tooth",
    "encéphalopathie traumatique chronique", "etc", "mucoviscidose",
    "déficiences en cytochrome c oxydase", "cytochrome c oxydase",
    "syndrome d'ehlers-danlos", "ehlers-danlos", "sed",
    "fibrodysplasie ossifiante progressive", "fibrodysplasie ossifiante",
    "ataxie de friedreich", "friedreich",
    "démence frontotemporale", "dft", "maladies cardiovasculaires", "cardiovasculaire",
    "maladie de huntington", "huntington",
    "dystrophie neuroaxonale infantile", "neuroaxonal infantile",
    "kératocône", "kératoglobe", "leucodystrophies", "leucodystrophie",
    "dmla", "syndrome de marfan", "marfan",
    "myopathies mitochondriales", "myopathie mitochondriale",
    "syndrome de déplétion de l'adn mitochondrial", "déplétion de l'adn mitochondrial",
    "syndrome de mueller-weiss", "mueller-weiss", "atrophie multisystématisée", "ams",
    "dystrophies musculaires", "dystrophie musculaire",
    "céroïde-lipofuscinose neuronale", "céroïde neuronale",
    "maladies de niemann-pick", "niemann-pick", "arthrose",
    "hypertension artérielle pulmonaire", "htap", "maladies à prions", "prion",
    "paralysie supranucléaire progressive", "psp", "rétinite pigmentaire",
    "polyarthrite rhumatoïde", "rhumatoïde", "maladie de sandhoff", "sandhoff",
    "amyotrophie spinale", "asm",
    "panencéphalite sclérosante subaiguë", "sclérosante subaiguë",
    "trouble lié à l'usage de substances", "usage de substances",
    "maladie de tay-sachs", "tay-sachs", "démence vasculaire", "vasculaire", "lewy"],

    "maladies génétiques":
    # https://my.clevelandclinic.org/health/diseases/21751-genetic-disorders
    # https://www.genome.gov/For-Patients-and-Families/Genetic-Disorders
    ["maladie génétique", "syndrome de down", "trisomie 21", "trisomie", "syndrome de l'x fragile", "x fragile",
    "klinefelter", "triple x", "triple-x", "turner", "trisomie 18", "trisomie 13",
    "trouble chromosomique", "maladie coronarienne", "diabète", "migraine", "maux de tête",
    "spina bifida", "malformation cardiaque congénitale", "duchenne",
    "hypercholestérolémie", "hémochromatose", "neurofibromatose",
    "drépanocytose", "amylose aa", "adrénoleucodystrophie", "ald",
    "mitochondrial", "usher", "cri du chat", "maladie de dercum", "syndrome de duane",
    "maladie de gaucher", "hémophilie", "phénylcétonurie", "anomalie de poland",
    "maladie de wilson", "vélocardiofacial"],

    "autres handicaps ou maladies":
    ["asthme", "épilepsie", "problème de mobilité", "problème neurologique",
    "paralysie", "faiblesse physique", "crise", "malformation"],

    "termes insultants":
    # https://www.gov.uk/government/publications/inclusive-communication/inclusive-language-words-to-use-and-avoid-when-writing-about-disability
    # https://disability.stanford.edu/sites/g/files/sbiybj26391/files/media/file/disability-language-guide-stanford_1.pdf
    ["handicapé", "handicapé mental", "déficient mental", "attardé", "subnormal",
    "infirme", "invalide", "spastique", "fou", "dingue", "nain", "nabot", "monstre",
    "psychopathe", "dérangé", "estropié"],
}

DISABILITIES_EN = {
    "motor disabilities":
    # https://accessibe.com/glossary/motor-impairment
    # https://www.nhs.uk/conditions/social-care-and-support-guide/care-services-equipment-and-care-homes/household-gadgets-and-equipment-to-make-life-easier/
    ["muscular distrophy", "distrophy", "myasthenia gravis", "parkingson", "parkingson's disease",
    "multiple sclerosis", "sclerosis", "ms", "cerebral palsy", "palsy", "cp",
    "developmental coordination disorder", "coordination disorder", "developmental disorder", "dcd",
    "dyspraxia", "attention deficit hyperactivity disorder", "adhd", "attention deficit disorder", "add",
    "stroke", "arthritis", "osteoporosis", "tremor", "rigisity", "spasticity",
    "spinal cord injury", "spinal cord injuries", "spinal cord", "injury", "injuries",
    "limb loss", "limb", "amputation", "musculoskeletal injuries", "musculoskeletal",
    "physical therapy", "therapy", "occupational therapy", "medication", "drug", "dopamine",
    "surgical intervention", "surgery", "surgeries",
    "motor disability", "motor impairment", "impairement", "muscule function", "motor disorder",
    "mobility", "adaptation", "adaptability", "disability",
    "walking stick", "walking frame", "zimmer frame", "rollators", "wheelchair",
    "mobility scooter", "scooter", "disability equipment", "equipment"],

    "sensory disabilities":
    # https://kines.rutgers.edu/dshw/disabilities/sensory/1061-sensory-disabilities
    # https://maplecommunity.com.au/types-of-sensory-disabilities-and-impairment/
    # https://www.washington.edu/doit/working-together-computers-and-people-sensory-impairments
    ["autism spectrum disorder", "austism spectrum", "austism", "autistic", "autist", "asd",
    "blindness", "partial blindness", "visual impairment", "blind", "low vision", "vison disability",
    "vision", "limited visual acuity", "color blindness", "color blind", "colorblind",
    "cataract", "macular degeneration", "glaucoma", "trachoma",
    "deafness", "partial deafness", "deaf", "hearning loss", "loss of hearing",
    "conductive hearinf loss", "sensorineural hearing loss", "inner hear", "hearing nerve",
    "mixed hearing loss", "auditory neuropathy spectrum disorder", "infection", "maternal rubella",
    "syphilis", "sensory processing disorder", "spd", "sensory integration dysfunction",
    "condition", "overwhelmed", "over stimulation", "overstimulation",
    "touch impairement", "taste impairement", "tactile impairement", "gustatory impairement",
    "smell impairement", "olfactory impairement", "developmental regression", "language delay",
    "assistive devices", "tactile skill learning", "sensory training",
    "rehabilitation services", "environment modification",
    "mobility help", "cane", "guide dog", "hearing aid", "cochlear implant",
    "prosthetic devices", "prosthetic", "screen readers", "braille devices", "glasses", "screen magnifier"],

    "mental disabilities":
    # https://www.psychiatry.org/patients-families/eating-disorders/what-are-eating-disorders
    # https://www.who.int/news-room/fact-sheets/detail/mental-disorders
    # https://www.epa.gov/americaschildrenenvironment/health-neurodevelopmental-disorders
    # https://www.unafam.org/troubles-et-handicap-psy/troubles-psychiques/les-principaux-troubles
    ["anxiety disorder", "generalized anxiety", "panic", "panic attack", "social anxiety",
    "seperation anxiety", "depression", "excessive huilt", "low self-worth", "hopelessness",
    "suicidal thought", "suicidal", "suicide", "bipolar disorder", "manic symptoms",
    "euphoria", "irritability", "post traumatic stress disorder", "post-traumatic stress disorder",
    "ptsd", "trauma", "schizophrenia", "delusion", "hallucination",
    "eating disorder", "anorexia", "bulimia", "binge-eating", "binge eating",
    "avoidant restrictive food intake disorder", "arfid", "pica",
    "disruptive behavious", "dissocial disorder", "neurodevelopmental disorder",
    "intellectual disability", "attention-deficit", "hypercativity", "learning disabilities",
    "mental retardation", "conduct disorder", "obessive compulsive disorder", "ocd",
    "beorderline personality disorder",
    "psychoeducation", "psychological treatment", "psychosocial rehabilitation", "rehabilitation"],

    "degenerative diseases":
    # https://medlineplus.gov/degenerativenervediseases.html
    # https://en.wikipedia.org/wiki/Degenerative_disease
    ["alzheimer's disease", "alzheimer", "amyotrophic lateral sclerosis", "als",
    "amytrophic lateral sclerosis", "cancers", "cancer",
    "charcot–marie–tooth disease", "charcot–marie–tooth",
    "chronic traumatic encephalopathy", "cte", "cystic fibrosis", "cf",
    "cytochrome c oxidase deficiencies", "cytochrome c oxidase",
    "ehlers–danlos syndrome", "ehlers–danlos", "eds",
    "fibrodysplasia ossificans progressiva", "fibrodysplasia ossificans",
    "friedreich's ataxia", "friedreich", "freidreich ataxia",
    "frontotemporal dementia", "ftd", "cardiovascular diseases", "cardiovascular",
    "huntington's disease", "huntington",
    "infantile neuroaxonal dystrophy", "infantile neuroaxonal",
    "keratoconus", "kc", "keratoglobus", "leukodystrophies", "leukodystrophy",
    "amd", "marfan's syndrome", "marfan",
    "mitochondrial myopathies", "bpd", "mitochondrial myopathy",
    "mitochondrial dna depletion syndrome", "mitochondrial dna depletion",
    "mueller–weiss syndrome", "mueller–weiss", "multiple system atrophy", "msa",
    "muscular dystrophies", "muscular dystrophy",
    "neuronal ceroid lipofuscinosis", "neuronal ceroid",
    "niemann–pick diseases", "niemann–pick", "osteoarthritis",
    "pulmonary arterial hypertension", "pah", "prion diseases", "prion",
    "progressive supranuclear palsy", "psp", "retinitis pigmentosa", "rp",
    "rheumatoid arthritis", "rheumatoid", "sandhoff disease", "sandhoff",
    "spinal muscular atrophy", "sma",
    "subacute sclerosing panencephalitis", "subacute sclerosing",
    "substance use disorder", "substance use",
    "tay–sachs disease", "tay–sachs", "vascular dementia", "vascular", "lewy"],

    "genetic disorders":
    # https://my.clevelandclinic.org/health/diseases/21751-genetic-disorders
    # https://www.genome.gov/For-Patients-and-Families/Genetic-Disorders
    ["genetic disorder", "down syndrome", "trisomy 21", "trisomy", "fragile x syndrome", "fragile x",
    "klinefelter", "triple x", "triple-x", "turner", "trisomy 18", "trisomy 13",
    "chromosomal disorder", "coronary artery disease", "diabetes", "migraine", "headaches",
    "spina bifida", "congenital heart defect", "duchenne",
    "hyercholosterolemia", "hemochromotosis", "neurofibromatosis",
    "sickle cell", "tay-sachs", "tay sachs", "aa amyloidosis", "adrenoleukodystrophy", "ald",
    "mitochondrilal", "usher", "cat's cry", "dercum disease", "duane syndrome",
    "gaucher disease", "hemophilia", "phenylketonuria", "pland anomaly",
    "wilson disease", "velocardiofacial"],

    "other disabilities or diseases":
    ["asthma", "epilepsy", "mobility problem", "neurological problem",
    "paralysis", "physical weakness", "seizure", "malformation"],

    "insulting terms":
    # https://www.gov.uk/government/publications/inclusive-communication/inclusive-language-words-to-use-and-avoid-when-writing-about-disability
    # https://disability.stanford.edu/sites/g/files/sbiybj26391/files/media/file/disability-language-guide-stanford_1.pdf
    ["disabled", "mentally handicapped", "mentally deffective", "retarded", "subnormal",
    "cripple", "ivalid", "spastic", "insane", "mad", "dwarf", "midget", "freak",
    "psycho", "deranged", "crip"],
}

PATTERN_FR = re.compile(r'(?i)(?<!\w)(?:' + '|'.join(rf"{re.escape(word)}s?" for words in DISABILITIES_FR.values() for word in words) + r')(?!\w)')
PATTERN_EN = re.compile(r'(?i)(?<!\w)(?:' + '|'.join(rf"{re.escape(word)}s?" for words in DISABILITIES_EN.values() for word in words) + r')(?!\w)')

REVERSE_LOOKUP_FR = {word: cat for cat, words in DISABILITIES_FR.items() for word in words}
REVERSE_LOOKUP_EN = {word: cat for cat, words in DISABILITIES_EN.items() for word in words}

def get_categories(hits, reverse_lookup):
    seen = []
    for hit in hits:
        cat = reverse_lookup.get(hit)
        if cat is None:
            # hit may be a pluralised form — strip trailing 's' and retry
            cat = reverse_lookup.get(hit.rstrip('s'))
        if cat and cat not in seen:
            seen.append(cat)
    return seen

def collect_hits(pattern, text):
    if pd.isna(text):
        return []
    hits = [hit.lower() for hit in pattern.findall(text)]
    return hits

def apply_disability_detection(folder):
    rows = []
    for file in (f for f in os.scandir(folder) if f.name.endswith('.txt')):
        with open(file.path, 'r', encoding='utf-8') as f:
            text = f.read()
            rows.append({"doc_id": file.name, "disabilities": collect_hits(PATTERN_FR, text)})

    df_dis = pd.DataFrame(rows)
    df_dis["categories"] = df_dis["disabilities"].apply(lambda hits: get_categories(hits, REVERSE_LOOKUP_FR))
    
    return df_dis