"""Gender detection system for THIRD person singular in French. Based on morpho-syntactic gender markers
and leveraging semantic information. Adapted from gender_detection_fr.py that was for first person singular."""

import os
import json
import pandas as pd
from collections import Counter
import spacy


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_GENDER_FOLDER = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..")
)
BIOGRAPHY_DIR = os.path.join(
    SCRIPT_DIR,
    "biographies_gender_benchmark"
)


nlp = spacy.load("fr_dep_news_trf")

def resolve_gender(counter):
    """Resolve a gender Counter into a single label."""
    if len(counter) > 0:
        res = counter.most_common(1)[0][0]
    else:
        res = "Neutre"
    counter_val = counter.values()
    if len(counter_val) > 1 and len(set(counter_val)) == 1:
        res = "Ambigu"
    return res


def refers_to_excluded_agent(token, excluded):
    """Return True if any ancestor of `token` in the dependency tree is one of the
    excluded relation nouns (e.g. 'mari', 'fille'). Used to filter out tokens that
    describe someone OTHER than the patient, e.g. 'poète' in 'son futur mari, un poète'
    where 'poète' is an apposition of 'mari', or 'boulanger' in "Fille d'un boulanger"
    where 'boulanger' is an nmod of 'Fille'."""
    return any(ancestor.text.lower() in excluded for ancestor in token.ancestors)


def modifies_unmarked_epicene(token, effective_head, epicene):
    """Return True if `token` is an ADJ modifying an épicène profession noun
    (e.g. 'soprano', 'juge') that has no gender-determining article. In such
    cases the ADJ's grammatical gender reflects the noun's conventional default,
    not the referent's actual gender (e.g. 'premier soprano' is masc even for a
    female singer). Mirrors the existing logic used for the épicène noun itself."""
    if token.pos_ != "ADJ":
        return False
    if effective_head.text.lower() not in epicene:
        return False
    has_gendered_det = any(
        c.dep_ == "det" and c.text.lower() in ("un", "le", "une", "la")
        for c in effective_head.children
    )
    return not has_gendered_det


# Relation nouns REMOVED from agents_hum entirely: they almost never describe the
# patient herself in clinical/biographical text — they describe a relative or
# close contact, so we don't want them to fire as agent markers on their own.
EXCLUDED_FROM_AGENTS = {
    "personne", "mari", "épouse", "époux", "monsieur", "madame"
}

# Superset used ONLY for filtering DESCENDANTS in the dependency tree. We keep
# 'fille' and 'fils' in agents_hum because patient-introduction patterns like
# "Fille d'un boulanger, elle..." legitimately mark the patient — but anything
# hanging UNDER such a fille/fils node ('boulanger' here) is about the relative,
# so we use this wider list when walking ancestors.
RELATIVE_NOUNS = EXCLUDED_FROM_AGENTS | {"fille", "fils"}


def get_gender(text, details=False):
    """Apply linguistic rules based on Spacy tags to detect the THIRD person singular gender
    markers in a text.

    Args:
        text (str): The text to be analyzed (= for which we want to find the author's gender).
        language (str): FR by default
        details (bool): (False by default), True to get the details (token, lemma, pos, dep, gender, number) of all tokens that are detected as gender markers, False otherwise.

    Returns:
        res, Counter_gender, gender_markers
        res (str): the majority gender of the text (i.e. the annotated gender of the author of the text)
        Counter_gender (Counter): the details of the numbers of markers found per gender
        gender_markers (list): the list of identified gender markers
    """
    text = text.replace("  ", " ")
    doc = nlp(text)

    #list of gender-neutral (épicène) job titles from DELA, with Profession:fs:ms, to check and filter out if they're identified as Masc when used without a masc DET
    with open(f"{RESOURCES_GENDER_FOLDER}/epicene_fr.json", encoding="utf-8") as f: #MODIFIED: path to file
        epicene_jobs = json.load(f)

    with open(f"{RESOURCES_GENDER_FOLDER}/lexical_res_P3_fr.json", encoding="utf-8") as f: #MODIFIED: path to file
        agents_hum = json.load(f)

    # Removing nouns referring to relatives or close contacts that often appear (in
    # clinical/biographical cases) but to mention another person, not the patient
    agents_hum = [el for el in agents_hum if el not in EXCLUDED_FROM_AGENTS]

    # list of identified gender tags in the adj/verbs of the text
    gender = []
    # list of the tokens that have a gender tag (and are adj/verbs)
    gender_markers = []
    # to take into account the presence of a person's name/initial of a name as the first word of the text
    prenom_initiale = []
    # Often, the group "laissant derrière elle/lui" ("leaving behind her/him") is present and indicates the gender of the patient, so we manually add it to the list of gender markers
    for sent in doc.sents:
        if "laissant derrière elle" in sent.text:
            gender.append("Fem")
            gender_markers.append("laissant derrière elle")
        elif "laisse derrière elle" in sent.text:
            gender.append("Fem")
            gender_markers.append("laisse derrière elle")
        elif "laissant derrière lui" in sent.text:
            gender.append("Masc")
            gender_markers.append("laissant derrière lui")
        elif "laisse derrière lui" in sent.text:
            gender.append("Masc")
            gender_markers.append("laisse derrière lui")

        this_sent = []
        for token in sent:
            this_sent.append(token.text.lower() + "-" + token.dep_)
            # Note: We can't have a rule to look for third-person singular pronouns as they can refer to non-human entities and we need coreference systems to know (vs 1-person singular that always refers to a human entity)

            # 2a. The token is a noun referring to a human agent AND is not an "agentive oblique" (e.g. in a prepositional group, see https://universaldependencies.org/fr/dep/obl-agent.html)
            cond_agt = token.text.lower() in agents_hum and token.pos_=="NOUN"  #and "obl" not in token.dep_ #"obl:" in token.dep_
            if len(this_sent) == 1 and ((token.pos_ == "PROPN" and "nsubj" in token.dep_) or (token.text.isupper() and len(token)==2 or len(token)==4 and "." in token.text)):
                prenom_initiale.append(token.text)
            

            # Check that the sentence contains a marker of P3 agent
            cond_agt_avt = [s for s in this_sent if "nsubj" in s] and [s for s in this_sent if "nsubj" in s][-1].split("-")[0] in agents_hum

            # 2b. The token is an adjective or past participle that refers to a agent noun (epithet),
            # and it does not have the auxiliary "avoir" (= it refers to a state and not an action)
            cond_pos = (token.pos_ == "ADJ" or token.pos_ == "VERB")
            cond_noavoir = (("a-aux:tense" not in this_sent and "avoir-aux:tense" not in this_sent) or ("a-aux:tense" in this_sent and "été-aux:pass" in this_sent))
            # For coordinated adjectives ("intellectuel et culturel"), token.head is the
            # first adjective, not the noun being modified. Walk up through ADJ ancestors
            # to find the actual head noun (e.g. 'héritage' in 'un héritage intellectuel et culturel').
            effective_head = token.head
            while effective_head.pos_ == "ADJ" and effective_head.head != effective_head:
                effective_head = effective_head.head
            cond_adj_pp = cond_pos and (
                    ((effective_head.text.lower() in agents_hum or (
                                prenom_initiale and effective_head.text == prenom_initiale[0])) and cond_noavoir) or (
                            effective_head.pos_ != "NOUN" and cond_noavoir and cond_agt_avt))
            # Manually fix Spacy mistakes (mislabeling some Feminine words as Masculine ones)
            erreurs_genre = ["inscrite", "technicienne"]

            # 2c. The token is a root with a proper noun as subject
            cond_propn_subj = cond_pos and cond_noavoir and (token.head == token or token.head.dep_ == "ROOT") and any(
                    t.pos_ == "PROPN" and "nsubj" in t.dep_ for t in token.head.children) and not any(
                    t.pos_ == "NOUN" and t.text.lower() not in agents_hum and "attr" in t.dep_ for t in token.head.children)      

            # The candidate must be a noun referring to an agent OR an adj/past participle,
            # AND it must be singular (plurals like 'reconnus' refer to several people, not the subject),
            # AND it must not be attached (directly or transitively) to a relative noun
            # (e.g. 'boulanger' in "Fille d'un boulanger" hangs under 'Fille' as nmod;
            #  'poète' in 'son futur mari, un poète' hangs under 'mari'),
            # AND if it's an ADJ, it must not modify an épicène profession noun without
            # a gendered determiner (e.g. 'premier' in 'premier soprano' doesn't tell us
            # about the referent's gender, only the noun's conventional grammatical default)
            if (cond_agt or cond_adj_pp or cond_propn_subj) \
                    and "Plur" not in token.morph.get("Number") \
                    and not refers_to_excluded_agent(token, RELATIVE_NOUNS) \
                    and not modifies_unmarked_epicene(token, effective_head, epicene_jobs):
                token_gender = token.morph.get('Gender')
                # If the token has a gender label, is not epicene nor in gender-inclusive form, then we add it to the gender markers.
                if token_gender and token.text.lower() not in epicene_jobs and "(" not in token.text.lower() and token.text.lower() not in erreurs_genre: #(e
                    gender.append(token_gender[0])
                    gender_markers.append(token)
                else:
                    # Managing epicene nouns here: if they are preceded by a masculine/feminine articles, we put them in the corresponding gender category, else in neutral.
                    if (token.text.lower() in epicene_jobs and len(this_sent)>1 and this_sent[-2] in ["un-det", "le-det"]) or token.text.lower()=="chef" and "chef" not in [str(tok) for tok in gender_markers]:
                        gender.append("Masc")
                        gender_markers.append(token)
                    if (token.text.lower() in epicene_jobs and len(this_sent)>1 and this_sent[-2] in ["une-det", "la-det"]) or token.text.lower() in erreurs_genre:  # or token.text=="Femme":
                        gender.append("Fem")
                        gender_markers.append(token)
                    if "(" in token.text.lower():
                        gender.append("Neutre")
                        gender_markers.append(token)

            if details:
                print(token.text.lower(), token.pos_, token.dep_, token.lemma_, token.morph.get("Gender"), token.morph.get("Number"))

    pronoun_counts = {"il": 0, "elle": 0}
    for token in doc:
        if token.pos_ == "PRON" and "nsubj" in token.dep_ and token.lower_ in pronoun_counts:
            pronoun_counts[token.lower_] += 1

    if pronoun_counts["il"] > pronoun_counts["elle"]:
        gender.extend(["Masc", "Masc"])
        gender_markers.append("il (pronom majoritaire)")
    elif pronoun_counts["elle"] > pronoun_counts["il"]:
        gender.extend(["Fem", "Fem"])
        gender_markers.append("elle (pronom majoritaire)")

    Counter_gender = Counter(gender)
    res = resolve_gender(Counter_gender)

    if res in ("Neutre", "Ambigu"):
        if pronoun_counts["il"] > pronoun_counts["elle"]:
            res = "Masc"
        elif pronoun_counts["elle"] > pronoun_counts["il"]:
            res = "Fem"

    return res, Counter_gender, gender_markers


def apply_gender_detection(txts_path=BIOGRAPHY_DIR):
    """Apply gender detection system (from function get_gender and get_gender_from_names) on the generations contained in a CSV file and append
        the results (manual annotations) in a new CSV file.

        Args:
            txts_path: A string -> the path of the texts files containing the generated biographies.

        Returns:
            df_out (DataFrame): One row per doc_id with columns:
                - genre_auto       : detected gender ('Masc', 'Fem', 'Neutre', 'Ambigu')
                - Detailed_counter : Counter with nb of markers per gender
                - Detailed_markers : list of identified gender markers
    """
    total_name_files = []
    total_texts = []
    total_gender   = []
    total_counter  = []
    total_markers  = []


    for file in (f for f in os.scandir(txts_path) if f.name.endswith('.txt')):
        with open(file.path, 'r', encoding='utf-8') as f:
            text = f.read()

            res, counter, markers = get_gender(text)

            total_name_files.append(file.name)
            total_texts.append(text)
            total_gender.append(res)
            total_counter.append(counter)
            total_markers.append(markers)


    df_out = pd.DataFrame({
        "doc_id":        total_name_files,
        "bio":              total_texts,
        "genre_auto":       total_gender,
        "Detailed_counter": total_counter,
        "Detailed_markers": total_markers,
    })

    df_out.to_csv(f"{SCRIPT_DIR}/gender_detection_benchmark.csv")

    return df_out



if __name__ == "__main__":
    df = apply_gender_detection(BIOGRAPHY_DIR)
    print(f"{len(df)} documents processed")
    print(df[["doc_id", "genre_auto"]].head())