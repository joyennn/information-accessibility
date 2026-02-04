import csv
import stanza
from tqdm import tqdm


stanza.download('en')
nlp = stanza.Pipeline('en')
doc = nlp(text)

# CoNLL-U form setting
for sentence in doc.sentences:
    print("# text =", sentence.text)
    for word in sentence.words:
        print(f"{word.id}\t{word.text}\t{word.lemma}\t{word.upos}\t{word.xpos}\t{word.feats if word.feats else '_'}\t{word.head}\t{word.deprel}\t_\t_")
    print()


### Target Sentences Extraction ###

with open("bnc_filtered.txt", "r", encoding="utf-8") as f:
    data = [line.strip() for line in f if line.strip()]

#preposing
def is_preposing(sent):
    for word in sent.words:
        if word.deprel == "root" and word.upos in {"NOUN", "PROPN", "PRON"}:
            root_noun = word.text
            root_noun_id = word.id
            for w in sent.words:
                if w.deprel == "acl:relcl" and w.upos == "VERB" and w.head == root_noun_id:
                    return True, root_noun
    return False, None

#passive
def is_passive(sent):
    root_verb_id = None
    has_be_auxpass = False
    by_agent_ok = False
    passive_subject = None

    for word in sent.words:
        if word.deprel == "root" and word.upos == "VERB":
            root_verb_id = word.id
            break
    if root_verb_id is None:
        return False, None

    for word in sent.words:
        if word.lemma == "be" and word.deprel == "aux:pass" and word.head == root_verb_id:
            has_be_auxpass = True
            break
    if not has_be_auxpass:
        return False, None

    for word in sent.words:
        if word.text.lower() == "by":
            head_id = word.head
            for candidate in sent.words:
                if candidate.id == head_id and candidate.upos in {"NOUN", "PROPN", "PRON"}:
                    by_agent_ok = True
                    break

    for word in sent.words:
        if word.deprel == "nsubj:pass" and word.upos in {"NOUN", "PROPN", "PRON"}:
            passive_subject = word.text

    return has_be_auxpass and by_agent_ok and passive_subject is not None, passive_subject

#inversion1
def is_inversion_type1(sent):
    first = sent.words[0]
    if first.deprel not in {"advmod", "case", "csubj"}:
        return False, None

    root_idx = -1
    for i, word in enumerate(sent.words):
        if word.deprel == "root":
            root_idx = i
            break

    if root_idx == -1:
        return False, None

    for i in range(root_idx):
        if sent.words[i].text == ",":
            return False, None

    root = sent.words[root_idx]
    if root.upos not in {"NOUN", "PROPN", "PRON"}:
        return False, None

    has_cop = False
    candidates = []

    for word in sent.words:
        if word.deprel == "cop" and word.head == root.id:
            has_cop = True
        if word.upos in {"NOUN", "PROPN", "PRON"}:
            if word.head == root.id:
                candidates.append(word.text)
            else:
                for mid in sent.words:
                    if mid.id == word.head and mid.deprel in {"advmod", "csubj"} and mid.head == root.id:
                        candidates.append(word.text)
                        break

    return has_cop and len(candidates) > 0, ", ".join(candidates)


#inversion2
def is_inversion_type2(sent):
    first = sent.words[0]
    if first.deprel not in {"advmod", "case", "csubj"}:
        return False, None

    root_idx = -1
    for i, word in enumerate(sent.words):
        if word.deprel == "root":
            root_idx = i
            break

    if root_idx == -1:
        return False, None

    for i in range(root_idx):
        if sent.words[i].text == ",":
            return False, None

    root = sent.words[root_idx]
    if root.upos != "VERB":
        return False, None

    has_obj_or_obl = False
    candidates = []

    for word in sent.words:
        if word.deprel in {"obj", "obl"} and word.upos in {"NOUN", "PROPN", "PRON"}:
            has_obj_or_obl = True

        if word.upos in {"NOUN", "PROPN", "PRON"}:
            if word.head == root.id:
                candidates.append(word.text)
            else:
                for mid in sent.words:
                    if mid.id == word.head and mid.deprel in {"advmod", "csubj"} and mid.head == root.id:
                        candidates.append(word.text)
                        break

    return has_obj_or_obl and len(candidates) > 0, ", ".join(candidates)



def is_inversion(sent):
    match1, noun1 = is_inversion_type1(sent)
    match2, noun2 = is_inversion_type2(sent)

    if match1:
        return True, noun1
    elif match2:
        return True, noun2
    return False, None


#stanza initial setting
nlp = stanza.Pipeline(
    'en',
    use_gpu=True,
    processors='tokenize,mwt,pos,lemma,depparse',
    tokenize_batch_size=64,
    pos_batch_size=64,
    depparse_batch_size=64
)

df = data[600001:] #index setting

preposing = []
passive = []
inversion = []

import pickle
from tqdm import tqdm

start_index = 0

for i in tqdm(range(start_index, len(df)), desc="Processing"):
    text = df[i]

    try:
        doc = nlp(text)

        for sent in doc.sentences:
            match, noun = is_preposing(sent)
            if match:
                preposing.append([text, noun])

            match, noun = is_inversion(sent)
            if match:
                inversion.append([text, noun])

            if "by" in text.lower():
                match, noun = is_passive(sent)
                if match:
                    passive.append([text, noun])

    except Exception as e:
        print(f"{i}번째 문장에서 오류 발생: {e}")
        continue

    if i % 100 == 0:
        with open(f"{save_dir}/preposing.pkl", "wb") as f:
            pickle.dump(preposing, f)
        with open(f"{save_dir}/inversion.pkl", "wb") as f:
            pickle.dump(inversion, f)
        with open(f"{save_dir}/passive.pkl", "wb") as f:
            pickle.dump(passive, f)
        with open(f"{save_dir}/checkpoint.pkl", "wb") as f:
            pickle.dump(i + 1, f)


# Preposing - Save
with open("preposing.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(preposing)

# Passive - Save
with open("passive.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(passive)

# Inversion - Save
with open("inversion.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(inversion)

