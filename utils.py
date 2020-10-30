import re
import os
from transformers import BertTokenizer, BertModel, BertForMaskedLM

# Load pre-trained model tokenizer (vocabulary)
modelpath = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(modelpath)


def make_segments_ids(tokenized_text):
    segment_idx = 0
    segments_ids = []
    for w in tokenized_text:
        if w == "[SEP]":
            segment_idx += 1
        segments_ids.append(segment_idx)
    assert len(tokenized_text) == len(segments_ids)

    return segments_ids


def bert_tokenize(text, masked_word, masked_idx):
    #punct = [".", ",", ":", ";", "?", "!", "–", "(", ")", "—"]
    text = (" ".join([w for w in text if w.isalpha() or w in punct])).replace(
        "...", "."
    )

    bert_tokenized_sent = tokenizer.tokenize(text)

    # add the [SEP] tag
    tokenized_text = ["[CLS]"]
    for w in bert_tokenized_sent:
    #    if w in punct:
        tokenized_text.append(w)
#            tokenized_text.append("[SEP]")  # first punct, then [SEP]
 #       else:
  #          tokenized_text.append(w)

    if masked_word in tokenized_text:
        masked_sent = tokenized_text
        try:
            # look in window of 20 words AFTER masked_idx (cause BERT tokenizer makes sents longer, not shorter)
            new_masked_index = masked_idx + masked_sent[
                masked_idx : masked_idx + 20
            ].index(masked_word)
            masked_sent[new_masked_index] = "[MASK]"
        except ValueError:
            new_masked_index = masked_sent.index(masked_word)
            masked_sent[new_masked_index] = "[MASK]"

        return tokenized_text, masked_sent, new_masked_index, True

    else:  # then the word is NOT in BERT's vocab intact, e.g. allegation = 'all', '##ega', '##tion'

        return text, bert_tokenized_sent, masked_idx, False


def read_sents(file):
    """ Opens the file with the parser output."""

    # read and format sentences
    sents = []
    lines = []

    marker = False
    with open(file, "r") as f:
        for line in f:
            line = re.sub("\s{2,}", "\t", line)
            lines.append(line)

    for line in lines:
        if line.startswith("#"):
            pass
        elif line.split("\t")[0].isdigit():
            marker = True
            if line.split("\t")[0] == "1":  # first word of new sentence
                sents.append({})
            if marker:
                sents[-1].update(format_line(line))
        else:
            marker = False

    return sents


def format_line(raw_line):
    """ Formats word features into a list of features. Format:
        {tokenID: { word:X, POS:Y, label:x, headID:0}, ...} """

    raw_features = raw_line.strip().split("\t")
    features = {
        "word": None,
        "lemma": None,
        "POS": None,
        "TIGER": None,
        "headID": None,
        "label": None,
    }

    tokenID = raw_features[0]
    features["word"] = raw_features[1].replace("_", "").lower()
    features["lemma"] = raw_features[2].replace("_", "").lower()
    features["POS"] = raw_features[3].replace("_", "")
    features["TIGER"] = raw_features[4].replace("_", "")
    if raw_features[6].isdigit():
        features["headID"] = int(raw_features[6])
    else:
        try:
            features["headID"] = int(re.search("\d+", raw_features[6]).group(0).strip())
            features["label"] = (
                re.search("\D+", raw_features[6].replace("_", "")).group(0).strip()
            )
        except AttributeError:
            print(raw_features)
    if not features["label"]:
        features["label"] = raw_features[7]

    output = {int(tokenID): features}

    return output


def read_phrases(SP_dir):

    files = [i for i in os.listdir(SP_dir) if i.endswith("txt")]
    phrases = {}

    for file in files:
        with open(os.path.join(SP_dir, file), "r", encoding="utf-8") as f:
            phrases[file[:-15].replace("wino_", "")] = []
            types = file[:-15].replace("wino_", "").split("_")
            if len(types) == 1 and not "amod" in types:
                label_A = None
                label_B = types[0]
            elif len(types) == 1 and "amod" in types:
                label_A = None
                label_B = types[0]
            else:
                label_A = types[0]
                label_B = types[1]

            for line in f:
                temp = []
                temp.append(line.split("\t")[0])
                temp.append((line.split("\t")[1], label_B, label_A))
                temp.append(float(line.split("\t")[2].strip()))
                #            temp.append(types)
                phrases[file[:-15].replace("wino_", "")].append(temp)

    return phrases
