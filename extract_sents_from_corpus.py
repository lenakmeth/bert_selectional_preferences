import re
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import os
from utils import bert_tokenize, make_segments_ids, read_sents, read_phrases
import io
import json
from configure import parse_args

args = parse_args()

# open files
UK_DIR = args.corpus_dir
SP_dir = args.sp_dir

uk_files = [i for i in os.listdir(UK_DIR) if i.endswith("final")]
sp_files = [i for i in os.listdir(SP_dir) if not "readme" in i]

# Open the phrases

phrases = read_phrases(SP_dir)
type_phrases = args.type_phrases

# Make tsv to save sentences
new_file = open(
    "sentences/" + type_phrases + "/" + parsed_file[6:-6] + "_sents.tsv",
    "w+",
    encoding="utf-8",
)
new_file.write("Prompt file\tPrompt\tUnmasked\tIndex\n")


for n, parsed_file in enumerate(uk_files):
    print(n + " out of " + len(uk_files))

    sentences = read_sents(os.path.join(UK_DIR, parsed_file))

    for sent in sentences:

        labels = [word["label"] for word_id, word in sent[0].items()]
        words = [word["word"] for word_id, word in sent[0].items()]

        # not subordinate and relative clauses
        # not questions
        # length of sentence 4-15 tokens
        if (
            "?" not in words
            and 4 < len(words) < 15
            and "sconj" not in labels
            and "acl" not in labels
            and "acl:rel" not in labels
        ):

            words_lemmatized = {
                w["lemma"]: k for k, w in sent.items()
            }  # we need lemmas to match with SP-10K

            for phrase in phrases[type_phrases]:

                word_A = phrase[0].lower()  # first word of phrase (head)
                word_B = phrase[1][0].lower()  # second word of phrase (complement)

                # first objective, both words in sentence (lemmatized)
                if word_A in words_lemmatized and word_B in words_lemmatized:
                    # find position of word
                    word_A_position = words_lemmatized[
                        word_A
                    ]  # position in parsed from 1
                    word_B_position = words_lemmatized[
                        word_B
                    ]  # position in parsed from 1

                    # distance of word A and word B no more than 5 words
                    if abs(word_B_position - word_A_position) < 5:

                        # if label of word_B matches the syntactic relation we look for
                        if sent[word_B_position]["label"] == phrase[1][1]:
                            word = sent[word_B_position]

                            # word B must be dependent to the word A
                            if word["headID"] == word_A_position:

                                # if word B is the root, we will get an exception
                                try:  # word_B is dependent
                                    word_B_head = sent[
                                        word["headID"]
                                    ]  # find head of word_B

                                    # if word_B is dependent to a word_A which is not root, we need to check the label of word_A too
                                    # this happens to amod phrases, head
                                    if phrase[1][2]:  # this means that we look for a specific syntactic relation in
                                        if (word_B_head["label"] == phrase[1][2]
                                            and word_B_head["POS"] == "N"):
                                            unmasked_sent, masked_sent, idx, continue_flag = bert_tokenize(
                                                                                                    [w["word"] for w in sent.values()],
                                                                                                    word_B,
                                                                                                    word_B_position - 1
                                                                                                          )
                                            if continue_flag: 
                                                new_file.write("\t".join([
                                                                            type_phrases,
                                                                            word_A,
                                                                            word_B,
                                                                            " ".join(unmasked_sent),
                                                                            " ".join(masked_sent),
                                                                            str(idx),
                                                                         ]))
                                                new_file.write("\n")
                                                

                                    # if word_B is dependent to a root, it has to be a root and a VERB
                                    else:
                                        if word_B_head["POS"] == "V":
                                            unmasked_sent, masked_sent, idx, continue_flag = bert_tokenize(
                                                                                                [w["word"] for w in sent.values()],
                                                                                                word_B,
                                                                                                word_B_position - 1
                                                                                                          )
                                            if continue_flag: 
                                                new_file.write("\t".join([
                                                                            type_phrases,
                                                                            word_A,
                                                                            word_B,
                                                                            " ".join(unmasked_sent),
                                                                            " ".join(masked_sent),
                                                                            str(idx),
                                                                         ]))
                                                new_file.write("\n")

                                except KeyError:  # word_B is the root (0), phrase not useful
                                    pass
                                except TypeError:  # word_B is the root (0), phrase not useful
                                    pass

    new_file.close()
