import argparse
import sys


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="BERT model. Options: bert-base-uncased/bert-large-uncased, default is 'bert-base-uncased'",
    )
    parser.add_argument(
        "--type_phrases",
        default="",
        type=str,
        help="SP-10K syntactic relation types. Options: nsubj/nsubj_amod/amod/dobj/dobj_amod",
    )
    parser.add_argument(
        "--type_attention",
        default="standard",
        type=str,
        help="standard, head, context, or control",
    )
    parser.add_argument(
        "--corpus_dir",
        default="",
        type=str,
        help="The directory of the corpus to extract sents from.",
    )
    parser.add_argument(
        "--sp_dir", default="", type=str, help="The directory where SP-10K is saved."
    )
    parser.add_argument(
        "--sent_file", default="", type=str, help="The file with the preprocessed sentences."
    )

    args = parser.parse_args()

    return args
