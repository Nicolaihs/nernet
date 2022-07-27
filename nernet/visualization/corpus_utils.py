import streamlit as st
from nltk import Text
from nltk.corpus import PlaintextCorpusReader


@st.cache(persist=True, allow_output_mutation=True)
def create_corpus(corpus_dir: str, filename:str='.*\.txt', encoding: str='utf8'):
    """Create nltk corpus from text files."""
    corpus = PlaintextCorpusReader(corpus_dir, filename, encoding=encoding)
    text = Text(corpus.words())
    return text


def query_corpus(corpus: Text, query: str, width: int=20) -> list:
    """Query corpus."""
    return corpus.concordance_list(query, width=width)