import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from sklearn import preprocessing

dir_path = os.path.dirname(os.path.realpath(__file__))

model_path = os.path.join(dir_path, '../../data/models/')

@st.cache(allow_output_mutation=True)
def load_model(filename):
    if filename[0] not in ('~', '/'):
        filepath = os.path.join(model_path, filename)
    else:
        filepath = filename
    if filename[-3:] == 'bin':
        model = KeyedVectors.load_word2vec_format(datapath(filepath), binary=True)
    else:
        full_model = KeyedVectors.load(filepath)
        model = full_model.wv
        del full_model
    return model

def lookup_sims(word, min_similarity=0.6):
    """"""
    if '+' in word:
        words = [model_wv[single.strip()] for single in word.split('+')]
        word = sum(words)
    else:
        word = word.strip()
    sims = model_wv.most_similar(word, topn=300)
    sims = [(sim, score) for sim, score in sims if score >= min_similarity]
    return sims


def distance(word, target):
    if '+' in target:
        targets = [model_wv[single.strip()] for single in target.split('+')]
        target = sum(targets)

        most_similar = model_wv.most_similar(positive=[target,], topn=1)
        dist = model_wv.distance(word, most_similar[0][0])
    else:
        dist = model_wv.distance(word, target)
    return dist


def calculate_schmidt_score(sims, x_left, x_right, y_down, y_up):
    x_vector = model_wv[x_left] - model_wv[x_right]
    y_vector = model_wv[y_down] - model_wv[y_up]

    words = [sim for sim, _score in sims]
    distance_to_x_vec = model_wv.distances(x_vector, words)
    distance_to_y_vec = model_wv.distances(y_vector, words)

    df_schmidt = pd.DataFrame(words, columns=['word'])
    df_schmidt['x'] = distance_to_x_vec
    df_schmidt['y'] = distance_to_y_vec

    return df_schmidt


def calculate_distance_scores(sims, x_left, x_right, y_down, y_up):

    df_sims = pd.DataFrame(sims, columns=['word', 'score'])

    x_values, y_values = [], []
    for sim, _score in sims:
        left_val = distance(sim, x_left)
        right_val = model_wv.distance(sim, x_right)
        x_values.append((left_val, right_val, left_val-right_val))

        down_val = model_wv.distance(sim, y_down)
        up_val = model_wv.distance(sim, y_up)
        y_values.append((down_val, up_val, down_val-up_val))

    x_array = np.array([x[2] for x in x_values])
    y_array = np.array([y[2] for y in y_values])
    norm_x = preprocessing.normalize([x_array])
    norm_y = preprocessing.normalize([y_array])

    df_sims['x'] = norm_x[0]
    df_sims['y'] = norm_y[0]
    return df_sims


st.title('MeMo in Space!')
st.subheader('Based on work by Peter Leonard, based on work by Ben Schmidt')

with st.sidebar:
    model_files = [{'name': 'Memo, 200 features, using R (Peter Leonard)',
      'filename': 'memo_vectors.bin'},
      {'name': 'Memo, 500 features, using gensim',
      'filename': 'memo_m5_f500_epoch10_w10.model.w2v.bin'},
      {'name': 'Memo, 500 features, with bigrams, using gensim',
      'filename': 'memo_m3_f500_epoch10_w10_bigram.model.w2v.bin'},
      {'name': 'Modern Danish, 500 features, using gensim (DSL)',
      'filename': '/Users/nhs/Arkiv/korpus/models/dsl_500.model'}
      ]
    select_model = st.selectbox('Select word embeddings', [row['name'] for row in model_files])
    if select_model:
        selected = [row['filename'] for row in model_files if row['name'] == select_model][0]
        model_wv = load_model(selected)

    min_sim = st.slider('Min. similarity', 0.1, 0.9, 0.6)
    input_seed = st.text_input('Seed word').strip()
    tops = st.columns(3)
    middle = st.columns(3)
    bottom = st.columns(3)
    x_left = middle[0].text_input('X axis, left', 'ung').strip()
    x_right = middle[2].text_input('X axis, right', 'gammel').strip()
    y_up = tops[1].text_input('Y axis, up', 'kvinde').strip()
    y_down = bottom[1].text_input('Y axis, down', 'mand').strip()

    include_axis_words = st.checkbox('Include the words from the axis', False)

if input_seed and x_left and x_right and y_up and y_down:
    input_seed = input_seed.strip()
    sims = []
    for seed in input_seed.split(','):
        for self_seed in seed.split('+'):
            if self_seed not in [sim for sim, _score in sims]:
                sims.append((self_seed, 1.0))
        sims += lookup_sims(seed, min_similarity=min_sim)

    if include_axis_words:
        axis_words = x_left.split('+') + x_right.split('+') + y_down.split('+') + y_up.split('+')
        for word in axis_words:
            if word not in [sim for sim, _score in sims]:
                sims.append((word, 0.0))

    if st.checkbox('show words'):
        st.dataframe(pd.DataFrame(sims, columns=['Word', 'Similarity']))

    measure = st.selectbox('Use formula', ['Distance score', 'Ben Schmidt\'s original score'])
    if measure == 'Distance score':
        df_graph = calculate_distance_scores(sims, x_left, x_right, y_down, y_up)
    else:
        df_graph = calculate_schmidt_score(sims, x_left, x_right, y_down, y_up)

    fig = px.scatter(df_graph, x="x", y="y", text="word",
                     labels={'x': f'{x_left} <-----> {x_right}',
                             'y': f'{y_down} <--------> {y_up}'}) #, log_x=True, size_max=60)
    fig.update_layout(
        title=f'Show 2-dim vector space for words similar to "{input_seed}"'
    )
    st.plotly_chart(fig)

else:
    st.image('memo_space1.png')
    st.markdown('''
    # Instructions

    1. Choose some words to visualize by inserting a seed word
        1. Add more seed words delimited by a comma (,), e.g. "mand,kvinde"
        1. Fine-tune the point in vector space by combining terms with a plus sign (+), e.g. "mand+dreng,kvinde+pige"
    1. Inspect the words by clicking Show words. The score shows the semantic distance to the seed word or words
       1. If necessary (ie. too few or too many words), adjust the slider to allow for more or less semantic distance
    1. Choose axis on the graph, e.g. mand <-> kvinde
    1. Watch the axis -- a word on 0 on an axix is equally distant from the two axis words
       1. **Note:** 0 might not be centered on the diagram
       2. Select "Include the words from the axis" in order to max out the graph

    # Problems:

    **Note**: This a prototype only. Beware of:

    1. It is only possible to use single words on the axis

    # Literature

    * Original blog post by Ben Schmidt: [Word Embeddings for the digital humanities](http://bookworm.benschmidt.org/posts/2015-10-25-Word-Embeddings.html)
       ''')

