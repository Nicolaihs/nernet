import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from sklearn import preprocessing


model_path = '/Users/nhs/Udvikling/dighum/nernet/data/models/'

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
        dist = model_wv.distance(sim, target)
    return dist


st.title('MeMo in Space')
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
    input_seed = st.text_input('Seed word', 'farve')
    tops = st.columns(3)
    middle = st.columns(3)
    bottom = st.columns(3)
    x_left = middle[0].text_input('X axis, left', 'ung')
    x_right = middle[2].text_input('X axis, right', 'gammel')
    y_up = tops[1].text_input('Y axis, up', 'kvinde')
    y_down = bottom[1].text_input('Y axis, down', 'mand')

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
    if st.checkbox('show words'):
        st.dataframe(df_sims)

    fig = px.scatter(df_sims, x="x", y="y", text="word",
                     labels={'x': f'{x_left} <-----> {x_right}',
                             'y': f'{y_down} <--------> {y_up}'}) #, log_x=True, size_max=60)
    fig.update_layout(
        title=f'Analyzing {input_seed}'
    )
    st.plotly_chart(fig)

#    import ipdb; ipdb.set_trace()
#    x_axis = model_wv[x_right] - model_wv[x_left]

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

    # Problems:

    **Note**: This a prototype only. Beware of:

    1. The main measurement does not follow Leonard and Schmidt's formula but simply
    measures the distance to each of the axis words. This is probably less precise.
    2. It is only possible to use single words on the axis
       ''')

