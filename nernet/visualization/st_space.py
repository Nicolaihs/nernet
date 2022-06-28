
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from gensim.models import Word2Vec
from sklearn import preprocessing


model_path = '../../data/models/memo_m5_f500_epoch10_w10.model'

@st.cache(allow_output_mutation=True)
def load_model():
    model = Word2Vec.load(model_path)
    model_wv = model.wv
#    del model
    return model_wv

model_wv = load_model()

def lookup_sims(word, min_similarity=0.6):
    """"""
    word = word.strip()
    sims = model_wv.most_similar(word, topn=50)
    sims = [(sim, score) for sim, score in sims if score >= min_similarity]
    return sims

def add_more_sims(sims):
    new_sims = sims.copy()
    for sim, score in sims:
        candidates = lookup_sims(sim)
        print(sim, candidates)
        for candidate, candidate_score in candidates:
            if candidate not in [sim for sim, _score in new_sims]:
                new_sims.append((candidate, candidate_score))

    return new_sims


st.title('A Peter Leonard Plotter')
with st.sidebar:
    input_seed = st.text_input('Seed word')
    col1, col2 = st.columns(2)
    x_left = col1.text_input('X axis, left', 'hvid')
    x_right = col2.text_input('X axis, right', 'sort')
    col1, col2 = st.columns(2)
    y_down = col1.text_input('Y axis, down', 'mand')
    y_up = col2.text_input('Y axis, up', 'kvinde')

if input_seed and x_left and x_right and y_up and y_down:
    input_seed = input_seed.strip()
    sims = []
    for seed in input_seed.split(','):
        sims += lookup_sims(seed)
    sims = add_more_sims(sims)
    if len(sims) < 50:
        sims = add_more_sims(sims)
#    sims = add_more_sims(sims)
    df_sims = pd.DataFrame(sims, columns=['word', 'score'])

    x_values, y_values = [], []
    for sim, _score in sims:
        left_val = model_wv.distance(sim, x_left)
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

