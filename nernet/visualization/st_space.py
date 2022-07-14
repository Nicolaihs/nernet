import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from math import sqrt
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

def lookup_sims(model, word, min_similarity=0.6):
    """"""
    if '+' in word:
        words = [model[single.strip()] for single in word.split('+')]
        word = sum(words)
    else:
        word = word.strip()
    sims = model.most_similar(word, topn=300)
    sims = [(sim, score) for sim, score in sims if score >= min_similarity]
    return sims


def distance(model, word, target):
    if '+' in target:
        targets = [model[single.strip()] for single in target.split('+')]
        target = sum(targets)

        most_similar = model.most_similar(positive=[target,], topn=1)
        dist = model.distance(word, most_similar[0][0])
    else:
        dist = model.distance(word, target)
    return dist


def calculate_schmidt_score(model, sims, x_left, x_right, y_down, y_up):
    x_vector = model[x_left] - model[x_right]
    y_vector = model[y_down] - model[y_up]

    existing_sims = []
    nonexisting_sims = []
    for sim, score in sims:
        if not model.key_to_index.get(sim):
            nonexisting_sims.append((sim, score))
        else:
            existing_sims.append((sim, score))

    words = [sim for sim, _score in existing_sims]
    distance_to_x_vec = model.distances(x_vector, words)
    distance_to_y_vec = model.distances(y_vector, words)

    df_schmidt = pd.DataFrame(words, columns=['word'])
    df_schmidt['x'] = distance_to_x_vec
    df_schmidt['y'] = distance_to_y_vec

    return df_schmidt, pd.DataFrame(nonexisting_sims, columns=['word', 'score'])


def calculate_distance_scores(model, sims, x_left, x_right, y_down, y_up):

    existing_sims = []
    nonexisting_sims = []
    x_values, y_values = [], []
    for sim, score in sims:
        if not model.key_to_index.get(sim):
            nonexisting_sims.append((sim, score))
            continue
        left_val = model.distance(sim, x_left)
        right_val = model.distance(sim, x_right)
        x_values.append((left_val, right_val, left_val-right_val))

        down_val = model.distance(sim, y_down)
        up_val = model.distance(sim, y_up)
        y_values.append((down_val, up_val, down_val-up_val))
        existing_sims.append((sim, score))

    df_sims = pd.DataFrame(existing_sims, columns=['word', 'score'])
    df_nonexisting_sims = pd.DataFrame(nonexisting_sims, columns=['word', 'score'])

    x_array = np.array([x[2] for x in x_values])
    y_array = np.array([y[2] for y in y_values])
    norm_x = preprocessing.normalize([x_array])
    norm_y = preprocessing.normalize([y_array])

    df_sims['x'] = norm_x[0]
    df_sims['y'] = norm_y[0]
    return df_sims, df_nonexisting_sims


def calculate_distances(model, sims, measure, x_left, x_right, y_down, y_up):
    if measure == 'Distance score':
        df_graph, df_skipped = calculate_distance_scores(model, sims, x_left, x_right, y_down, y_up)
    else:
        df_graph, df_skipped = calculate_schmidt_score(model, sims, x_left, x_right, y_down, y_up)
    return df_graph, df_skipped


def draw_diagram(df: pd.DataFrame, x_left: str, x_right: str, y_down: str, y_up: str):
    if not 'color' in df.columns:
        df['color'] = 'white'
    fig = px.scatter(df, x="x", y="y", text="word", color="color",
                     labels={'x': f'{x_left} <-----> {x_right}',
                             'y': f'{y_down} <--------> {y_up}'}) #, log_x=True, size_max=60)
    fig.update_traces(textposition='top center')
    fig.update_layout(
        title=f'Show 2-dim vector space for words similar to "{input_seed}"'
    )
    st.plotly_chart(fig)
    return fig


def draw_annotated_diagram(df: pd.DataFrame, arrows: list, x_left: str, x_right: str, y_down: str, y_up: str):
    if not 'color' in df.columns:
        df['color'] = 'white'

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df['x'], y=df['y'],
            mode='markers+text', name='markers',
            marker=dict(
                color=df['color'],
                size=4,
                line=dict(
                    color=df['color'],
                    width=2
                )
            ),
            text=df['word'],
            textposition="top center"))

    fig.update_layout(
#        title="Plot Title",
        xaxis_title=f'← {x_left}              {x_right} →',
        yaxis_title=f'← {y_down}              {y_up} →',
        legend_title="Legend Title",
        font=dict(
            family="Source Sans Pro, sans-serif",
            size=14,
            color="#444"
        )
    )
    for arrow in arrows:
        fig.add_annotation(x=arrow['x_to'], y=arrow['y_to'],
                           ax=arrow['x_from'], ay=arrow['y_from'],
                           text="",
                           arrowcolor="lightgrey",
                           showarrow=True,
                           arrowhead=5,
                           arrowwidth=1.5,
                           axref="x", ayref="y")

    st.plotly_chart(fig)
    return fig


st.title('MeMo in Space!')
st.caption('Based on work by Peter Leonard, based on work by Ben Schmidt')

with st.sidebar:
    df_models = pd.read_csv(os.path.join(model_path, 'space_embeddings.csv'), delimiter=";")
    select_model = st.selectbox('Select word embeddings', df_models['name'])

    if select_model:
        selected_filepath = df_models.loc[df_models['name']==select_model, 'filepath'].tolist()[0]
        model_wv = load_model(selected_filepath)

    input_seed = st.text_input('Seed word or words').strip()
    st.caption('Input one or more words to investigate. \n Use + to combine words into a single vector.\nUse , to enter multiple words.\nE.g.: "mand+dreng,kvinde+pige"')

    st.markdown('---')
    expand_with_sims = st.checkbox('Expand with sims', False)
    min_sim = st.slider('Min. similarity', 0.1, 0.9, 0.6)

    st.markdown('---')
    compare = st.checkbox('Compare with other model?', False)
    if compare:
        select_comparison = st.selectbox('Select comparison model', df_models[df_models.name != select_model]['name'])
        if select_comparison:
            comparison_filepath = df_models.loc[df_models.name==select_comparison, 'filepath'].tolist()[0]
            compare_wv = load_model(comparison_filepath)
    else:
        compare_wv = None
    st.markdown('---')

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
        if expand_with_sims:
            sims += lookup_sims(model_wv, seed, min_similarity=min_sim)

    if include_axis_words:
        axis_words = x_left.split('+') + x_right.split('+') + y_down.split('+') + y_up.split('+')
        for word in axis_words:
            if word not in [sim for sim, _score in sims]:
                sims.append((word, 0.0))

    st.metric('Number of selected words', len(sims))

    if st.checkbox('show words'):
        st.dataframe(pd.DataFrame(sims, columns=['Word', 'Similarity']))

    measure = st.selectbox('Use formula', ['Distance score', 'Ben Schmidt\'s original score'])

    df_graph, df_skipped = calculate_distances(model_wv, sims, measure, x_left, x_right, y_down, y_up)

    draw_diagram(df_graph, x_left, x_right, y_down, y_up)

    if compare_wv:
        st.header('Comparison model')
        st.write(f'Comparing with this model: {select_comparison}')
        df_comp, df_comp_skipped = calculate_distances(compare_wv, sims, measure, x_left, x_right, y_down, y_up)
        draw_diagram(df_comp, x_left, x_right, y_down, y_up)
        if len(df_skipped) > 0:
            st.markdown('The following words were not found in comparison model')
            st.dataframe(df_skipped)

        st.header('Side-by-side')
        st.markdown(f'Comparing:\n * 1: {select_model}\n * 2: {select_comparison} (²)')

        rows = df_models.loc[df_models.name.isin([select_model, select_comparison])]
        blankIndex = [''] * len(rows)
        rows.index = blankIndex
        st.dataframe(rows[['name', 'features', 'texttypes', 'comments', 'year_from', 'year_to']])
        st.markdown('**BEWARE!**\n * Differences in a word''s coordinates might arise from differences in the underlying corpora''s texttypes or other.\n * Movement might also result from changes in the axis words'' semantics.\n * The numeric scores and distances means nothing: The scores are normalized according to the words in the graph -- only the relative movement and the relative distances are significant.')

        two = []
        changes = []
        for _, row in df_graph.iterrows():
            word = row['word']
            x = row['x']
            y = row['y']
            comp_row = df_comp.loc[df_comp['word'] == word]
            if not comp_row.empty:
                comp_x = comp_row['x'].values[0]
                comp_y = comp_row['y'].values[0]
                two.append((f'{word}¹', x, y, '#3cb44b'))
                two.append((f'{word}²', comp_x, comp_y, '#f58231'))

                arrow = {
                    'word': word,
                    'change': sqrt((x-comp_x)**2 + (y-comp_y)**2),
                    'change x': comp_x-x, 'change y': comp_y-y,
                    'x_from': x, 'y_from': y,
                    'x_to': comp_x, 'y_to': comp_y
                }
                changes.append(arrow)


        df_two = pd.DataFrame(two, columns=['word', 'x', 'y', 'color'])
        fig = draw_annotated_diagram(df_two, changes, x_left, x_right, y_down, y_up)

        display_changes = [
            (item['word'], item['change'], item['change x'],
             item['change x'] > 0 and '→' or '←',
             item['change y'],
            item['change y'] > 0 and '↑' or '↓') for item in changes]
        df_changes = pd.DataFrame(display_changes, columns=['word', 'change', 'change x', 'left/right', 'change y', 'up/down'])
        df_changes.sort_values('change', inplace=True, ascending=False)

        st.dataframe(df_changes)


else:
    st.image('memo_space1.png')
    st.header('Available models')
    st.dataframe(df_models)
    st.header('Instructions')
    st.markdown('''
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

