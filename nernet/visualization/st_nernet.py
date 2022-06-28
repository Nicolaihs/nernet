import glob
import os
import pandas as pd
import plotly.express as px
import streamlit as st

from nernet.visualization.st_utils import load_nodes, create_network, create_network_from_edges
from nernet.visualization.st_utils import split_by_timestamp, create_nodes, count_networks
from nernet.visualization.st_utils import get_teams

st.title('NERnet')
st.subheader('Named Entity Recognition Networks')

SELECTION_NO_PROT = [0, 1, 2, 3, 4]
FILEPATH = '../../data/processed/memo/'
DEFAULT_MINIMUM_INTERACTIONS = 5


all_metadata, all_nodes, all_edges, all_timestamped_edges = load_nodes(FILEPATH)
df_filtered = pd.DataFrame.from_dict(all_metadata, orient='index')
#df_filtered.reset_index(drop=True, inplace=True)


selected_books = []
with st.sidebar:
    selected_no_prot = st.multiselect('No. of protagonists', SELECTION_NO_PROT)
    selected_years = st.multiselect('Select one or more years', sorted(list(set([book['year'] for book in all_metadata.values()]))))
    if selected_no_prot:
        df_filtered = df_filtered.loc[df_filtered['no. of protagonists'].isin(selected_no_prot)]
    if selected_years:
        df_filtered = df_filtered.loc[df_filtered['year'].isin(selected_years)]

    selected_books = st.multiselect('Select one or more books:', df_filtered.index)

#st.dataframe(df_filtered[['author', 'title']])

def present_network(header, df_nodes, df_edges, minimum_interactions, teams={}):
    """Show a network."""
    st.markdown('---')
    st.header(header)
    cols = st.columns(2)
    cols[0].metric('No. of network characters', len(df_nodes))
    df_nodes_reduced = df_nodes[df_nodes['Interactions'] > minimum_interactions]
    cols[1].metric(f'No. of network characters\n(more than {minimum_interactions} interactions)', len(df_nodes_reduced))

    create_network(df_nodes_reduced, df_edges, teams=teams)


if len(selected_books) == 0:
    counts = count_networks(all_metadata)
    all = pd.DataFrame.from_dict(counts, orient='index').fillna(0)
    fig = px.bar(all, x=all.index, y=all.columns.sort_values())
    fig.update_layout(
        title="No of network protagonists by year",
        legend_title="No. of protagonists")
    st.plotly_chart(fig)
else:
    for selected_book in selected_books:

        metadata = all_metadata[selected_book]
        df_nodes = all_nodes[selected_book]
        df_edges = all_edges[selected_book]
        df_timestamped_edges = all_timestamped_edges[selected_book]
        st.header(f'{metadata["author"]}: {metadata["title"]}')

        team_ratio = st.slider('Team ratio', 1.5, 6.0, 3.0)
        teams = get_teams(df_nodes, df_edges, metadata['protagonists'], team_ratio)

        cols = st.columns(2)
        cols[0].metric('No. network protagonists', len(metadata['protagonists']))

        minimum_interactions = st.slider('Minimum interactions', 0, 10, DEFAULT_MINIMUM_INTERACTIONS)

        present_network('I-III', df_nodes, df_edges, minimum_interactions, teams=teams)
        # # df_nodes_reduced = df_nodes[df_nodes['Interactions'] > minimum_interactions]
        # # cols = st.columns(2)
        # # cols[1].metric('No. network characters', len(df_nodes))
        # # cols[1].metric(f'No. network characters\n(more than {minimum_interactions} interactions)', len(df_nodes_reduced))

        # create_network(df_nodes_reduced, df_edges)

        df_edges_1, df_edges_2, df_edges_3 = split_by_timestamp(df_timestamped_edges)
        df_nodes_1 = create_nodes(df_edges_1)
        df_nodes_2 = create_nodes(df_edges_2)
        df_nodes_3 = create_nodes(df_edges_3)

        present_network('I', df_nodes_1, df_edges_1, minimum_interactions, teams=teams)
        present_network('II', df_nodes_2, df_edges_2, minimum_interactions, teams=teams)
        present_network('III', df_nodes_3, df_edges_3, minimum_interactions, teams)



