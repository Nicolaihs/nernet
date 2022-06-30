import glob
import os
import re
import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from collections import defaultdict
from pyvis.network import Network

TEAM_COLORS = [ '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#1f77b4', ]

@st.cache(persist=True)
def load_nodes(input_dir, protagonist_score='Interactions', protagonist_ratio=1.75):
    """Load all nodes from from input_dir and initialise database."""
    all_nodes = {}
    all_edges = {}
    all_timed_edges = {}
    metadata = {}

    filepath = os.path.join(input_dir, '*.paragraphed_nodes.csv')
    no_of_files = len(glob.glob(filepath))

    for i, f in enumerate(sorted(glob.glob(filepath))):
        # Get filename of glob
        filename = os.path.basename(f)
        book_name = filename.split('.paragraphed_nodes.csv')[0]
        nodes = pd.read_csv(f)
        nodes.sort_values(by='Interactions', ascending=False, inplace=True)
        metadata[book_name] = {
            'author': book_name.split('_')[1],
            'title': book_name.split('_')[2],
            'year': book_name.split('_')[0],
            'no. of protagonists': None,
            'protagonists': None,
            'name': book_name
        }
        protagonists = get_protagonists(nodes, protagonist_score, protagonist_ratio)
        metadata[book_name]['protagonists'] = protagonists
        metadata[book_name]['no. of protagonists'] = len(protagonists)

        all_nodes[book_name] = nodes

        # Read edges
        df_edges = pd.read_csv(os.path.join(input_dir, f'{book_name}.paragraphed_edges.csv'))
        all_edges[book_name] = df_edges

        df_timed_edges = pd.read_csv(os.path.join(input_dir, f'{book_name}.paragraphed_edges_timestamped.csv'))
        all_timed_edges[book_name] = df_timed_edges

    return metadata, all_nodes, all_edges, all_timed_edges


@st.cache
def count_networks(all_metadata, from_year=1870, to_year=1899):
    """Statistics for network."""
    counts = {}
    years = {}
    for year in range(from_year, to_year+1):
        years[str(year)] = 0
        counts[str(year)] = defaultdict(int)
    for book in all_metadata.values():
        if book['year'] in years.keys():
            years[book['year']] += 1
        counts[book['year']][len(book['protagonists'])] += 1
    return counts


def get_protagonists(df_nodes: pd.DataFrame, col_name, protagonist_ratio) -> list:
    """Determine how many protagonists there are in the network."""
#    print(f"Determining protagonists...{col_name}, {protagonist_ratio}")
    df_copy = df_nodes.sort_values(by=col_name, ascending=False)
    prev_value = None
    protagonists = []
    for _, row in df_copy.iterrows():
        if prev_value is None:
            prev_value = row[col_name]
            protagonists.append(row['Id'])
            continue
        row_ratio = prev_value / row[col_name]
        if row_ratio >= protagonist_ratio:
            break
        protagonists.append(row['Id'])
        prev_value = row[col_name]
        if len(protagonists) >= 5:
            protagonists = []
            break
    return protagonists


def show_network(G):
    """Show the network using pyvis."""
    ner_net = Network(height='1024px', width='800px', bgcolor='#222222', font_color='white')
    ner_net.from_nx(G)

    path = '/tmp'
    ner_net.save_graph(f'{path}/pyvis_graph.html')
    HtmlFile = open(f'{path}/pyvis_graph.html','r', encoding='utf-8')

    # Load HTML file in HTML component for display on Streamlit page
    components.html(HtmlFile.read(), height=1024, width=800)



import numpy as np

def lighter(color, percent):
    '''assumes color is rgb between (0, 0, 0) and (255, 255, 255)'''
    color = np.array(hex2rgb(color))
    white = np.array([255, 255, 255])
    vector = white-color
    ligthened = tuple(color + vector * percent)
    ligthened = int(ligthened[0]), int(ligthened[1]), int(ligthened[2])
    return rgb2hex(ligthened)

def hex2rgb(hex):
    hex = hex.lstrip('#')
    return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))

def rgb2hex(rgb):
    return '#%02x%02x%02x' % rgb


def create_network(df_nodes, df_edges, size_from='Degree', teams: dict={}):
    """Create networkx network from nodes and edges."""
    protagonists = list(teams.keys())
    team_members = {}
    for protagonist, members in teams.items():
        for member, affinity_score in members:
            team_members[member] = {'leader': protagonist, 'affinity_score': affinity_score}

    if teams:
        max_teamness = max([score for xs in teams.values() for _name, score in xs])

    G = nx.Graph()
    memory = []
    for _, row in df_nodes.iterrows():
        memory.append(str(row['Id']))
        color = None
        node = row['Id']
        if team_members.get(node):
            team_leader = team_members[node]['leader']
            team = teams[team_leader]
            team_idx = protagonists.index(team_leader)
            team_color = TEAM_COLORS[team_idx]
            teamness = [score for name, score in team if name == row['Id']][0]
            teamness = teamness / max_teamness
            color = lighter(team_color, 1-(teamness/2+0.5))
        elif node in protagonists:
            team_idx = protagonists.index(node)
            team_color = TEAM_COLORS[team_idx]
            color = team_color

        G.add_node(str(row['Id']), size=int(row[size_from]), weight=int(row['Interactions']),
                   color=color)

    for _, row in df_edges.iterrows():
        if str(row['source']) in memory and str(row['target']) in memory:
            G.add_edge(str(row['source']), str(row['target']), value=int(row['weight'])*int(row["weight"]))
    show_network(G)


def create_network_from_edges(df_edges):
    """Create network from edges only."""
    G = nx.Graph()
    for _, row in df_edges.iterrows():
        G.add_edge(str(row['source']), str(row['target']), value=int(row['weight'])*int(row["weight"]))

    degrees = dict(G.degree)
    nx.draw(G, nodelist=degrees.keys(), node_size=[v * 100 for v in degrees.values()])

    show_network(G)


def split_by_timestamp(df, no_of_slices=3):
    """Split df in equal parts."""

    size = int(len(df)/no_of_slices)
    N = no_of_slices
    df_slices = [ df.iloc[i*size:(i+1)*size].copy() for i in range(N+1) ]
    # cols = st.columns(3)
    # cols[0].metric('Len. I', len(df_slices[0]))
    # cols[1].metric('Len. I', len(df_slices[1]))
    # cols[2].metric('Len. I', len(df_slices[2]))

    df_slices = [timestamp_to_weight(df_slice) for df_slice in df_slices]
    return df_slices[0], df_slices[1], df_slices[2]


def timestamp_to_weight(df):
    """Return dataframe with timestamp column converted into weight."""
    df_grouped = df.groupby(['source','target']).count()
    df_grouped.reset_index(inplace=True)
    df_grouped.rename(columns={'timestamp': 'weight'}, inplace=True)
    df_grouped['type'] = 'undirected'
    return df_grouped


@st.cache
def create_nodes(df_edges):
    """Return dataframe with nodes from edges file."""

    G = nx.from_pandas_edgelist(df_edges, edge_attr='weight')

    nodes = []
    for node_id in G.nodes:
        interactions = sum(df_edges[df_edges['source'] == node_id]['weight']) \
                       + sum(df_edges[df_edges['target'] == node_id]['weight'])
        node = {
            'Id': node_id,
            'Connections': G.degree[node_id],
            'Interactions': interactions,
            'Degree': G.degree[node_id]
        }
        nodes.append(node)

    return pd.DataFrame(nodes)

@st.cache
def get_teams(df_nodes: pd.DataFrame, df_edges: pd.DataFrame, protagonists: list, team_ratio: int = 3) -> list:
    """Determine teams for networks with more than one central character."""
    if len(protagonists) < 2:
        return {}

    edges_to_protagonists = {}
    for _, row in df_nodes.iterrows():
        if row['Id'] in protagonists:
            continue
        current_edges = {}
        for protagonist in protagonists:
            count = 0
            cell = df_edges.loc[(df_edges['source'] == row['Id'])
                                & (df_edges['target'] == protagonist), 'weight']
            if cell.values:
                count += int(cell.values[0])
            cell = df_edges.loc[(df_edges['source'] == protagonist)
                                & (df_edges['target'] == row['Id']), 'weight']
            if cell.values:
                count += int(cell.values[0])
            current_edges[protagonist] = count
        edges_to_protagonists[row['Id']] = current_edges

    teams = defaultdict(list)
    for character, character_counts in edges_to_protagonists.items():
        sorted_counts = {k: v for k, v in sorted(character_counts.items(), key=lambda item: item[1])}
        counts = list(sorted_counts.values())
        protagonist = list(sorted_counts.keys())[-1]
        if counts[-1] == 0 and counts[-2] == 0:
            continue
        elif counts[-2] == 0:
            teams[protagonist].append((character, counts[-1]))
        else:
            affinity_score = counts[-1] / counts[-2]
            if affinity_score >= team_ratio:
                teams[protagonist].append((character, affinity_score))
    for protagonist, character_teams in teams.items():
        # Sort teams by disruption score
        character_teams.sort(key=lambda x: x[1], reverse=True)
    return teams