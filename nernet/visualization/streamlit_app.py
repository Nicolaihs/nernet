
import glob
import os
import re
import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from collections import defaultdict
from pyvis.network import Network

import plotly.express as px

edges_file = None
nodes_file = None
input_dir = '../../data/processed/memo/'
protagonist_base = 'Interactions'
protagonist_ratio = 1.75

disruptor_ratio = 3

TEAM_COLORS = [ '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#1f77b4', ]
#TEAM_COLORS = [ 'red', 'green', '#2ca02c', '#d62728', '#9467bd',]

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

def get_central_characters(df_nodes: pd.DataFrame, col_name, protagonist_ratio) -> list:
    """Determine how many protagonists there are in the network."""
    print(f"Determining protagonists...{col_name}, {protagonist_ratio}")
    df_nodes.sort_values(by=protagonist_base, ascending=False, inplace=True)
    prev_interactions = None
    protagonists = []
    for _, row in df_nodes.iterrows():
        if prev_interactions is None:
            prev_interactions = row[col_name]
            protagonists.append(row['Id'])
            continue
        row_ratio = prev_interactions / row[col_name]
#        st.write(f'{row["Id"]}: {prev_interactions}/{row[col_name]} = {row_ratio}')
        if row_ratio >= protagonist_ratio:
            break
        protagonists.append(row['Id'])
        prev_interactions = row[col_name]
        if len(protagonists) >= 5:
            protagonists = []
            break
    return protagonists

def get_disruptors(df_nodes: pd.DataFrame, df_edges: pd.DataFrame, central_characters: list) -> list:
    """Determine disruptors for networks with more than one central character."""
    if len(central_characters) < 2:
        return []

    edges_to_central_characters = {}
    for _, row in df_nodes.iterrows():
        if row['Id'] in central_characters:
            continue
        current_edges = {}
        for central_character in central_characters:
            count = 0
            cell = df_edges.loc[(df_edges['source'] == row['Id'])
                                & (df_edges['target'] == central_character), 'weight']
            if cell.values:
                count += int(cell.values[0])
            cell = df_edges.loc[(df_edges['source'] == central_character)
                                & (df_edges['target'] == row['Id']), 'weight']
            if cell.values:
                count += int(cell.values[0])
            current_edges[central_character] = count
        edges_to_central_characters[row['Id']] = current_edges

    disruptors = defaultdict(list)
    for character, character_counts in edges_to_central_characters.items():
        sorted_counts = {k: v for k, v in sorted(character_counts.items(), key=lambda item: item[1])}
        counts = list(sorted_counts.values())
        central_character = list(sorted_counts.keys())[-1]
        if counts[-1] == 0 and counts[-2] == 0:
            continue
        elif counts[-2] == 0:
            disruptors[central_character].append((character, counts[-1]))
        else:
            disruption_score = counts[-1] / counts[-2]
            if disruption_score >= disruptor_ratio:
                disruptors[central_character].append((character, disruption_score))
    for central_character, character_disruptors in disruptors.items():
        # Sort disruptors by disruption score
        character_disruptors.sort(key=lambda x: x[1], reverse=True)
    return disruptors

# Open directory
books = {}
for f in sorted(glob.glob(os.path.join(input_dir, '*.paragraphed_nodes.csv'))):
    # Get filename of glob
    filename = os.path.basename(f)
    book_name = filename.split('.paragraphed_nodes.csv')[0]
    nodes = pd.read_csv(f)
    nodes.sort_values(by='Interactions', ascending=False, inplace=True)
    books[book_name] = {
        'nodes': nodes,
        'edges': None,
        'year_of_publication': '1800',
        'central_characters': None
    }
    year_match = re.match('\d\d\d\d', book_name)
    if year_match is not None:
        books[book_name]['year_of_publication'] = year_match.group(0)

years = {}
counts = {}
for year in range(1870, 1900):
    years[str(year)] = 0
    counts[str(year)] = defaultdict(int)
def analyze_books():
    for book in books.values():
        book['central_characters'] = get_central_characters(book['nodes'], protagonist_base, protagonist_ratio)
        if book['year_of_publication'] in years.keys():
            years[book['year_of_publication']] += 1
        counts[book['year_of_publication']][len(book['central_characters'])] += 1

def bar_chart(data, y_label):
    """Display plotly chart."""
    fig = px.bar(x=data.keys(), y=data.values(),
                labels={'x': 'Year', 'y': y_label})
    st.plotly_chart(fig)

# Set header title
st.title('Network Graph Visualization')

analyze_books()

st.header('No of books by year')
all = pd.DataFrame.from_dict(counts, orient='index').fillna(0)
fig = px.bar(all, x=all.index, y=all.columns.sort_values())
fig.update_layout(
    title="No of network protagonists by year",
    legend_title="No. of protagonists")
st.plotly_chart(fig)

with st.sidebar:
    books_to_view = books
    no_of_pro = st.multiselect('No. of protagonists', [0, 1, 2, 3, 4])
#    if no_of_pro is not None:
#        books_to_view = {key: value for key, value in books.items() if len(value['central_characters']) in no_of_pro}
    selected_years = st.multiselect('Select one or more years', sorted(list(set([book['year_of_publication'] for book in books.values()]))))
#    if selected_years:
#        books_to_view = {key: book for key, book in books.items() if book['year_of_publication'] in selected_years}
#    books_to_view = st.multiselect('Select one or more books', books_to_view.keys())
#    protagonist_base = st.radio('Select central character measure', ['Interactions', 'Connections', 'Centrality'])
#    protagonist_ratio = st.slider('Select central character ratio', 0.5, 3.0, 1.7)

    books_reduced = [(key, book['year_of_publication'], len(book['central_characters'])) for key, book in books.items()]
    df_filtered_books = pd.DataFrame(books_reduced, columns=['index', 'year', 'no of prot.'])
    if no_of_pro:
        df_filtered_books = df_filtered_books.loc[df_filtered_books['no of prot.'].isin(no_of_pro)]
    if selected_years:
        df_filtered_books = df_filtered_books.loc[df_filtered_books['year'].isin(selected_years)]

    books_to_view = st.multiselect('Books', df_filtered_books)

if len(books_to_view) == 0:
    st.write('No books selected')
else:
    for book_to_view in books_to_view:
        st.header(book_to_view)
        df_nodes = books[book_to_view]['nodes']

        # Load edges
        if books[book_to_view]['edges'] is None:
            df_interact = pd.read_csv(os.path.join(input_dir, book_to_view + '.paragraphed_edges.csv'))
            books[book_to_view]['edges'] = df_interact
        else:
            df_interact = books[book_to_view]['edges']

        # Remove nodes with few edges
        df_nodes = df_nodes[df_nodes['Interactions'] > 5]

        central_characters = get_central_characters(df_nodes, protagonist_base, protagonist_ratio)
        st.metric('No. of central characters', len(central_characters))
        teams = get_disruptors(df_nodes, df_interact, central_characters)
        if teams:
            st.header('Teams')
            cols = st.columns(len(teams))
            for i, (central_character, members) in enumerate(teams.items()):
                cols[i].subheader(central_character)
                display = [f'{member} ({score})' for member, score in members]
                cols[i].write('; '.join(display))

            character_team = {}
            for _, row in df_nodes.iterrows():
                for central_character, members in teams.items():
                    if row['Id'] in [ member for member, _score in members]:
                        character_team[row['Id']] = central_character

        # Set header 2
        st.header('All characters in the network')
        st.dataframe(df_nodes.sort_values(by=['Connections'], ascending=False))

        # Select only edges where both source and target are in the nodes list
        df_interact = df_interact[df_interact['source'].isin(df_nodes['Id']) & df_interact['target'].isin(df_nodes['Id'])]


        # Define selection options and sort alphabetically
        df_most_interactions = df_nodes.sort_values(by=['Interactions'], ascending=False)[:20]
        node_list = df_most_interactions['Id'].tolist()
        #node_list.sort()

        # Implement multiselect dropdown menu for option selection
        selected_nodes = st.multiselect('Select characters to visualize', node_list)

        # Set info message on initial site load
        if len(selected_nodes) == 0:
            df_select = df_interact
        # Create network graph when user selects >= 1 item
        else:
            # Code for filtering dataframe and generating network
            df_select = df_interact.loc[df_interact['source'].isin(selected_nodes) | \
                                        df_interact['target'].isin(selected_nodes)]
            df_select = df_select.reset_index(drop=True)
            df_nodes = df_nodes.loc[df_nodes['Id'].isin(df_select['source']) | df_nodes['Id'].isin(df_select['target'])]

        # Create network graph
        G = nx.Graph()
        for _, row in df_nodes.iterrows():
            color = None
            if teams:
                if row['Id'] in character_team.keys():
                    team_leader = character_team[row['Id']]
                    team = teams[team_leader]
                    team_color = TEAM_COLORS[central_characters.index(team_leader)]
                    max_teamness = max([score for xs in teams.values() for _name, score in xs])
                    teamness = [score for name, score in team if name == row['Id']][0]
                    teamness = teamness / max_teamness
                    color = lighter(team_color, 1-(teamness/2+0.5))
                elif row['Id'] in central_characters:
                    team_color = TEAM_COLORS[central_characters.index(row['Id'])]
                    color = team_color

            G.add_node(str(row['Id']), size=int(row['Connections']), weight=row['Interactions'], color=color) # color=network_color(color_network_option), title=hover_info, borderWidth=4)

        for _, row in df_select.iterrows():
            G.add_edge(row['source'], row['target'], value=row['weight']*row["weight"])

        ner_net = Network(height='1024px', width='800px', bgcolor='#222222', font_color='white')
        ner_net.from_nx(G)

        path = '/tmp'
        ner_net.save_graph(f'{path}/pyvis_graph.html')
        HtmlFile = open(f'{path}/pyvis_graph.html','r', encoding='utf-8')

        # Load HTML file in HTML component for display on Streamlit page
        components.html(HtmlFile.read(), height=1024, width=800)

        #if st.checkbox('Save this grpah as GraphML'):
        #    nx.write_graphml(G, f'/tmp/{book_to_view}.ml')
