
import glob
import os
import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

edges_file = None
nodes_file = None
input_dir = '../../data/processed/memo/'
protagonist_base = 'Interactions'
protagonist_ratio = 1.75


def central_characters(df_nodes: pd.DataFrame, col_name, protagonist_ratio) -> list:
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
        st.write(f'{row["Id"]}: {prev_interactions}/{row[col_name]} = {row_ratio}')
        if row_ratio >= protagonist_ratio:
            break
        protagonists.append(row['Id'])
        prev_interactions = row[col_name]
        if len(protagonists) >= 5:
            protagonists = []
            break
    return protagonists

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
        'central_characters': None}

def analyze_books():
    for book in books.values():
        book['central_characters'] = central_characters(book['nodes'], protagonist_base, protagonist_ratio)

# Set header title
st.title('Network Graph Visualization')

with st.sidebar:
    books_to_view = st.multiselect('Select one or more books', books.keys())
    protagonist_base = st.radio('Select central character measure', ['Interactions', 'Connections', 'Centrality'])
    protagonist_ratio = st.slider('Select central character ratio', 0.5, 3.0, 1.7)

if len(books_to_view) == 0:
    st.write('No books selected')
else:
    for book_to_view in books_to_view:
        st.header(book_to_view)
        df_nodes = books[book_to_view]['nodes']

        central_characters = determine_protagonists(df_nodes, protagonist_base, protagonist_ratio)
        st.metric('No. of central characters', len(central_characters))


        # Load edges
        if books[book_to_view]['edges'] is None:
            df_interact = pd.read_csv(os.path.join(input_dir, book_to_view + '.paragraphed_edges.csv'))
            books[book_to_view]['edges'] = df_interact
        else:
            df_interact = books[book_to_view]['edges']

        # Remove nodes with few edges
        df_nodes = df_nodes[df_nodes['Interactions'] > 5]


        # Set header 2
        st.header('All characters in the network')
        st.dataframe(df_nodes.sort_values(by=['Connections']))

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
            G.add_node(row['Id'], size=row['Connections'], weight=['Interactions']) # color=network_color(color_network_option), title=hover_info, borderWidth=4)

        for _, row in df_select.iterrows():
            G.add_edge(row['source'], row['target'], value=row['weight']*row["weight"])

        ner_net = Network(height='1024px', width='1024px', bgcolor='#222222', font_color='white')
        ner_net.from_nx(G)

        path = '/tmp'
        ner_net.save_graph(f'{path}/pyvis_graph.html')
        HtmlFile = open(f'{path}/pyvis_graph.html','r', encoding='utf-8')

        # Load HTML file in HTML component for display on Streamlit page
        components.html(HtmlFile.read(), height=1024, width=1024)

