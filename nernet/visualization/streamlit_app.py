
import glob
import os
import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

edges_file = None
nodes_file = None

# Open directory
books = {}
for f in glob.glob('../../data/processed/memo/*paragraphed_nodes.csv'):
    # Get filename of glob
    filename = os.path.basename(f)
    books[filename] = pd.read_csv(f)

books_to_view = st.multiselect('Select one or more books', books.keys())
if len(books_to_view) == 0:
    st.write('No books selected')
else:
    for book_to_view in books_to_view:
        st.header(book_to_view)
        st.dataframe(books[book_to_view])

# Read dataset
edges_file = st.file_uploader("Choose an edge file", type=["csv"])
if edges_file is not None:
    #nodes_file = None
    df_interact = pd.read_csv(edges_file)

nodes_file = st.file_uploader("Choose a nodes file", type=["csv"])
if nodes_file is not None:
    df_nodes = pd.read_csv(nodes_file)

if edges_file is None or nodes_file is None:
    st.text(f'Please upload both an edge and a nodes file ({edges_file} and {nodes_file})')
else:
    # Remove nodes with few edges
    df_nodes = df_nodes[df_nodes['Interactions'] > 5]

    # Set header title
    st.title('Network Graph Visualization')

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


    # Grapping the weights of the edges
    #weights = [d['weight'] for (u, v, d) in G.edges(data=True)]

    # # Create networkx graph object from pandas dataframe
    # G = nx.from_pandas_edgelist(df_select, 'source', 'target', 'weight')

    # # Initiate PyVis network object
    # ner_net = Network(height='1024px', width='1024px', bgcolor='#222222', font_color='white')

    # # Take Networkx graph and translate it to a PyVis graph format
    # ner_net.from_nx(G)

    # # Generate network with specific layout settings
    # #ner_net.repulsion(node_distance=420, central_gravity=0.33,
    # #                    spring_length=110, spring_strength=0.10,
    # #                    damping=0.95)

    ner_net = Network(height='1024px', width='1024px', bgcolor='#222222', font_color='white')
    ner_net.from_nx(G)

    path = '/tmp'
    ner_net.save_graph(f'{path}/pyvis_graph.html')
    HtmlFile = open(f'{path}/pyvis_graph.html','r', encoding='utf-8')

    # Load HTML file in HTML component for display on Streamlit page
    components.html(HtmlFile.read(), height=1024, width=1024)