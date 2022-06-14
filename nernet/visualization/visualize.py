# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from collections import Counter
from pathlib import Path
from jaal import Jaal
from jaal.datasets import load_got
from dotenv import find_dotenv, load_dotenv


def normalize(list_of_dict, key, multiplier=10):
    """Normalize list_of_dict."""
    values = [item[key] for item in list_of_dict]
    for item in list_of_dict:
        item[key] = (item[key] - min(values)) \
                    /(max(values) - min(values)) * multiplier


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
def main(input_filepath):
    """ Visualize network
    """
    logger = logging.getLogger(__name__)
    logger.info('Visualizing data')

    logger.info('Reading input file ...')
    df = pd.read_csv(input_filepath)
    df = df.loc[df['Weight']>1, :]

    # Connections: Number of different edges
    connection_list = list(df['Source'].tolist() +
                           df['Target'].tolist())
    connection_list = Counter(connection_list)
    nodes = [{'id': node_name, 'label': node_name, 'shape': 'dot', 'connections': size}
             for node_name, size in connection_list.items()]
    normalize(nodes, 'connections')

    # Interactions: Total number of weight of all edges
    records = df.to_dict(orient='records')
    sources = [{'id': item['Source'], 'interactions': item['Weight']} for item in records]
    targets = [{'id': item['Target'], 'interactions': item['Weight']} for item in records]
    sources_dict = {item['id']: item for item in sources}
    target_dict = {item['id']: item for item in targets}
    interactions = []
    for row in sources:
        print(row)
        weight = target_dict.get('id', {}).get('interactions', 0)
        row['interactions'] += weight
        interactions.append(row)
    for row in targets:
        if row['id'] not in sources_dict:
            interactions.append(row)
    normalize(interactions, 'interactions')
    interactions = {item['id']: item for item in interactions}
    for row in nodes:
        row['interactions'] = interactions[row['id']]['interactions']

    edges = []
    for row in df.to_dict(orient='records'):
        source, target, weight = row['Source'], row['Target'], row['Weight']
        edges.append({'id': f'{source}_{target}', 'from': source, 'to': target, 'width': weight})
    normalize(edges, 'width')

    edge_df, node_df = load_got()
    import ipdb; ipdb.set_trace()
    Jaal(pd.DataFrame(edges), pd.DataFrame(nodes)).plot()

    # load the data
    #edge_df, node_df = load_got()
    # init Jaal and run server
    #Jaal(edge_df, node_df).plot()


if __name__ == '__main__':
#    progressbar.streams.wrap_stderr()
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
