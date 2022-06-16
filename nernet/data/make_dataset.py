# -*- coding: utf-8 -*-
import click
import csv
import glob
import logging
import os
import networkx as nx
from collections import defaultdict
from pathlib import Path

from progressbar import ProgressBar
from dotenv import find_dotenv, load_dotenv

import dacy
from dacy.sentiment import da_vader_getter
from spacy.tokens import Span, Doc
from torch import neg_

Doc.set_extension("vader_da", getter=da_vader_getter)

import warnings
warnings.filterwarnings("ignore")

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def normalize(dict_of_nodes: dict, key: str, round_to: int) -> dict:
    """Normalize value for key in dict_of_nodes."""
    list_of_nodes = dict_of_nodes.values()
    values = [item[key] for item in list_of_nodes]
    for item in list_of_nodes:
        try:
            item[f'{key}_norm'] = round((item[key] - min(values)) \
                        /(max(values) - min(values)), round_to)
        except ZeroDivisionError:
            import ipdb; ipdb.set_trace()
    output = {}
    for item in list_of_nodes:
        output[item['id']] = item
    return output


class NetworkDoc:
    round_to = 2

    def __init__(self, nlp) -> None:
        self.counter = 0
        self.current_time = 0
        self.nlp = nlp
        self.current_doc = None
        self.ne_from_previous_doc = None
        self.nodes = {}
        self.network = defaultdict(int)
        self.edges_timestamped = []
        self.ner_memory = defaultdict(int)
        self.ner_sentiments = defaultdict(int)
        self.ner_sentence_sentiments = defaultdict(int)
        self.current_polarity = 0

    def process_doc(self, doc):
        """Process a doc and retrieve network."""
        self.counter += 1
        self.current_doc = doc
        self.calculate_sentiment_score()
        self.expand_ners()
        self.create_simplified_network()
        self.calculate_sentiments()

    def handle_inflection(self, entity_text):
        """Detect and remove inflection"""
        if len(entity_text) > 3 and entity_text[-2:] == "'s":
            entity_text = entity_text[:-2]
        elif entity_text[-1] == "'":
            entity_text = entity_text[:-1]
        elif entity_text[-1] == 's' and entity_text[:-1] in self.ner_memory:
            entity_text = entity_text[:-1]
        self.ner_memory[entity_text] += 1
        return entity_text

    def expand_ners(self):
        """Expand the NERs to the left and right."""
        left_expands = ('Frøken', 'Frøknerne', 'Fru', 'Hr.', 'Herr')
        for ent in self.current_doc.ents:
            if ent.start > 0 and self.current_doc[ent.start-1].lemma_ in left_expands:
                span = Span(self.current_doc, ent.start-1, ent.start+1, label='PERSON')
                self.current_doc.set_ents([span], default="unmodified")

    def create_simplified_network(self):
        """Create simplified network of NER PERSON but connecing PER to next PER only."""
        pers = [ent for ent in self.current_doc.ents if ent.label_ == 'PER']
        if len(pers) == 0:# If paragraph has no PER, last ne is deleted
            self.ne_from_previous_doc = None
        if self.ne_from_previous_doc is not None:
            pers = [self.ne_from_previous_doc] + pers
        if pers:
            self.current_time += 1
            self.ne_from_previous_doc = pers[-1] # Save for next paragraph
        i = 0
        while i < len(pers) - 1:
            ent_1 = self.handle_inflection(pers[i].text)
            ent_2 = self.handle_inflection(pers[i+1].text)
            if ent_1 != ent_2:
                ent_source, ent_target = sorted((ent_1, ent_2))
                self.network[(ent_source, ent_target)] += 1
                self.edges_timestamped.append((ent_source, ent_target, self.current_time))
            self.ner_sentiments[ent_1] += self.current_polarity

            i += 1

    def analyze_nodes(self):
        """_summary_
        """
        connections = defaultdict(int)
        interactions = defaultdict(int)
        for source, target in self.network:
            connections[source] += 1
            interactions[source] += self.network[(source, target)]
            connections[target] += 1
            interactions[target] += self.network[(source, target)]

        for node in connections:
            self.nodes[node] = {
                'id': node,
                'polarity': round(self.ner_sentiments.get(node, 0), self.round_to),
                'polarity_sentence': round(self.ner_sentence_sentiments.get(node, 0), self.round_to),
                'connections': connections[node],
                'interactions': interactions[node]
            }
        self.nodes = normalize(self.nodes, 'polarity', self.round_to)
        self.nodes = normalize(self.nodes, 'polarity_sentence', self.round_to)
        self.nodes = normalize(self.nodes, 'connections', self.round_to)
        self.nodes = normalize(self.nodes, 'interactions', self.round_to)

    def calculate_centrality_scores(self):
        """Calculate centrality scores for nodes."""
        G = nx.from_edgelist(self.network)
        centrality = nx.eigenvector_centrality(G)
        for node in centrality:
            self.nodes[node]['centrality'] = round(centrality[node], 2)
        degree = nx.degree(G)
        for node, deg in degree:
            self.nodes[node]['degree'] = deg
        betweenness = nx.centrality.betweenness.betweenness_centrality(G)
        for node in betweenness:
            self.nodes[node]['betweenness'] = round(betweenness[node], self.round_to)
        self.nodes = normalize(self.nodes, 'centrality', self.round_to)
        self.nodes = normalize(self.nodes, 'degree', self.round_to)
        self.nodes = normalize(self.nodes, 'betweenness', self.round_to)

    def calculate_sentiment_score(self):
        """Calculate whether or not the doc is positive or negative"""
        sentiment = self.current_doc._.vader_da
        self.current_polarity = sentiment['compound']

    def calculate_sentiments(self):
        """Calculate sentiments for each entity in document.

        This is done by calculating the average sentiment of the sentence
        and adding to score for each entity, multiple times if the
        entity is mentioned multiple times in the sentence."""
        for sent in self.current_doc.sents:
            if not sent.ents: # Only if there are entities
                continue
            sentence_doc = self.nlp(sent.text)
            sentiment = sentence_doc._.vader_da
            for ent in sent.ents:
                self.ner_sentence_sentiments[ent.text] += sentiment['compound']


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info('making final data set from raw data')

    logger.info('Loading model ...')
    nlp = dacy.load("da_dacy_medium_trf-0.1.0", "/Users/nhs/Udvikling/models/dacy")

    if os.path.isdir(input_filepath):
        documents = sorted(glob.glob(os.path.join(input_filepath, f'*.txt')))
    else:
        # Asssume a single document
        documents = [input_filepath,]

    for filepath in documents:
        process_file(nlp, filepath, output_filepath)

def process_file(nlp, input_filepath, output_filepath):
    """Process one file."""
    logger.info('Reading input file ...')
    with open(input_filepath) as f:
        texts = f.readlines()

    network_doc = NetworkDoc(nlp)
    logger.info(f'Processing {input_filepath} ...')
    logger.info('Processing input paragraphs ...')
    with ProgressBar(max_value=len(texts), redirect_stdout=True) as bar:
        for i, doc in enumerate(nlp.pipe(texts, batch_size=1)):
            network_doc.process_doc(doc)
            bar.update(i)

    logger.info('Analyzing nodes ...')
    network_doc.analyze_nodes()
    logger.info('Calculating centrality scores ...')
    network_doc.calculate_centrality_scores()

    logger.info('Saving edges to data/processed/')
    output_filename = f'{Path(input_filepath).stem}_edges.csv'
    with open(Path.joinpath(Path(output_filepath), Path(output_filename)), 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(('source', 'target', 'type', 'weight'))
        for source, target in network_doc.network.keys():
            writer.writerow((source, target, 'Undirected', network_doc.network[(source, target)]))

    logger.info('Saving timestamped edges to data/processed/')
    output_filename = f'{Path(input_filepath).stem}_edges_timestamped.csv'
    with open(Path.joinpath(Path(output_filepath), Path(output_filename)), 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(('source', 'target', 'timestamp'))
        for source, target, timestamp in network_doc.edges_timestamped:
            writer.writerow((source, target, timestamp))

    logger.info('Saving nodes to data/processed/')
    output_filename = f'{Path(input_filepath).stem}_nodes.csv'
    with open(Path.joinpath(Path(output_filepath), Path(output_filename)), 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(('Id',
                         'Polarity', 'Polarity_normalized',
                         'Polarity_sentence', 'Polarity_sentence_normalized',
                         'Connections', 'Connections_normalized',
                         'Interactions', 'Interactions_normalized',
                         'Degree', 'Degree_normalized',
                         'Centrality', 'Centrality_normalized',
                         'Betweenness', 'Betweenness_normalized'))
        for node in network_doc.nodes.values():
            writer.writerow((node['id'],
                             node['polarity'], node['polarity_norm'],
                             node['polarity_sentence'], node['polarity_sentence_norm'],
                             node['connections'], node['connections_norm'],
                             node['interactions'], node['interactions_norm'],
                             node['degree'], node['degree_norm'],
                             node['centrality'], node['centrality_norm'],
                             node['betweenness'], node['betweenness_norm']))


if __name__ == '__main__':

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
