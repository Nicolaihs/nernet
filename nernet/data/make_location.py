import click
import csv
import dacy
import glob
import logging
import os
from dacy.sentiment import da_vader_getter
from spacy.tokens import Span, Doc
from pathlib import Path


from progressbar import ProgressBar
from dotenv import find_dotenv, load_dotenv


log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


class LocationDoc:
    def __init__(self, nlp) -> None:
        self.nlp = nlp
        self.counter = 0
        self.current_doc = None
        self.output = []

    def process_doc(self, doc):
        """Process a doc and retrieve network."""
        self.counter += 1
        self.current_doc = doc
        self.locate_locations()

    def locate_locations(self):
        pers = [ent for ent in self.current_doc.ents if ent.label_ == 'PER']
        loc = [ent for ent in self.current_doc.ents if ent.label_ == 'LOC']
        if loc or pers:
            res = {
                'timestamp': self.counter,
                'pers': pers,
                'loc': loc
            }
        if loc:
            self.output.append(res)
            print('---------------')
            print(f'PERSONER: {pers}')
            print(f'LOKATIONER: {loc}')


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--pattern', type=click.STRING)
def main(input_filepath, output_filepath=None, pattern=None):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info('making final data set from raw data')

    logger.info('Loading model ...')
    nlp = dacy.load("da_dacy_medium_trf-0.1.0", "/Users/nhs/Udvikling/models/dacy")

    if pattern:
        file_pattern = f'{pattern}*.txt'
    else:
        file_pattern = '*.txt'
    filepath = os.path.join(input_filepath, file_pattern)
    logger.info(f'Reading data from: {filepath}')
    if os.path.isdir(input_filepath):
        documents = sorted(glob.glob(filepath))
    else:
        # Asssume a single document
        documents = [input_filepath,]

    for filepath in documents:
        process_file(nlp, filepath, output_filepath=output_filepath)

def process_file(nlp, input_filepath, output_filepath=None):
    """Process one file."""
    logger.info('Reading input file ...')
    with open(input_filepath) as f:
        texts = f.readlines()

    location_doc = LocationDoc(nlp)
    logger.info(f'Processing {input_filepath} ...')
    logger.info('Processing input paragraphs ...')
    with ProgressBar(max_value=len(texts), redirect_stdout=True) as bar:
        for i, doc in enumerate(nlp.pipe(texts, batch_size=1)):
            location_doc.process_doc(doc)
            bar.update(i)
#            if location_doc.output:
#                break

    logger.info('Saving nodes to data/processed/')
    output_filename = f'{Path(input_filepath).stem}_pers.csv'
    with open(Path.joinpath(Path(output_filepath), Path(output_filename)), 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(('timestamp', 'pers'))
        for row in location_doc.output:
            for pers in row['pers']:
                writer.writerow((row['timestamp'], str(pers)))
    output_filename = f'{Path(input_filepath).stem}_loc.csv'
    with open(Path.joinpath(Path(output_filepath), Path(output_filename)), 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(('timestamp', 'loc'))
        for row in location_doc.output:
            for loc in row['loc']:
                writer.writerow((row['timestamp'], str(loc)))


if __name__ == '__main__':

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
