import glob
import os
import pandas as pd
import plotly.express as px
import streamlit as st
from collections import defaultdict

INPUT_DIR = '/Users/nhs/Arkiv/korpus/txt/MeMo/no_pages'

st.title('Count Stuff')

@st.cache(persist=True)
def count_this(input_dir: str, countables: list) -> list:
    """Load all documents from input_dir."""
    filepath = os.path.join(input_dir, '*.paragraphed.txt')

    books = []

    for i, f in enumerate(sorted(glob.glob(filepath))):
        # Get filename of glob
        filename = os.path.basename(f)
        book_name = filename.split('.paragraphed.txt')[0]
        year = book_name[:4]
        book_name = book_name[5:]

        counts = defaultdict(int)

        # Load file
        with open(f, 'r') as doc_file:
            for line in doc_file:
                line = line.lower()
                for countable in countables:
                    counts[countable] += line.count(countable)

                # Count number of words in line
                counts['words'] += len(line.split())

        record = {
            'year': year,
            'name': book_name,
            'counts': counts
        }
        books.append(record)

    return books


def calculate_stats(books: list, countables: list) -> dict:
    """Calculate statistics for books."""
    stats = {}

    for book in books:
        year = book['year']
        if year not in stats:
            stats[year] = {}
            stats[year]['counts'] = defaultdict(int)

        for countable in countables:
            stats[year]['counts'][countable] += book['counts'][countable]
            stats[year]['counts']['total'] += book['counts'][countable]
            stats[year]['counts']['words'] += book['counts']['words']

    for year in stats:
        stats[year]['avg'] = {}
        stats[year]['avg']['total'] = stats[year]['counts']['total'] / stats[year]['counts']['words']
        for countable in countables:
            stats[year]['avg'][countable] = stats[year]['counts'][countable] / stats[year]['counts']['words']

    return stats


MAX_SIZE = 10

def clear_form():
    """Clear input fields."""

    for i in range(MAX_SIZE):
        st.session_state[f'input{i}'] = ""


countables = []
with st.sidebar:
    size = st.slider('Number of countables', 1, MAX_SIZE, 3)
    boxes = ['' for i in range(size)]
    with st.form(key='countables'):
        for i in range(0, size):
            boxes[i] = st.text_input(f'Count this ({i+1})', key=f'input{i}')
        cols = st.columns(2)
        submit = cols[0].form_submit_button('Count')
        clear = cols[1].form_submit_button('Clear', on_click=clear_form, help="Clear all input fields")
    if submit:
        countables = list(set([box.lower() for box in boxes if box]))
#    if clear:
#        boxes = ['' for i in range(size)]

if len(countables) > 0:
    books = count_this(INPUT_DIR, countables=countables)
    stats = calculate_stats(books, countables=countables)

    table = []
    for year in stats:
        row = [year,]
        row.append(stats[year]['avg']['total'])
        for countable in countables:
            row.append(stats[year]['avg'][countable])
        table.append(row)
    df = pd.DataFrame(table, columns=['year', 'total'] + countables)

    for countable in countables:
        fig = px.line(df, x=df.year, y=df[countable])
        fig.update_layout(
            title=f"Average number of '{countable}' per word in MeMo",
        )
        st.plotly_chart(fig)

    st.header('Trendlines for all counts')
    fig = px.scatter(df, x=df.year, y=df.columns[2:], trendline='ols')
    fig.update_layout(
        title=f"Trendlines for all counts",
    )
    st.plotly_chart(fig)

    fig = px.scatter(df, x=df.year, y=df['total'], trendline='ols')
    fig.update_layout(
        title=f"All counts combined",
    )
    st.plotly_chart(fig)
else:
    st.header('Instructions')
    st.markdown('''
    * Enter the strings you want to count in the sidebar.
    * Move slider to add more input fields for countables
    * Examples:
      * 'københavn' / 'kjøbenhavn' to study the trend for these words
      * '.', '?', '!' to count punctuation
      * 'england', 'frankrig', 'amerika' to observe the declining trend. Where do the novels move to?
    ''')