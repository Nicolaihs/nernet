import glob
import os
import re
import pandas as pd
import plotly.express as px
import streamlit as st
from collections import defaultdict
from nernet.visualization.corpus_utils import create_corpus, query_corpus
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode


INPUT_DIR = '/Users/nhs/Arkiv/korpus/txt/MeMo/no_pages'
st.set_page_config(page_title="Simple Graphs", layout="wide")

st.title('Simple Graphs')

@st.cache(persist=True)
def count_this(input_dir: str, countables: list, regex_countables: list) -> list:
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

                for countable in regex_countables:
                    counts[countable] += len(re.findall(countable, line))

                # Count number of words in line
                counts['words'] += len(line.split())

        record = {
            'filename': filename,
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
        stats[year]['avg']['total'] = stats[year]['counts']['total'] / stats[year]['counts']['words'] * 1000
        for countable in countables:
            stats[year]['avg'][countable] = stats[year]['counts'][countable] / stats[year]['counts']['words'] * 1000

    return stats


MAX_SIZE = 10

def clear_form():
    """Clear input fields."""
    for i in range(MAX_SIZE):
        st.session_state[f'input{i}'] = ""
    for i in range(MAX_SIZE):
        st.session_state[f'regex_input{i}'] = ""


if 'countables' not in st.session_state:
    st.session_state['countables'] = []
if 'regex_countables' not in st.session_state:
    st.session_state['regex_countables'] = []

with st.sidebar:
    st.header('Simple strings')
    size = st.slider('Number of input fields', 1, MAX_SIZE, 2, key='size')
    boxes = ['' for i in range(size)]
    for i in range(0, size):
        boxes[i] = st.text_input(f'Count this ({i+1})', key=f'input{i}')

    st.header('Regular expressions')
    regex_size = st.slider('Number of input fields', 1, MAX_SIZE, 2, key='regex_size')
    regex_boxes = ['' for i in range(regex_size)]
    for i in range(0, regex_size):
        regex_boxes[i] = st.text_input(f'Count this ({i+1})', key=f'regex_input{i}')

    cols = st.columns(2)
    submit = cols[0].button('Count')
    cols[1].button('Clear', on_click=clear_form, help="Clear all input fields")
    if submit:
        st.session_state.countables = list(set([box.lower() for box in boxes if box]))
        st.session_state.regex_countables = list(set([box.lower() for box in regex_boxes if box]))

if len(st.session_state.countables) > 0 or len(st.session_state.regex_countables) > 0:
    books = count_this(INPUT_DIR,
                       countables=st.session_state.countables,
                       regex_countables=st.session_state.regex_countables)
    all_countables = st.session_state.countables + st.session_state.regex_countables
    stats = calculate_stats(books, countables=all_countables)

    table = []
    for year in stats:
        row = [year,]
        row.append(stats[year]['avg']['total'])
        for countable in all_countables:
            row.append(stats[year]['avg'][countable])
        table.append(row)
    df = pd.DataFrame(table, columns=['year', 'total'] + all_countables)

    for countable in all_countables:
        fig = px.line(df, x=df.year, y=df[countable])
        fig.update_layout(
            title=f"Number of '{countable}' per 1,000 word in MeMo",
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

    st.header('Counts by books')
    st.markdown('''
    * Select one or more years to see counts for these specific years. If no years are selected, counts for all years are shown.
    * The maximum value in the selection is marked by yellow.
    * Sort by a column by clicking on the header.''')
    book_stats = []
    for book in books:
        row = [book['year'], book['name'], book['counts']['words']]
        total = 0
        for countable in all_countables:
            row.append(book['counts'][countable])
            total += book['counts'][countable]
        for countable in all_countables:
            row.append(book['counts'][countable] / book['counts']['words'] * 1000)
        row.append(total)
        row.append(total / book['counts']['words'] * 1000)
        row.append(book['filename'])
        book_stats.append(row)

    selected_years = st.multiselect('Select years', list(stats.keys()))
    df_books = pd.DataFrame(book_stats, columns=['year', 'name', 'no of words'] + all_countables + [f'"{cc}" per 1,000' for cc in all_countables] + ['total', 'total per 1,000', 'filename'])
    if selected_years:
        df_books = df_books[df_books.year.isin(selected_years)]
    df_books.sort_values(by='total per 1,000', ascending=False, inplace=True)
    st.dataframe(df_books.style.highlight_max(axis=0))

    st.header('On the fly concordance creator')
    st.markdown('Select one or more books to create concordances.')
    gb = GridOptionsBuilder.from_dataframe(df_books)
    gb.configure_pagination()
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gridOptions = gb.build()
    selection = AgGrid(df_books, gridOptions=gridOptions,
                           enable_enterprise_modules=True,
                           allow_unsafe_jscode=True,
                           update_mode=GridUpdateMode.SELECTION_CHANGED)
    selected_files = [row['filename'] for row in selection['selected_rows']]

    for filename in selected_files:
        st.markdown(f'### {filename}')
        concordancer = create_corpus(INPUT_DIR, filename)

        for countable in st.session_state.countables:
            st.markdown(f'#### {countable}')
            conc = query_corpus(concordancer, countable, width=30)
            conc_txt = ''
            for sentence in conc:
                left = ' '.join(sentence.left)
                right = ' '.join(sentence.right)
                conc_txt += f'{left.rjust(50)} {sentence.query} {right.ljust(50)}\n'
            st.text(conc_txt)

else:
    st.subheader('Count strings by year')
    st.header('Instructions')
    st.markdown('''
    * Enter the strings you want to count in the sidebar.
    * **Note**: The strings are not word-aware. The string 'hus' will also match 'husk', and 'husar'. Use regular expressions to better control of words.
    * Move slider to add more input fields for countables
    * Examples:
      * 'københavn' / 'kjøbenhavn' to study decrease of the "kj" form
      * '.', '?', '!' to count development of punctuation
      * 'england', 'frankrig', 'amerika' to observe the declining trend. Where do the novels move to?
      * 'han', 'hun' to observe the increase in the relative frequency of 'hun'. Are female characters becoming fashionable?
    ''')