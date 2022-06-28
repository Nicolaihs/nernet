import glob
import os
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

import numpy as np
import pydeck as pdk

from geopy import distance


from tools3.ws_utilities import call_word2dict

INPUT_DIR = '../../data/processed/memo_loc'

#input = st.text_input('Search for')
#if input:
##    sims = call_word2dict(input)
 #   st.write(sims)


def load_location(input_dir,):
    """Load all location files from from input_dir."""

    filepath = os.path.join(input_dir, '*.paragraphed_loc.csv')

    all_books = {}
    for i, f in enumerate(sorted(glob.glob(filepath))):
        # Get filename of glob
        filename = os.path.basename(f)
        book_name = filename.split('.paragraphed_loc.csv')[0]
        df_loc = pd.read_csv(f)
        all_books[book_name] = {
            'name': book_name,
            'loc': df_loc
        }
    return all_books


@st.cache(persist=True)
def lookup_location(query):
    """Get best location matches from query."""
    BASE_URL = 'https://api.dataforsyningen.dk/stednavne2'

    response = requests.get(f"{BASE_URL}?q={query}")
    results = response.json()
    best = [row for row in results if row['navn'].lower() == query.lower() and row.get('brugsprioritet') == 'primær']
    return best


def get_best_location(places):

    counties = [place for place in places if place.get('sted', {}).get('undertype') == 'landsdel']
    cities = [place for place in places if place.get('sted', {}).get('undertype') == 'by' and place['sted']['egenskaber']['indbyggerantal'] > 100]
    bydel = [place for place in places if place.get('sted', {}).get('undertype') == 'bydel']
    roads = [place for place in places if place.get('sted', {}).get('hovedtype') == 'Vej']
    buildings = [place for place in places if place.get('sted', {}).get('undertype') == 'andenBygning']

    if counties:
        return counties[0]
    elif cities:
        return cities[0]
    elif bydel:
        return bydel[0]
    elif roads:
        return roads[0]
    elif buildings:
        return buildings[0]
    return None


def create_locations(df_loc):
    locations = []
    memory = []
    for _, row in df_loc.iterrows():
        places = lookup_location(row['loc'])
        place = get_best_location(places)
        if place:
            lon, lat = place['sted']['visueltcenter']
            subtype = place['sted']['undertype']
            locations.append((lat, lon, place['navn'], subtype, row['timestamp']))
            memory.append(place['navn'])

    return locations


def analyze_from_point(locations, point, radius=5000):
    counts = 0
    for location in locations:
        dist = distance.distance(point, (location[0], location[1]))
        if dist < radius:
            counts += 1
    return counts


def display_centre_metrics(locations, centres):
    for center, point, radius in centres:
        count = analyze_from_point(locations, point, radius)
        cols = st.columns(2)
        cols[0].metric(f'Locations in {center}', count)
        cols[1].metric(f'Pct. of all', round(100*count/len(locations), 1))

def draw_map(locations):
    df = pd.DataFrame(locations,
        columns=['lat', 'lon', 'name', 'type', 'timestamp'])

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=55.40,
            longitude=11.35,
            zoom=6,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
            'HexagonLayer',
            data=df,
            get_position='[lon, lat]',
            radius=1000,
            elevation_scale=4,
            elevation_range=[0, 1000],
            pickable=True,
            extruded=True,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=df,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=2000,
        ),
        ],
    ))


def main():
    st.title('LOCATION INVESTIGATOR')
    all_books = load_location(INPUT_DIR)
    with st.sidebar:
        st.title('Emerging cities')
        st.image('emerging_city.jpg')
        selected_books = st.multiselect('Select one or more books', all_books.keys())

    if not selected_books:
        st.dataframe(pd.DataFrame(all_books.keys()))
    else:
        for book in selected_books:
            book = all_books[book]
            st.header(book['name'])
            cols = st.columns(2)
            cols[0].metric('No. of locations', len(book['loc']))
            locations = create_locations(book['loc'])
            cols = st.columns(2)
            cols[0].metric('No. of found geolocations', len(locations))
            cols[1].metric('No. of different geolocations', len(set([loc[2] for loc in locations])))
            centres = (
                ('København', (55.70552403, 12.56284182), 5),
            )

            max_time = max([loc[4] for loc in locations])
            locations_1 = [loc for loc in locations if loc[4] < max_time / 3]
            locations_2 = [loc for loc in locations if loc[4] < 2*max_time / 3 and loc[4] >= max_time/3]
            locations_3 = [loc for loc in locations if loc[4] >= 2*max_time/3]

            display_centre_metrics(locations, centres)
            st.header('I')
            display_centre_metrics(locations_1, centres)
            st.header('II')
            display_centre_metrics(locations_2, centres)
            st.header('III')
            display_centre_metrics(locations_3, centres)

            show_locations = st.checkbox('Show raw locations', key='ch_loc')
            if show_locations:
                st.dataframe(book['loc'])
            if st.checkbox('Show identified geolocations', key='cb_geo'):
                st.dataframe(pd.DataFrame(locations))

            draw_map(locations)

            st.header('I')
            draw_map(locations_1)
            st.header('II')
            draw_map(locations_2)
            st.header('III')
            draw_map(locations_3)

if __name__ == "__main__":
    main()