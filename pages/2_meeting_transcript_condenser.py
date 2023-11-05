import sys

#sys.path.append('D:\\projects\\interactive_nlp\\packages')

pkg = 'packages'
if pkg not in sys.path:
    sys.path.append('packages')
print(sys.path)

import streamlit as st
import streamlit_analytics
st.set_page_config(layout="wide")

import numpy as np
import pandas as pd
import PyPDF2
import spacy
from textacy import extract
from collections import Counter
import re
import matplotlib.pyplot as plt
from nlp_utils import document_deconstructor as decon
import nlp_utils
from nlp_utils import vtt_parser


nlp = spacy.load("en_core_web_sm")

#https://docnavigator.streamlit.app/

streamlit_analytics.start_tracking()
st.title("Upload a WebVTT file (the transcript you get when you record calls)")

st.write("Instructions: Upload a document, find the sections you're interested in, then download a csv of them.")

def convert_span_to_string(li):
    #return len(li)
    text =''
    for l in li:
        for t in l:
            text = text + t.text + ' '

    return text

def convert_span_to_string_set(li):
    li = ', '.join(list(set(li)))


    return li


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:

    df = vtt_parser.parse_vtt_simple(uploaded_file)
    df['nss_string'] = df['no_short_sents'].apply(convert_span_to_string)
    df['named_locations_str'] = df['named_locations'].apply(convert_span_to_string_set)
    df['named_organizations_str'] = df['named_organizations'].apply(convert_span_to_string_set)


    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')


    csv = convert_df(df)

    st.header("Download the resulting csv for offline analysis.")
    st.write("Or see the key fields you'll download inline below.")
    st.download_button(
        "Download Results",
        csv,
        "file.csv",
        "text/csv",
        key='download-csv'
    )

    st.table(df[['name','nss_string','named_locations_str','named_organizations_str']][df['nss_string'] != ''])

streamlit_analytics.stop_tracking()












    #st.header("Mapping Demo")
    #st.write("""Can pull GPE entities out of the doc and map them... but would need to make calls to
    #a Google API and I don't feel like putting my API key up here right now... So enjoy this random
    #map.""")
    #map_data = pd.DataFrame(
    #    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    #    columns=['lat', 'lon'])

    #st.map(map_data, use_container_width=False)

