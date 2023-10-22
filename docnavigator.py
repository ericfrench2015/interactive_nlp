import streamlit as st
st.set_page_config(layout="wide")

import numpy as np
import pandas as pd
import PyPDF2
import spacy
from textacy import extract
from collections import Counter
import re
#import matplotlib
#import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")

#https://docnavigator.streamlit.app/


st.title("Document Analyzer POC (PDF Only for now)")

st.write("Instructions: Upload a document, find the sections you're interested in, then download a csv of them.")


#map_data = pd.DataFrame(
#    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
#    columns=['lat', 'lon'])

#st.map(map_data)


from io import StringIO

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    #bytes_data = uploaded_file.getvalue()
    #st.write(bytes_data)

    # To convert to a string based IO:
    #stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #st.write(stringio)

    # To read file as string:
    #string_data = stringio.read()
    #st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    #dataframe = pd.read_csv(uploaded_file)
    #st.write(dataframe)

    reader = PyPDF2.PdfReader(uploaded_file)
    document = reader.pages

    # load text
    working_text_l = []
    working_text = ''
    for i in range(len(document)):
        working_text += ' ' + document[i].extract_text()
        pg = document[i].extract_text().split("\n")
        for l in pg:
            working_text_l.append(['x', i, l])

    doc = nlp(working_text)


    st.header("First lets do some quick profiling of the doc")

    #get top 10 ngrams...just because
    counter = Counter()
    x = list(extract.ngrams(doc, 2))
    x_txt = []
    for i in x:
        x_txt.append(i.text.lower())
    counter = Counter(x_txt)
    df_bigrams = pd.DataFrame(counter.most_common(10),columns=['bigram','count'])
    df_bigrams.set_index('bigram', inplace=True)
    try:
        st.pyplot(df_bigrams.sort_values(by='count').plot.barh(title='most common bigrams').figure, use_container_width=False)
    except:
        st.subheader("error generating matplotlib chart... enjoy a dataframe instead!")
        df_bigrams






    st.header("Now you can search for specific words or phrases and see them all in one place.")
    st.write("""You can select one word to search for, put in an arbitrary word, or copy the whole line below.
    If you want to search multiple terms at once, separate them with a pipe '|'
    "If you want to ensure you match just that word and not a part of it, add a space.
    eg. 'differ' matches 'differ' and 'different', 'differ ' just matches 'differ '.""")

    st.write("if you're familiar with regular expressions, you can use those too.")


    strong_indicators = ['due to', 'because', 'lead to', 'leads to', 'result of', 'caused by', 'therefore', 'thus',
                         'thereby']
    possible_indicators = ['as', 'why', 'which', 'since', 'after']

    copy_paste_strong = ' |'.join(strong_indicators)
    copy_paste_possible =  ' |'.join(possible_indicators)

    st.text(f"suggestion for terms that are strong indicators: {copy_paste_strong}")
    st.text(f"suggestion for terms that are possible indicators: {copy_paste_possible}")

    search_text = st.text_input('Word or short phrase of interest', 'due to')
    st.header('Results of search:', search_text)
    st.write("""What you are seeing is the results of a search of the full document for the term(s) you input.
    The formatting aligns your search term(s) in the middle column, with the a text snippet directly before and
    after the term in question so you can get a sense of the context.""")

    search_term = re.compile(search_text)

    matched_terms = list(extract.keyword_in_context(doc, search_term, window_width=60, pad_context=True))
    df = pd.DataFrame(matched_terms, columns=['text_before_term','search_term','text_after_term'])
    df
    #https://docs.streamlit.io/library/api-reference/data/st.dataframe



    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')


    csv = convert_df(df)

    st.header("Download the resulting sentence fragments to csv for offline analysis.")
    st.download_button(
        "Download Results",
        csv,
        "file.csv",
        "text/csv",
        key='download-csv'
    )

    st.header("Mapping Demo")
    st.write("""Can pull GPE entities out of the doc and map them... but would need to make calls to 
    a Google API and I don't feel like putting my API key up here right now... So enjoy this random
    map.""")
    map_data = pd.DataFrame(
        np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
        columns=['lat', 'lon'])

    st.map(map_data, use_container_width=False)