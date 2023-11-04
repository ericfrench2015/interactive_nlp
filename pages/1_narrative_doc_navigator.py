import sys

#sys.path.append('D:\\projects\\interactive_nlp\\packages')

pkg = 'packages'
if pkg not in sys.path:
    sys.path.append('packages')
print(sys.path)

import streamlit as st
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


nlp = spacy.load("en_core_web_sm")

#https://docnavigator.streamlit.app/


st.title("Document Analyzer POC (PDF Only for now)")

st.write("Instructions: Upload a document, find the sections you're interested in, then download a csv of them.")


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:

    exploded_df = decon.deconstruct_from_pdf(uploaded_file)
    doc = decon.get_full_doc_from_pdf(uploaded_file)

    #exploded_df[['starting_page_num' ,'paragraph_num' ,'paragraph_text','sentence_text']]
    st.header("First lets do some quick profiling of the doc")

    col1, col2, col3 = st.columns(3)

    counter = Counter()
    x = list(extract.ngrams(doc, 2))
    x_txt = []
    for i in x:
        x_txt.append(i.text.lower())
    counter = Counter(x_txt)
    df_bigrams = pd.DataFrame(counter.most_common(10),columns=['bigram','count'])
    df_bigrams.set_index('bigram', inplace=True)
    try:
        #st.pyplot(df_bigrams.sort_values(by='count').plot.barh(title='most common bigrams').figure, use_container_width=False)
        #https://discuss.streamlit.io/t/cannot-change-matplotlib-figure-size/10295/8
        with col1:
            #st.write("DataFrame 1")
            #st.write(df1)
            st.subheader("Top 10 most common 2 word sequences.")
            df_bigrams
    except Exception as error:
        print(error)
        st.subheader("error generating matplotlib chart... have a dataframe instead!")
        df_bigrams

    #get top 10 entities

    elist=[]
    for e in doc.ents:
        if e.label_ in ['GPE','ORG','NORP','PERSON']:
            elist.append(e.text)
            #st.write(e)
    counter_ents = Counter(elist)

    df_ents = pd.DataFrame(counter_ents.most_common(10), columns=['entities', 'count'])
    with col2:
        st.subheader("Top 10 most common places, people, and organizations in the doc.")
        df_ents


    svos = extract.subject_verb_object_triples(doc)
    svo_list = []
    for svo in svos:
        subjects, verbs, objects = svo

        subject = ' '.join([x.text for x in subjects])
        verb = ' '.join([x.text for x in verbs])
        object = ' '.join([x.text for x in objects])

        svo_list.append([subject, verb, object])
    df_svos = pd.DataFrame(svo_list, columns=['subject', 'verb', 'object'])

    with col3:
        st.subheader("Subject/Verb/Object combinations.")
        df_svos




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

    st.write("The following are some terms that may indicate CAUSAL statements.")
    st.text(f"strong indicators: {copy_paste_strong}")
    st.text(f"possible indicators: {copy_paste_possible}")

    search_text = st.text_input('Word or short phrase of interest', copy_paste_strong)
    st.header('Results of search:', search_text)
    st.write("""What you are seeing is the results of a search of the full document for the term(s) you input.
    The formatting aligns your search term(s) in the middle column, with the a text snippet directly before and
    after the term in question so you can get a sense of the context.""")

    search_term = re.compile(search_text)

    #matched_terms = list(extract.keyword_in_context(doc, search_term, window_width=60, pad_context=True))
    #df = pd.DataFrame(matched_terms, columns=['text_before_term','search_term','text_after_term'])
    #df

    #https://docs.streamlit.io/library/api-reference/data/st.dataframe

    #VERSION 2
    def get_kwic(doc, search_term):
        l = list(extract.keyword_in_context(doc, search_term, window_width=60, pad_context=True))
        if len(l) > 0:
            ll = l[0]
            return pd.Series({'search_left': ll[0], 'search_term': ll[1], 'search_right': ll[2]})
        else:
            return pd.Series({'search_left': None, 'search_term': None, 'search_right': None})


    exploded_df[['search_left', 'search_term', 'search_right']] = exploded_df.apply(
        lambda x: get_kwic(x['sentence_spacy'], search_term), axis=1)

    df_show = exploded_df[['search_left', 'search_term', 'search_right']][exploded_df['search_term'].isnull() == False]
    st.dataframe(df_show, width=1000)












    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')


    csv = convert_df(exploded_df)

    st.header("Download the resulting sentence fragments to csv for offline analysis.")
    st.download_button(
        "Download Results",
        csv,
        "file.csv",
        "text/csv",
        key='download-csv'
    )

    #st.header("Mapping Demo")
    #st.write("""Can pull GPE entities out of the doc and map them... but would need to make calls to
    #a Google API and I don't feel like putting my API key up here right now... So enjoy this random
    #map.""")
    #map_data = pd.DataFrame(
    #    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    #    columns=['lat', 'lon'])

    #st.map(map_data, use_container_width=False)

