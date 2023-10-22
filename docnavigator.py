import streamlit as st
st.set_page_config(layout="wide")

import numpy as np
import pandas as pd
import PyPDF2
import spacy
from textacy import extract

nlp = spacy.load("en_core_web_sm")


chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)


map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)


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

    df_f = pd.DataFrame(list(extract.keyword_in_context(doc, "as", window_width=120, pad_context=True)))
    df_f