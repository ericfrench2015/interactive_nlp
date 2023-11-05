import sys

#sys.path.append('D:\\projects\\interactive_nlp\\packages')
#?analytics=on

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


nlp = spacy.load("en_core_web_sm")

#https://docnavigator.streamlit.app/
streamlit_analytics.start_tracking()

st.title("Document Processing Utilities")

st.write("This app serves as a platform to experiment make analysts lives easier, one burdensome task at a time.")
st.write("Philosophically, it's intended to simplify common tasks, but not replace human intervention.")



st.subheader("Inventory of Utilities")
st.write(" - narrative doc navigator: To help you quickly identify and view sentences matching certain criteria")
st.write("- meeting transcript condenser: To reformat of transcripts, strip out probable noise, and extract Named Entities"
         " for easy searching.")


streamlit_analytics.stop_tracking()

