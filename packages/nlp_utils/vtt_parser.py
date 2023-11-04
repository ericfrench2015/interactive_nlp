import pandas as pd
import numpy as np
import re
from io import StringIO
import spacy
from textacy import extract

nlp = spacy.load("en_core_web_md")


def categorize_row(row):
    if re.fullmatch(r'\d+', row) is not None:
        return 'statement_index'
    elif re.search(r'\d+\d:', row) is not None:
        return 'time_code'
    else:
        return 'content'


def trim_rows(df):
    while df.iloc[0]['row_category'] != 'statement_index':
        df = df.drop(index=0).reset_index(drop=True)

    while df.iloc[-1]['row_category'] != 'content':
        df = df.drop(index=0).reset_index(drop=True)

    cl = len(df[df['row_category'] == 'content'])
    sl = len(df[df['row_category'] == 'statement_index'])
    tl = len(df[df['row_category'] == 'time_code'])
    if ((cl == sl) and (sl == tl)):
        print('lengths consistent')

    return df





def parse_content(raw_input, row_category):
    #if row_category == 'statement_index':
    x = {'first':raw_input, 'second':'foo'}

    return pd.Series(x)


def remove_filler_phrases(text):
    filler_phrases = [
        "um", "uh", "you know", "i mean", "so", "basically", "actually", "kind of", "well",
        "in terms of", "to be honest", "of course", "at the end of the day", "as i mentioned",
        "in my opinion", "the fact of the matter is", "moving forward", "so on and so forth"
    ]

    # Create a regular expression pattern to match the filler phrases
    pattern = r'\b(?:' + '|'.join(filler_phrases) + r')\b[.,!?]*'

    # Remove filler phrases from the input string
    cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()

    return cleaned_text


def split_raw_input(row):
    x = row.split(":")
    print(f"x is {x}")
    if len(x) == 1:
        output = [np.nan,x[0]]
    elif len(x) == 2:
        output = [x[0],x[1]]
    elif len(x) > 2:
        output = [x[0],' '.join(x[1:])]
    else:
        output = ['','']

    return pd.Series({'name':output[0],'content':output[1]})

def remove_short_sentences(doc, min_len=8):
    sents = []
    for s in doc.sents:
        if len(s) < min_len:
            next
        else:
            sents.append(s)
    return sents


def remove_junk_sentences(doc, min_len=8):
    sents = []
    junk_flag = 0
    for s in doc.sents:
        for t in s:
            if t.text == 'mute':
                junk_flag = 1
        if junk_flag == 0:
            sents.append(s)

    return sents

def extract_entities(doc, ent_type=['GPE']):
    ents = []
    for e in doc.ents:
        if e.label_ in ent_type:
            ents.append(e.text)

    ents = list(set(ents))

    return ents



def parse_vtt_simple2(file):
    #with open(file, "r") as f

    f = StringIO(file.getvalue().decode("utf-8"))
    transcript = [r.replace("\n", "") for r in f]
    return transcript

def parse_vtt_simple(file):

    f = StringIO(file.getvalue().decode("utf-8"))
    transcript = []
    for r in f:
        r = r.replace("\n","")
        r = r.replace("\r", "")

        transcript.append(r)




    df = pd.DataFrame(transcript, columns=['raw_input'])
    df = df[df['raw_input'] != '']
    df['row_category'] = df['raw_input'].apply(categorize_row)
    print(len(df))

    try:
        df = trim_rows(df)
    except:
        print("TODO... see why this is happening")

    df_simple = df[['raw_input']][df['row_category'] == 'content'].copy()
    df_simple[['name', 'content']] = df_simple['raw_input'].apply(split_raw_input)
    print(df_simple['content'].tolist())
    df_simple['name'] = df_simple['name'].ffill()
    df_simple = df_simple[['name', 'content']]
    # Group the DataFrame by contiguous names
    grouped = df_simple['name'].ne(df_simple['name'].shift()).cumsum()

    # Add the group number as a new column
    df_simple['group_number'] = grouped
    df_simple_collapsed = df_simple.groupby(['name', 'group_number'])['content'].agg(', '.join).reset_index()
    df_simple_collapsed = df_simple_collapsed.sort_values(by='group_number').reset_index(drop=True)



    df_simple_collapsed['content'] = df_simple_collapsed['content'].apply(remove_filler_phrases)

    # clean up leading and trailing whitespace
    df_simple_collapsed['content'] = df_simple_collapsed['content'].apply(lambda x: x.strip())
    df_simple_collapsed['nlp'] = df_simple_collapsed['content'].apply(nlp)

    df_simple_collapsed['no_short_sents'] = df_simple_collapsed['nlp'].apply(remove_short_sentences)
    # df_simple_collapsed['no_short_sents_orJunk'] = df_simple_collapsed['no_short_sents'].apply(remove_junk_sentences)d
    df_simple_collapsed['named_locations'] = df_simple_collapsed['nlp'].apply(extract_entities)
    df_simple_collapsed['named_organizations'] = df_simple_collapsed['nlp'].apply(
        lambda x: extract_entities(x, ent_type=['ORG']))

    #drop the nlp column on return to avoid arrow error
    return df_simple_collapsed.drop(columns=['nlp'])







