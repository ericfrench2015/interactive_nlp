import PyPDF2
import pandas as pd
import spacy
import re

nlp = spacy.load("en_core_web_sm")


def get_full_doc_from_pdf(file):

    reader = PyPDF2.PdfReader(file)
    document = reader.pages

    working_text_l = []
    working_text = ''
    for i in range(len(document)):
        working_text += ' ' + document[i].extract_text()
        pg = document[i].extract_text().split("\n")
        for l in pg:
            working_text_l.append(['x', i, l])


    doc = nlp(working_text)
    return doc

def deconstruct_from_pdf(file):

    def collapse_into_paragraphs(working_text_l):
        formatted_paras = []
        current_para = ''
        pg_start = 0

        for r in working_text_l:
            file_name = r[0].name ## .name for actual uploaded files

            pg = r[1]
            l = r[2]

            if len(l) < 2:
                if len(current_para) > 2:
                    formatted_paras.append([file_name, pg_start, current_para.strip()])

                # reset vars
                pg_start = r[1]
                para_separator = 'no'
                current_para = ''
            else:
                current_para += l

        return formatted_paras

    def paragraph_to_sentence(paragraph, split_tokens=".?!"):
        # input paragraph of text
        # output list of sentences

        paragraph = paragraph.replace('\n', '')
        delimiter_pattern = f'[{split_tokens}]'
        sentences = [s.strip() for s in re.split(delimiter_pattern, paragraph) if len(s) > 2]

        return sentences


    reader = PyPDF2.PdfReader(file)
    document = reader.pages

    # load text
    working_text_l = []
    working_text = ''
    for i in range(len(document)):
        working_text += ' ' + document[i].extract_text()
        pg = document[i].extract_text().split("\n")
        for l in pg:
            working_text_l.append([file,i,l])






    formatted_paras = collapse_into_paragraphs(working_text_l)


    df = pd.DataFrame(formatted_paras, columns=['file_name' ,'starting_page_num' ,'paragraph_text'])
    df['paragraph_num'] = df.reset_index().index
    df = df[['file_name' ,'starting_page_num' ,'paragraph_num' ,'paragraph_text']]

    df['sentence_text'] = df['paragraph_text'].apply(paragraph_to_sentence)
    exploded_df = df.explode('sentence_text', ignore_index=True)
    exploded_df = exploded_df.reset_index()
    exploded_df = exploded_df.rename(columns={'index': 'sentence_num'})
    exploded_df['sentence_spacy'] = exploded_df['sentence_text'].apply(nlp)
    exploded_df = exploded_df[['file_name','starting_page_num','paragraph_num','sentence_num','paragraph_text','sentence_text','sentence_spacy']]


    return exploded_df


if __name__ == '__main__':
    test_file = "d:\\projects\\_external_files\\202310_test_docs\\20231010 Libya Humanitarian Update_HC cleared.pdf"
    exploded_df = deconstruct_from_pdf(test_file)
    print(exploded_df)






