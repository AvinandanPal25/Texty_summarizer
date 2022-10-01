# Importing Packages:---
from summarizer_creator import summarizer #the function to create the summary...
import streamlit as st
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words='english', max_features=10)
# nltk.download('stopwords')
Stop_words = nltk.corpus.stopwords.words('english')
# nltk.download('wordnet') #for importing lemmatizer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer() 
# nltk.download('punkt') #for importing tokenizer functionality
import en_core_web_sm  #for part of speech tagging
nlp = en_core_web_sm.load()
from nltk.tokenize import sent_tokenize
#------------------------------------------------------------------------------------------------------
# loading reqd. csv files:---

df_25 = pd.read_csv('word_cnt_division wise a values__25%.csv').drop('Unnamed: 0', axis=1)
df_33 = pd.read_csv('word_cnt_division wise a values__33%.csv').drop('Unnamed: 0', axis=1)
df_all = pd.read_csv('word_cnt_division wise a values__all.csv').drop('Unnamed: 0', axis=1)
#------------------------------------------------------------------------------------------------------
st.set_page_config(page_title='Extractive Summarizer', page_icon=None, layout='wide', initial_sidebar_state='collapsed')

header = st.beta_container()
about = st.beta_container()
text_box = st.beta_container()
details = st.beta_container()
result = st.beta_container()

with header:
    st.markdown("<h1 style='text-align: center; color:#AD2225;'>Welcome to 'TEXTY' Summarizer APP!!!</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color:#225117;'><b>Generate an Extractive Summary of desired word_length for your Article</b> </h1>", unsafe_allow_html=True)

with about:
    st.write('-------------------------------------------')
    st.markdown(""" #### ** (1). Enter/ Type your article in the text area below. Press <code>_Clt+Enter_</code> to apply the changes.**""", unsafe_allow_html=True)
    st.markdown(""" #### **(2). Choose Summary length, and summary type, (3). Get the Summary...**""")

with text_box:
    article = st.text_area(label="", height= 300)
    line_by_line = sent_tokenize(article) 
    line_by_line = [i.replace('\n', '').strip() for i in line_by_line]
    article = '. '.join(line_by_line)
    inp_word_cnt = len(article.split(' ')) # no of words in the article entered by the user

with details:
    col1, col2, col3, col4, col5= st.beta_columns((3,5,3,6,2))

with result:
    r_col1, r_col2, r_col3 = st.beta_columns((3,10,3))

if article != '': #only when some text is entered
    col2.markdown("### ** No. of Words in the article**")
    col2.info(f"# **{inp_word_cnt}**")
    summary_length = col4.selectbox("Specify Summary Length with respect to your article:-- ", options=['select from drop-down', 'One-third', 'One-fourth', 'Some other Value of your Choice'], index=0)
    if summary_length == 'One-third':
        sum_len = 100/3
        df = df_33
    elif summary_length == 'One-fourth':
        sum_len = 100/4
        df = df_25
    elif summary_length == 'Some other Value of your Choice':
        sum_len = col4.number_input("Specify in percentage (between 10% and 75%)", min_value = 10, max_value = 75, value = 50, step = 10)
        df = df_all

    if summary_length != 'select from drop-down':
        user_opinion = col4.radio("According to YOU, The first and the last sentences in an article are:--", options=['more important than other sentences', 'all have similar importances'], index = 0)
        if user_opinion == 'more important than other sentences':
            if inp_word_cnt<300:
                a = df[df.word_cnt_division == '<300'].iloc[1]['a value']
            elif inp_word_cnt<500:
                a = df[df.word_cnt_division == '<500'].iloc[1]['a value']
            elif inp_word_cnt<800:
                a = df[df.word_cnt_division == '<800'].iloc[1]['a value']
            else:
                a = df[df.word_cnt_division == '>=800'].iloc[1]['a value']
        else:
            if inp_word_cnt<300:
                a = df[df.word_cnt_division == '<300'].iloc[0]['a value']
            elif inp_word_cnt<500:
                a = df[df.word_cnt_division == '<500'].iloc[0]['a value']
            elif inp_word_cnt<800:
                a = df[df.word_cnt_division == '<800'].iloc[0]['a value']
            else:
                a = df[df.word_cnt_division == '>=800'].iloc[0]['a value']
        #------------------------------------------------------------------------------------
        # Summarization
        summary = summarizer(article, a, sum_len) #summary is created

        with result:
            temp_space = col4.empty()
            do_summarize = temp_space.button("!!! Generate Summary !!!")
            if do_summarize:
                details.write('------')
                r_col2.title('Summary:--')
                r_col2.success(summary)
                r_col2.markdown('##### **.. you can change the summary length if needed ..**')
                
                
                summary_word_cnt = len(summary.split(' ')) # no of words in the summary generated
                col2.markdown(f" ### ** No. of Words in the summary**")
                col2.info(f"# **{summary_word_cnt}**")
                temp_space.empty()
            