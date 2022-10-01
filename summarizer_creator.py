import numpy as np, pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words='english', max_features=10)
from sklearn.metrics.pairwise import cosine_similarity
# nltk.download('stopwords')
Stop_words = nltk.corpus.stopwords.words('english')

# nltk.download('wordnet') #for importing lemmatizer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# nltk.download('punkt') #for importing tokenizer functionality

import en_core_web_sm  #for part of speech tagging
nlp = en_core_web_sm.load()
from nltk.tokenize import sent_tokenize

def summarizer(text, a_value, scale):
    '''
    Step1: Processing the Input String, that includes the following steps namely,---
    (a) case-lowering, (b) Contraction Removal, (c) Punctuation Removal, (d) Tokenization, (e) Stop Words Removal, (f) Lemmatization
    
    Step2: Selecting the most impactful lines: Six Factors for determining this.
        1. TF-IDF value (sk)
        ----> TF(Term Freq): No. of occurrences of a term in an Article/ Total no. of terms in that article,
        ----> DF(Document Freq): No. of sentences in which the term is present/ Total no. of sentences,
    
        2. Sentence length based Importance (sw) Calculation:--
        ----> The higher/ lower is the no. of words in a functions, higher will be s_rl (sentence relative length), 
        And lower will be the sw value.
        
        3. Importance on the basis of Line position (pk):-- 
        ----> The starting sentences and ending sentences will get the highest importance. 
        And the values gradually decreases, as it gets the  middle-most position.

        4. Sentence similarity score (ss):--
        ----> The first and the last sentence of a piece is the most important lines is an article.
        This factor is to determine the most similar sentences wrt the first and the last sentence.
        cosine_simalrity is calcualated for each sentences with the 1st and the last sentences, and then avg is taken.
         
        5. Importance on the basis of presencen of Proper Noun (pn):--
        ----> Proper nouns are important in any article, as the lines containing them.. generally bears important details
        related to the article. Hence presence of NNP and NNPS are determined.
         
        6. Importance on the basis of presencen of Indicators (in):--
        ----> Indicators are those words or combo of words that generally are used while telling about something imp. or
        while drawing a conclusion. e.g. 'To conclude', 'Finally', 'recently', 'in spite of' etc.
    ---------------------------------------------------------------------------------------------------------------
    
    Step3: ADDing all three determining Parameters to get a aggregate ranking_score
    Step4: Creating the Summary with a constraint in the Word-count ---> some% of word-count of the actual input text.
           Untill the count is exceeded, new sentences are added. If a current sent has more words than the empty space, next sent is considered
    ---------------------------------------------------------------------------------------------------------------------
    '''

    # Processing the Input String---    

    # A dataframe that stores all infos about each line in the String.
    text_df = pd.DataFrame(columns = ['line_no', 'text_string', 'tokens'])

    def text_processor(text):
        #1. lower
        text = text.lower()

        #2. Remove contractions
        replace_map = {"'cause": "because", "'ve": " have", "won't": "will not", "can't": "can not", "n't": " not",
                "'d": " would", "'ll": " will", "let's": "let us", "'s": " is", "I'm": "I am", " i ": " I ", "ma'am": "madam", 
                "o'clock": "of the clock", "'re": " are", "y'all": "you all", "\n":"","\t":"", "no.":"num", 'No.':'num', 
                "etc.":"etc", "e.g.":"eg", 'i.e.':'that is', 'asap': 'as soon as possible', "fyi": "for your information",
                "imho": "in my humble opinion", "omg": "oh my god", "pov": "point of view", "fomo": "fear of missing out",
                "idk": "i do not know", "btw": "by the way", "tbh": "to be honest", "eod": "end of day", "rsvp": "please reply", 
                "lmk": "let me know", "dob": "date of birth", "tba" : "to be announced", "tbc": "to be confirmed", "aka": "also known as"}

        for tbr, trw in replace_map.items(): #to_be_replaced, #to_replace_with
            text = text.replace(tbr, trw)

        #3. Remove punctuation
        regular_punct = list(string.punctuation)
        for punc in regular_punct:
            if punc in text:
                text = text.replace(punc, '')
            text =  text.strip() #Strip Function removes extra spaces from the extream ends (starting and ending)

        #4. Tokenize
        tokenized_text = text.split(' ') #Spliting sentences into words

        #5. Stop Words removal
        new_tokens = [words for words in tokenized_text if words not in Stop_words]
            
        #6. Lemmatization
        final_tokens = [lemmatizer.lemmatize(token, pos = 'v') for token in new_tokens]
        final_tokens = [lemmatizer.lemmatize(token, pos = 'n') for token in final_tokens]

        final_tokens = ['less' if token=='le' else token for token in final_tokens] #due to Noun lemmatization, less --> le, HENCE
        return final_tokens

    line_by_line = sent_tokenize(text) 
    line_by_line = [i.replace('\n', '') for i in line_by_line]


    for index, line in enumerate(line_by_line):
        line_no = index+1
        text_string = line
        tokens = text_processor(text_string)
        while '' in tokens:
            tokens.remove('')
        line_details = [line_no, text_string, tokens]
        text_df.loc[len(text_df.index)] = line_details
#-------------------------------------------------------------------------------------------------    
    #Selecting the most impactful lines:
    
    ## ~~ 1. TF-IDF value (sk)

    unique_tokens = text_processor(text) 
    n_tokens = len(set(unique_tokens)) #there can be duplicate tokens in one long text
    token_df = pd.DataFrame(columns=['Tokens'])
    token_df['Tokens'] = unique_tokens
    token_df = token_df.Tokens.value_counts().reset_index().rename(columns={'index':'Tokens', 'Tokens':'Counts'}) 
            #automatically duplicate tokens are removed here
            
    #==> (i). TF-IDF value Calculation:--
    token_df['TF'] = token_df.Counts/n_tokens

    n_sents = len(text_df)
    doc_freq_df = pd.DataFrame(columns=['Tokens', 'Doc_count'])
    for token in list(token_df.Tokens.values):
        count = 0
        for line in list(text_df.text_string.values):
            if token in line.lower():
                count+=1
        doc_freq_df.loc[len(doc_freq_df.index)] = [token, count]
    
    tf_idf_df = token_df.merge(doc_freq_df, on='Tokens',how='left')
    tf_idf_df['IDF'] = np.log10(list(n_sents/(tf_idf_df.Doc_count+0.1).values))
    tf_idf_df = tf_idf_df[tf_idf_df.Tokens!='']
    tf_idf_df['tf*idf'] = tf_idf_df.TF*tf_idf_df.IDF

    #==> (ii). line-wise sum of TF-IDF value, addition of all tf*idf values of the words present in each sentence.
    tf_idf_sum_list = []
    for i in range(len(text_df)):
        tf_idf_sum = 0
        for token in text_df.iloc[i].tokens:
            tf_idf_sum += tf_idf_df[tf_idf_df.Tokens==token]['tf*idf'].values[0]
        tf_idf_sum_list.append(tf_idf_sum)

    text_df['tf_idf_sum'] = tf_idf_sum_list

    ## ~~ 2. Sentence length based Importance (sw) Calculation

    # Formula used: 
    # L_avg = Avg length of the sentance in a piece of Article
    # L = line of each sentance
    # s_rl = abs(L_avg-L)/L_avg
    # sw = log(1/s_rl) 

    sent_lengths = []
    for line in list(text_df.text_string.values):
        line = line.replace(',','')
        sent_lengths.append(len(line.split(' ')))
    L_avg = round(sum(sent_lengths)/len(sent_lengths))
    s_rl = [abs(L_avg - L)/L_avg for L in sent_lengths]
    sw = [np.log10(1/(rl+0.01)) for rl in s_rl]

    ## ~~ 3. Importance on the basis of Line position (pk):-- 
    
    # Formula used: 
    # mid_pos = the middle most line in the article, 
    #           for odd no. of lines=> length of article/2 ==> returns a float line no., so no values will be zero in the formula
    #           for even no. of lines=>length of article+1/2 ==> returns a float line no.

    # k = Position of a particular line.
    # pk = pow(abs(k-mid_pos),2)/4*a ------> 'a' can be a tunable Factor.''' small value --> first and last line gets more value

    a = a_value #changable
    if len(text_df)%2==0:
        mid_pos = (len(text_df)+1)/2 #EVEN no. of lines in the Article
    else:
        mid_pos = len(text_df)/2

    pk = []
    for line_no in list(text_df.line_no.values):
        k = line_no   
        pk.append(pow((k-mid_pos),2)/(4*a))

    ## ~~ 4. Sentence_Similarity (ss) Calculation:--
    
    line_1 = text_df.text_string[0]
    line_n = text_df.text_string[len(text_df)-1]
    ss_1 = []
    ss_n = []
    def cosine_sim_finder(ref_string, test_string):
        cv.fit(ref_string)
        ref_matrix = cv.transform(ref_string)
        ref_vec = ref_matrix.toarray()

        out_matrix = cv.transform(test_string)
        out_vec = out_matrix.toarray()
        return round(cosine_similarity(ref_vec, out_vec)[0][0],4)

    ## ~~ 5. Proper Noun Presence flag (pn):--
    
    pn_flag = []
    def properNoun_tag_finder(string):
        doc = nlp(string)
        tag_flag = 0
        for token in doc:   
            if (token.tag_ == 'NNP') or (token.tag_ == 'NNPS'):
                tag_flag += 1
        return tag_flag

    ## ~~ 6. Indicator Presence flag (in):--
    
    indicators = ["as", "because", "since", "for", "due to", "in spite of", "in addition", "considering", "owing to", 
                "initially", "additionally", "accordingly", "thus", "as a result", "to conclude", "to draw a conclusion", 
                "in a nutshell", "in brief", "consequently", "we may conclude", "mainly", "recently", "evident", "obvious", 
                "therefore", "finally", "finalized", "overall", "given that"]
    in_flag = []
    
    def indicator_tag_finder(string):
        if any(x in string.lower() for x in indicators):
            return 1
        else:
            return 0

    for line_i in list(text_df.text_string.values):
        ss_1.append(cosine_sim_finder([line_1], [line_i]))
        ss_n.append(cosine_sim_finder([line_n], [line_i]))

        pn_flag.append(properNoun_tag_finder(line_i)/len(text_df))
        in_flag.append(indicator_tag_finder(line_i)/len(text_df))

    mean_ss = [(a+b)/2 for a,b in zip(ss_1, ss_n)]

    # ADDing all three determining Parameters:
    text_df['sent_weight'] = sw
    text_df['positional_weight'] = pk
    text_df['sent_similarity'] = mean_ss
    text_df['has_proper_noun'] = pn_flag
    text_df['has_indicators'] = in_flag

    text_df['Total_weight'] = text_df.tf_idf_sum + text_df.sent_weight + text_df.positional_weight + text_df.sent_similarity + text_df.has_proper_noun + text_df.has_indicators
    imp_text_df = text_df.sort_values(['Total_weight'], ascending=False).reset_index(drop=True)
#---------------------------------------------------------------------------------------------------------------------------
    # Summary Size Selection: ---> 33% word-count of the actual input text.
    scaling_factor = 100/scale

    input_word_cnt = len(text.split(' '))
    summary_word_cnt = round(input_word_cnt/scaling_factor) 

    summary = ''

    l_nos = []
    for index, line in enumerate(list(imp_text_df.text_string.values)):
        if (summary_word_cnt-len(line.split(' ')))>=0: #runs only when summ_word_cnt is stil greater than len(curr line)
            l_nos.append(imp_text_df.iloc[index].line_no)
            summary_word_cnt = summary_word_cnt-len(line.split(' '))
    
    #if they want summary with this many sentences---
    # 
    # q_sents = 6
    # if q_sents<=len(imp_text_df):
    #     l_nos = [imp_text_df.head(q_sents).line_no]

    l_nos = sorted(l_nos)
    for l_no in l_nos:
        summary = summary +'. '+ imp_text_df[imp_text_df.line_no == l_no].text_string.item()
            
    while '..' in summary:
        summary = summary.replace('..', '.')
    summary = summary[2:]
    return summary