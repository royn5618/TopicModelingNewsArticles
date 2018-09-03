# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 13:07:00 2018

@author: NRoy
"""
import pandas as pd

#NLTK

from nltk.corpus import wordnet as wn

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

#misc
import os
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
#from corpus_creation import df
from pandas import ExcelWriter

def token_corpus_build(data):
    #stopword removal and lemmatisation
    token_corpus = []
    for text in data:
        token_list = []
        for token in gensim.utils.simple_preprocess(text, deacc=True):
            if token not in gensim.parsing.preprocessing.STOPWORDS:
                token_list.append(token)
        token_corpus.append(token_list)
    return token_corpus


def n_gram_generation(data):
    bigram = gensim.models.Phrases(data, min_count=5,
                                   threshold=0.50,
                                   scoring='npmi')
    trigram = gensim.models.Phrases(bigram[data],
                                    threshold=0.70,
                                    scoring='npmi')
    quadgram = gensim.models.Phrases(trigram[bigram[data]],
                                    threshold=0.85,
                                    scoring='npmi')
    bigram_phraser = gensim.models.phrases.Phraser(bigram)
    trigram_phraser = gensim.models.phrases.Phraser(trigram)
    quadgram_phraser = gensim.models.phrases.Phraser(quadgram)
    bi_tri_grammed_corpus = [quadgram_phraser[trigram_phraser[bigram_phraser[text]]] for text in data]
    return bi_tri_grammed_corpus

def lemma_generation(data):
    lemmatized_data = []
    for each_token_list in data:
        lemma = [token if wn.morphy(token) is None else wn.morphy(token) for token in each_token_list]
        lemmatized_data.append(lemma)
    return lemmatized_data

#MAchine Learning for LanguagE Toolkit
def run_mallet_lda(corpus, id2word, num_topics):
    os.environ.update({'MALLET_HOME':r'C:/mallet-2.0.8'})
    mallet_path = 'C:/mallet-2.0.8/bin/mallet'
    mallet_lda_model = gensim.models.wrappers.LdaMallet(mallet_path, 
                                                        corpus = corpus, 
                                                        num_topics = num_topics, 
                                                        id2word = id2word,
                                                        iterations=1000)
    return mallet_lda_model
    
def compute_coherence_values(id2word, corpus, lemmatized_corpus, x):
    print('Computing coherence values')
    coherence_values = []
    models = []
    for num_topics in x:
        model = run_mallet_lda(corpus, id2word, num_topics)
        models.append(model)
        cm = CoherenceModel(model=model, texts=lemmatized_corpus, dictionary=id2word, coherence='u_mass')
        coherence_values.append(cm.get_coherence())
        print('Completed topics ' + str(num_topics))
    return models, coherence_values

def get_optimal_model(id2word, corpus, lemmatized_corpus, x):
    models, coherence_values = compute_coherence_values(id2word, corpus, lemmatized_corpus, x)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.legend(("No. of Topics vs. Coherence Scores"), loc='best')
    plt.show()
    for i, j in zip(x, coherence_values):
        print("Num Topics:", i, "Coherence Value:", round(j, 2))
    #For exploring the topics
#    for model in models:
#        topics_list.append(model.show_topic(topn = 15)) 
    max_cv = max(coherence_values)
    cv_index = coherence_values.index(max_cv)
    return models[cv_index]
    
def get_dominant_topics_df(model, corpus):
    doc_topics_df = pd.DataFrame()
    for doc in model[corpus]:
        #each row ~ document
        doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for n, (num_topic, perc_contrib) in enumerate(doc):
            if n == 0:  # => dominant topic
                temp = model.show_topic(num_topic, topn = 15)
                keywords = "; ".join([topic for topic, perc in temp])
                doc_topics_df = doc_topics_df.append(pd.Series([int(num_topic), round(perc_contrib*100, 2), keywords]), ignore_index=True)
            else:
                break
    doc_topics_df.columns = ['Dominant_Topic', 'Percentage_Contribution', 'Contributing_Keywords']
    return doc_topics_df

def df_to_excel(df, key):
    writer = ExcelWriter('topics_dataset_' + key + '.xlsx')
    df.to_excel(writer)
    writer.save()
    
def begin_process(df, key):
    data = df.Body.values.tolist()
    print('Data procured')
    print(len(data))
    tokenized_corpus = token_corpus_build(data)
    print('Data tokenized')
    print(len(tokenized_corpus))
    bi_tri_grammed_corpus = n_gram_generation(tokenized_corpus)
    print('Data n-grammatized')
    print(len(bi_tri_grammed_corpus))
    lemmatized_corpus = lemma_generation(bi_tri_grammed_corpus)
    print('Data lemmatized')
    print(len(lemmatized_corpus))
    id2word = corpora.Dictionary(lemmatized_corpus) #dictionary
    print('Data dictionarised')
    print(len(id2word))
    corpus = [id2word.doc2bow(text) for text in lemmatized_corpus]
    print('Corpus created')
    print(len(corpus))
    #declare stating number of topics and so on
    #start=10; step=2; limit = 50
    #x = range(start, limit, step)
    #topics_list = []
    #model = get_optimal_model(id2word, corpus, lemmatized_corpus, x)
    #print('Optimal model obtained')
    model = run_mallet_lda(corpus, id2word, num_topics = 5)
    print('LDA mallet model ready')
    topic_list.append([model.print_topics(num_words=15)])
    df_dom_top = get_dominant_topics_df(model, corpus)
    print('Dominant topics dataframe ready')
    df_dom_top['Date'] = list(df.Date)
    df_dom_top['Publication'] = list(df.Publication)
    #topics_df = get_dominant_topics_df(model, corpus)
    #df_to_excel(topics_df, key)
    print('Final dataset created.. Good job!')
    df_to_excel(df_dom_top, key)
    print('Dataset saved to excel. Congratulations!!!')
    

############# LOAD DATASET FROM EXCEL##################
#extracting to list
topic_list = []
df = pd.read_excel('daily_dataset_Topic_Modelling.xlsx') #Output of the corpus_creation code
years = [2013, 2014,2015]
step = 3
for year in years:
    for n, i in enumerate(range(0 , 12 , step)):
        df_temp = df[(df.Month >= (i + 1)) & (df.Month <= (i + step)) & (df.Year == year)]
        print('######################HEAD########################')
        print(df_temp.head())
        print('######################TAIL########################')
        print(df_temp.tail())
        print('Beginning to cluster from ' + str(i+1) + 'to' + str(i+step) + 'of' + str(year))
        key = 'Q' + str(n+1)+ '_' + str(year)
        begin_process(df_temp, key)
print(topic_list)
print('Congratulations!! One step ahead for disseratation...')

###########Topics Details to Excel##############################
#import pandas as pd
#Load new = ''' topics text '''
#df_topics_list = pd.DataFrame()
#for n, i in enumerate(range(1, len(new))):
#    for j in range(1, len(new[i]), 2):
#        item = new[i][j]
#        print(item)
#        item = item.split('+')
#        for each in item:
#            print(each)
#            each_split = each.split('*')
#            print(each_split)
#            df_topics_list = df_topics_list.append(pd.Series([n, each_split[0], each_split[1]]), ignore_index=True)
#df_topics_list.columns = ['Indexing', 'Perc', 'Keyword']