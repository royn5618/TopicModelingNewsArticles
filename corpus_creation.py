# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 19:09:43 2018

@author: NRoy
"""
import datetime
from nltk.tag import StanfordNERTagger
import os
import pandas as pd
from pandas import ExcelWriter

#Reading Corpus

def split_documents(all_doc):
    #print('splitting docs')
    docs_split = all_doc.split('DOCUMENTS')
    del docs_split[0]
    for each in docs_split:
        read_article(each)
        
def get_article_date(date):
    if '\n' in date:
        date = date.split('\n')[0]
    date = date.strip()
    date = datetime.datetime.strptime(date, '%B %d, %Y %A').date()
    return date

def read_article(article):
    #print('splitting article')
    article_split = article.split('\n\n')
    
    #strip the empty and beginning of body
    begin_marker = ['LENGTH: ', 'BYLINE: ', 'SECTION: ', 'DATELINE: ']
    for marker in begin_marker:
        for each_split in article_split:
            if marker in each_split or each_split is '':
                article_split.remove(each_split)
    
    #storing the properties of each news
    publication = article_split[0].strip().replace('\n','')
    date = get_article_date(article_split[1])
    title = article_split[2].strip()
    
    #extract the list of the body, append whole body, remove \n and strip from end
    i = 3
    body = []
    while ('LOAD-DATE' not in article_split[i]):
        body.append(article_split[i])
        i = i + 1
    body = ' '.join(body)
    body =  body.replace('\n', ' ')
    end_marker = ['For Reprint Rights:', 'Published by HT Syndication']
    for each_sep in end_marker:
        if each_sep in body:
            body = body.split(each_sep, 1)[0]
    df.loc[len(df.index)] = [publication, date, title, body, date.month, date.year]
    
    #RAKE implementation
#    from rake_nltk import Rake
#    r = Rake(language='english',min_length=2, max_length=4)
#    r.extract_keywords_from_text(body)
#    temp = r.get_ranked_phrases_with_scores()
#    print(temp[:10])
    
def get_daily_texts_df(df):
    daily_df = pd.DataFrame()
    grouped_df = df.groupby(['Date'])
    list_of_dates = list(set(df['Date']))
    list_of_dates.sort()
    for each_date in list_of_dates:
        temp_df = grouped_df.get_group(each_date)
        all_text = ' '.join(temp_df['Body'])
        daily_df = daily_df.append(pd.Series([each_date, all_text]), ignore_index=True)
    daily_df.columns = ['Date', 'Body']
    return daily_df

def df_to_excel(df):
    writer = ExcelWriter('daily_dataset_Topic_Modelling.xlsx')
    df.to_excel(writer)
    writer.save()

def load_corpus():
    path = '$/Dissertation/Corpus_final'
    for root, dirs, files in os.walk(path, topdown=True):  
        if not files:
            continue
        else:
            for file in files:
                full_path = root+ '/' + file
                print(full_path)
                text = open(full_path, encoding="utf8").read()
                split_documents(text)
                
df = pd.DataFrame(columns=['Publication', 'Date', 'Title', 'Body', 'Month', 'Year'])
load_corpus()
#df = get_daily_texts_df(df)
df_to_excel(df)
print(df.tail())
