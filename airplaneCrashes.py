# -*- coding: utf-8 -*-
"""
Created on Tue May 15 16:38:28 2018

@author: s149545
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
import plotly
import plotly.graph_objs as go
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

sns.set_style('whitegrid')

weeklist = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
monthlist = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep','Oct', 'Nov', 'Dec']
state_codes = {
    'District of Columbia' : 'dc','Mississippi': 'MS', 'Oklahoma': 'OK', 
    'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 
    'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 
    'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 
    'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 
    'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 
    'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 
    'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI',
    'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 
    'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 
    'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 
    'Alabama': 'AL', 'Rhode Island': 'RI','South Dakota': 'SD', 
    'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 
    'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 
    'Nevada': 'NV', 'Maine': 'ME'}
state_names =  {v: k for k, v in state_codes.items()}

# Get data and remove NA if few NA
df = pd.read_csv('Airplane_Crashes_and_Fatalities_Since_1908.csv')
sns.heatmap(df.isnull(), yticklabels=False, cbar=False)
plt.show()
df.drop('Flight #', axis= 1, inplace=True)
df = df[df.loc[:,'Aboard'].notnull()]
df = df[df.loc[:,'Location'].notnull()]

# Create df with city and country/state split

def lengthcheck(x) :
    if len(x) == 2:
        return x[1] 
    elif len(x) == 3:
        return x[2]
    else:
        return None
    
df2 = df['Location'].apply(lambda x: x.split(', '))
df['City'] = df2.apply(lambda x: x[0])
df['Country/State'] = df2.apply(lengthcheck)
df = df[df.loc[:,'Country/State'].notnull()]


# Clean the countries and states
def codes2names(x):
    if x in list(state_names.keys()):
        return state_names[x]
    else:
        return x

def countryfun(x):
    if x in list(state_codes.keys()):
        return 'United States'
    else:
        return x

df['Country/State'] = df['Country/State'].apply(codes2names)
df['Country/State'] = df['Country/State'].apply(lambda x: x.lstrip().rstrip())
df['Country/State'].replace('Ilinois','Illinois', inplace = True)
df['Country/State'].replace('Massachutes','Massachusetts', inplace = True)
df['Country/State'].replace('Cailifornia','California', inplace = True)
df['Country/State'].replace('Wisconson','Wisconsin', inplace = True)
df['Country/State'].replace('Washington DC','Washington', inplace = True)
df['Country/State'].replace('Washingon','Washington', inplace = True)
df['Country/State'].replace('Virginia.','Virginia', inplace = True)
df['Country/State'].replace('Alaksa','Alaska', inplace = True)
df['Country/State'].replace('Alakska','Alaska', inplace = True)
df['Country/State'].replace('Amsterdam','Netherlands', inplace = True)
df['Country/State'].replace('Azores (Portugal)','Azores', inplace = True)
df['Country/State'].replace('Azores','Portugal', inplace = True)
df['Country/State'].replace('Inodnesia','Indonesia', inplace = True)
df['Country/State'].replace('Azores','Portugal', inplace = True)
df['Country/State'].replace('Russian','Russia', inplace = True)

df['State'] = df[df['Country/State'].apply(lambda x: x in list(state_codes.keys()))]['Country/State']
df['State'].fillna('None', inplace=True)
df['State'] = pd.Categorical(df['State'], list(state_codes.keys()))

df['Country'] = df['Country/State'].apply(countryfun)


# Add columns with time, date, years, hours, months, weekdays
df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y'))
df['Date'] = df['Date'].apply(lambda x: x.date())

df['Time'] = df['Time'].apply(lambda x: str(x).replace("c:", ""))
df['Time'] = df['Time'].apply(lambda x: str(x).replace("c", ""))
df['Time'] = df['Time'].apply(lambda x: str(x).replace("'", ":"))
df['Time'] = df['Time'].apply(lambda x: str(x).replace(".", ":"))
df['Time'] = df['Time'].apply(lambda x: str(x).replace("114", "14"))
df['Time'] = df['Time'].apply(lambda x: str(x).replace("0943", "09:43"))

df['Time'] = df['Time'].apply(lambda x: x.lstrip())
df['Time'] = df[df.loc[:,'Time'] != 'nan']['Time'].apply(lambda x: datetime.strptime(str(x), '%H:%M'))
df['Time'] = df[df.loc[:,'Time'].notnull()]['Time'].apply(lambda x: x.time())
df['Hour'] = df[df['Time'].notnull()]['Time'].apply(lambda x: x.hour)

df['Weekday'] = df['Date'].apply(lambda x: x.weekday())
df['Weekday'] = df['Weekday'].map({0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'})
df['Weekday'] = pd.Categorical(df['Weekday'], weeklist)

df['Month'] = df['Date'].apply(lambda x: x.month)
df['Month'] = df['Month'].map({1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'})
df['Month'] = pd.Categorical(df['Month'], monthlist)
df['Year'] = df['Date'].apply(lambda x: x.year)

# Visualize updated df
sns.heatmap(df.isnull(), yticklabels=False, cbar=False)
plt.show()

# Visualize some stuff on Fatalities
df2 = df.groupby(df['Year']).sum()
plt.figure(figsize=(10,5))
plt.bar(df2.index,df2['Aboard'])
plt.bar(df2.index,df2['Fatalities'], color='red')
plt.legend(['Aboard','Fatalities'])
plt.xlabel('Year')
plt.ylabel('People')
plt.show()

df2 = df.groupby(df['Month']).sum()
plt.figure(figsize=(10,5))
sns.barplot(df2.index,df2['Aboard'], color='blue',label='Aboard')
sns.barplot(df2.index,df2['Fatalities'], color='red', label='Fatalities')
plt.legend()
plt.xlabel('Month')
plt.ylabel('People')
plt.show()

df2 = df.groupby(df['Weekday']).sum()
plt.figure(figsize=(10,5))
sns.barplot(df2.index,df2['Aboard'], color='blue',label='Aboard')
sns.barplot(df2.index,df2['Fatalities'], color='red', label='Fatalities')
plt.legend()
plt.xlabel('Weekday')
plt.ylabel('People')
plt.show()

df2 = df.groupby(df['Hour']).sum()
plt.figure(figsize=(10,5))
plt.bar(df2.index,df2['Aboard'])
plt.bar(df2.index,df2['Fatalities'], color='red')
plt.legend(['Aboard','Fatalities'])
plt.xlabel('Hour')
plt.ylabel('People')
plt.show()

perState = df.groupby(df['State'], as_index=False).count()
perState['Code'] = perState['State'].apply(lambda x : state_codes[x])

geodata = dict(type = 'choropleth',
            locations = perState['Code'],
            locationmode = 'USA-states',
            colorscale= 'Reds',
            text = perState['State'],
            z= perState['Date'],
            colorbar = {'title':'Crashes'},
            marker = dict(
                line = dict (
                    color = 'rgb(180,180,180)',
                    width = 0.5
                )))
geolayout = dict(geo = dict(scope='usa', showlakes = True, lakecolor='rgb(230, 255, 247)', showcoastlines = False))

choromap = go.Figure(data = [geodata],layout = geolayout)
plotly.offline.plot(choromap, filename="USmap.html")
plt.show()

df['Country'].replace('Soviet Union','Russia', inplace = True)
df['Country'].replace('USSR','Russia', inplace = True)

perCountry = df.groupby(df['Country'], as_index=False).count()

geodata = dict(type = 'choropleth',
            locations = perCountry['Country'],
            locationmode = 'country names',
            colorscale= 'Reds',
            z= perCountry['Date'],
            colorbar = {'title':'Crashes'},
            marker = dict(
                line = dict (
                    color = 'rgb(180,180,180)',
                    width = 0.5
                )))
geolayout = dict(geo = dict(showlakes = True, lakecolor='rgb(230, 255, 247)', showcoastlines=False))

choromap = go.Figure(data = [geodata],layout = geolayout)
plotly.offline.plot(choromap, filename="Worldmap.html")
plt.show()


# Natural Language Processing

def goodtext(x):
    tagged_list = []
    words = nltk.word_tokenize(x)
    tagged = nltk.pos_tag(words)
    for word,pos in tagged:
        if re.match('NN|NNS|JJ.?',pos):
            tagged_list.append(word.lower()) 
    return tagged_list
#   nopunctuation = [c for c in x if c not in string.punctuation.replace('-','')]
#   nopunctuation = ''.join(nopunctuation).lower()
#   goodtext = [word for word in nopunctuation.split() if word not in set(stopwords.words('english'))]
#   try:
#       goodtext = [ps.stem(word) for word in goodtext]
#   except: 
#       pass
#   return goodtext

dfSummary = df[df.loc[:,'Summary'].notnull()]

from sklearn.feature_extraction.text import CountVectorizer
ps = PorterStemmer()
BOW_transformer = CountVectorizer(analyzer=goodtext).fit(dfSummary['Summary'])
summaryBOW = BOW_transformer.transform(dfSummary['Summary'])

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(summaryBOW)
summaryTFIDF = tfidf_transformer.transform(summaryBOW)

from sklearn.decomposition import TruncatedSVD
clf = TruncatedSVD(5)
words_pca = clf.fit_transform(summaryTFIDF)

import scipy.cluster.hierarchy as sch

dendogram = sch.dendrogram(sch.linkage(words_pca,method='ward'))
plt.xlabel('Index')
plt.ylabel('Euclidean distance')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=4, linkage='ward')
yhc = hc.fit_predict(words_pca)

trace1 = go.Scatter3d(
    x=words_pca[:,0],
    y=words_pca[:,1],
    z=words_pca[:,2],
    mode='markers',
    marker=dict(
        size=6,
        color=yhc,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
    )
)
 
layout = go.Layout(
    margin=dict(
        l=0, r=0, b=0, t=0
    )
)
figPCA = go.Figure(data=[trace1], layout=layout)
plotly.offline.plot(figPCA, filename="PCA.html")
plt.show()

dfSummary['cluster'] = yhc
         
df0 = dfSummary[dfSummary['cluster']==0]
BOW0 = CountVectorizer(analyzer=goodtext).fit(df0['Summary'])
summaryBOW0 = BOW0.transform(df0['Summary'])
tfidf0 = TfidfTransformer().fit(summaryBOW0)
summaryTFIDF0 = tfidf0.transform(summaryBOW0)
sum_words0bow = summaryBOW0.sum(axis=0)
sum_words0tfidf = summaryTFIDF0.sum(axis=0)
words_freq0 = [(word, sum_words0tfidf[0, idx],sum_words0bow[0, idx]) for word, idx in BOW0.vocabulary_.items()]
words_freq0 =sorted(words_freq0, key = lambda x: x[1], reverse=True)

df1 = dfSummary[dfSummary['cluster']==1]
BOW1 = CountVectorizer(analyzer=goodtext).fit(df1['Summary'])
summaryBOW1 = BOW1.transform(df1['Summary'])
tfidf1 = TfidfTransformer().fit(summaryBOW1)
summaryTFIDF1 = tfidf1.transform(summaryBOW1)
sum_words1bow = summaryBOW1.sum(axis=0)
sum_words1tfidf = summaryTFIDF1.sum(axis=0)
words_freq1 = [(word, sum_words1tfidf[0, idx],sum_words1bow[0, idx]) for word, idx in BOW1.vocabulary_.items()]
words_freq1 =sorted(words_freq1, key = lambda x: x[1], reverse=True)


df2 = dfSummary[dfSummary['cluster']==2]
BOW2 = CountVectorizer(analyzer=goodtext).fit(df2['Summary'])
summaryBOW2 = BOW2.transform(df2['Summary'])
tfidf2 = TfidfTransformer().fit(summaryBOW2)
summaryTFIDF2 = tfidf2.transform(summaryBOW2)
sum_words2bow = summaryBOW2.sum(axis=0)
sum_words2tfidf = summaryTFIDF2.sum(axis=0)
words_freq2 = [(word, sum_words2tfidf[0, idx],sum_words2bow[0, idx]) for word, idx in BOW2.vocabulary_.items()]
words_freq2 =sorted(words_freq2, key = lambda x: x[1], reverse=True)

df3 = dfSummary[dfSummary['cluster']==3]
BOW3 = CountVectorizer(analyzer=goodtext).fit(df3['Summary'])
summaryBOW3 = BOW3.transform(df3['Summary'])
tfidf3 = TfidfTransformer().fit(summaryBOW3)
summaryTFIDF3 = tfidf3.transform(summaryBOW3)
sum_words3bow = summaryBOW3.sum(axis=0)
sum_words3tfidf = summaryTFIDF3.sum(axis=0)
words_freq3 = [(word, sum_words3tfidf[0, idx],sum_words3bow[0, idx]) for word, idx in BOW3.vocabulary_.items()]
words_freq3 =sorted(words_freq3, key = lambda x: x[1], reverse=True)


wordsdf0 = pd.DataFrame(words_freq0[0:15], columns=['Word','Importance','Count']) 
wordsdf1 = pd.DataFrame(words_freq1[0:15], columns=['Word','Importance','Count']) 
wordsdf2 = pd.DataFrame(words_freq2[0:15], columns=['Word','Importance','Count']) 
wordsdf3 = pd.DataFrame(words_freq3[0:15], columns=['Word','Importance','Count']) 

# What cluster has the most fatalities proportional to aboard?
fatal_prop0 = dfSummary[dfSummary['cluster']==0]['Fatalities'].mean()/dfSummary[dfSummary['cluster']==0]['Aboard'].mean()
fatal_prop1 = dfSummary[dfSummary['cluster']==1]['Fatalities'].mean()/dfSummary[dfSummary['cluster']==1]['Aboard'].mean()
fatal_prop2 = dfSummary[dfSummary['cluster']==2]['Fatalities'].mean()/dfSummary[dfSummary['cluster']==2]['Aboard'].mean()
fatal_prop3 = dfSummary[dfSummary['cluster']==3]['Fatalities'].mean()/dfSummary[dfSummary['cluster']==3]['Aboard'].mean()
