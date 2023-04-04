import streamlit as st
import tiktoken
import openai
from openai.embeddings_utils import distances_from_embeddings, get_embedding, cosine_similarity
import backoff  

from tqdm import tqdm
import spacy
from spacymoji import Emoji
import string
from sklearn.feature_extraction.text import CountVectorizer
import umap
import hdbscan
import operator

import mysql.connector
from sqlalchemy import create_engine,exc, MetaData, Table, Column, Integer, String,DateTime,Float,Text
import http.client, urllib.request, urllib.parse, urllib.error

import pandas as pd
import numpy as np
import os
from datetime import date, datetime, timedelta
import time
from itertools import islice
import re
import pickle

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import chart_studio
import chart_studio.tools as tls
import chart_studio.plotly as py

pd.options.mode.chained_assignment = None

openai.organization = st.secrets['openaiorg'] 
openai.api_key = st.secrets['openaikey']

# @st.cache_resource
def connect_bdd():
    
    HOST=st.secrets['HOST']
    DATABASE=st.secrets['DATABASE']
    PORT=st.secrets['PORT']
    USER=st.secrets['USER']
    PASSWORD=st.secrets['PASSWORD']
    engine = create_engine('mysql+mysqlconnector://'+USER+':'+PASSWORD+'@'+HOST+':'+PORT+'/'+DATABASE, 
                                   echo=False,
                                   encoding="utf8")
    return engine
    
    
@st.cache_data
def query_fb(_engine,importation_ids,start,end):
    #on prépare notre requête SQL
    sql_fb='''
    SELECT 
        p.id as id, 
        p.uniq_id as uniq_id, 
        p.social_media as social_media, 
        p.created_time as created_time, 
        p.url as post_url, 
        p.author_id as author_id, 
        p.author_name as author_name, 
        p.text as text, 
        p.share_count as shares, 
        p.comments_count as comments,
        p.total_reactions_count as reactions, 
        p.likes_count as likes,
        p.fb_toplevel_comments_count as top_comments,
        p.fb_reactions_love_count as love,
        p.fb_reactions_care_count as care,
        p.fb_reactions_wow_count as wow,
        p.fb_reactions_haha_count as haha,
        p.fb_reactions_sad_count as sad,
        p.fb_reactions_angry_count as angry,
        p.fb_reactions_sad_count + p.fb_reactions_angry_count as negative_reactions,
        p.fb_reactions_love_count + p.fb_reactions_wow_count + p.fb_reactions_care_count as positive_reactions,
        p.fb_reactions_haha_count as haha_reactions,
        p.fb_reactions_care_count + p.fb_reactions_angry_count+p.fb_reactions_sad_count+p.fb_reactions_love_count + p.fb_reactions_wow_count + p.fb_reactions_haha_count as total_reactions_sauf_likes
    FROM post p
    WHERE p.importation_id IN ('''+parse_ids(importation_ids)+''')
    AND p.created_time BETWEEN \"'''+start+'''\" AND \"'''+end+'''\"'''
    
    df=execute_sql(_engine,sql_fb)
    return df


@st.cache_data
def execute_sql(_engine,sql):
    """
    Cette fonction permet d'executer une requête SQL et de retourner un dataframe

    """
    res=_engine.execute(sql)
    df = pd.DataFrame(res.fetchall())
    if df.empty :
        raise Exception("Le rapport généré est vide.")
    df.columns = res.keys()
    return df

@st.cache_data
def parse_ids(l_ids):
    """
    Cette fonction permet de préparer une liste d'IDs numériques pour une requête SQL

    """
    ids=''
    for p in l_ids:
        ids=ids+','+str(p)
    ids= ids.strip(',')  
    return ids 

###########################################################################################
# UTILS
###########################################################################################

@st.cache_data
def load_pickle(projet):
    with open(os.path.join('.',projet,'data.pkl'), 'rb') as f:
        df = pickle.load(f)
    return df

@st.cache_data
def write_pickle(df, projet):
    with open(os.path.join('.',projet,'data.pkl'), 'wb') as f:
        pickle.dump(df, f)
    return df


@st.cache_data
def create_dir(path):
    """
    Cette fonction permet de créer un répertoire sur votre PC, en fournissant un chemin

    """
    if not os.path.exists(path):
        os.makedirs(path)
    print(path, "- répertoire créé")
    return path

def save_dataframe_csv(df,dir_csv,projet,name):
    """
    Cette fonction permet de sauvegarder un dataframe au format CSV, dans un répertoire projet

    """
    names=df.columns
    df.to_csv(os.path.join(dir_csv,projet+"_"+name+".csv"), header=names, sep=';',encoding='utf-8',index=False, decimal="," )
    print("FICHIER SAUVEGARDE : ",os.path.join(dir_csv,projet+"_"+name+".csv"))

###########################################################################################
# TRAITEMENT DE DONNEES
###########################################################################################
    
    
@st.cache_data
def preprocess(df, days):
    df['total_engagement']=df['comments']+df['shares']+df['reactions']
    df['replies']=df['comments']-df['top_comments']
    df['percentage_replies']=df['replies']/df['comments']
    df['percentage_replies']=df['percentage_replies'].fillna(0)

    # on supprime les doublons a posteriori (pourrait être fait dans la requête SQL), on a quelques posts en doublons entre nos 2 importations
    df.drop_duplicates(subset="uniq_id",inplace=True)
    df.dropna(subset=["author_id"], inplace=True)

    #calcul de la distribution des reactions
    cols_fb_reactions=['negative_reactions', 'positive_reactions', 'haha_reactions']
    cols_fb_reactions_detailed=['care', 'love','wow','haha','sad','angry']
    df['top_reaction']=df[cols_fb_reactions].apply(lambda row: row.idxmax() if row.sum() > 0 else 'no_reactions', axis=1)
    df['top_reaction_detailed']=df[cols_fb_reactions_detailed].apply(lambda row: row.idxmax() if row.sum() > 0 else 'no_reactions', axis=1)

    # on calcule nos performances moyennes pour une liste de métriques
    cols=['comments','shares','reactions','care','love','wow','haha','sad','angry','likes']

    df= avg_performance(df, cols, days)  

    # on calcule les taux de sur-réaction pour notre liste de métriques
    col_sur_engagement=['tx_'+col for col in cols]
    df=kpi_reaction(df, cols)
    df[col_sur_engagement]=df[col_sur_engagement].fillna(-1)

    # on supprime nos colonnes contenant la performance moyenne (on ne devrait plus en avoir besoin)
    cols = [c for c in df.columns if c.lower()[:4] != 'avg_']
    df=df[cols]

    # on catégorise les formes de réaction
    cols_fb=['comments','shares','reactions']
    cols_fb_sur=['tx_'+col for col in cols_fb]

    df=get_reactions_type(df, cols_fb_sur, "categorie_engagement")
    df['text']=df['text'].fillna(' ')

    return df

@st.cache_data
def avg_performance(df, cols, days):
    """
    La fonction prend en entrée un dataframe et une liste d'indicateurs.
    Elle retourne un dataframe contenant la performance hebdo moyenne des indicateurs)
    """
    df_final=pd.DataFrame()
    # pour chaque ID d'auteur
    for author in list(df['author_id'].unique()):
        # on sélectionne les lignes correspondantes
        d=df[df['author_id'] == author]
        # on transforme la date en supprimant les heures / min / sec
        d['date']=d['created_time'].dt.date
        # pour chaque indicateur de notre liste
        for col in cols:
            avg_perf=[]
            for i, row in tqdm(d.iterrows(), total=d.shape[0], desc="ID : "+author+" calcul des moyennes sur "+str(days)+" jours - "+col):
                # on détermine la période de 7 jours flottantes 
                delta=row['date']-timedelta(days=days)
                #on calcule la perf moyenne de ces 7 jours
                avg_perf.append(d[(d["date"]>delta) & (d["date"]<=row['date'])][col].mean())
            # on ajoute notre perf à notre dataframe
            d['avg_'+col]=avg_perf
        #on concatène
        df_final=pd.concat([df_final,d])
    return df_final.drop(columns='date')

@st.cache_data
def get_reactions_type(df, cols, col_dest):
    all_val=[]
    
    for i,row in tqdm(df.iterrows(), total=df.shape[0], desc="qualification des posts"):
        str_val=''
        count=0
        for col in cols:
            if row[col]>0:
                str_val=str_val+' '+col.replace('tx_', 'sur-')
                count=count+1
        if count==0:
            str_val="sous reaction"
        if count==len(cols):
            str_val="sur reaction totale"
        all_val.append(str_val.strip())
            
    df[col_dest]=all_val       
    return df

@st.cache_data
def kpi_reaction(df, cols):
    """
    Cette fonction prend un dataframe et une liste de colonnes en entrée.
    Pour chaque colonne, on va calculer le taux de sur-réaction.
    """
    for col in cols:
        df['tx_'+col]=(df[col]-df['avg_'+col])/(df[col]+df['avg_'+col])
    return df

    
###########################################################################################
# OPENAI
###########################################################################################

@st.cache_resource
def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch
        


@st.cache_resource
def chunked_tokens(text, encoding_name, chunk_length):
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    chunks_iterator = batched(tokens, chunk_length)
    yield from chunks_iterator
    
@st.cache_data
def len_safe_get_embedding(text,  model='text-embedding-ada-002', max_tokens=8191, encoding_name="cl100k_base", average=True):
    rate_limit_per_minute = 3500
    delay = 60.0 / rate_limit_per_minute

    chunk_embeddings = []
    chunk_lens = []
    for chunk in chunked_tokens(text, encoding_name=encoding_name, chunk_length=max_tokens):
        chunk_embedding = query_embeddings(delay, chunk, model)
#         chunk_embedding = query_embeddings_backoff(chunk, model)
        chunk_embeddings.append(chunk_embedding)
        chunk_lens.append(len(chunk))

    if chunk_embeddings and average:
        chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)  # normalizes length to 1
        chunk_embeddings = chunk_embeddings.tolist()
    return chunk_embeddings

@st.cache_data
def query_embeddings(delay,text,MODEL_EMBEDDING):
    time.sleep(delay)
    response = openai.Embedding.create(input=text,engine=MODEL_EMBEDDING)
    return response['data'][0]['embedding']

# @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
# def query_embeddings_backoff(text,MODEL_EMBEDDING):
#     response = openai.Embedding.create(input=text,engine=MODEL_EMBEDDING)
#     return response['data'][0]['embedding']

def search_reviews(df, prompt, n=3, pprint=True, engine='text-embedding-ada-002'):
    question_embedding = get_embedding(prompt,engine)
    print(len(question_embedding))
    print(len(df['embeddings'][0]))
    df["similarity"] = df.embeddings.apply(lambda x: cosine_similarity(x, question_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
    )
    if pprint:
        for i, r in results.iterrows():
            print(r.text)
            print()
    return results

###########################################################################################
# DATAVIZ
###########################################################################################   

@st.cache_data
def scatter3D(df,size_dots, colors,titre, col_categorie, col_x, col_y, col_z, x_title, y_title, z_title):
    
    df["color"]=colors
    fig=go.Figure()
    
    for i,s in enumerate(df[col_categorie].unique()):
        df_viz = df[df[col_categorie] == s]

        fig.add_trace(go.Scatter3d( 
                            x=df_viz[col_x], 
                            y=df_viz[col_y], 
                            z=df_viz[col_z],
                            
                            mode='markers',
                            name=s,
                            marker_color=df_viz['color'],
                            marker_size=size_dots,
                            hovertemplate = 
                                    '<b>#ID '+df_viz["uniq_id"].astype(str)+'</b><br>'+
                                    'Auteur : '+df_viz["author_name"].astype(str)+'<br>'+
                                    df_viz["short_text"].astype(str)+'<br>'+
                                    'Réactions : '+df_viz["reactions"].astype(str)+'<br>'+
                                    'Comments : '+df_viz["comments"].astype(str)+'<br>'+
                                    'Shares : '+df_viz["shares"].astype(str)+'<br>', 

        ))


    camera = dict(
        up=dict(x=1, y=0, z=2),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=2, y=1.25, z=0.5)
    )


    fig.update_layout(title=titre,
        scene = dict(
            xaxis_title=x_title,
            yaxis_title=y_title,
            zaxis_title=z_title,
            xaxis = dict(nticks=5, range=[-1,1],),
            yaxis = dict(nticks=5, range=[-1,1],),
            zaxis = dict(nticks=5, range=[-1,1],),
        
        ),
                      
        width=1000,
        height=1000,
        legend=dict(
        title="Type de réaction",
        orientation="h",
        yanchor="bottom",
        y=-0.08,
        xanchor="center",
        x=0.5,
        itemsizing= 'constant'
        )
    )

    fig.update_traces(marker=dict(line=dict(width=1,color='white')),selector=dict(mode='markers'))
    return fig
    
@st.cache_data
def scale_values(values, min_size, max_size):
    min_value = min(values)
    max_value = max(values)
    new_min = 0
    new_max = 1
    scaled_values = []
    for value in values:
        scaled_value = ((value - min_value) / (max_value - min_value)) * (new_max - new_min) + new_min
        scaled_values.append(round(scaled_value * (max_size - min_size) + min_size,0))
    return scaled_values

@st.cache_data
def cr_comparison(df1,df2):
        df1=df1[["uniq_id","total_engagement","created_time"]]
        df1['date']=df1['created_time'].dt.strftime('%Y-%m-%d')

        df2=df2[["uniq_id","total_engagement", "created_time"]]
        df2['date']=df2['created_time'].dt.strftime('%Y-%m-%d')

        df1_gb=df1.groupby("date").agg({"uniq_id":"nunique", "total_engagement":"sum"})
        df2_gb=df2.groupby("date").agg({"uniq_id":"nunique", "total_engagement":"sum"})

        merged_df=pd.merge(df1_gb,df2_gb, how='left', on="date", suffixes=('_all', '_subset')).reset_index()
        return merged_df

@st.cache_data
def barplot_comparison(merged_df, x, y_bar1, y_bar2, color_bar1, color_bar2, title, xtitle, ytitle):
   
    fig = go.Figure(data=[
        go.Bar(name='All', x=merged_df[x], y=merged_df[y_bar1], marker_color=color_bar1),
        go.Bar(name='Sujet', x=merged_df[x], y=merged_df[y_bar2], marker_color=color_bar2)
    ])

    fig.update_layout(
        title=title,
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        barmode='overlay'
    )
    return fig



# @st.cache_data
# def resonance_compared(df1,df2):
    
#     df1=df1[["total_engagement","created_time"]]
#     df1['date']=df1['created_time'].dt.strftime('%Y-%m-%d')
    
#     df2=df2[["total_engagement","created_time"]]
#     df2['date']=df2['created_time'].dt.strftime('%Y-%m-%d')
    
#     df1_gb=df1.groupby("date").agg({"total_engagement":"sum"})
#     df2_gb=df2.groupby("date").agg({"total_engagement":"sum"})
    
#     merged_df=pd.merge(df1_gb,df2_gb, how='left', on="date", suffixes=('_all', '_subset'))
#     merged_df['%_engagement']=(merged_df['total_engagement_subset']/merged_df['total_engagement_all'])*100
    
#     fig = go.Figure(data=[
#         go.Bar(name='All', x=merged_df.index, y=merged_df['total_engagement_all'], marker_color='silver'),
#         go.Bar(name='Sujet', x=merged_df.index, y=merged_df['total_engagement_subset'], marker_color='steelblue')
#     ])

#     fig.update_layout(
#         title='Résonance du sujet',
#         xaxis_title='Date',
#         yaxis_title='#Résonance',
#         barmode='overlay'
#     )
    
    return fig

@st.cache_data
def couverture_resonance(df):
    
    df['date']=df['created_time'].dt.strftime('%Y-%m-%d')
    df=df.groupby(["date"]).agg({'uniq_id':'nunique','total_engagement':'sum'}).reset_index() 
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=df['date'].unique(), y=df['total_engagement'], name="#engagements",mode='lines',line=dict(color='indianred', width=4)),
        secondary_y=True,
    )
    # Add traces
    fig.add_trace(
        go.Bar(x=df['date'].unique(), y=df['uniq_id'], name="#posts",marker_color='lightpink',opacity=0.6),
        secondary_y=False,

    )

    # Add figure title
    fig.update_layout(
        title_text="Publication & Résonance globale",xaxis_tickangle=0
    )

    # Set x-axis title
    fig.update_xaxes(title_text="date")

    # Set y-axes titles
    fig.update_yaxes(title_text="#Résonance", secondary_y=False)
    fig.update_yaxes(title_text="#Publications", secondary_y=True)  
    
    return fig

@st.cache_resource
def generate_wordcloud(text, n, width=3000, height=1500, dpi=300):
    # Create a CountVectorizer object
    vectorizer = CountVectorizer()

    # Fit and transform the text data using the CountVectorizer
    word_count = vectorizer.fit_transform(text)

    # Get the count of each word
    word_freq = dict(zip(vectorizer.get_feature_names_out(), word_count.sum(axis=0).tolist()[0]))

    # Get the top n words by frequency
    top_n_words = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[:n])

    # Generate a wordcloud of the top n words
    wordcloud = WordCloud(width=3000, height=1500, background_color='white').generate_from_frequencies(top_n_words)

    # Plot the wordcloud
    plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    return plt, top_n_words

@st.cache_data
def scatter_clustering(df, x, y, hover_data, size, color, title):
    fig_clustering = px.scatter(df,x=x, y=y, hover_data=hover_data, size=size, color=color, title=title )   
    fig_clustering.update_layout( width=1000,height=1000)
    return fig_clustering


##############################################################################################
# NLP
##############################################################################################

@st.cache_resource
def load_spacy(spacy_model="fr_core_news_sm"):
    nlp=spacy.load(spacy_model,disable=["attribute_ruler","entity_ruler","sentencizer", "textcat"])
    nlp.add_pipe("emoji", first=True)
    return nlp

@st.cache_data
def tokenize_text(spacy_model, df, col_text,col_tokens, pos_to_keep, batch_size, n_process):
    nlp=load_spacy(spacy_model)
#     nlp=spacy.load("fr_core_news_sm",disable=["attribute_ruler","entity_ruler","sentencizer", "textcat"])
#     nlp.add_pipe("emoji", first=True)
    lemmas, all_emojis=[],[]
    # n_process=2 charge 2x le modèle (2 coeurs de processeurs nécessaires) pour paralléliser les traitements 
    for doc in tqdm(nlp.pipe(df[col_text].astype('unicode').values,                               
                        batch_size=batch_size,                                                               
                        n_process=n_process)):   
           
        if doc.has_annotation("DEP"):
            #on parcoure chaque mot du document
            current_lemmas=''
            emojis=[]     
            for token in doc:
                if token.pos_ in pos_to_keep:
                    lemme=token.lemma_.lower().translate(str.maketrans('', '', string.punctuation+'’'))
                    current_lemmas=current_lemmas+' '+lemme
                if token._.is_emoji:
                    emojis.append(str(token))
            lemmas.append(current_lemmas.strip())
            all_emojis.append(emojis)

        else:
            lemmas.append(None)
            all_emojis.append(None)
    df[col_tokens] = lemmas
    df["emojis"]=all_emojis
    
    return df


@st.cache_data
def process_umap(feat, n_neighbors, n_components, min_dist, metric):
    
    
    UMAP_obj = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist,
                                   metric=metric,random_state=42).fit(feat)
    embeddings=UMAP_obj.embedding_
    
    return embeddings
    
@st.cache_data  
def hdbscan_clustering(embeddings, algorithm='best', alpha=1.0, cluster_selection_epsilon=0.0, approx_min_span_tree=True,
                       gen_min_span_tree=True, leaf_size=40, metric='euclidean', min_cluster_size=5, min_samples=None,
                       p=None, cluster_selection_method='eom', prediction_data = True):
    
    
    clusterer = hdbscan.HDBSCAN(algorithm=algorithm, alpha=alpha, cluster_selection_epsilon=cluster_selection_epsilon, 
                                approx_min_span_tree=approx_min_span_tree,gen_min_span_tree=gen_min_span_tree,
                                leaf_size=leaf_size,metric=metric,min_cluster_size=min_cluster_size, 
                                min_samples=min_samples,p=p,cluster_selection_method=cluster_selection_method,
                                prediction_data = prediction_data)

    clusterer.fit(embeddings)
    
    return clusterer,clusterer.labels_,clusterer.probabilities_

@st.cache_data
def word_count(corpus, top_keywords):
    vectorizer = CountVectorizer()
    word_count_matrix = vectorizer.fit_transform(corpus)

    # Get the word-to-index dictionary from the vectorizer
    word_to_index = vectorizer.vocabulary_

    # Convert the word-to-index dictionary to a word-to-count dictionary
    word_to_count = {word: word_count_matrix[:, index].sum() for word, index in word_to_index.items()}

    # Sort the words by their count
    most_frequent_words = sorted(word_to_count.items(), key=operator.itemgetter(1), reverse=True)

    # Check if the length of most_frequent_words is less than top_keywords
    if len(most_frequent_words) < top_keywords:
        # Append default values to most_frequent_words until its length equals top_keywords
        for i in range(top_keywords - len(most_frequent_words)):
            most_frequent_words.append(('NA', 0))

    # Create two lists, one containing the words and the other containing the counts
    words = [word for word, count in most_frequent_words[:top_keywords]]
    counts = [count for word, count in most_frequent_words[:top_keywords]]

    return words, counts