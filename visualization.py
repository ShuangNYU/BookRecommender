#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 22:52:16 2020

@author: gaoshuang
"""

from pyspark.ml.recommendation import ALSModel
from sklearn.manifold import TSNE  
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import pickle

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

import sys
file_path = 'hdfs:/user/ym1970/'
model_name = sys.argv[1]
genres = [0,4,5]

def prepare(model_name, file_path, genres):

    model = ALSModel.load(file_path + model_name)
    print('Model loaded...')
    item_factors = model.itemFactors
    book_genre = spark.read.parquet(file_path + 'book_genre.parquet')
    item_factor_genre = item_factors.join(book_genre, ['id'], 'inner')
    item_factor_genre = item_factor_genre.filter(item_factor_genre['genre'].isin(genres))
    item_data_flatten = item_factor_genre.select('id','features', 'genre').rdd.flatMap(lambda x:[x]).collect()
    if len(item_data_flatten) > 50000:
        index = random.sample(range(len(item_data_flatten)),50000)
        item_data_temp = []
        for i in index:
            item_data_temp.append(item_data_flatten[i])
        item_data_flatten = item_data_temp
    item_id_list, item_feature_list, item_genre_list = [], [], []
    for i in range(len(item_data_flatten)):
        item_id_list.append(item_data_flatten[i].id)
        item_feature_list.append(item_data_flatten[i].features)
        item_genre_list.append(item_data_flatten[i].genre)
    item_features_df = pd.DataFrame(item_feature_list)
    print('item features prepared...')

    user_factors = model.userFactors
    user_data_flatten = user_factors.select('id','features').rdd.flatMap(lambda x:[x]).collect()
    if len(user_data_flatten) > 50000:
        index = random.sample(range(len(user_data_flatten)),50000)
        user_data_temp = []
        for i in index:
            user_data_temp.append(user_data_flatten[i])
        user_data_flatten = user_data_temp
    user_id_list, user_feature_list = [], []
    for i in range(len(user_data_flatten)):
        user_id_list.append(user_data_flatten[i].id)
        user_feature_list.append(user_data_flatten[i].features)
    user_features_df = pd.DataFrame(user_feature_list)
    print('user features prepared...')

    with open('user_feature_df_'+model_name, 'wb') as f:
        pickle.dump(user_features_df, f)
    with open('item_feature_df_'+model_name, 'wb') as f:
        pickle.dump(item_features_df, f)
    with open('item_genre_list_'+model_name, 'wb') as f:
        pickle.dump(item_genre_list, f)
    
    return user_features_df, item_features_df, item_genre_list
    
    
def train_item(item_features_df, item_genre_list):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
    tsne_results = tsne.fit_transform(item_features_df)
    df_item = pd.DataFrame()
    df_item['genre'] = item_genre_list
    df_item['tsne-2d-one'] = tsne_results[:,0]
    df_item['tsne-2d-two'] = tsne_results[:,1]
    return df_item
    
def train_user(user_features_df):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
    tsne_results = tsne.fit_transform(user_features_df)
    df_user = pd.DataFrame()
    df_user['tsne-2d-one'] = tsne_results[:,0]
    df_user['tsne-2d-two'] = tsne_results[:,1]
    return df_user
    
def visualize(model_name, file_path, genres):
    
    '''
    model_name: string, the name of saved model
    file_path: string, e.g. 'hdfs:/user/sg5963/'
    genres: a list of integers, representing the genres of items taking into visualization, e.g. [0,4,5]
            children: 0
            comics, graphic: 1
            fantasy, paranormal: 2
            fiction: 3
            history, historical fiction, biography: 4
            mystery, thriller, crime: 5
            non-fiction: 6
            poetry: 7
            romance: 8
            young-adult: 9
    '''
    print('start data preparing ...')
    user_features_df, item_features_df, item_genre_list = prepare(model_name, file_path, genres)
    print('start dedimensionality ...')
    df_item = train_item(item_features_df, item_genre_list)
    print('tsne has been trained for items ...')
    df_user = train_user(user_features_df)
    print('tsne has been trained for users ...')
    
    plt.switch_backend('agg')
    
    plt.figure(figsize=(16,10))
    sns.scatterplot(x='tsne-2d-one', y='tsne-2d-two', data=df_user, legend='full', alpha=0.3)
    plt.savefig('user_' + model_name + '.png')
    print('figure for users has been saved ...')
    plt.figure(figsize=(16,10))
    sns.scatterplot(x='tsne-2d-one', y='tsne-2d-two', hue='genre', \
                    palette=sns.color_palette('hls', len(genres)), \
                    data=df_item,legend='full', alpha=0.3)
    plt.savefig('item_' + model_name + '.png')
    print('figure for items has been saved ...')
    
if __name__ == '__main__':
    visualize(model_name, file_path, genres)
    
