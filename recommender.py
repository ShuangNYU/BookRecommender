#!/usr/bin/env python

# Please import this file in pyspark interface and call the two functions subsequently to prepare datasets ready to train models.

from pyspark.ml.recommendation import ALS
from pyspark.ml.recommendation import ALSModel
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import pickle
import time
from pyspark.sql.types import IntegerType

from pyspark.context import *
sc = SparkContext.getOrCreate()

def csv_to_parquet(spark, file_path_from, file_path_to):

	'''
	Convert original csv file to parquet file, saving us time processing data.
	file_path_from: the directory followed by file name.
									such as hdfs:/user/bm106/pub/goodreads/goodreads_interactions.csv
	file_path_to: the directory where we want to store parquet files.
								such as hdfs:/user/ym1970/.
	'''

	interactions = spark.read.csv(file_path_from, header=True, schema='user_id int, book_id int, is_read int, rating int, is_reviewed int')

	interactions_repa = interactions.repartition('user_id')
	interactions_repa.write.parquet(file_path_to + 'interactions_repa.parquet')
	interactions = spark.read.parquet(file_path_to + 'interactions_repa.parquet')

	interactions.createOrReplaceTempView('interactions')

	df = interactions.select('user_id', 'book_id', 'rating')
	df = df.withColumnRenamed('user_id', 'user')
	df = df.withColumnRenamed('book_id', 'item')
	df_valid = df.groupBy('user').count().filter('count >= 10')
	df_new = df.join(df_valid, 'user')
	df_new = df_new.drop('count')
	
	df_new.write.parquet(file_path_to + 'interactions_preprocessed_repa.parquet')


def preprocess(spark, file_path, percent=None):

	'''
	Split data to train, val and test.
	file_path: the directory where we want to read preprocessed parquet file 
							and store to be splitted parquet files.
							e.g., hdfs:/user/ym1970/.
	percent: the ratio of downsampling, such as 1, 5 or 25 as suggested in instructions.
	'''

	df = spark.read.parquet(file_path + 'interactions_preprocessed_repa.parquet')
	print('Start downsampling...')
	if percent:
		df_downsample_id, _ = [i for i in df.select('user').distinct().randomSplit([percent/100, 1-percent/100], 123)]
		df = df.join(df_downsample_id, 'user', 'left_semi')

	print('Start splitting...')
	
	df_train_id, df_val_id, df_test_id = [i for i in df.select('user').distinct().randomSplit([0.6, 0.2, 0.2], 123)]

	print('Select records based on user id...')
	df_train = df.join(df_train_id, 'user', 'left_semi')
	df_val = df.join(df_val_id, 'user', 'left_semi')
	df_test = df.join(df_test_id, 'user', 'left_semi')
	
	print('Sample user id to be moved to train...')	
	window = Window.partitionBy('user').orderBy('item')
	df_val_window = (df_val.select('user', 'item', 'rating', F.row_number().over(window).alias('row_number')))
	df_test_window = (df_test.select('user', 'item', 'rating', F.row_number().over(window).alias('row_number')))

	print('Move to train...')
	df_val_to_train = df_val_window.filter(df_val_window.row_number % 2 == 1).select('user', 'item', 'rating')
	df_test_to_train = df_test_window.filter(df_test_window.row_number % 2 == 1).select('user', 'item', 'rating')

	df_train = df_train.union(df_val_to_train).union(df_test_to_train)

	df_val = df_val.subtract(df_val_to_train)
	df_test = df_test.subtract(df_test_to_train)
	
	print('Delete based on book id...')
	df_val = df_val.join(df_train, 'item', 'left_semi')
	df_test = df_test.join(df_train, 'item', 'left_semi')
	
	print('Write to parquet...')
	df_train.write.parquet(file_path + 'interactions_train_' + str(percent) + '.parquet')
	df_val.write.parquet(file_path + 'interactions_val_' + str(percent) + '.parquet')
	df_test.write.parquet(file_path + 'interactions_test_' + str(percent) + '.parquet')
