#!/usr/bin/env python


from pyspark.ml.recommendation import ALS
from pyspark.ml.recommendation import ALSModel
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import functions as F
import pickle
import time
import numpy as np
from pyspark.sql.types import IntegerType
from pyspark.context import *
from pyspark.sql import SparkSession
import sys

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()

file_path = 'hdfs:/user/ym1970/'
percent = sys.argv[1]
ranks = [150]
regParams = [0.01]

def train(df, name, rank, maxIter, regParam):
	'''
	Train model.
	'''

	als = ALS(rank=rank, maxIter=maxIter, regParam=regParam)
	model = als.fit(df)
	model.save(name)

	return model

def evaluation(df, model, ks):
	'''
	Evaluate the model.
	ks: a list of parameter k used in precision at k and NDCG at k.
	'''

	print(' Make predictions...')
	predictions = model.recommendForUserSubset(df, 500)

	print(' Prepare ground truth set and predicted set...')
	labels = df.groupBy('user').agg(F.collect_set('item')).collect()
	user_pred = predictions.select('user','recommendations.item').rdd.flatMap(lambda x:[x]).collect()
	labels = sorted(labels, key = lambda x: x.user)
	user_pred = sorted(user_pred, key = lambda x: x.user)
	print(' Combine ground truth set and predicted set...')
	predictionAndLabels = []
	for i in range(len(user_pred)):
		predictionAndLabels.append((user_pred[i].item, labels[i][1]))
	print(' Parallelize...')
	predictionAndLabels = sc.parallelize(predictionAndLabels, numSlices=2000)
	print(' Calculate metrics...')
	metrics = RankingMetrics(predictionAndLabels)
	eval_results = []
	eval_results.append(metrics.meanAveragePrecision)
	for k in ks:
		eval_results.append(metrics.precisionAt(k))
		eval_results.append(metrics.ndcgAt(k))

	return eval_results


def parameter_tuning(file_path, percent, ranks, regParams, maxIter=10, ks=[10, 200, 500]):
	'''
	Tune parameters.
	'''

	print('Load train parquet...')
	df_train = spark.read.parquet(file_path + 'interactions_train_' + str(percent) + '.parquet')
	print('Load val parquet...')
	df_val = spark.read.parquet(file_path + 'interactions_val_' + str(percent) + '.parquet')
	print('Load test parquet...')
	df_test = spark.read.parquet(file_path + 'interactions_test_' + str(percent) + '.parquet')
	
	tuning_dict = {}
	for rank in ranks:
		for regParam in regParams:
			print('Tune parameters: rank={} and reg={}...'.format(rank, regParam))
			model_name = 'rank_' + str(rank) + '_regParam_' + str(regParam) + '_downsample_' + str(percent)
			print('Train...')
			try:
				model = ALSModel.load(model_name)
			except:
				model = train(df=df_train, name=model_name, rank=rank, maxIter=maxIter, regParam=regParam)
			print('Evaluate...')
			eval_results = evaluation(df_test, model, ks)
			tuning_dict[model_name] = eval_results
	
	metrics_name = 'metrices_test' + str(round(time.time())) + '_downsample_' + str(percent) + '.pkl'
	with open(metrics_name, 'wb') as f:
		pickle.dump(tuning_dict, f)
	
	return tuning_dict

if __name__ == '__main__':
	results = parameter_tuning(file_path, percent, ranks, regParams, maxIter=10, ks=[10, 200, 500])
	print('Resulting metrices: {}.'.format(results))

