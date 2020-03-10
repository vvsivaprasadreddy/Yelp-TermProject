import pandas as pd
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.mllib.recommendation import ALS
import csv
import math
from io import StringIO
from sklearn import preprocessing
from sklearn.utils import resample

#Class to read csv
class MyDialect(csv.Dialect):
    strict = True
    skipinitialspace = True
    quoting = csv.QUOTE_ALL
    delimiter = ','
    quotechar = '"'
    lineterminator = '\n'

#Class to remove restaurants and users not present in training
def get_test(city_train,city_test):
    #Select ids in	 test if present in train
    user_ids = [id for id in city_test.user_id.unique() if id in city_train.user_id.unique()]
    business_ids = [id for id in city_test.business_id.unique() if id in city_train.business_id.unique()]
    final_test = city_test[city_test.user_id.apply(lambda a: a in user_ids)].copy()
    final_test = city_test[city_test.user_id.apply(lambda a: a in user_ids)].copy()
    final_test = final_test[final_test.business_id.apply(lambda a: a in business_ids)].copy()
    final_test = final_test.reset_index()
    final_test = final_test.drop('index', axis = 1)
    return final_test

#Class to evaluate parameters of ALS model
def mat_fact_val(city, rank=5, iter_=20, lambda_ = .01):
    toronto_train = pd.read_csv('D:/Study/TermProject/yelp-dataset/data/'+city+'_train.csv',encoding = 'iso8859_15')
    toronto_val = pd.read_csv('D:/Study/TermProject/yelp-dataset/data/'+city+'_val.csv',encoding = 'iso8859_15')
    toronto_val = get_test(toronto_train,toronto_val)
    le_user_id = preprocessing.LabelEncoder()
    le_user_id = le_user_id.fit(toronto_train.user_id)
    user_id_enc = le_user_id.transform(toronto_train.user_id)
    toronto_train['user_id_enc'] = user_id_enc
    toronto_val['user_id_enc'] = le_user_id.transform(toronto_val.user_id)
    le_business_id = preprocessing.LabelEncoder()
    le_business_id = le_business_id.fit(toronto_train.business_id)
    business_id_enc = le_business_id.transform(toronto_train.business_id)
    toronto_train['business_id_enc'] = business_id_enc
    toronto_val['business_id_enc'] = le_business_id.transform(toronto_val.business_id)
    sqlCtx = SQLContext(sc)
    toronto_train_sp = sqlCtx.createDataFrame(toronto_train[['user_id_enc','business_id_enc','stars_review']])
    toronto_train_sp = toronto_train_sp.withColumn("stars_review", toronto_train_sp["stars_review"].cast("double"))
    toronto_val_sp = sqlCtx.createDataFrame(toronto_val[['user_id_enc','business_id_enc','stars_review']])
    toronto_val_sp = toronto_val_sp.withColumn("stars_review", toronto_val_sp["stars_review"].cast("double"))
    model = ALS.train(toronto_train_sp, rank, seed=0, iterations=iter_,lambda_=lambda_)
    prediction=model.predictAll(toronto_train_sp.rdd.map(lambda line: (line[0],line[1]))).map(lambda d: ((d[0],d[1]),d[2]))
    true_and_pred = toronto_train_sp.rdd.map(lambda d: ((d[0],d[1]),d[2])).join(prediction).map(lambda r:(r[0],r[1][0], r[1][1]))
    true_and_pred.map(lambda line:(line[0],line[1],5 if line[2]>=5 else line[2]))
    error = math.sqrt(true_and_pred.map(lambda r: (math.fabs(r[1]-r[2]))**1).mean())
    print('Training: ',error)
    prediction=model.predictAll(toronto_val_sp.rdd.map(lambda line: (line[0],line[1]))).map(lambda d: ((d[0],d[1]),d[2]))
    true_and_pred = toronto_val_sp.rdd.map(lambda d: ((d[0],d[1]),d[2])).join(prediction).map(lambda r:(r[0],r[1][0], r[1][1]))
    true_and_pred.map(lambda line:(line[0],line[1],5 if line[2]>=5 else line[2]))
    error = math.sqrt(true_and_pred.map(lambda r: (math.fabs(r[1]-r[2]))**1).mean())
    print('Validation: ',error)

#Class to check error on test data
def mat_fact_test(city, rank=5, iter_=20, lambda_ = .01):
    toronto_train = pd.read_csv('D:/Study/TermProject/yelp-dataset/data/'+city+'_train.csv',encoding = 'iso8859_15')
    toronto_test = pd.read_csv('D:/Study/TermProject/yelp-dataset/data/'+city+'_test.csv',encoding = 'iso8859_15')
    toronto_test = get_test(toronto_train,toronto_test)
    toronto_test = resample(toronto_test,n_samples=int(np.ceil(0.8 * toronto_test.shape[0])))
    toronto_test = toronto_test.reset_index()
    toronto_test = toronto_test.drop('index', axis = 1)
    le_user_id = preprocessing.LabelEncoder()
    le_user_id = le_user_id.fit(toronto_train.user_id)
    user_id_enc = le_user_id.transform(toronto_train.user_id)
    toronto_train['user_id_enc'] = user_id_enc
    toronto_test['user_id_enc'] = le_user_id.transform(toronto_test.user_id)
    le_business_id = preprocessing.LabelEncoder()
    le_business_id = le_business_id.fit(toronto_train.business_id)
    business_id_enc = le_business_id.transform(toronto_train.business_id)
    toronto_train['business_id_enc'] = business_id_enc
    toronto_test['business_id_enc'] = le_business_id.transform(toronto_test.business_id)
    sqlCtx = SQLContext(sc)
    toronto_train_sp = sqlCtx.createDataFrame(toronto_train[['user_id_enc','business_id_enc','stars_review']])
    toronto_train_sp = toronto_train_sp.withColumn("stars_review", toronto_train_sp["stars_review"].cast("double"))
    toronto_test_sp = sqlCtx.createDataFrame(toronto_test[['user_id_enc','business_id_enc','stars_review']])
    toronto_test_sp = toronto_test_sp.withColumn("stars_review", toronto_test_sp["stars_review"].cast("double"))
    model = ALS.train(toronto_train_sp, rank, seed=0, iterations=iter_,lambda_=lambda_)
    prediction=model.predictAll(toronto_train_sp.rdd.map(lambda line: (line[0],line[1]))).map(lambda d: ((d[0],d[1]),d[2]))
    true_and_pred = toronto_train_sp.rdd.map(lambda d: ((d[0],d[1]),d[2])).join(prediction).map(lambda r:(r[0],r[1][0], r[1][1]))
    true_and_pred.map(lambda line:(line[0],line[1],5 if line[2]>=5 else line[2]))
    error = math.sqrt(true_and_pred.map(lambda r: (math.fabs(r[1]-r[2]))**1).mean())
    print('Training: ',error)
    prediction=model.predictAll(toronto_test_sp.rdd.map(lambda line: (line[0],line[1]))).map(lambda d: ((d[0],d[1]),d[2]))
    true_and_pred = toronto_test_sp.rdd.map(lambda d: ((d[0],d[1]),d[2])).join(prediction).map(lambda r:(r[0],r[1][0], r[1][1]))
    true_and_pred.map(lambda line:(line[0],line[1],5 if line[2]>=5 else line[2]))
    error = math.sqrt(true_and_pred.map(lambda r: (math.fabs(r[1]-r[2]))**1).mean())
    print('Test: ',error)

mat_fact('toronto', 10, 10,.001)

mat_fact_test('toronto', 15, 10,.5)
