import sys
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier

from pyspark.sql import SparkSession, functions as f, types, SQLContext, Row

spark = SparkSession.builder.appName('amenity neighbourhoods').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
sqlContext = SQLContext(spark)

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

''' RUN: spark-submit code/amenities_nh_analysis.py cleaned-data-crime cleaned-data-amenities amnt-nh-data '''

crime_schema = types.StructType([
    types.StructField('type', types.StringType()),
    types.StructField('year', types.StringType()),
    types.StructField('month', types.StringType()),
    types.StructField('day', types.StringType()),
    types.StructField('neighbourhood', types.StringType()),
    types.StructField('latitude', types.FloatType()),
    types.StructField('longitude', types.FloatType()),
    types.StructField('c_id', types.StringType()),
])



amnt_schema = types.StructType([
    types.StructField('amenity', types.StringType()),
    types.StructField('lat', types.FloatType()),
    types.StructField('lon', types.FloatType()),
    types.StructField('name', types.StringType()),
    types.StructField('category', types.StringType()),
    types.StructField('amnt_id', types.IntegerType()),
])

def main(crime, amnt, outp):

    crime_data = spark.read.csv(crime, schema = crime_schema, header = True).cache()
    amnt_data = spark.read.csv(amnt, schema = amnt_schema, header = True)

    ''' Get neighborhood feature for amnt_data: 
        1. train_test_split on crime_data where (X: lon, lat. y: neighborhood). 
        2. Use model on amnt's lon lat features to predict neighborhood. '''

    ''' Conversion to Pandas is safe, as data only contains crime info from 2010 - 2020 '''
    ''' Conversion to Pandas is safe, as only 'interesting' amenities were kept '''
    crime_data_pd = crime_data.toPandas()
    amnt_data_pd = amnt_data.toPandas()

    nh_X = crime_data_pd[['latitude', 'longitude']]
    nh_y = crime_data_pd['neighbourhood']

    nh_X_train, nh_X_valid, nh_y_train, nh_y_valid = train_test_split(nh_X, nh_y)

    nh_model = make_pipeline(
        KNeighborsClassifier(n_neighbors = 5)
    )

    nh_model.fit(nh_X_train, nh_y_train)

    print(f'MODEL TRAINING SCORE: {nh_model.score(nh_X_train, nh_y_train)}  ||  MODEL VALIDATION SCORE: {nh_model.score(nh_X_valid, nh_y_valid)}')

    amnt_data_pd['neighbourhood'] = nh_model.predict(amnt_data_pd[['lat', 'lon']])
    amnt_data = spark.createDataFrame(amnt_data_pd)

    amnt_data.coalesce(1).write.csv(outp, mode = 'overwrite', header = True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])