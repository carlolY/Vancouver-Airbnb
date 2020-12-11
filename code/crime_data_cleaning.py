import sys
from pyproj import Proj
from pyspark.sql import SparkSession, functions as f, types

spark = SparkSession.builder.appName('crime data cleaning').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

''' RUN: spark-submit code/crime_data_cleaning.py data/crimedata_all_years.csv cleaned-data-crime '''

schema = types.StructType([
    types.StructField('type', types.StringType()),
    types.StructField('year', types.StringType()),
    types.StructField('month', types.StringType()),
    types.StructField('day', types.StringType()),
    types.StructField('hour', types.StringType()),
    types.StructField('minute', types.StringType()),
    types.StructField('hundred_block', types.StringType()),
    types.StructField('neighbourhood', types.StringType()),
    types.StructField('X', types.FloatType()),
    types.StructField('Y', types.FloatType())
])


def main(inp, outp):
    '''DATA: https://geodash.vpd.ca/opendata/ '''
    data = spark.read.csv(inp, schema = schema, header = True).cache()

    # counts = data.groupBy('type').count()
    # counts.show(20, False)

    ''' Conversion to Pandas is safe, as data only contains crime info from 2010 - 2020 '''
    data_cleaned = data.filter(
        (data.neighbourhood.isNotNull()) &
        (data.X != 0) &
        (data.Y != 0) &
        # keep intentional crimes (vehicle collision are more likely accidental)
        (data.type != 'Vehicle Collision or Pedestrian Struck (with Fatality)') & 
        (data.type != 'Vehicle Collision or Pedestrian Struck (with Injury)') &
        (data.year >= 2010)
    ).toPandas()

    '''REFERENCE: https://ocefpaf.github.io/python4oceanographers/blog/2013/12/16/utm/'''
    vanProj = Proj("+proj=utm +zone=10K, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

    lon, lat = vanProj(data_cleaned['X'].values, data_cleaned['Y'].values, inverse=True)

    data_cleaned['latitude'] = lat
    data_cleaned['longitude'] = lon

    data_cleaned = spark.createDataFrame(data_cleaned).drop('hour', 'minute', 'hundred_block', 'X', 'Y')

    data_cleaned = data_cleaned.withColumn('c_id', f.monotonically_increasing_id())

    # data_cleaned.show()

    ''' Coalescion is fine, as crime data only spans from 2010 - 2020 '''
    data_cleaned.coalesce(1).write.csv(outp, mode = 'overwrite', header = True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])