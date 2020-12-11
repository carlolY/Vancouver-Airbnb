import sys
from pyspark.sql import SparkSession, functions as f, types

spark = SparkSession.builder.appName('listings data cleaning').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

''' RUN: spark-submit code/listings_data_cleaning.py data/listings.csv cleaned-listings '''

schema = types.StructType([
    types.StructField('id', types.IntegerType()),
    types.StructField('name', types.StringType()),
    types.StructField('host_id', types.IntegerType()),
    types.StructField('host_name', types.StringType()),
    types.StructField('neighbourhood_group', types.StringType()),
    types.StructField('neighbourhood', types.StringType()),
    types.StructField('latitude', types.FloatType()),
    types.StructField('longitude', types.FloatType()),
    types.StructField('room_type', types.StringType()),
    types.StructField('price', types.FloatType()),
    types.StructField('minimum_nights', types.IntegerType()),
    types.StructField('number_of_reviews', types.IntegerType()),
    types.StructField('last_review', types.TimestampType()),
    types.StructField('reviews_per_month', types.FloatType()),
    types.StructField('calculated_host_listings_count', types.StringType()),
    types.StructField('availability_365', types.StringType()),
])


def main(inp, outp):
    listings = spark.read.csv(inp, header = True, schema = schema)
    listings = listings.drop('host_id', 'host_name', 'neighbourhood_group', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365')

    # listings.describe(['minimum_nights']).show()

    # Give a range to minimum_nights
    listings = listings.filter("minimum_nights > 0 AND minimum_nights <= 30")

    # Remove listings with last_review posted more than 2 years ago
    listings = listings.withColumn('curr', f.current_date())
    listings = listings.withColumn('diff', f.datediff(f.col('curr'), f.col('last_review')))
    cleaned_listings = listings.filter(listings['diff'] < 730).drop('curr','diff', 'number_of_reviews', 'last_review')

    ''' Coalescion is fine, as listing data only contains airbnb locations in Vancouver '''
    cleaned_listings.coalesce(1).write.csv(outp, mode = 'overwrite', header = True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

