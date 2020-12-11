import sys
from pyspark.sql import SparkSession, functions as f, types
from pyspark.sql.functions import monotonically_increasing_id

spark = SparkSession.builder.appName('amenity data cleaning').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

''' RUN: spark-submit code/amenities_data_cleaning.py data/amenities-vancouver.json.gz cleaned-data-amenities '''

schema = types.StructType([
    types.StructField('amenity', types.StringType()),
    types.StructField('lat', types.FloatType()),
    types.StructField('lon', types.FloatType()),
    types.StructField('name', types.StringType()),
    types.StructField('tags', types.StringType()),
    types.StructField('timestamp', types.TimestampType()),
])


def main(inp, outp):
    data = spark.read.json(inp, schema = schema)

    #data.select('amenity').distinct().sort('amenity').show()

    # Interesting amenities
    amnt = ['Observation Platform', 'arts_centre', 'atm', 'atm;bank', 'bank', 'bar', 'bbq', 'bicycle_rental', 'biergarten',
            'bistro', 'boat_rental', 'bureau_de_change', 'bus_station','cafe', 'car_rental', 'car_sharing','casino', 'cinema', 'clock', 
            'conference_centre', 'courthouse','events_venue', 'fast_food', 'ferry_terminal', 'fountain', 'gym', 'ice_cream',
            'internet_cafe', 'juice_bar', 'leisure', 'library', 'luggage_locker', 'marketplace', 'meditation_centre','monastery',
            'motorcycle_rental', 'nightclub', 'office|financial', 'park', 'photo_booth', 'place_of_worship', 'playground', 
            'police', 'pub', 'public_building', 'research_institute', 'restaurant', 'science', 'seaplane terminal', 
            'shop|clothes', 'spa', 'stripclub', 'studio', 'taxi', 'theatre','townhall', 'university', 'workshop']

    # Amenities in categories
    leisure = ['casino','cinema','gym','internet_cafe','leisure','library','marketplace','meditation_centre','photo_booth',
            'playgroud','shop|clothes', 'spa']
    attraction = ['Observation Platform', 'arts_centre','clock','fountain','monastery','park','place_of_worship','theatre',
                'university']
    business = ['atm', 'atm;bank', 'bank','bureau_de_change','conference_centre','courthouse','event_venue','luggage_locker',
            'office|financial','police','public_building','research_institute','science','studio','workshop','townhall']
    food = ['bbq', 'biergarten', 'bistro','cafe','fast_food','ice_cream','juice_bar','restaurant']
    transportation = ['bicycle_rental', 'boat_rental','car_rental','car_sharing','ferry_terminal','motorcycle_rental',
                    'seaplane terminal','taxi','bus_station']
    nightlife = ['bar', 'nightclub','pub','stripclub']

    # Keep only the amenities in city of Vancouver
    data_cleaned = data[data.amenity.isin(amnt)].filter(data.lon.between(-123.264760,-123.023715) & data.lat.between(49.2, 49.317244)).drop('tags','timestamp')

    data_cleaned = data_cleaned.withColumn('category', f.when(f.col('amenity').isin(leisure), 'Leisure')
                                .when(f.col('amenity').isin(attraction), 'Attraction')
                                .when(f.col('amenity').isin(business), 'Business')
                                .when(f.col('amenity').isin(food), 'Food')
                                .when(f.col('amenity').isin(transportation), 'Transportation')
                                .otherwise('Nightlife'))
    # data_cleaned.show()

    # Get rid of places without name
    data_cleaned = data_cleaned.filter(~(f.isnull(data_cleaned.name)))
    data_cleaned = data_cleaned.withColumn('amnt_id', monotonically_increasing_id())

    ''' Coalescion is fine, as amenities data only contains 'interesting' things in Vancouver '''
    data_cleaned.coalesce(1).write.csv(outp, mode = 'overwrite', header = True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])