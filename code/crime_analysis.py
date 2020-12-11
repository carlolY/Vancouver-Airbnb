import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from scipy import stats
from pyspark.sql import SparkSession, functions as f, types, SQLContext, Row

spark = SparkSession.builder.appName('crime and amnt analysis').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
sqlContext = SQLContext(spark)

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

''' RUN: time spark-submit code/crime_analysis.py cleaned-data-crime amnt-nh-data '''

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
    types.StructField('neighbourhood', types.StringType())
])



def getData(crime, amnt):
    crime_data = spark.read.csv(crime, schema = crime_schema, header = True).cache()
    amnt_data = spark.read.csv(amnt, schema = amnt_schema, header = True).cache()

    return crime_data, amnt_data
    


''' SCATTERPLOT CRIME (Visualization) '''
def makePlots(amnt_data_pd, crime_data_pd):
    crime_data_pd = crime_data_pd.rename(columns={'type': 'Crime Type'})
    amnt_data_pd = amnt_data_pd.rename(columns={'category': 'Amenity Category'})

    plt.figure()
    sns.set(font_scale = 0.5)

    with sns.cubehelix_palette(8):
        crime_plt = sns.scatterplot(x='longitude', y='latitude', data=crime_data_pd, hue='Crime Type', s=10, marker='.', linewidth=0)

    ''' Put a legend below current axis '''
    ''' REFERENCE: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot '''
    box = crime_plt.get_position()
    crime_plt.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

    crime_plt.legend(loc='upper center', bbox_to_anchor=(0.45, -0.15), fancybox=True, ncol=4)
    crime_plt.set(xlabel='Longitude', ylabel='Latitude')
    crime_plt.set_title('Crimes by Lat / Lon')


    ''' SCATTERPLOT AMENITY (Visualization) '''
    with sns.color_palette('GnBu_d'):
        amnt_plt = sns.scatterplot(x='lon', y='lat', data=amnt_data_pd, hue='Amenity Category', s=10, marker='.', linewidth=0)

    box = amnt_plt.get_position()
    amnt_plt.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

    amnt_plt.legend(loc='upper center', bbox_to_anchor=(0.45, -0.15), fancybox=True, ncol=4)
    amnt_plt.set(xlabel='Longitude', ylabel='Latitude')
    amnt_plt.set_title('Crime and Amenities by Lat / Lon')

    amnt_fig = amnt_plt.get_figure()
    amnt_fig.savefig('figures/Crime_Amenities.png', dpi=300, bbox_inches = 'tight')

    amnt_fig.clf()



def makeKDEPlots(crime_data_pd):
    sns.set(font_scale = 0.6)
    plt.figure()
    plt.suptitle('Two-Variable Probability Density by Crime Type')
    
    KDE_plt = sns.FacetGrid(crime_data_pd, col='type', col_wrap=4)
    KDE_plt.map(sns.kdeplot, 'longitude', 'latitude', cmap="GnBu", shade=True, shade_lowest=False)

    KDE_plt.savefig('figures/Crime_Density.png', dpi=300, bbox_inches = 'tight')



''' Are different amenities related to different crimes?
    1. Find the distance of each amenity from each crime in the same neighbourhood 
    2. Filter to only show relations where distance is within 500 meters 
    3. Summary df to cleanly display the above information '''

def nearestCrime(amnt_data, crime_data):
    amnt_data.registerTempTable('amnt_table')
    crime_data.registerTempTable('crime_table')

    '''REFERENCE: https://stackoverflow.com/questions/49925904/find-closest-points-using-pyspark'''
    distance_query = 'SELECT amnt_id, category, amenity, name, amnt_table.neighbourhood, c.c_id, c.type as crime_type,\
                    6371000 * DEGREES(ACOS(COS(RADIANS(c.latitude))\
                    * COS(RADIANS(lat))\
                    * COS(RADIANS(c.longitude) - RADIANS(lon))\
                    + SIN(RADIANS(c.latitude))\
                    * SIN(RADIANS(lat)))) AS distance_in_meters\
            FROM amnt_table\
            INNER JOIN (\
                SELECT c_id, type, neighbourhood, latitude, longitude\
                FROM crime_table\
            ) AS c ON c.neighbourhood = amnt_table.neighbourhood'

    crime_distance = sqlContext.sql(distance_query)

    crime_distance.registerTempTable('crime_distance_table')

    filter_distances = 'SELECT *\
        FROM crime_distance_table\
        WHERE distance_in_meters <= 500\
        ORDER BY amnt_id, distance_in_meters'

    crime_distance = sqlContext.sql(filter_distances)

    crime_distance.registerTempTable('crime_distance_table')

    # crime_distance.show(100)

    num_rows = sqlContext.sql('SELECT COUNT(*) FROM crime_distance_table').collect()[0]['count(1)']
    dist_amnt_types = sqlContext.sql('SELECT COUNT(DISTINCT amenity) FROM crime_distance_table').collect()[0]['count(DISTINCT amenity)']
    dist_amnt = sqlContext.sql('SELECT COUNT(DISTINCT amnt_id) FROM crime_distance_table').collect()[0]['count(DISTINCT amnt_id)']
    dist_amnt_cat = sqlContext.sql('SELECT COUNT(DISTINCT category) FROM crime_distance_table').collect()[0]['count(DISTINCT category)']

    summary = spark.createDataFrame(
        [
            (num_rows, dist_amnt_types, dist_amnt, dist_amnt_cat),
        ],
        ['num_rows', 'distinct_amnt_types', 'distinct_amnt_id', 'dist_amnt_cat']
    )

    summary.show()

    return crime_distance



def makeBarplot(crime_distance):
    crime_by_amnt = crime_distance.select(
        crime_distance['category'],
        crime_distance['crime_type']
    ).cache()

    plt.figure()
    sns.set(font_scale = 0.5)
    
    counts = crime_by_amnt.groupBy('category', 'crime_type').count()
    counts = f.broadcast(counts)
    crime_by_amnt = crime_by_amnt.join(counts, on=['category', 'crime_type']).toPandas().drop_duplicates()
    crime_by_amnt = crime_by_amnt.rename(columns={'crime_type': 'Crime Type'})

    crime_by_amnt_plt = sns.barplot(x='category', y='count', hue='Crime Type', data=crime_by_amnt, palette='deep')
    crime_by_amnt_plt.set(xlabel='Amenity Category', ylabel='Frequency')
    crime_by_amnt_plt.set_title('Count of Crime Types by Amenity Category')
    
    crime_by_amnt_fig = crime_by_amnt_plt.get_figure()
    crime_by_amnt_fig.savefig('figures/Crime_Barplot.png', dpi=300, bbox_inches = 'tight')

    crime_by_amnt_fig.clf()



def crime_amnt_analysis(crime_distance):
    amnt_to_crime_count = crime_distance.select(
        crime_distance['category'],
        crime_distance['crime_type']
    )

    amnt_to_crime_all = amnt_to_crime_count.groupBy('category').pivot('crime_type').count().na.fill(0).cache()

    ''' Conform to Chi-Square requirement: Each observed value should be >= 5 '''
    amnt_to_crime_all = amnt_to_crime_all.filter(
        (amnt_to_crime_all['Break and Enter Commercial'] > 5) &
        (amnt_to_crime_all['Break and Enter Residential/Other'] > 5) &
        (amnt_to_crime_all['Mischief'] > 5) &
        (amnt_to_crime_all['Other Theft'] > 5) &
        (amnt_to_crime_all['Theft from Vehicle'] > 5) &
        (amnt_to_crime_all['Theft of Bicycle'] > 5) &
        (amnt_to_crime_all['Theft of Vehicle'] > 5)
    )

    amnt_to_crime_all.show()

    contingency = amnt_to_crime_all.drop('category').toPandas().values.tolist()

    _, p, _, _ = stats.chi2_contingency(contingency)

    if p < 0.05:
        print(f'p-value is: {p}; we can conclude that amenity category has some relation to nearby ( <= 500m ) crimes committed.')
    else:
        print(f'p-value is: {p}; we fail to conclude that amenity category has some relation to nearby ( <= 500m ) crimes committed.')


    ''' Chi-square analysis for 'petty' crimes only '''
    amnt_to_crime_petty = amnt_to_crime_count.filter(
        (amnt_to_crime_count['crime_type'] != 'Theft of Vehicle') &
        (amnt_to_crime_count['crime_type'] != 'Break and Enter Residential/Other') &
        (amnt_to_crime_count['crime_type'] != 'Break and Enter Commercial')
    )

    amnt_to_crime_petty = amnt_to_crime_petty.groupBy('category').pivot('crime_type').count().na.fill(0).cache()

    ''' Conform to Chi-Square requirement: Each observed value should be >= 5 '''
    amnt_to_crime_petty = amnt_to_crime_petty.filter(
        (amnt_to_crime_petty['Mischief'] > 5) &
        (amnt_to_crime_petty['Other Theft'] > 5) &
        (amnt_to_crime_petty['Theft from Vehicle'] > 5) &
        (amnt_to_crime_petty['Theft of Bicycle'] > 5)
    )

    amnt_to_crime_petty.show()

    contingency = amnt_to_crime_petty.drop('category').toPandas().values.tolist()

    _, p, _, _ = stats.chi2_contingency(contingency)

    if p < 0.05:
        print(f'p-value is: {p}; we can conclude that amenity category has some relation to nearby ( <= 500m ) petty crimes committed.')
    else:
        print(f'p-value is: {p}; we fail to conclude that amenity category has some relation to nearby ( <= 500m ) petty crimes committed.')


if __name__=='__main__':
    crime_data, amnt_data = getData(sys.argv[1], sys.argv[2])

    ''' Conversion to Pandas is safe, as data only contains crime info from 2010 - 2020 '''
    ''' Conversion to Pandas is safe, as only 'interesting' amenities were kept '''
    amnt_data_pd = amnt_data.toPandas()
    crime_data_pd = crime_data.toPandas()

    makePlots(amnt_data_pd, crime_data_pd)
    makeKDEPlots(crime_data_pd)

    crime_distance = nearestCrime(amnt_data, crime_data).cache()

    makeBarplot(crime_distance)

    crime_amnt_analysis(crime_distance)