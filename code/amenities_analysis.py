import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gmplot
from sklearn.cluster import KMeans
from pyspark.sql import SparkSession, functions as f, types, SQLContext, Row
from pyspark.sql.window import Window

spark = SparkSession.builder.appName('crime and amnt analysis').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
sqlContext = SQLContext(spark)

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

''' RUN: spark-submit code/amenities_analysis.py cleaned-data-amenities cleaned_listings '''


amen_schema = types.StructType([
    types.StructField('amenity', types.StringType()),
    types.StructField('lat', types.FloatType()),
    types.StructField('lon', types.FloatType()),
    types.StructField('name', types.StringType()),
    types.StructField('category', types.StringType()),
    types.StructField('amnt_id', types.IntegerType()),
])

list_schema = types.StructType([
    types.StructField('id', types.IntegerType()),
    types.StructField('name', types.StringType()),
    types.StructField('neighbourhood', types.StringType()),
    types.StructField('latitude', types.FloatType()),
    types.StructField('longitude', types.FloatType()),
    types.StructField('room_type', types.StringType()),
    types.StructField('price', types.FloatType()),
    types.StructField('minimum_nights', types.IntegerType())
])


def getData(amenities, listings):
    amenities = spark.read.csv(amenities, header = True, schema = amen_schema)
    listings = spark.read.csv(listings, header = True, schema = list_schema).cache()

    return amenities, listings


######################### KMEANS ANALYSIS ##############################

def getClusterData(amenities):
    n_clust = 5
    x = amenities.select('lat','lon').collect()
    model = KMeans(n_clusters = n_clust, random_state = 353).fit(x)
    clusters = model.predict(x)
    cluster = clusters.tolist()

    centres = model.cluster_centers_

    # convert list to a dataframe
    df = sqlContext.createDataFrame([(l,) for l in cluster], ['cluster'])
    df = df.withColumn("index",f.row_number().over(Window.orderBy(f.monotonically_increasing_id()))-1)

    amnt = amenities.join(df, amenities.amnt_id == df.index).drop("index", 'amnt_id')
    # amnt.show()

    lat = amnt.select('lat').collect()
    lon = amnt.select('lon').collect()
    cluster = amnt.select('cluster').collect()

    return lat, lon, cluster, centres, amnt


######################### MAKE PLOTS #####################################

def makePlots(lat, lon, cluster, centres, amnt):
    sns.set(font_scale = 0.5)
    plt.figure()
    
    scatter = plt.scatter(lon, lat, c = cluster, s=20, edgecolor='w', linewidth=0.5, cmap="viridis")
    plt.legend(*scatter.legend_elements(), loc="lower left", title="Cluster")
    plt.scatter(centres[:,1], centres[:,0], marker="x", color='red')

    plt.title('Amenities Cluster')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig('figures/Cluster_5.png', dpi=300, bbox_inches = 'tight')

    # group cluster 2 and 4 because cluster 4 has too little data
    amnt = amnt.withColumn('clusters', f.when(amnt.cluster == 0, 0).when(amnt.cluster == 1, 1).when(amnt.cluster == 3, 3).otherwise(2)).drop('cluster')

    ## calculate new mean for cluster 2
    new_clust2_mean_row = amnt.select('lat', 'lon','clusters').filter(amnt.clusters == 2).groupBy().avg().collect()
    new_clust2_mean = [new_clust2_mean_row[0][0], new_clust2_mean_row[0][1]]

    centres2 = np.delete(centres, [2,4], 0)
    centres2 = np.insert(centres2,2,new_clust2_mean, axis = 0)
    #centres2

    cluster2 = amnt.select('clusters').collect()

    # Plot again
    plt.figure()
    scatter2 = plt.scatter(lon, lat, c = cluster2, s=20, edgecolor='w', linewidth=0.3, cmap="viridis")
    plt.legend(*scatter2.legend_elements(), loc="lower left", title="Cluster")

    plt.scatter(centres2[:,1], centres2[:,0], marker="x", color='red')
    plt.title('Modified Amenities Cluster')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig('figures/Cluster_4.png', dpi=300, bbox_inches = 'tight')

    return amnt, centres2


############################### MAKE CENTRE DF #################################################

def getCentre(centres):
    centre = pd.DataFrame(centres, columns = ['clust_lat', 'clust_lon'])
    centre['cluster'] = [0,1,2,3]

    centre = spark.createDataFrame(centre, schema = 'CLat:float, CLon:float, cluster:int')

    # amnt = amnt.join(centre, amnt.clusters == centre.cluster)
    # centre.count()

    return centre


############################## CALCULATE DIST ################################################

# https://stackoverflow.com/questions/60086180/pyspark-how-to-apply-a-python-udf-to-pyspark-dataframe-columns

def distance(CLat, CLon, data, col_name):
    return data.withColumn('CLon', f.lit(CLon)).withColumn('CLat',f.lit(CLat)).withColumn("dlon", f.radians(f.col("CLon")) - f.radians(f.col("longitude"))).withColumn("dlat", f.radians(f.col("CLat")) - f.radians(f.col("latitude"))).withColumn(col_name, f.asin(f.sqrt(
                                         f.sin(f.col("dlat") / 2) ** 2 + f.cos(f.radians(f.col("latitude")))
                                         * f.cos(f.radians(f.col("CLat"))) * f.sin(f.col("dlon") / 2) ** 2
                                         )
                                    ) * 2 * 6371 * 1000) \
              .drop("dlon", "dlat",'CLon', 'CLat')


################################### TOP LISTING IN EACH CLUSTER ###############################

def getClusterListings(c, listings):
    centre = c.collect()

    for i in range(c.count()):
        listings = distance(centre[i][0], centre[i][1], listings, "clust_dist" + str(i))

    # listings.show()

    Clust0_listings = (listings.sort(listings['clust_dist0']).drop('clust_dist0','clust_dist1','clust_dist2','clust_dist3','clust_dist4')
                        .withColumn('cluster', f.lit(0)).first())
    Clust1_listings = (listings.sort(listings['clust_dist1']).drop('clust_dist0','clust_dist1','clust_dist2','clust_dist3','clust_dist4')
                        .withColumn('cluster', f.lit(1)).first())
    Clust2_listings = (listings.sort(listings['clust_dist2']).drop('clust_dist0','clust_dist1','clust_dist2','clust_dist3','clust_dist4')
                        .withColumn('cluster', f.lit(2)).first())
    Clust3_listings = (listings.sort(listings['clust_dist3']).drop('clust_dist0','clust_dist1','clust_dist2','clust_dist3','clust_dist4')
                        .withColumn('cluster', f.lit(3)).first())

    op_listings = [Clust0_listings, Clust1_listings, Clust2_listings, Clust3_listings]

    listings_schema = types.StructType([
        types.StructField('id', types.IntegerType()),
        types.StructField('name', types.StringType()),
        types.StructField('neighbourhood', types.StringType()),
        types.StructField('latitude', types.FloatType()),
        types.StructField('longitude', types.FloatType()),
        types.StructField('room_type', types.StringType()),
        types.StructField('price', types.FloatType()),
        types.StructField('minimum_nights', types.IntegerType()),
        types.StructField('cluster', types.StringType()),
    ])

    listings_df = spark.createDataFrame(op_listings, schema=listings_schema)
    listings_df.show(4, False)
    
    ''' Coalescion of 4 rows of data '''
    listings_df.coalesce(1).write.csv('top-airbnb', mode = 'overwrite', header = True)

    return centre, op_listings


####################################### DRAW THE MAP #################################################

def drawMap(lat, lon, centre, op_listings):
    lat2 = []
    for i in range(len(lat)):
        lat2.append(lat[i][0])
        
    lon2 = []
    for i in range(len(lon)):
        lon2.append(lon[i][0])

    centre_lat = []
    centre_lon = []

    for i in range(len(centre)):
        centre_lat.append(centre[i][0])
        centre_lon.append(centre[i][1])

    # plot updated amenities list
    gmap = gmplot.GoogleMapPlotter(49.248903, -123.115505, 13) 

    # plot optimum airbnb/hotel location
    for i in range(len(op_listings)):
        gmap.marker(op_listings[i][3], op_listings[i][4],'plum', size = 40, title = 'Airbnb_id:' + str(op_listings[i][0]))

    # plot mean of each amenity 
    # for i in range(centre.count()):
    #     gmap.marker(centre2[i][0], centre2[i][1], size = 20, title = 'Cluster_mean' + str(i))
    gmap.scatter(lat2, lon2, 'darkcyan', size = 20, marker = False) 
    gmap.scatter(centre_lat, centre_lon, 'red', size = 60, marker = False)

    # personal api key
    gmap.apikey = "AIzaSyCj4I7U8Bc7ZhQkLGp_kncaMUka62ZACns"
    gmap.draw( "airbnb_map.html" ) 


######################### PROPORTIONS ANALYSIS #####################################

def plotProportions(amnts):
    Clust0 = amnts.filter(amnts['clusters'] == 0)
    Clust1 = amnts.filter(amnts['clusters'] == 1)
    Clust2 = amnts.filter(amnts['clusters'] == 2)
    Clust3 = amnts.filter(amnts['clusters'] == 3)
    Clust0_n = Clust0.groupBy('category').count()
    Clust1_n = Clust1.groupBy('category').count()
    Clust2_n = Clust2.groupBy('category').count()
    Clust3_n = Clust3.groupBy('category').count()

    Clust0_n = Clust0_n.withColumn('percent', Clust0_n['count'] / Clust0.count()).drop('count')
    Clust1_n = Clust1_n.withColumn('percent', Clust1_n['count'] / Clust1.count()).drop('count')
    Clust2_n = Clust2_n.withColumn('percent', Clust2_n['count'] / Clust2.count()).drop('count')
    Clust3_n = Clust3_n.withColumn('percent', Clust3_n['count'] / Clust3.count()).drop('count')

    Clust0_n = Clust0_n.withColumn('cluster', f.lit('Clust 0'))
    Clust1_n = Clust1_n.withColumn('cluster', f.lit('Clust 1'))
    Clust2_n = Clust2_n.withColumn('cluster', f.lit('Clust 2'))
    Clust3_n = Clust3_n.withColumn('cluster', f.lit('Clust 3'))

    Full = Clust0_n.union(Clust1_n).union(Clust2_n).union(Clust3_n)

    Cat0 = Full.filter(Full['category'] == 'Food').cache()
    Cat1 = Full.filter(Full['category'] == 'Attraction').cache()
    Cat2 = Full.filter(Full['category'] == 'Nightlife').cache()
    Cat3 = Full.filter(Full['category'] == 'Leisure').cache()
    Cat4 = Full.filter(Full['category'] == 'Transportation').cache()
    Cat5 = Full.filter(Full['category'] == 'Business').cache()

    Sum0 = Cat0.groupBy().sum('percent').collect()[0][0]
    Sum1 = Cat1.groupBy().sum('percent').collect()[0][0]
    Sum2 = Cat2.groupBy().sum('percent').collect()[0][0]
    Sum3 = Cat3.groupBy().sum('percent').collect()[0][0]
    Sum4 = Cat4.groupBy().sum('percent').collect()[0][0]
    Sum5 = Cat5.groupBy().sum('percent').collect()[0][0]

    Cat0 = Cat0.withColumn('sc percent', Cat0['percent'] / Sum0).cache()
    Cat1 = Cat1.withColumn('sc percent', Cat1['percent'] / Sum1).cache()
    Cat2 = Cat2.withColumn('sc percent', Cat2['percent'] / Sum2).cache()
    Cat3 = Cat3.withColumn('sc percent', Cat3['percent'] / Sum3).cache()
    Cat4 = Cat4.withColumn('sc percent', Cat4['percent'] / Sum4).cache()
    Cat5 = Cat5.withColumn('sc percent', Cat5['percent'] / Sum5).cache()

    total = Cat0.union(Cat1).union(Cat2).union(Cat3).union(Cat4).union(Cat5)

    ''' Conversion to Pandas is okay as each dataframe only consists amnt of one category (small lists) '''
    Cat0_n = Cat0.toPandas()
    Cat1_n = Cat1.toPandas()
    Cat2_n = Cat2.toPandas()
    Cat3_n = Cat3.toPandas()
    Cat4_n = Cat4.toPandas()
    Cat5_n = Cat5.toPandas()
    total = total.toPandas()

    plt.figure()
    sns.set(font_scale = 0.8)

    plt.bar(Cat0_n['cluster'], Cat0_n['percent'], width=0.8, label='Food', color='cornflowerblue')
    plt.bar(Cat1_n['cluster'], Cat1_n['percent'], width=0.8, label='Attraction', color='sandybrown', bottom=Cat0_n['percent'])
    plt.bar(Cat2_n['cluster'], Cat2_n['percent'], width=0.8, label='Nightlife', color='lightgreen', bottom=Cat0_n['percent']+Cat1_n['percent'])
    plt.bar(Cat3_n['cluster'], Cat3_n['percent'], width=0.8, label='Leisure', color='indianred', bottom=Cat0_n['percent']+Cat1_n['percent']+Cat2_n['percent'])
    plt.bar(Cat4_n['cluster'], Cat4_n['percent'], width=0.8, label='Transportation', color='mediumpurple', bottom=Cat0_n['percent']+Cat1_n['percent']+Cat2_n['percent']+Cat3_n['percent'])
    plt.bar(Cat5_n['cluster'], Cat5_n['percent'], width=0.8, label='Business', color='sienna', bottom=Cat0_n['percent']+Cat1_n['percent']+Cat2_n['percent']+Cat3_n['percent']+Cat4_n['percent'])
    plt.legend(loc="center right",bbox_to_anchor=(1.4, 0.5), borderaxespad=0)
    plt.subplots_adjust(right=0.7)
    plt.title('Proportions of Amenity Category per Cluster')
    plt.xlabel('')
    plt.ylabel('Percent')
    plt.savefig('figures/Amnt_Prop_by_Clus.png', dpi=300, bbox_inches = 'tight')

    plt.figure()
    g = sns.catplot(x="cluster", y="sc percent", hue="category", data=total, height=6, kind="bar", palette="muted")
    g.despine(left=True)
    g.set_ylabels("Proportion")
    g.set_xlabels("Cluster")

    plt.title('Proportions of Amenity Category per Cluster w/ Equal Category Proportions')
    plt.savefig('figures/Amnt_Prop_by_Amnt.png', dpi=300, bbox_inches = 'tight')


if __name__=='__main__':
    amenities, listings = getData(sys.argv[1], sys.argv[2])

    lat, lon, cluster, centres, amenities = getClusterData(amenities)

    amnt, centres = makePlots(lat, lon, cluster, centres, amenities)

    centre = getCentre(centres)

    centre, cluster_listings = getClusterListings(centre, listings)

    drawMap(lat, lon, centre, cluster_listings)

    plotProportions(amnt)