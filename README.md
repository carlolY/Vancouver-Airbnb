# CMPT353_Final

Libraries required:

- numpy
- pandas
- matplotlib
- seaborn
- statsmodels
- sklearn
- scipy
- gmplot
- pyspark


Program order of excecution:

RUN:    spark-submit code/crime_data_cleaning.py data/crimedata_all_years.csv cleaned-data-crime
>> Produces cleaned-data-crime folder with .csv file of cleaned crime data

RUN:    spark-submit code/amenities_data_cleaning.py data/amenities-vancouver.json.gz cleaned-data-amenities
>> Produces cleaned-data-amenities folder with .csv file of cleaned amenity data

RUN:    spark-submit code/amenities_nh_analysis.py cleaned-data-crime cleaned-data-amenities amnt-nh-data
>> Produces amnt-nh-data folder with .csv file of cleaned amenity data with a new 'neighbourhood' column

RUN:    spark-submit code/listings_data_cleaning.py data/listings.csv cleaned_listings
>> Produces cleaned_listings folder with .csv file of cleaned listings data

RUN:    spark-submit code/crime_analysis.py cleaned-data-crime amnt-nh-data                                     
>> Will take 4 - 5 minutes to run
>> Produces two printed chi-squares of amenity category by crime type + relevant figures

RUN:    spark-submit code/amenities_analysis.py cleaned-data-amenities cleaned_listings
>> Produces top 4 Airbnb listings in a .csv in the top-airbnb folder + relevant figures