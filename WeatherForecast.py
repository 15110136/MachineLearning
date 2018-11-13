from datetime import datetime, timedelta
import time
from collections import namedtuple
import pandas as pd
import requests
import matplotlib.pyplot as plt
#API_KEY = 'dfb4893708875ce0ce1b45592854ec4a'
#BASE_URL = "https://api.darksky.net/forecast/dfb4893708875ce0ce1b45592854ec4a/37.8267,-122.4233"
API_KEY = '7052ad35e3c73564'
BASE_URL = "http://api.wunderground.com/api/{}/history_{}/q/NE/Lincoln.json"
target_date = datetime(2016, 5, 16)
features = ["date", "meantempm", "meandewptm", "meanpressurem", "maxhumidity", "minhumidity", "maxtempm",
            "mintempm", "maxdewptm", "mindewptm", "maxpressurem", "minpressurem", "precipm"]
DailySummary = namedtuple("DailySummary", features)
def extract_weather_data(url, api_key, target_date, days):
    records = []
    for _ in range(days):
        request = BASE_URL.format(API_KEY, target_date.strftime('%Y%m%d'))
        response = requests.get(request)
        if response.status_code == 200:
            data = response.json()['history']['dailysummary'][0]
            records.append(DailySummary(
                date=target_date,
                meantempm=data['meantempm'],
                meandewptm=data['meandewptm'],
                meanpressurem=data['meanpressurem'],
                maxhumidity=data['maxhumidity'],
                minhumidity=data['minhumidity'],
                maxtempm=data['maxtempm'],
                mintempm=data['mintempm'],
                maxdewptm=data['maxdewptm'],
                mindewptm=data['mindewptm'],
                maxpressurem=data['maxpressurem'],
                minpressurem=data['minpressurem'],
                precipm=data['precipm']))
        time.sleep(6)
        target_date += timedelta(days=1)
    return records
    records = extract_weather_data(BASE_URL, API_KEY, target_date, 500)
    records += extract_weather_data(BASE_URL, API_KEY, target_date, 500)
    df = pd.DataFrame(records, columns=features).set_index('date')
    tmp = df[['meantempm', 'meandewptm']].head(10)
    tmp
    N = 1

    # target measurement of mean temperature
    feature = 'meantempm'

    # total number of rows
    rows = tmp.shape[0]

    # a list representing Nth prior measurements of feature
    # notice that the front of the list needs to be padded with N
    # None values to maintain the constistent rows length for each N
    nth_prior_measurements = [None]*N + [tmp[feature][i-N] for i in range(N, rows)]

    # make a new column name of feature_N and add to DataFrame
    col_name = "{}_{}".format(feature, N)
    tmp[col_name] = nth_prior_measurements
    tmp
def derive_nth_day_feature(df, feature, N):
    rows = df.shape[0]
    nth_prior_measurements = [None]*N + [df[feature][i-N] for i in range(N, rows)]
    col_name = "{}_{}".format(feature, N)
    df[col_name] = nth_prior_measurements
    for feature in features:
        if feature != 'date':
            for N in range(1, 4):
                derive_nth_day_feature(df, feature, N)
    df.columns
    to_remove = [feature
                 for feature in features
                 if feature not in ['meantempm', 'mintempm', 'maxtempm']]

    # make a list of columns to keep
    to_keep = [col for col in df.columns if col not in to_remove]

    # select only the columns in to_keep and assign to df
    df = df[to_keep]
    df.columns
    Index(['meantempm', 'meandewptm', 'meanpressurem', 'maxhumidity',
           'minhumidity', 'maxtempm', 'mintempm', 'maxdewptm', 'mindewptm',
           'maxpressurem', 'minpressurem', 'precipm', 'meantempm_1', 'meantempm_2',
           'meantempm_3', 'meandewptm_1', 'meandewptm_2', 'meandewptm_3',
           'meanpressurem_1', 'meanpressurem_2', 'meanpressurem_3',
           'maxhumidity_1', 'maxhumidity_2', 'maxhumidity_3', 'minhumidity_1',
           'minhumidity_2', 'minhumidity_3', 'maxtempm_1', 'maxtempm_2',
           'maxtempm_3', 'mintempm_1', 'mintempm_2', 'mintempm_3', 'maxdewptm_1',
           'maxdewptm_2', 'maxdewptm_3', 'mindewptm_1', 'mindewptm_2',
           'mindewptm_3', 'maxpressurem_1', 'maxpressurem_2', 'maxpressurem_3',
           'minpressurem_1', 'minpressurem_2', 'minpressurem_3', 'precipm_1',
           'precipm_2', 'precipm_3'],
          dtype='object')
    df.info()
    df = df.apply(pd.to_numeric, errors='coerce')
    df.info()
    # Call describe on df and transpose it due to the large number of columns
    spread = df.describe().T

    # precalculate interquartile range for ease of use in next calculation
    IQR = spread['75%'] - spread['25%']

    # create an outliers column which is either 3 IQRs below the first quartile or
    # 3 IQRs above the third quartile
    spread['outliers'] = (spread['min'] < (spread['25%'] - (3 * IQR))) | (spread['max'] > (spread['75%'] + 3 * IQR))

    # just display the features containing extreme outliers
    spread.ix[spread.outliers,]
    #%matplotlib inline
    'exec(%matplotlib inline)'
    plt.rcParams['figure.figsize'] = [14, 8]
    #xem tính năng áp suất tối thiểu                ppppp
    df.maxhumidity_1.hist()
    plt.title('Distribution of maxhumidity_1')
    plt.xlabel('maxhumidity_1')
    plt.show()
    df.minpressurem_1.hist()
    plt.title('Distribution of minpressurem_1')
    plt.xlabel('minpressurem_1')
    plt.show()
    for precip_col in ['precipm_1', 'precipm_2', 'precipm_3']:
        # create a boolean array of values representing nans
        missing_vals = pd.isnull(df[precip_col])
        df[precip_col][missing_vals] = 0
    df = df.dropna()
