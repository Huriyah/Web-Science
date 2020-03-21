import pymongo
from pymongo import MongoClient
import json
import twitter
import tweepy
from threading import Thread
from pprint import pprint
from datetime import datetime
from datetime import timedelta
import time


API_KEY = ""
API_SECRET_KEY = ""
ACCESS_TOKEN = ""
ACCESS_TOKEN_SECRET = ""

auth = tweepy.OAuthHandler(API_KEY,API_SECRET_KEY)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)


client = MongoClient('localhost', 27017)
database = client.tweets_database

RUN_TIME = 60
WOEID = 23424975
COLLECTION_NAME = "enhanced_crawler"
duplicates = 0
time_ended = False


#Coverts to date into a python object
def datetime_conversion(status):
    json_tweet = status._json
    json_tweet["Created"] = datetime.strptime(json_tweet["created_at"], '%a %b %d %H:%M:%S +0000 %Y')
    return json_tweet

# Insert tweet to the database and if there is a duplicate count them
def insert_to_database(tweet):
    try:
        database[COLLECTION_NAME].insert_one(tweet)
    except pymongo.errors.DuplicateKeyError:
        global duplicates
        duplicates = duplicates + 1



# Used for streaming, when a tweet comes in add it to the database
class Listener(tweepy.StreamListener):
    def on_status(self, status):
        tweet = datetime_conversion(status)
        insert_to_database(tweet)
        return True

    def on_error(self, status_code):
        if status_code == 420:
            return False
        print(status_code)


# Checks the amount of tweets
def tweet_volume(tweets):
    if tweets is None:
        return 0
    return tweets


# Gets all the trends in the county and querys to find them tweets and insert them into the database
def trends_in_country():
    country_trends = api.trends_place(WOEID)
    data = country_trends[0] 
    trends = data['trends']
    sorted_trends = sorted(trends, key=lambda k: tweet_volume(k['tweet_volume']))
    for trend in sorted_trends:
        trend_name = trend["name"]
        for item in tweepy.Cursor(api.search, q=trend_name, count=150, lang="en").items():
            database[COLLECTION_NAME].insert_one(datetime_conversion(item))
            if time_ended:
                break
        if time_ended:
            break
           


api = tweepy.API(auth_handler=auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

listener = Listener(api=api)
stream = tweepy.Stream(auth=auth, listener=listener)

# Time to last for the hour set
start_time = datetime.now()
time_end = start_time + timedelta(minutes=RUN_TIME)

# Start the thread so the tweets from the trends come in
rest_thread = Thread(target=trends_in_country)
rest_thread.start()

# Sample tweets in english
stream.sample(languages=['en'], is_async=True)

while datetime.now() < time_end:
    time.sleep(20)

time_ended = True
rest_thread.join()
stream.disconnect()

print("Start Time: ", start_time)
print("End Time: ", time_end)
print("Number of duplicates:", duplicates)




    
    

    