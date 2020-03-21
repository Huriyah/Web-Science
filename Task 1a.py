import pymongo
from pymongo import MongoClient
import json
import twitter
import tweepy
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
COLLECTION_NAME = "Stream_Crawler"
duplicates = 0


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


# Time to last for the hour set
start_time = datetime.now()
time_end =  start_time + timedelta(minutes=RUN_TIME)

twitterStream = tweepy.Stream(auth, Listener())

# Sample tweets in english
twitterStream.sample(languages=["en"], is_async=True)

while datetime.now() < time_end:
    time.sleep(30)

twitterStream.disconnect()
print("Start Time: ", start_time)
print("End Time: ", time_end)
print("Number of duplicates:", duplicates)



