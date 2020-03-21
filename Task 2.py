""" Methods to get and extract important information from tweets was used from
https://github.com/ugis22/analysing_twitter/blob/master/Jupyter%20Notebook%20files/Interaction%20Network.ipynb?fbclid=IwAR1QYLmuh_PFNnj3CAbb86RsD98SeDz4aZ7IraiCBpsrWDO6H_B4dmjFwBo 

K-Means Clustering - https://pythonprogramminglanguage.com/kmeans-text-clustering/

Feature Extraction - https://towardsdatascience.com/k-means-clustering-8e1e64c1561c """
 
import pymongo
from pymongo import MongoClient
import json
import re
import twitter
import tweepy
import numpy
from threading import Thread
from pprint import pprint
from datetime import datetime
from datetime import timedelta
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import pandas as pd
import numpy as np
from scipy import stats
from operator import itemgetter
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx




client = MongoClient('localhost', 27017)
database = client.tweets_database


df = pd.DataFrame(list(database['Partition Data'].find()))

# To clean the tweets for clustering
def clean_tweet(tweet): 
        tweet.replace('RT', '')
        tweet.replace('rt', '')
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", tweet).split())



tweets_dataframe = pd.DataFrame(columns = ["created_at", "id", "in_reply_to_screen_name", "in_reply_to_status_id", "in_reply_to_user_id",
                                      "retweeted_id", "retweeted_screen_name", "user_mentions_screen_name", "user_mentions_id", "hashtags"
                                       "text", "user_id", "screen_name", "followers_count", "cluster"])




equal_columns = ["created_at", "id", "text"]
tweets_dataframe[equal_columns] = df[equal_columns]


#Code help from references above
def get_basics(tweets_dataframe):
    tweets_dataframe["screen_name"] = df["user"].apply(lambda x: x["screen_name"])
    tweets_dataframe["user_id"] = df["user"].apply(lambda x: x["id"])
    tweets_dataframe["followers_count"] = df["user"].apply(lambda x: x["followers_count"])
    return tweets_dataframe


# Get the user mentions, with screen name and id
def get_usermentions(tweets_dataframe):
    tweets_dataframe["user_mentions_screen_name"] = df["entities"].apply(lambda x: x["user_mentions"][0]["screen_name"] if x["user_mentions"] else np.nan)
    tweets_dataframe["user_mentions_id"] = df["entities"].apply(lambda x: x["user_mentions"][0]["id_str"] if x["user_mentions"] else np.nan)
    return tweets_dataframe

# Get the user mentions , with screen name and id
def get_hashtags(tweets_final):
    tweets_dataframe["hashtags"] = df["entities"].apply(lambda x: x["hashtags"][0]["text"] if x["hashtags"] else np.nan)
    return tweets_dataframe


# Get retweets, with screen name and id
def get_retweets(tweets_dataframe):
    tweets_dataframe["retweeted_screen_name"] = df["retweeted_status"].apply(lambda x: x["user"]["screen_name"] if x is not np.nan else np.nan)
    tweets_dataframe["retweeted_id"] = df["retweeted_status"].apply(lambda x: x["user"]["id_str"] if x is not np.nan else np.nan)
    return tweets_dataframe

# Get the information about replies
def get_in_reply(tweets_final):
    tweets_dataframe["in_reply_to_screen_name"] = df["in_reply_to_screen_name"]
    tweets_dataframe["in_reply_to_status_id"] = df["in_reply_to_status_id"]
    tweets_dataframe["in_reply_to_user_id"]= df["in_reply_to_user_id"]
    return tweets_dataframe

#Fill in the new dataframe with information
def populate_dataframe(tweets_dataframe):
    get_basics(tweets_dataframe)
    get_usermentions(tweets_dataframe)
    get_retweets(tweets_dataframe)
    get_hashtags(tweets_dataframe)
    get_in_reply(tweets_dataframe)
    return tweets_dataframe

# Get the interactions between the different users
def get_interactions(row):
   
    user = row["user_id"], row["screen_name"]
   
    if user[0] is None:
        return (None, None), []
   
    interactions = set()
    

    interactions.add((row["in_reply_to_user_id"], row["in_reply_to_screen_name"]))
   
    interactions.add((row["retweeted_id"], row["retweeted_screen_name"]))
  
    interactions.add((row["user_mentions_id"], row["user_mentions_screen_name"]))
    
    interactions.discard((row["user_id"], row["screen_name"]))
    
    interactions.discard((None, None))

    return user, interactions

tweets_dataframe = populate_dataframe(tweets_dataframe)
tweets_dataframe = tweets_dataframe.where((pd.notnull(tweets_dataframe)), None)




# Clean the tweets to get rid of useless information

text = tweets_dataframe['text']
for t in text:
    clean_tweet(t)


# Defining vectorising parameters
vectorizer = TfidfVectorizer(stop_words='english')
x = vectorizer.fit_transform(text)
tf_idf_array = x.toarray()

true_k = 5

model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(x)

prediction = model.predict(x)

clusters = model.labels_.tolist()

#Putting clusters into the tweets dataframe 
tweets_dataframe['cluster'] = clusters


# Printing the top 5 terms per cluster. Code help from references above
print("Top 5 terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :5]:
        print(' %s' % terms[ind]),
    print




# Top features for each cluster and scores and hashtags. Code help from references above
def get_top_features_cluster(tf_idf_array, prediction, num_of_features):
    labels = np.unique(prediction)
    top_features = []
    for label in labels:
        # Finding indices where each cluster has the label
        id_temp = np.where(prediction==label) 

        # Average score across cluster
        average_score = np.mean((tf_idf_array[id_temp]), axis = 0) 

        # Top score indices 
        sorted_means = np.argsort(average_score)[::-1][:num_of_features] 
        
        features = vectorizer.get_feature_names()

        # Put best features into a dataframe with their scores 
        best_features = []
        for i in sorted_means:
            best_features.append((features[i], average_score[i]))
        df = pd.DataFrame(best_features, columns = ['features', 'score'])
        top_features.append(df)


        # Return top hashtag for that cluster
        hashtags_with_the_label = tweets_dataframe[tweets_dataframe.cluster.isin([label])]
        print("Size of Cluster " + str(label) + "  ",  hashtags_with_the_label.size)
        top_hashtags = hashtags_with_the_label['hashtags'].value_counts().idxmax()
        print("Top hashtags for cluster " + str(label) + " ",  top_hashtags)


        # Plot the best features with their scores
        plt.figure(dpi=100)
        plt.title("Cluster" + " " + str(label), fontsize=18)
        plt.bar(df['features'],df['score'], color=(0.2, 0.4, 0.6, 0.6))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel("Top Features", fontsize = 16)
        plt.ylabel("Score", fontsize = 16)
        plt.show()
    return top_features

dfs = get_top_features_cluster(tf_idf_array, prediction, 10)


print(dfs)