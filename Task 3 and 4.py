""" Methods to get and extract important information from tweets was used from and graph interaction
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
from itertools import combinations
import jgraph as ig
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
def fill_df(tweets_dataframe):
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

tweets_dataframe = fill_df(tweets_dataframe)
tweets_dataframe = tweets_dataframe.where((pd.notnull(tweets_dataframe)), None)


#initialise graph
graph = nx.Graph()


    
#Adding an edge to both user ids. Code help from references above
for index, tweet in tweets_dataframe.iterrows():
    user, interactions = get_interactions(tweet)
    user_id, user_name = user
    tweet_id = tweet["id"]
    for interaction in interactions:
        int_id, int_name = interaction
        graph.add_edge(user_id, int_id, tweet_id=tweet_id)
        

        graph.node[user_id]["name"] = user_name
        graph.node[int_id]["name"] = int_name



# Degrees the graph has
degrees = [val for (node, val) in graph.degree()]

largest_subgraph = max(nx.connected_component_subgraphs(graph), key=len)

graph_centrality = nx.degree_centrality(largest_subgraph)

max_degree = max(graph_centrality.items(), key=itemgetter(1))

graph_closeness = nx.closeness_centrality(largest_subgraph)

max_closeness = max(graph_closeness.items(), key=itemgetter(1))

graph_betweenness = nx.betweenness_centrality(largest_subgraph, normalized=True, endpoints=False)


max_bet = max(graph_betweenness.items(), key=itemgetter(1))



# Change the graph to directed to get the triads
directed_graph = nx.triadic_census(graph.to_directed())
print(directed_graph)


node_and_degree = largest_subgraph.degree()
colors_central_nodes = ['blue', 'red']
central_nodes = [max_degree[0],max_closeness[0] ]

pos = nx.spring_layout(largest_subgraph, k=0.05)


plt.figure(figsize = (20,20))
nx.draw(largest_subgraph, pos=pos, edge_color="black", linewidths=0.3, node_size=60, alpha=0.6, with_labels=False)
nx.draw_networkx_nodes(largest_subgraph, pos=pos, nodelist=central_nodes, node_size=300, node_color=colors_central_nodes)
plt.savefig('interaction_graph.png')
plt.show() 