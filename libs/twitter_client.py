
import re
import tweepy
import os
from tweepy import OAuthHandler
from textblob import TextBlob
from libs.stream_event import MyStreamListener
if os.environ.get('TWITTER_CONSUMER_KEY') is None:
    from .config import *

#If the env keys are set the get from env
if os.environ.get('TWITTER_CONSUMER_KEY') is not None:
    TWITTER_CONSUMER_KEY = os.environ.get('TWITTER_CONSUMER_KEY')
    TWITTER_CONSUMER_SECRET = os.environ.get('TWITTER_CONSUMER_SECRET')
    TWITTER_ACCESS_TOKEN = os.environ.get('TWITTER_ACCESS_TOKEN')
    TWITTER_ACCESS_TOKEN_SECRET = os.environ.get('TWITTER_ACCESS_TOKEN_SECRET')

class TwitterClient(object):
    '''
    Generic Twitter Class for sentiment analysis.
    '''
    def __init__(self):
        '''
        Class constructor or initialization method.
        '''
        # keys and tokens from the Twitter Dev Console
        print("init", TWITTER_CONSUMER_KEY)
        # attempt authentication
        try:
            # create OAuthHandler object
            self.auth = OAuthHandler(TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET)
            # set access token and secret
            self.auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET)
            # create tweepy API object to fetch tweets
            self.api = tweepy.API(self.auth)
            # create tweepy Stream API

        except Exception as e:
            print("TwitterClient: __init__ Failed", e)
 
    def get_tweets(self, query, count = 10):
        '''
        Main function to fetch tweets and parse them.
        '''
        # empty list to store parsed tweets
        tweets = []

        try:
            # call twitter api to fetch tweets
            fetched_tweets = self.api.search(q = query, count = count)
            # parsing tweets one by one
            for tweet in fetched_tweets:
                # empty dictionary to store required params of a tweet
                parsed_tweet = {}
                # saving text of tweet
                parsed_tweet['text'] = tweet.text
                parsed_tweet['url'] = 'https://twitter.com/statuses/' + tweet.id_str
                if tweet.retweet_count > 0:
                    # if tweet has retweets, ensure that it is appended only once
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)
 
            # return parsed tweets
            return tweets
 
        except tweepy.TweepError as e:
            # print error (if any)
            print("Error : " + str(e))
 
    def get_location(self):
        print("Starting stream")
        # api.geo_search(query="USA", granularity="country")
        places = list(self.api.geo_search(query="New Zealand", granularity="country"))

        foundPlaceId = ""
        for place in places:
            if place.country_code == 'NZ':
                foundPlaceId = place.id
                return foundPlaceId 

    # - create dictionary of id to tweet_id
    # - create array of just tweet_ids
    # - send to API
    # - loop through return and get the valid tweet ids
    #   - return list of mapped db ids to update  
    def validate_tweets(self, tweets):
        # create dictionary and api
        tweet_db_map = {}
        tweet_ids = list()
        for tweet in tweets:
            tweet_id = tweet.get('tweet_id')
            tweet_del_status = {
                'id' : tweet.get('id'),
                'deleted' : True,
            }
            tweet_db_map[tweet_id] = tweet_del_status
            tweet_ids.append(tweet_id)
        
        #send to api
        tweets_statuses = self.api.statuses_lookup(tweet_ids)
        valid_tweets = list()
        for tweet in tweets_statuses:
            tweet_db_map[tweet.id_str]['deleted'] = False

        return list(map(lambda x: tweet_db_map[x] ,tweet_db_map))

    #https://twitter.com/intent/user?user_id=XXXX
    def validate_users(self, tweets):
        # create dictionary and api
        tweet_db_map = {}
        user_ids = list()
        for tweet in tweets:
            user_id = tweet.get('user_id')
            tweet_del_status = {
                'id' : tweet.get('id'),
                'deleted' : True,
            }
            tweet_db_map[user_id] = tweet_del_status
            user_ids.append(user_id)
        
        tweets_statuses = self.api.lookup_users(user_ids)
        valid_tweets = list()
        for tweet in tweets_statuses:
            tweet_db_map[tweet.id_str]['deleted'] = False

        return list(map(lambda x: tweet_db_map[x] ,tweet_db_map))


    def get_stream(self):   
        while True:
            self.stream = tweepy.Stream(auth = self.auth, listener=MyStreamListener())
            try:
                # Connect/reconnect the stream
                self.stream.filter(track=['https t co'], stall_warnings=True)
            except Exception as e:
                # Oh well, reconnect and keep trucking
                print("get_stream Exception:", e)
                self.stream.disconnect()
                continue
            except KeyboardInterrupt:
                # Or however you want to exit this loop
                self.stream.disconnect()
                break
