import os
import sys
here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(here, "../vendored"))

import psycopg2
import pandas as pd

from libs.links import get_link_domain, get_without_prefix
import re

if os.environ.get('DB_HOST') is None:
    from .config import *

#If the env keys are set the get from env
if os.environ.get('DB_HOST') is not None:
    DB_HOST = os.environ.get('DB_HOST')
    DB_USER = os.environ.get('DB_USER')
    DB_NAME = os.environ.get('DB_NAME')
    DB_PASSWORD = os.environ.get('DB_PASSWORD')

whitelist = [
    'youtu.be', 
    'm.youtube', 
    'youtube',
    'google',
    'youtu', 
    'instagram',
    'facebook',
    'twitch',
    'spotify',
    'shazam',
    'apple',
    'soundcloud',
    'swarmapp',
    'reddit',
    '511ny',
    'careerarc',
    'businessinsider',
    'pdora',
    'pandora',
    'sigalert',
    'aimhightips',
    'cnn',
    'nationalgeographic',
    'nytimes',
    'news5cleveland',
    'bigleaguepolitics',
    'untp',
    'nyti',
    'wikipedia',
    'foxnews',
    'cnbc',
    'djwilliedynamite',
    'nypost',
    'vimeo',
    'scientificamerican',
    'chuffed',
    'secondnexus',
    'relrules',
    'politi',
    'huffingtonpost',
    'abc',
    'amzn',
    'linkedin',
    'masslive',
    'bbc',
    'huffpost',
    'vox',
    'gq',
    'buzzfeed',
    'washingtonpost',
    'newsweek',
    'bible',
    'medium',
    'kob',
    'fox26houston',
    'vine',
    'nbcnews',
    'amazon',
    'yahoo',
    'pscp',
    'curvesofcocoa',
    'theguardian',
    'pscp',
    'thehill',
    'nylon',
    'army',
    'townhall',
    'endomondo',
    'zerohedge',
    'wired',
    'thoughtco',
    'integrallife',
    'imgur',
    'mytherapist',
    'armstronglegal',
    'zendesk',
    'newsok',
    'ny1',
    'celebritylovess',
    'thetalentedmsrealestat',
    ]

class DbClass():
    def __init__(self):
        print("DB_HOST", DB_HOST)

    def connect(self):
        return psycopg2.connect("""
            host={} 
            dbname={}
            user={}
            password={}
        """.format(DB_HOST, DB_NAME, DB_USER, DB_PASSWORD))

    def get_untagged_tweets(self, conn, from_date):
        sql = """
            SELECT id, tweet_id, tweet_raw->'user'->'id_str' as user_id
                FROM tweets
                WHERE created_at <= '{}' AND (tweet_deleted IS NULL OR user_deleted is NULL);
        """.format(from_date)
        print("sql",sql)
        cur = conn.cursor()
        cur.execute(sql)
        untagged_tweets = results = cur.fetchall()
        formatted_tweets = list(
            map(lambda x: {
                "id": x[0],
                "tweet_id": x[1],
                "user_id": x[2], }, untagged_tweets))
        cur.close()
        return formatted_tweets

    def get_max_migrated_tweet(self, conn):
        sql = """
            SELECT MAX(created_at) from tweet_features;
        """
        cur = conn.cursor()
        cur.execute(sql)
        results = cur.fetchall()
        cur.close()
        return results[0][0]

    def get_max_deleted_tweet(self, conn):
        sql = """
            SELECT MAX(created_at)
            FROM tweets
            WHERE tweet_deleted is not null
        """
        cur = conn.cursor()
        cur.execute(sql)
        results = cur.fetchall()
        cur.close()
        return results[0][0]

    def get_tweet_raw(self, conn, from_date, end_date):
        columns = [
            'tweet_raw', 'tweet_id', 'created_at', 'county_code', 'tweet_text',
            'tweet_deleted', 'url_1', 'url_1_redirects'
        ]
        column_sql = str.join(", ", map(lambda x: '{}'.format(x), columns))
        sql = """
            SELECT {columns}
                FROM tweets 
                WHERE created_at >= '{from_date}' AND created_at <= '{end_date}'
                AND url_1_retweet = FALSE;
        """.format(
            from_date=from_date, end_date=end_date, columns=column_sql)
        cur = conn.cursor()
        cur.execute(sql)
        tweet_raw = results = cur.fetchall()
        cur.close()
        df = pd.DataFrame(data=tweet_raw, columns=columns)
        return df

    def update_tweet_del_status(self, conn, tweets):
        sql = '''
            UPDATE tweets 
            SET tweet_deleted = %(deleted)s
            WHERE id = %(id)s
        '''
        cur = conn.cursor()
        cur.executemany(sql, tweets)
        conn.commit()
        cur.close()

    def update_user_del_status(self, conn, tweets):
        sql = '''
            UPDATE tweets 
            SET user_deleted = %(deleted)s
            WHERE id = %(id)s
        '''
        cur = conn.cursor()
        cur.executemany(sql, tweets)
        conn.commit()
        cur.close()

  

                  
    def save_tweet(self, conn, tweet):
        sql = """
        INSERT INTO tweets
        (
            tweet_id,
            created_at,
            tweet_text,
            county_code,
            google_unsafe,
            url_1,
            url_2,
            url_3,
            url_1_retweet,
            url_2_retweet,
            url_3_retweet,
            url_1_redirects,
            url_2_redirects,
            url_3_redirects,
            url_1_redirects_chain,
            url_2_redirects_chain,
            url_3_redirects_chain,
            tweet_raw
        )
        VALUES
            (%s, %s, %s, %s, %s,
             %s, %s, %s, 
             %s, %s, %s, 
             %s, %s, %s, 
             %s, %s, %s, 
             %s);
        """
        cur = conn.cursor()
        cur.execute(
            sql, (tweet.get('tweet_id'), tweet.get('created_at'),
                  tweet.get('tweet_text'), tweet.get('county_code'),
                  tweet.get('google_unsafe'), tweet.get('url_1'),
                  tweet.get('url_2'), tweet.get('url_3'),
                  tweet.get('url_1_retweet'), tweet.get('url_2_retweet'),
                  tweet.get('url_3_retweet'), tweet.get('url_1_redirects'),
                  tweet.get('url_2_redirects'), tweet.get('url_3_redirects'),
                  tweet.get('url_1_redirects_chain'),
                  tweet.get('url_2_redirects_chain'),
                  tweet.get('url_3_redirects_chain'), tweet.get('tweet_raw')))

        conn.commit()
        cur.close()
        print("saved_tweet", tweet.get('tweet_text'))

    def format_feature_insert(self, tweet_features, cur):
        print("format_feature_insert")
        tweet_tuple = (int(tweet_features.get('tweet_id')),
                       int(tweet_features.get('user_follower_count')),
                       int(tweet_features.get('user_following_count')),
                       int(tweet_features.get('user_lists_count')),
                       int(tweet_features.get('user_favourites_count')),
                       int(tweet_features.get('user_account_age')),
                       bool(tweet_features.get('user_description_exists')),
                       bool(tweet_features.get('user_is_verified')),
                       int(tweet_features.get('tweet_len')),
                       int(tweet_features.get('tweet_retweet_count')),
                       int(tweet_features.get('tweet_favorite_count')),
                       int(tweet_features.get('tweet_url_count')),
                       int(tweet_features.get('tweet_hashtag_count')),
                       int(tweet_features.get('tweet_mention_count')),
                       bool(tweet_features.get('tweet_geolocation_exists')),
                       int(tweet_features.get('domain_url_len')),
                       int(tweet_features.get('domain_domain_len')),
                       int(tweet_features.get('domain_url_dot_count')),
                       int(tweet_features.get('domain_redirect_count')),
                       tweet_features.get('created_at'),
                       tweet_features.get('county_code'),
                       tweet_features.get('tweet_text'),
                       bool(tweet_features.get('tweet_deleted')))
        args_str = cur.mogrify(
            """ (%s,
            %s, %s, %s, %s, %s, %s, %s, 
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s
            )""", tweet_tuple)

        return args_str.decode(encoding='UTF-8')

    def save_tweet_features_migration(self, conn, tweet_inserts, cur):
        sql = """
        INSERT INTO tweet_features
        (
            tweet_id,

            user_follower_count,
            user_following_count,
            user_lists_count,
            user_favourites_count,
            user_account_age,
            user_description_exists,
            user_is_verified,

            tweet_len,
            tweet_retweet_count,
            tweet_favorite_count,
            tweet_url_count,
            tweet_hashtag_count,
            tweet_mention_count,
            tweet_geolocation_exists,

            domain_url_len,
            domain_domain_len,
            domain_url_dot_count,
            domain_redirect_count,

            created_at,
            county_code,
            tweet_text,
            tweet_deleted
        )
        VALUES {};
        """.format(tweet_inserts)
        print("inserting")
        cur.execute(sql)

    #Main method to get the tweets
    def get_features(self, conn, from_date=None, to_date=None, limit=False):
        columns = [
            'tweet_id',
            'user_follower_count',
            'user_following_count',
            'user_lists_count',
            'user_favourites_count',
            'user_account_age',
            'user_description_exists',
            'user_is_verified',
            'tweet_len',
            # 'tweet_retweet_count',
            # 'tweet_favorite_count',
            'tweet_url_count',
            'tweet_hashtag_count',
            'tweet_mention_count',
            'tweet_geolocation_exists',
            'domain_url_len',
            'domain_domain_len',
            'domain_url_dot_count',
            'domain_redirect_count',
            'created_at',
            'county_code',
            'tweet_text',
            'tweet_deleted'
        ]
        column_sql = str.join(", ", columns)
        sql = """
            SELECT {columns}
                FROM tweet_features
        """.format(columns=column_sql)
        if from_date is not None and to_date is not None:
            sql += """ WHERE created_at >= '{from_date}' AND created_at < '{to_date}'
                """.format(
                from_date=from_date, to_date=to_date)
        if limit:
            sql += 'LIMIT {}'.format(limit)
        sql += ';'
        print(sql)
        cur = conn.cursor()
        cur.execute(sql)
        tweet_raw = results = cur.fetchall()
        cur.close()
        return pd.DataFrame(data=tweet_raw, columns=columns)
    #Main method to get the tweets

    def get_features_tweet(self, conn, from_date=None, to_date=None, sample=False, label="user_deleted", relabel=True):
        columns = [
            'tweet_id',
            'user_follower_count',
            'user_following_count',
            'user_lists_count',
            'user_favourites_count',
            'user_account_age',
            'user_description_exists',
            'user_is_verified',
            'tweet_len',
            # 'tweet_retweet_count',
            # 'tweet_favorite_count',
            'tweet_url_count',
            'tweet_hashtag_count',
            'tweet_mention_count',
            # 'tweet_geolocation_exists',
            'domain_url_len',
            'domain_domain_len',
            'domain_url_dot_count',
            'domain_redirect_count',
            'created_at',
            'county_code',
            'tweet_text',
            'tweet_deleted',
        ]
        columns_select = map(lambda x: 'tf.'+x, columns)
        column_sql = str.join(", ", columns_select)
        sql = """
            SELECT t.url_1 as url, {columns}, t.user_deleted, t.google_unsafe
                FROM tweet_features tf
                INNER JOIN tweets t ON t.tweet_id = tf.tweet_id
        """.format(columns=column_sql)
        if from_date is not None and to_date is not None:
            sql += """ WHERE tf.created_at >= '{from_date}' AND tf.created_at < '{to_date}'
                """.format(
                from_date=from_date, to_date=to_date)
        sql += ';'
        print(sql)
       
        cur = conn.cursor()
        cur.execute(sql)
        tweet_raw = results = cur.fetchall()
        cur.close()
        columns.insert(0, 'url')
        columns.append("user_deleted")
        columns.append('google_unsafe')
        df = pd.DataFrame(data=tweet_raw, columns=columns)
        df['url_temp'] = df['url'].apply(lambda x: get_link_domain(x))
        df.url_temp
        # Transform google unsafe into user_Deleted (true label)
        google_unsafe = df[df['google_unsafe'] == True]
        google_unsafe[label] = True
        print("google_unsafe", google_unsafe)

        # Get the phishing tweets, filter them out, use a concat of the filtered and non-phishing
        df_phishing = df[df[label] == True]
        df_non_phishing = df[df[label] == False]
        df_non_phishing = df_non_phishing.sort_values(by=['created_at'])

        if sample:
            df_non_phishing = df_non_phishing.sample(frac=0.3, random_state=123)
        print("df", df.shape)
        print("df_phishing shape", df_phishing.shape)
        print("df_non_phishing", df_non_phishing.shape)

        # Whitelisting items
        df_non_phishing_whitelist = df_phishing[df_phishing['url_temp'].isin(whitelist)]
        df_phishing_whitelist = df_phishing[~df_phishing['url_temp'].isin(whitelist)]
        if relabel:
            df_non_phishing_whitelist[label] = False
        # whitelist_count = self.get_counts(df_phishing_whitelist)
        df = df_non_phishing.append([df_phishing_whitelist, df_non_phishing_whitelist])
        df['tweet_text'] = df.apply(lambda x: re.sub(r'http\S+', get_without_prefix(x['url']), x['tweet_text']), axis=1)
        df['tweet_text'] = df['tweet_text'].apply(
            lambda x: x.replace('gt', '').replace('.', ' ').replace('/','').replace(':',''))
        df = df.drop(['google_unsafe', 'url_temp'],axis=1)
        # self.printCsv(df, from_date, "tweet_text")
        if label == "user_deleted":
            df['tweet_deleted'] = df[label]
            df = df.drop([label],axis=1)
        print("df", df.shape)

        df = df.sort_values(by=['created_at'])
        return df

    def get_counts(self, df):
        df["time"]= df["created_at"].apply(lambda x: x[0:10])
        count = df.groupby("time")['user_deleted'].count()
        df = df.drop(['time'],axis=1)
        return count

    def get_deleted_tweets(self, conn):
        sql = """
            SELECT created_at, tweet_id, tweet_text, url_1, county_code, tweet_deleted, user_deleted 
                FROM tweets 
                WHERE url_1_retweet = false AND user_deleted = true
                order by created_at;
        """

        cur = conn.cursor()
        cur.execute(sql)
        tweets = cur.fetchall()
        cur.close()
        return tweets

    def get_deleted_tweets_count(self, conn, from_date, to_date):
        print("hello", from_date, to_date)
        sql = '''
            SELECT COUNT(*),
            SUM(case when user_deleted = true then 1 else 0 END) as phishing,
            SUM(case when user_deleted = false then 1 else 0 END) as non_phishing
                FROM tweets
                WHERE created_at >= '{from_date}' AND created_at < '{to_date}';
        '''.format(from_date=from_date, to_date=to_date)
        cur = conn.cursor()
        cur.execute(sql)
        tweets = cur.fetchall()
        cur.close()
        return tweets


    def get_del_tweet_text(self, conn, from_date, to_date):
        print("hello", from_date, to_date)
        sql = ''' 
            SELECT county_code, created_at, tweet_text, user_deleted, tweet_deleted, url_1 as url
                FROM tweets 
                WHERE url_1_retweet = false AND tweet_deleted = true
                AND created_at >= '{from_date}' AND created_at < '{to_date}';
        '''.format(from_date=from_date, to_date=to_date)
        print("sql",sql)
        cur = conn.cursor()
        cur.execute(sql)
        tweet_raw = results = cur.fetchall()
        cur.close()
        df = pd.DataFrame(data=tweet_raw, columns=["county_code","created_at","tweet_text", "user_deleted", "tweet_deleted", "url"])
        df['url_temp'] = df['url'].apply(lambda x: get_link_domain(x))

        # Get the phishing tweets, filter them out, use a concat of the filtered and non-phishing
        df_phishing = df[df['tweet_deleted'] == True]
        print(df_phishing.shape)
        # df_phishing = df[df['user_deleted'] == True]

        # Whitelisting items
        df_phishing_whitelist = df_phishing[~df_phishing['url_temp'].isin(whitelist)]
        df = df_phishing
        df['tweet_text'] = df.apply(lambda x: re.sub(r'http\S+', get_without_prefix(x['url']), x['tweet_text']), axis=1)
        df['tweet_text'] = df['tweet_text'].apply(
            lambda x: x.replace('gt', '').replace('.', ' ').replace('/','').replace(':',''))

        return df   

    
    def get_deleted_tweets_csv(self, conn, from_date, to_date):
        sql = """
            SELECT t.url_1 as url, t.created_at, t.tweet_id, t.tweet_text, t.county_code, t.user_deleted 
                FROM tweets t
                INNER JOIN tweet_features tf ON t.tweet_id = tf.tweet_id
                WHERE 
                t.created_at >= '{from_date}' AND t.created_at < '{to_date}' AND t.user_deleted = true
                order by t.created_at;
        """.format(from_date=from_date, to_date=to_date)
        print(sql)
        cur = conn.cursor()
        cur.execute(sql)
        tweets = cur.fetchall()
        df = pd.DataFrame(tweets,
        columns=[
            'url', 'created_at', 'tweet_id', 'tweet_text', 'county_code',
            'user_deleted'
        ])

         # Whitelisting items
        df['url_temp'] = df['url'].apply(lambda x: get_link_domain(x))
        df['whitelist'] = df['url_temp'].apply(lambda x: x.lower() in whitelist)


        cur.close()
        return df

    def printCsv(self,df,from_date,name=""):
        # df = df.applymap(lambda x: x.encode('unicode_escape').
        #          decode('utf-8') if isinstance(x, str) else x)
        # df = df[df.url != 'https://www.instagram.com/p/BriNZqNFaF7/?utm_source=ig_twitter_share&igshid=390b40bb5qd8']
        writer = pd.ExcelWriter(
        './sample/clusters/del_tweets_{}_{}.xlsx'.format(from_date,name), engine='xlsxwriter',options={'strings_to_urls': False})
        df.to_excel(writer)
        writer.save()
