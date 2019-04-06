from libs.links import get_links_text, get_link_domain
from datetime import datetime
'''
    Library of features for tweets
    - Follower count (int) | user.followers_count
    - !Following count (int)   | is this friends_count? check online profile as ref 
    - Age of account in days (int) | date today - user.created_at in days
    - Favourites count (int) | user.favourites_count
    - !Subscribed to Lists count (int) | is this listed_count? check online profile as ref
    - Description exists (bool) | user.description
    - Is verified user (bool) | user.verified
'''


class UserFeatures():
    def __init__(self, tweet):
        user = tweet.get('user')
        self.follower_count = user.get('followers_count')
        self.following_count = user.get('friends_count')
        created_at = datetime.strptime(
            user.get('created_at'), "%a %b %d %H:%M:%S %z %Y").date()
        date_now = datetime.now().date()
        self.account_age = (date_now - created_at).days
        self.lists_count = user.get('listed_count')
        self.favourites_count = user.get('favourites_count')
        self.description_exists = user.get('description') is not None
        self.is_verified = user.get('verified')


'''
    Library of features for tweets
    - !Message length (int) | get length of tweet.text
    - Retweet count (int) | tweet.retweet_count
    - Favourites count (int) | tweet.favorite_count
    - URL count (int) | custom - tweet.links
    - Hashtag count (int) | len(tweet.text.split('#'))-1
    - Mention count (int) | len(tweet.text.split('@'))-1
    - Geolocation Exists (int) | tweet.geo_enabled (always due to Tweet Country req)
    - !Is sensitive in metadata (int) | ?
'''


class TweetFeatures():
    def __init__(self, tweet, links_len):
        self.tweet_len = len(tweet.get('text'))
        self.retweet_count = tweet.get('retweet_count')
        self.favorite_count = tweet.get('favorite_count')
        self.url_count = links_len
        self.hashtag_count = tweet.get('text').count('#')
        self.mention_count = tweet.get('text').count('@')
        self.geolocation_exists = tweet.get('user').get('geo_enabled')


'''
    Library of domain features
    - URL length | Number of characters found in a URL
    - Domain length | Number of characters found in the domain of a URL 
    - Redirection count |  Number of redirection hops between the initial URL and the final URL
    - Dot count | Number of dots (.) found in a URL. For example, www.auckland.ac.nz has 3 dots.
'''


class DomainFeatures():
    def __init__(self, link, redirects):
        self.url_len = len(link)
        self.domain_len = len(get_link_domain(link))
        self.url_dot_count = link.count('.')
        self.redirect_count = redirects