import tweepy 
import json
import os
from libs.links import get_links_text
from libs.sns import send_sns
from datetime import datetime

if os.environ.get('TOPIC_METADATA') is None:
    from .config import *

#If the env keys are set the get from env
if os.environ.get('TOPIC_METADATA') is not None:
    TOPIC_METADATA = os.environ.get('TOPIC_METADATA')

class MyStreamListener(tweepy.StreamListener):
    def __init__(self):
        super(MyStreamListener, self).__init__()
        self.valid_countries = ['AU', 'NZ', 'SG', 'US']

    #create_at is utc
    def on_status(self, status):
        try:
            if 'RT' not in status.text[0:3] and status.user.lang == 'en' and status.place is not None and status.place.country_code in self.valid_countries:
                links = get_links_text(status.text)
                if links:
                    status._json['created_at'] = status.created_at.isoformat()
                    json_string = json.dumps(status._json)
                    send_sns(TOPIC_METADATA, json_string)
        except Exception as e:
            print('[' + datetime.utcnow().isoformat() + ']'+ "Error: Stream on_status:", e)