"""
A simple example script to get all posts on a user's timeline.
Originally created by Mitchell Stewart.
<https://gist.github.com/mylsb/10294040>
"""
import facebook
import requests
import json
import urllib.request
data = []

def some_action(post):
    """ Here you might want to do something with each post. E.g. grab the
    post's message (post['message']) or the post's picture (post['picture']).
    In this implementation we just print the post's created time.
    """
    data.append(post)

# You'll need an access token here to do anything.  You can get a temporary one
# here: https://developers.facebook.com/tools/explorer/
access_token = 'EAACEdEose0cBABRkHA3B7WWSIUOEH8qOjdrzUeWlL5iZBHgSX6lP4N5WEBasGxVU8si62dXH5LDA3ZCLed5B4mBWxFnP7UmYYOWGuNZCX4zEoqYoQFDsSaNRf7j3QmoNxw6PYhXnd1rJzDXBOtOZAJFrYultSfXOOXS17L3x5gZDZD'
# Look at Bill Gates's profile for this example by using his Facebook id.
user = 'eclipse'

urleey = "https://graph.facebook.com/oauth/access_token?client_id=1796256433992124&client_secret=b77c8e647c2802263f1d2a54ca8a7305&grant_type=client_credentials"

response = requests.get(urleey)
token = response.text.replace('access_token=', '')

graph = facebook.GraphAPI(token)
profile = graph.get_object(user)
posts = graph.get_connections(profile['id'], 'posts')

# Wrap this block in a while loop so we can keep paginating requests until
# finished.
while True:
    try:
        # Perform some action on each post in the collection we receive from
        # Facebook.
        [some_action(post=post) for post in posts['data']]
        # Attempt to make a request to the next page of data, if it exists.
        posts = requests.get(posts['paging']['next']).json()
    except KeyError:
        # When there are no more pages (['paging']['next']), break from the
        # loop and end the script.
        break

with open('data.json', 'w') as outfile:
        json.dump(data, outfile)