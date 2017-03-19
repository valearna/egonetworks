"""this module is specialized for the representation of Twitter ego networks and their analysis"""

import os

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from egonetworks.core import ContactEgoNetwork, ContactAdditionalInfo
from egonetworks.error import *
from egonetworks.generic import DictReaderWithHeader, memoize
from typing import List, Dict, Iterator, Hashable, Tuple, Union
from collections import namedtuple, defaultdict
import csv
import json
import time
import datetime
import tweepy


__author__ = "Valerio Arnaboldi"
__license__ = "MIT"
__version__ = "1.0.1"


class TwitterUserClassifier(object):
    """Classify Twitter egos.

    This class represents a classifier used to divide Twitter accounts into different categories of users.
    """

    def __init__(self):
        """Create an empty classifier"""
        self.classifier = None
        self.accuracy = None

    def train_random_forest_from_csv(self, filename: str, sep: str = ' ', quote: str = '"', header: bool = False):
        """Train a random forest classifier from a csv file.

        :param filename: path to the csv file containing the classification dataset for training
        :param sep: the field separator of the csv file
        :param quote: field quoting used in the csv file
        :param header: indicates whether a header is present in the csv file or nor
        :type filename: str
        :type sep: str
        :type quote: str
        :type header: bool
        """
        x, y = TwitterUserClassCsvReader.read(filename, sep, quote, header)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)
        self.classifier = RandomForestClassifier()
        self.classifier.fit(x_train, y_train)
        self.accuracy = self.classifier.score(x_test, y_test)
        self.classifier.fit(x, y)

    def train_default_profile_classifier(self):
        """Train a classifier with the default training set provided with the module."""
        this_dir, this_filename = os.path.split(__file__)
        train_filepath = os.path.join(this_dir, "..", "datasets", "classified_profiles.csv")
        self.train_random_forest_from_csv(train_filepath, header=True)

    def classify_user_from_profile(self, ego) -> int:
        """Classify a user using its profile, with the trained classifier, if any.

        :param ego: a TwitterEgoNetwork object representing the user to classify
        :type ego: TwitterEgoNetwork
        :return: the class of the user
        :rtype: int
        """
        if self.classifier is not None:
            x = ego.get_user_profile_array()
            x = x.reshape(1, -1)
            return self.classifier.predict(x)[0]
        else:
            return None


class TweetAdditionalInfo(namedtuple("TweetAdditionalInfo", ["user_mentions", "hashtags", "urls"]),
                          ContactAdditionalInfo):
    """This class defines the specific format for additional information for tweets, such as the presence of hashtags,
    urls and user mentions."""

    def __new__(cls, user_mentions: List=None, hashtags: List=None, urls: List=None):
        if user_mentions is None:
            user_mentions = []
        if hashtags is None:
            hashtags = []
        if urls is None:
            urls = []
        return super().__new__(cls, user_mentions=frozenset(user_mentions), hashtags=frozenset(hashtags),
                               urls=frozenset(urls))

    def to_json_encoded_obj(self):
        """Convert the object into JSON."""
        tw_add_info_dict = self._asdict()
        tw_add_info_dict["user_mentions"] = list(self.user_mentions)
        tw_add_info_dict["hashtags"] = list(self.hashtags)
        tw_add_info_dict["urls"] = list(self.urls)
        return tw_add_info_dict

    @staticmethod
    def from_json(obj: Dict):
        """Load an object from a JSON formatted Dict.

        :param obj: The JSON formatted dictionary from which to load the object
        :type obj: Dict
        :return: the loaded object
        :rtype: TweetAdditionalInfo
        """
        return TweetAdditionalInfo(user_mentions=obj["user_mentions"], hashtags=obj["hashtags"], urls=obj["urls"])


class TwitterEgoNetwork(ContactEgoNetwork):
    """Twitter ego network.

    This is a special EgoNetwork class with properties to model Twitter user profile and contacts. See
    :class:`egonetworks.core.ContactEgoNetwork` for inherited methods.
    """

    def __init__(self, ego_id: int = None, last_time: int = 0, user_id: int = None,
                 screen_name: str = "", account_duration: int = 0, default_profile: bool = True,
                 default_profile_image: bool = True, description_not_null: bool = False, favourites_count: int = 0,
                 followers_count: int = 0, friends_count: int = 0, geo_enabled: bool = False, listed_count: int = 0,
                 location_not_null: bool = False, profile_background_image_not_null: bool = False,
                 protected: bool = False, statuses_count: int = 0, url_is_not_null: bool = False,
                 verified: bool = False):
        """Create a new TwitterEgoNetwork.

        :param ego_id: the id of the ego
        :param last_time: the time at which the data of the ego has been downloaded
        :param user_id: the Twitter id of the ego
        :param screen_name: the Twitter screen name of the ego
        :param account_duration: the duration of the Twitter account of the ego
        :param default_profile: whether the ego has a default Twitter profile or not
        :param default_profile_image: whether the ego has a default Twitter profile image or not
        :param description_not_null: whether the description of the Twitter profile of the ego is not null
        :param favourites_count: the number of tweets in the Twitter favourite list of the ego
        :param followers_count: the number of Twitter followers of the ego
        :param friends_count: the number of Twitter friends (followees) of the ego
        :param geo_enabled: whether the ego has a geo-enabled Twitter profile
        :param listed_count: the number of times the Twitter profile of the ego has been listed by other Twitter users
        :param location_not_null: whether the location field of the Twitter profile of the ego is not null
        :param profile_background_image_not_null: whether the background image of the Twitter profile of the ego is
            not null
        :param protected: whether the Twitter account of the ego is protected
        :param statuses_count: the number of tweets created by the ego
        :param url_is_not_null: whether the url in the Twitter profile of the ego is not null
        :param verified: whether the ego has a verified Twitter account
        :type ego_id: int
        :type last_time: int
        :type user_id: int
        :type screen_name: str
        :type account_duration: int
        :type default_profile: bool
        :type default_profile_image: bool
        :type description_not_null: bool
        :type favourites_count: int
        :type followers_count: int
        :type friends_count: int
        :type geo_enabled: bool
        :type listed_count: int
        :type location_not_null: bool
        :type profile_background_image_not_null: bool
        :type protected: bool
        :type statuses_count: int
        :type url_is_not_null: bool
        :type verified: bool
        :raise InvalidIdException: if user_id is negative
        :raise InvalidIntValueException: if one or more of the provided parameters that take integer values are negative
        :return a new ego network with Twitter extension
        :rtype: TwitterEgoNetwork
        """
        if user_id is not None and user_id < 0:
            raise InvalidIdException("Twitter user id cannot be negative")
        if account_duration < 0:
            raise InvalidIntValueException("Duration of Twitter account cannot be negative")
        if favourites_count < 0:
            raise InvalidIntValueException("Favourites count cannot be negative")
        if followers_count < 0:
            raise InvalidIntValueException("Followers count cannot be negative")
        if friends_count < 0:
            raise InvalidIntValueException("Friends count cannot be negative")
        if listed_count < 0:
            raise InvalidIntValueException("Listed count cannot be negative")
        if statuses_count < 0:
            raise InvalidIntValueException("Statuses count cannot be negative")

        super().__init__(ego_id, last_time)

        self.user_id = user_id
        self.screen_name = screen_name
        self.account_duration = account_duration
        self.default_profile = default_profile
        self.default_profile_image = default_profile_image
        self.description_not_null = description_not_null
        self.favourites_count = favourites_count
        self.followers_count = followers_count
        self.friends_count = friends_count
        self.geo_enabled = geo_enabled
        self.listed_count = listed_count
        self.location_not_null = location_not_null
        self.profile_background_image_not_null = profile_background_image_not_null
        self.protected = protected
        self.statuses_count = statuses_count
        self.url_is_not_null = url_is_not_null
        self.verified = verified

    def update_from_tweets_json_arr(self, tweets: str, overwrite_profile: bool = True, last_time: int = None):
        """Update ego network properties from a JSON array of Tweet Objects.

        The accepted format for tweet objects is the same as that defined by Twitter (see
        https://dev.twitter.com/overview/api/tweets) for more details.

        Note that account duration is automatically set as the number of seconds between the creation of the Twitter
        account and the actual time.

        :param tweets: a string containing a JSON array of tweet objects
        :param overwrite_profile: whether the Twitter profile data of the ego network has to be updated using
            information contained in the tweets
        :param last_time: the last time to be considered for the calculation of the ego network duration. If None, the
            actual time is used
        :type tweets: str
        :type overwrite_profile: bool
        :type last_time: int (*Unix Timestamp*)
        """
        tweets_dict = json.loads(tweets)
        for tweet in tweets_dict:
            if self.user_id is None and overwrite_profile:
                # read the profile information of the user from the first tweet of the array
                self.ego_id = tweet["user"]["id"]
                self.user_id = tweet["user"]["id"]
                self.screen_name = tweet["user"]["screen_name"]
                if last_time is None:
                    last_time = int(time.mktime(datetime.datetime.now().timetuple()))
                self.account_duration = last_time - int(time.mktime(
                    datetime.datetime.strptime(tweet["user"]["created_at"], "%a %b %d %H:%M:%S %z %Y").timetuple()))
                self.default_profile = tweet["user"]["default_profile"] == "true"
                self.default_profile_image = tweet["user"]["default_profile_image"] == "true"
                self.description_not_null = tweet["user"]["description"] != "null"
                self.favourites_count = tweet["user"]["favourites_count"]
                self.followers_count = tweet["user"]["followers_count"]
                self.friends_count = tweet["user"]["friends_count"]
                self.geo_enabled = tweet["user"]["geo_enabled"] == "true"
                self.listed_count = tweet["user"]["listed_count"]
                self.location_not_null = tweet["user"]["location"] != "null"
                self.profile_background_image_not_null = tweet["user"]["profile_background_image_url"] != "null"
                self.protected = tweet["user"]["protected"] == "true"
                self.statuses_count = tweet["user"]["statuses_count"]
                self.url_is_not_null = tweet["user"]["statuses_count"] != "null"
                self.verified = tweet["user"]["verified"] == "true"

            # identify the type of tweet (only direct communication is considered)
            contacted_alter_ids = []

            if tweet["in_reply_to_user_id"] is not None and tweet["in_reply_to_user_id"] != "null" \
                    and tweet["in_reply_to_status_id"] is not None and tweet["in_reply_to_status_id"] != "null" \
                    and tweet["in_reply_to_user_id"] != self.user_id and "entities" in tweet \
                    and len(tweet["entities"]["user_mentions"]) > 0:
                contact_type = "reply"
                # add the replied user to the contacts list
                contacted_alter_ids.append(int(tweet["in_reply_to_user_id"]))
            elif "retweeted_status" in tweet and tweet["retweeted_status"] is not None \
                    and tweet["retweeted_status"] != "" and tweet["retweeted_status"] != "null":
                contact_type = "retweet"
                # add the creator of the original tweet (retweeted by the ego) to the contacts list
                contacted_alter_ids.append(int(tweet["retweeted_status"]["user"]["id"]))
            elif "entities" in tweet and len(tweet["entities"]["user_mentions"]) > 0:
                contact_type = "mention"
                for mention in tweet["entities"]["user_mentions"]:
                    contacted_alter_ids.append(int(mention["id"]))
            else:
                contact_type = "__no_contact__"
                contacted_alter_ids.append(0)
            # additional tweet info
            tweet_entities = {"user_mentions": [], "hashtags": [], "urls": []}
            for mention in tweet["entities"]["user_mentions"]:
                tweet_entities["user_mentions"].append(int(mention["id"]))
            for hashtag in tweet["entities"]["hashtags"]:
                tweet_entities["hashtags"].append(hashtag["text"])
            for url in tweet["entities"]["urls"]:
                tweet_entities["urls"].append(url["display_url"])
            tweet_entities_frozen = TweetAdditionalInfo(user_mentions=tweet_entities["user_mentions"],
                                                        hashtags=tweet_entities["hashtags"],
                                                        urls=tweet_entities["urls"])
            for contacted_alter in contacted_alter_ids:
                self.add_contact(timestamp=int(time.mktime(datetime.datetime.strptime(tweet["created_at"],
                                                                                      "%a %b %d %H:%M:%S %z %Y")
                                                           .timetuple())),
                                 alter_id=contacted_alter, contact_type=contact_type, text=tweet["text"],
                                 additional_info=tweet_entities_frozen)
            downloaded_at = int(time.mktime(datetime.datetime.strptime(tweet["created_at"], "%a %b %d %H:%M:%S %z %Y")
                                            .timetuple()))
            if self.last_time is None or downloaded_at > self.last_time:
                self.last_time = downloaded_at

    # noinspection PyProtectedMember
    def update_from_twitter_api(self, user_id: int = None, screen_name: str = None, overwrite_profile: bool = True,
                                consumer_key: str = None, consumer_secret: str = None, access_token: str = None,
                                access_token_secret: str = None):
        """Download the tweets generated by the user and add them to the ego network.

        The tweets are retrieved from the timeline of the user identified by the provided Twitter user_id or
        screen_name through Twitter REST API. See https://dev.twitter.com/rest/reference/get/statuses/user_timeline for
        additional information on the API used by this method and https://dev.twitter.com/oauth/overview for a
        description of Twitter authentication process.

        :param user_id: the Twitter user_id of the account to be downloaded
        :param screen_name: the Twitter screen_name of the account to be downloaded
        :param overwrite_profile: whether the Twitter profile data of the ego network has to be updated using
            information contained in the tweets
        :param consumer_key: the consumer_key used to access Twitter API
        :param consumer_secret: the consumer_secret used to access Twitter API
        :param access_token: the access_token used to access Twitter API
        :param access_token_secret: the access_token_secret uset to access Twitter API
        :type user_id: int
        :type screen_name: str
        :type overwrite_profile: bool
        :type consumer_key: str
        :type consumer_secret: str
        :type access_token: str
        :type access_token_secret: str
        """
        # authenticate to Twitter and build the REST API object
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth, wait_on_rate_limit=True)

        # use user_id or screen_name and get
        if user_id is not None and user_id > 0:
            json_str = json.dumps([tweet._json for tweet in tweepy.Cursor(
                api.user_timeline, user_id=user_id, count=200).items()])
        elif screen_name is not None and screen_name != "":
            json_str = json.dumps([tweet._json for tweet in tweepy.Cursor(
                api.user_timeline, screen_name=screen_name, count=200).items()])
        else:
            raise InvalidTwitterRequestException("user_id or screen_name must be specified")

        self.update_from_tweets_json_arr(json_str, overwrite_profile)

    def to_json(self, compact=True, egonet_type="twitter_egonet"):
        """Convert the ego network into JSON format

        :param compact: use a compact JSON format, deleting spaces
        :type compact: bool
        :param egonet_type: type of ego network JSON object
        :type egonet_type: str
        :return the JSON representation of the ego network
        :rtype: str
        """
        return super().to_json(compact=compact, egonet_type=egonet_type)

    @staticmethod
    def create_from_json(egonet: str):
        """Create a new ego network from a JSON object.

        :param egonet: the JSON string representing the ego network
        :type egonet: str
        :return: a new ego network built from json data
        :rtype: TwitterEgoNetwork
        """
        dict_egonet = json.loads(egonet)
        ego = TwitterEgoNetwork()
        ego.__dict__.update({k: v for k, v in dict_egonet.items() if
                             k in ego.__dict__ and not k.startswith("_") and k != "contacts" and k != "last_time"})
        ego.last_time = dict_egonet["last_time"]
        for contact_type, contact_arr in dict_egonet["contacts"].items():
            for contact in contact_arr:
                if contact["additional_info"] is not None:
                    tweet_entities_frozen = TweetAdditionalInfo(**contact["additional_info"])
                else:
                    tweet_entities_frozen = TweetAdditionalInfo()
                del contact["additional_info"]
                if "contact_type" not in contact:
                    contact["contact_type"] = contact_type
                ego.add_contact(**contact, additional_info=tweet_entities_frozen)
        return ego

    @memoize
    def is_social(self, twitter_classifier: TwitterUserClassifier) -> bool:
        """Determine whether the ego is a socially relevant user or not.

        A socially relevant user represents an account used by a human to socialize with other people. All the other
        types of accounts are considered not socially relevant (e.g. companies, bots, spammers).

        :param twitter_classifier: a two-classes classifier. If a classifier with more than two classes is used, the
            output will only consider the first two classes, 0 and 1, and will ignore the other classes.
        :type twitter_classifier: TwitterUserClassifier
        :return: 1 if the ego is a socially relevant user, 0 if it is not socially relevant, and None in case of error.
        :rtype: bool
        """
        if isinstance(twitter_classifier, TwitterUserClassifier):
            return twitter_classifier.classify_user_from_profile(self) == 1

    def get_user_profile_array(self) -> np.array:
        """Return the profile data of the ego in the form of an array.

        :return: an array containing the profile data of the user
        :rtype: np.array
        """
        x = np.array(
            [self.account_duration, self.default_profile, self.default_profile_image, self.description_not_null,
             self.favourites_count, self.followers_count, self.friends_count, self.geo_enabled, self.listed_count,
             self.location_not_null, self.profile_background_image_not_null, self.protected, self.statuses_count,
             self.url_is_not_null, self.verified], dtype=float)
        return x

    @memoize
    def get_alters_brought_by_hashtags(self, contact_type: Hashable="__all__", active: bool=False) -> List:
        """Obtain the ids of the relationships that have an hashtag in their first contact.

        :param contact_type: the type of contact to consider
        :param active: if True, only active relationships are returned
        :type contact_type: Hashable
        :type active: bool
        :return: the ids of the alters brought by hashtags
        :rtype: List
        """
        alters_brought_by_htgs = []
        for alter_id, relationship in self.alters.items():
            if relationship.get_num_contacts(contact_type=contact_type) > 0 and not active or relationship.is_stable():
                sorted_contacts = relationship.get_sorted_contacts(contact_type=contact_type)
                if len(sorted_contacts[0].additional_info.hashtags) > 0:
                    alters_brought_by_htgs.append(alter_id)
        return alters_brought_by_htgs

    def get_hashtags_activating_relationships(self, alter_id: Hashable, contact_type: Hashable= "__all__") -> str:
        """Get the hashtag that activated the relationship, if present.

        A relationship is said to be "activated" by an hashtag if the hashtag in question is present in the first
        tweet of the relationship for the considered contact type.

        In case multiple activating hashtags are present, only the first one is returned.

        :param alter_id: the id of the alter to analyze
        :param contact_type: the type of contact to consider
        :type alter_id: Hashable
        :type contact_type: Hashable
        :return: the hashtag that activated the relationship, if present
        :rtype: str
        """
        sorted_contacts = self.alters[alter_id].get_sorted_contacts(contact_type=contact_type)
        if len(sorted_contacts[0].additional_info.hashtags) > 0:
            return next(iter(sorted_contacts[0].additional_info.hashtags))
        else:
            return None

    def get_relationships_most_used_hashtag(self, alter_id: Hashable, contact_type: Hashable = "__all__") -> str:
        """Get the hashtag that appears most frequently on the relationship, if present.

        In case multiple hashtags have the same frequency, only the first one is returned.

        :param alter_id: the id of the alter to analyze
        :param contact_type: the type of contact to consider
        :type alter_id: Hashable
        :type contact_type: Hashable
        :return: the hashtag that appears most frequently on the relationship
        :rtype: str
        """
        hashtags_count = defaultdict(int)
        contacts = self.alters[alter_id].get_contacts(contact_type=contact_type)
        for contact in contacts:
            if contact.additional_info.hashtags is not None:
                for hashtag in contact.additional_info.hashtags:
                    hashtags_count[hashtag] += 1
        hashtags_count_list = [(h, c) for h, c in hashtags_count.items()]
        if len(hashtags_count_list) > 0:
            return sorted(hashtags_count_list, key=lambda x: x[1], reverse=True)[0][0]
        else:
            return None

    def get_hashtag_total_usage(self, hashtag: str, contact_type: Hashable = "__all__") -> int:
        """Get the number of times that the specified hashtag has been used.

        :param hashtag: the hashtag to consider
        :param contact_type: the type of contact to consider
        :type hashtag: str
        :type contact_type: Hashable
        :return: the number of times that the hashtag has been used
        :rtype: str
        """
        htg_usage_count = 0
        for alter_id, relationship in self.alters.items():
            contacts = relationship.get_contacts(contact_type=contact_type)
            for contact in contacts:
                if hashtag in contact.additional_info.hashtags:
                    htg_usage_count += 1
        return htg_usage_count

    def get_relationship_n_hashtags(self, alter_id: Hashable, contact_type: Hashable= "__all__", unique: bool=False,
                                    hashtag: str=None) -> int:
        """Get the number of hashtags that the ego sent to the specified alter.

        :param alter_id: the id of the alter for which to retrieve the data
        :param contact_type: the type of contact to consider
        :param unique: if True, the number of unique hashtags is calculated, thus hashtags appearing multiple times
            are counted only once
        :param hashtag: if specified, the only hashtag to consider for the calculation
        :type alter_id: Hashable
        :type contact_type: Hashable
        :type unique: bool
        :type hashtag: str
        :return: the number of hashtags sent to the specified alter
        :rtype: int
        :raise KeyError: if the alter_id is not present in the ego network
        """
        if alter_id not in self.alters.keys():
            raise KeyError("alter not found")
        relationship = self.alters[alter_id]
        n_htgs = 0
        counted_htgs = set()
        for contact in relationship.get_contacts(contact_type=contact_type):
            contact_htgs = contact.additional_info.hashtags
            if hashtag is not None:
                n_htgs += hashtag in contact_htgs
            else:
                if not unique:
                    n_htgs += len(contact_htgs)
                else:
                    n_htgs += len([htg for htg in contact_htgs if htg not in counted_htgs])
                    counted_htgs.update(contact_htgs)
        return n_htgs

    def get_relationships_n_hashtags(self, contact_type: Hashable= "__all__", unique: bool=False, hashtag: str=None,
                                     alter_ids: List[Hashable]=None) -> Dict[Hashable, int]:
        """Obtain the number of hashtags for all the relationships.

        :param contact_type: the type of contacts to consider
        :param unique: if True, the number of unique hashtags is calculated, thus hashtags appearing multiple times
            are counted only once
        :param hashtag: if specified, the only hashtag to consider for the calculation
        :param alter_ids: an optional list of the alter_ids for which to retrieve the data. If None, the calculation is
            performed for all alters
        :type contact_type: Hashable
        :type unique: bool
        :type hashtag: str
        :type alter_ids: List[Hashable]
        :return: the number of hashtags for each relationship
        :rtype: Dict[Hashable, int]
        """
        rel_n_htgs = {}
        if alter_ids is not None:
            alter_ids = set(alter_ids)
        for alter_id in self.alters.keys():
            if alter_ids is None or alter_id in alter_ids:
                rel_n_htgs[alter_id] = self.get_relationship_n_hashtags(alter_id, contact_type=contact_type,
                                                                        unique=unique, hashtag=hashtag)
        return rel_n_htgs

    def generate_hashtags_network(self, contact_type: Hashable="__all__"):
        """Get the network of hashtags extracted from the tweets and create a new ego network containing the hashtags as
        alters. The contact type for the extracted contacts is set to "info_hashtag", whereas the alter_id for each
        hashtag is set to "__hashtag__" followed by the hashtag string.

        :param contact_type: a value indicating the type of contacts to use. Used to support ego networks with multiple
            types of social relationships. The special value "__all__" can be used for the network containing all the
            types of contacts
        :type contact_type: Hashable
        :return: a new ego network containing the hashtags as alters.
        :rtype: ContactEgoNetwork
        """
        hashtags_egonet = ContactEgoNetwork(ego_id=self.ego_id, last_time=self.last_time)
        for alter_id, alter in self.alters.items():
            contacts = alter.get_contacts(contact_type=contact_type)
            if contact_type == "__all__":
                contacts.extend(alter.get_contacts(contact_type="__no_contact__"))
            for contact in contacts:
                hashtags = contact.additional_info.hashtags
                for hashtag in hashtags:
                    hashtags_egonet.add_contact(timestamp=contact.timestamp, alter_id="__hashtag__" + hashtag,
                                                contact_type="info_hashtag", num_contacted_alters=len(hashtags))
        return hashtags_egonet


class InvalidTwitterRequestException(Exception):
    """The parameters needed for the Twitter request are not valid."""
    pass


class TwitterUserClassCsvReader(object):
    @staticmethod
    def read(filename, sep=' ', quote='', header=False):
        x = []
        y = []
        header_passed = not header
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=sep, quotechar=quote)
            for row in reader:
                if header_passed:
                    # leave out the first column (screen_name)
                    x.append([float(val) for val in row[1:-1]])
                    # save class as y
                    y.append(int(row[-1]))
                else:
                    header_passed = True
        return x, y


class TweetsCsvReader(object):
    """Load tweets from a csv file.

    Tweets in the file must be grouped by *user_id*.

    The file provided must have a header and must contain the following fields (not necessarily in this order):

      * **user_id**: the Twitter id of the user
      * **retweeted_status_id**: the id of the retweeted tweet if the tweet is a retweet. This field must be set to
        'null' if the tweet is not a retweet
      * **retweeted_user_id**: the id of the retweeted user if the tweet is a retweet. This field must be set to
        'null' if the tweet is not a retweet
      * **in_reply_to_status_id**: the id of the replied tweet if the tweet is a reply. This field must be set to
        'null' if the tweet is not a reply
      * **in_reply_to_user_id**: the id of the replied user if the tweet is a reply. This field must be set to
        'null' if the tweet is not a reply
      * **user_mention_ids**: an array containing the Twitter ids of the users mentioned in the tweet. The array must
        be delimited by square brackets and values must be separated by comma
      * **created_at**: the creation time of the tweet. The accepted format is Y-m-d H:M:S
      * **hashtags**: an array containing the text of the hashtags contained in the tweet. The array must be delimited
        by square brackets and values must be separated by comma
      * **urls**: an array containing the urls contained in the tweet. The array must be delimited by square brackets
        and values must be separated by comma
    """

    def __init__(self, filepath, separator):
        """Create a Tweets reader from the provided file.

        :param filepath: the path of the csv file containing the tweets
        :param separator: the field separator of the csv file
        :type filepath: str
        :type separator: str
        """
        self.filepath = filepath
        self.separator = separator

    def read(self) -> Iterator[TwitterEgoNetwork]:
        """Load the tweets from the csv file.

        :return: an ego network for each unique *user_id* found in the tweets with the tweets added as contacts
        :rtype: Iterator[TwitterEgoNetwork]
        """
        ego = TwitterEgoNetwork()
        for row in DictReaderWithHeader(open(self.filepath), self.separator):
            if int(row["user_id"]) != ego.ego_id:
                if ego.ego_id is not None:
                    yield ego
                ego = TwitterEgoNetwork(int(row["user_id"]))

            num_contacted_alters = 1
            if row["retweeted_status_id"] != "null" and row["retweeted_user_id"] != "null":
                contact_type = "retweet"
                alter_ids = [int(row["retweeted_user_id"])]
            elif row["in_reply_to_status_id"] != "null" and row["in_reply_to_user_id"] != "null":
                alter_ids = [int(row["in_reply_to_user_id"])]
                contact_type = "reply"
            elif row["user_mention_ids"] != "[]":
                alter_ids = []
                for mentioned_user in row["user_mention_ids"][1:-1].split(","):
                    alter_ids.append(int(mentioned_user))
                num_contacted_alters = len(row["user_mention_ids"][1:-1].split(","))
                contact_type = "mention"
            else:
                contact_type = "__no_contact__"
                alter_ids = [0]
            timestamp = int(time.mktime(datetime.datetime.strptime(row["created_at"], "%Y-%m-%d %H:%M:%S").timetuple()))
            text = row["text"]
            # read tweet entities
            tweet_entities = {"user_mentions": [], "hashtags": [], "urls": []}
            if row["user_mention_ids"] != "[]":
                for mention in row["user_mention_ids"][1:-1].split(","):
                    tweet_entities["user_mentions"].append(int(mention))
            if row["hashtags"] != "[]":
                for hashtag in row["hashtags"][1:-1].split(","):
                    tweet_entities["hashtags"].append(hashtag)
            if row["urls"] != "[]":
                for url in row["urls"][1:-1].split(","):
                    tweet_entities["urls"].append(url)
            tweet_entities_frozen = TweetAdditionalInfo(user_mentions=tweet_entities["user_mentions"],
                                                        hashtags=tweet_entities["hashtags"],
                                                        urls=tweet_entities["urls"])
            for alter_id in alter_ids:
                ego.add_contact(timestamp=timestamp, alter_id=alter_id, contact_type=contact_type, text=text,
                                num_contacted_alters=num_contacted_alters, additional_info=tweet_entities_frozen)
            downloaded_at = int(time.mktime(datetime.datetime.strptime(
                row["created_at"], "%Y-%m-%d %H:%M:%S").timetuple()))
            if ego.last_time is None or downloaded_at > ego.last_time:
                ego.last_time = downloaded_at
        else:
            yield ego


class ProfilesCsvReader(object):
    """Load Twitter profiles from a csv file.

    The file provided must have a header and must contain the following fields (not necessarily in this order):

      * **id**: the Twitter id of the profile
      * **screen_name**: the Twitter screen_name of the profile
      * **downloaded_at**: the time of the download of the profile. The accepted format is Y-m-d H:M:S
      * **created_at**: the time of the creation of the account. The accepted format is Y-m-d H:M:S
      * **default_profile**: 'true' if the profile settings are the default set by Twitter, 'false' otherwise
      * **default_profile_image**: 'true' if the profile image is the default one set by Twitter, 'false' otherwise
      * **description**: the text of the description field of the profile
      * **favourites_count**: the number of favourite tweets of the user
      * **followers_count**: the number of followers of the user
      * **friends_count**: the number of friends (followee) of the user
      * **listed_count**: the number of times the user appears in a Twitter list
      * **location**: the location associated to the profile
      * **protected**: 'true' if the account of the user is protected, 'false' otherwise
      * **statuses_count**: the number of tweets created by the user
      * **url**: the url associated to the profile
      * **verified**: 'true' if the account of the user is verified, 'false' otherwise

    """
    def __init__(self, filepath: str, separator: str):
        """Create a Twitter profiles reader from the provided file.

        :param filepath: the path of the csv file containing the profiles
        :type filepath: str
        :param separator: the field separator of the csv file
        :type separator: str
        """
        self.filepath = filepath
        self.separator = separator

    def read(self) -> Iterator[TwitterEgoNetwork]:
        """Read the profiles and return a :class:`TwitterEgoNetwork` for each of them.

        :return: a new ego network for each profile
        :rtype: Iterator[TwitterEgoNetwork]
        """
        for row in DictReaderWithHeader(open(self.filepath), self.separator):
            egonet = TwitterEgoNetwork(ego_id=int(row["id"]),
                                       screen_name=row["screen_name"],
                                       account_duration=(time.mktime(datetime.datetime.strptime(row["downloaded_at"],
                                                                                                "%Y-%m-%d %H:%M:%S")
                                                                     .timetuple()) - time.mktime(
                                           datetime.datetime.strptime(row["created_at"],
                                                                      "%Y-%m-%d %H:%M:%S").timetuple())) / 86400,
                                       default_profile=row["default_profile"] == "false",
                                       default_profile_image=row["default_profile_image"] == "false",
                                       description_not_null=row["description"] != "",
                                       favourites_count=int(row["favourites_count"]),
                                       followers_count=int(row["followers_count"]),
                                       friends_count=int(row["friends_count"]),
                                       geo_enabled=False,
                                       listed_count=int(row["listed_count"]),
                                       location_not_null=row["location"] != "",
                                       profile_background_image_not_null=False,
                                       protected=row["protected"] == "true",
                                       statuses_count=int(row["statuses_count"]),
                                       url_is_not_null=row["url"] != "null",
                                       verified=row["verified"] == "true",
                                       last_time=time.mktime(
                                           datetime.datetime.strptime(row["downloaded_at"], "%Y-%m-%d %H:%M:%S")
                                           .timetuple()))
            egonet.last_time = time.mktime(datetime.datetime.strptime(row["downloaded_at"], "%Y-%m-%d %H:%M:%S")
                                           .timetuple())
            egonet.first_time = time.mktime(datetime.datetime.strptime(row["created_at"], "%Y-%m-%d %H:%M:%S")
                                            .timetuple())
            yield egonet


class CombinedProfileTweetCsvReader(object):
    """Load Twitter ego networks from profile and tweets csv files.

    This class uses :class:`egonetworks.twitter.ProfilesCsvReader` and :class:`egonetworks.twitter.TweetsCsvReader` to
    generate combined ego networks.
    """
    def __init__(self, profile_path: str, tweets_path: str, separator: str="\t"):
        """Initialize a CombinedProfileTweetCsvReader object.

        :param profile_path: the path to the file containing profile information
        :type profile_path: str
        :param tweets_path: the path to the file containing tweets
        :type tweets_path: str
        :param separator: the column separator used in the input files
        :type separator: str
        """
        self.profile_reader = ProfilesCsvReader(profile_path, separator)
        self.tweets_reader = TweetsCsvReader(tweets_path, separator)

    def _load_profiles(self):
        egos = {}
        for egonet in self.profile_reader.read():
            egos[egonet.ego_id] = egonet
        return egos

    def read(self) -> Iterator[TwitterEgoNetwork]:
        """Load profile data from a csv file containing Twitter profiles and then read the related tweets and return
        a combined ego network.

        :return: a combined ego network built using both profile and tweets data
        :rtype: Iterator[TwitterEgoNetwork]
        """
        egos = self._load_profiles()
        for egonet in self.tweets_reader.read():
            egos[egonet.ego_id].update_contacts(egonet)
            yield egos[egonet.ego_id]
            del egos[egonet.ego_id]
