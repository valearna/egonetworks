import unittest
from egonetworks.twitter import TwitterEgoNetwork, TweetAdditionalInfo
import os


class TestTwitter(unittest.TestCase):

    def setUp(self):
        self. ego = TwitterEgoNetwork()
        urls = []
        info = TweetAdditionalInfo(user_mentions=[1, 2], hashtags=["a"], urls=urls)
        self.ego.add_contact(timestamp=1, alter_id=1, text="test", num_contacted_alters=1, additional_info=info)
        this_dir, this_filename = os.path.split(__file__)
        egonets_filepath = os.path.join(this_dir, "../datasets", "twitter_egonets.json")
        self.twitter_egonet = TwitterEgoNetwork()
        with open(egonets_filepath, "r") as twegonets:
            json_str_egonet = twegonets.readline().strip()
            complete_egonet = TwitterEgoNetwork.create_from_json(json_str_egonet)
            self.twitter_egonet = TwitterEgoNetwork(ego_id=complete_egonet.ego_id, last_time=complete_egonet.last_time)
            self.twitter_egonet.update_contacts(complete_egonet, contact_type=("retweet", "mention", "reply"))

    def test_create_from_json(self):
        ego2 = TwitterEgoNetwork.create_from_json(self.ego.to_json())
        self.assertEqual(ego2.ego_id, self.ego.ego_id, "wrong ego_id")

    def test_get_egonet_size(self):
        self.assertGreaterEqual(self.twitter_egonet.get_egonet_size(active=False, stable=False), 703)
        self.assertTrue(self.twitter_egonet.get_egonet_size(active=False, stable=True), 688)
        self.assertTrue(self.twitter_egonet.get_egonet_size(active=True, stable=True), 388)


if __name__ == "__main__":
    unittest.main()
