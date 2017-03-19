"""this simple script shows how to read TwitterEgoNetworks objects from .csv files and analyze them"""


from egonetworks.twitter import CombinedProfileTweetCsvReader, TwitterUserClassifier


__author__ = "Valerio Arnaboldi"
__license__ = "MIT"
__version__ = "1.0.1"


def main():
    reader = CombinedProfileTweetCsvReader("datasets/twitter_10_profiles.tsv", "datasets/twitter_10_tweets.tsv")
    for egonet in reader.read():
        print("Ego network id: " + str(egonet.ego_id))

        # calculate the size of the ego network both including inactive relationships outside the active network
        # and considering only the active network
        print("Total egonet size: " + str(egonet.get_egonet_size(active=False)))
        print("Active network size: " + str(egonet.get_egonet_size(active=True)))

        # compute stats about ego network circles
        c_prop = egonet.get_circles_properties(n_circles=5, contact_type=("reply", "mention"), active=True)

        # calculate the optimal number of circles of the ego network
        print("Optimal number of circles: " + str(egonet.get_optimal_num_circles()))

        print("Ego network duration: " + str(egonet.get_duration()))
        e_stab = egonet.get_stability(sim_types=("jaccard", "norm_spearman"), n_circles=5)
        # this is the actual stability value for the 5 rings - note that the second index is related to the "limit"
        # parameter, which, in this case, is not set
        print("Stability over time of the circles - Jaccard " + str(e_stab["jaccard"][0]))

        # this is an empty TwitterUser classifier
        classifier = TwitterUserClassifier()
        # train the classifier with the standard training set provided by the library
        classifier.train_default_profile_classifier()

        # determine whether the ego is a human being
        print("Ego is human? " + str(egonet.is_social(classifier)))

        # get_subnetwork returns a subset of the ego network in terms of time limits or contact types
        subnet_reply = egonet.get_subnetwork(contact_type="reply")
        subnet_mention = egonet.get_subnetwork(contact_type="mention")

        merged_subnet = subnet_reply.copy_egonet()
        merged_subnet.update_contacts(subnet_mention)

        # a more efficient way to obtain the merged subnetwork
        merged_subnet2 = egonet.get_subnetwork(contact_type=("reply", "mention"))

        # determine whether the ego network is stable
        print("Is the egonet stable? " + str(egonet.is_stable()))

        # add contacts manually
        egonet.add_contact(timestamp=1489937816, alter_id=1)

        # create a hashtag-based ego network, where alters are hashtags
        hashtags_egonet = egonet.generate_hashtags_network()
        egonet2 = egonet.copy_egonet()
        egonet2.update_contacts(hashtags_egonet)
        # consider only info_hashtags. Same as hashtags_egonet.get_egonet_size()
        egonet2.get_egonet_size(contact_type="info_hashtags")
        # overall size, considering both hashtags and social contacts (reply, mention, retweet). Count also unstable
        # and inactive relationships (outside the active network)
        egonet2.get_egonet_size(stable=False, active=False)

        # get a time series of the size of the ego network over time
        size_ts = egonet.get_size_time_series()

        # leave a blank line between the ego networks
        print()


if __name__ == '__main__':
    main()
