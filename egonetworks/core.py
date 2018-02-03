"""this is the core file containing the main data structures and methods for the creation and analysis of ego network
objects"""

import json
import math
from bisect import bisect_left
from jenks import jenks
from typing import List, Tuple, Dict, Hashable, Iterator, Union
import numpy as np
from scipy.stats import lognorm, pearsonr
from sklearn.cluster import MeanShift
from egonetworks.error import *
from egonetworks.generic import memoize, Memoizable, get_topics_from_text, get_topics_rel_network, \
    synchronized_method, calculate_dispersion, calculate_burstiness, count_bursts, get_mean_persistence, get_exit_jumps
import egonetworks.similarity as lists
from abc import ABCMeta, abstractmethod
from collections import namedtuple, defaultdict, deque, Counter
import igraph
from concurrent.futures import ThreadPoolExecutor
import logging


__author__ = "Valerio Arnaboldi"
__license__ = "MIT"
__version__ = "1.0.1"


CirclesProperties = namedtuple("CirclesProperties", ["num_circles", "sizes", "min_freqs", "total_freqs", "membership"])
AlterFreq = namedtuple("AlterFreq", ["alter_id", "frequency"])
AlterCount = namedtuple("AlterCount", ["alter_id", "count"])

# const shortcuts for timestamps
ONE_MINUTE = 60
ONE_HOUR = 60 * ONE_MINUTE
ONE_DAY = 24 * ONE_HOUR
ONE_YEAR = 365.25 * ONE_DAY
ONE_MONTH = ONE_YEAR / 12
SIX_MONTHS = ONE_YEAR / 2


def standardize_contact_type(contact_type: Union[Hashable, Tuple[Hashable]] = "__all__") -> Tuple[Hashable]:
    if not isinstance(contact_type, tuple):
        std_ct = contact_type,
    else:
        std_ct = contact_type
    if "__all__" in std_ct:
        std_ct = "__all__",
    return std_ct


class EgoNetwork(Memoizable, metaclass=ABCMeta):
    """**Abstract** class that represents an Ego Network graph.

    This is the main (**abstract**) class of egonetworks package and contains data structures and methods for ego
    network analysis. The class is a representation of an ego network and contains information about an individual
    person (ego) and the social contacts that this person has with other people (alters). Ego and alters are identified
    by unique numeric user ids

    The main classes that implement this abstract class are :class:`egonetworks.core.ContactEgoNetwork` which
    builds ego networks from information about single social contacts between the ego and its alters such as the
    timestamp of the contact and the related text and :class:`egonetworks.core.ContactEgoNetwork` which builds ego
    networks from information about the frequency of contact of the social relationships of the ego
    """

    @abstractmethod
    def __init__(self, ego_id: int = None):
        self.ego_id = ego_id
        self._freq_vector = defaultdict(list)
        super().__init__()

    def get_sorted_alter_freq_vector(self,
                                     contact_type: Union[Hashable, Tuple[Hashable]] = "__all__") -> List[AlterFreq]:
        """Get the vector of frequencies sorted in reverse order.

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :return: the sorted vector of frequencies
        :rtype: List[AlterFreq]
        """
        contact_type = standardize_contact_type(contact_type)
        if "__all__" in contact_type:
            freqs = self._freq_vector["__all__"]
        else:
            freqs = self._freq_vector[contact_type]
        if len(freqs) > 0:
            return sorted(freqs, key=lambda x: x.frequency, reverse=True)
        else:
            return []

    def get_alter_freq_list(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__",
                            active=False, limit: int = 0) -> List[AlterFreq]:
        """Get the vector of contact frequencies of the network with their alter_id.

        Calculate the frequencies of contact of the alters.

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :param active: if True, return only alters with active contact frequency (i.e. greater or equal to 1)
        :param limit: return only the specified maximum number of alters, taking those with highest contact frequency
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :type active: bool
        :type limit: int
        :return: the vector containing the tuples of (alter_id, frequency)
        :rtype: List[AlterFreq]
        """
        return [alter_freq for i, alter_freq in enumerate(self.get_sorted_alter_freq_vector(
            contact_type=contact_type)) if (not active or alter_freq.frequency >= 1) and (limit == 0 or i < limit)]

    def __get_freq_list(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__",
                        active=False, limit: int = 0) -> List[float]:
        """Get the vector of contact frequencies of the network, without any reference to alter_id.

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :param active: if True, return only alters with active contact frequency (i.e. greater or equal to 1)
        :param limit: return only the specified maximum number of alters, taking those with highest contact frequency
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :type active: bool
        :type limit: int
        :return: the vector containing the frequencies
        :rtype: List[float]
        """
        return [alter_freq.frequency for alter_freq in self.get_alter_freq_list(contact_type=contact_type,
                                                                                active=active, limit=limit)]

    @memoize
    def get_egonet_size(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__", active: bool = False) -> int:
        """Get the total size of the ego network as the total number of alters.

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :param active: if True, count only alters with active contact frequency (i.e. greater or equal to 1)
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :type active: bool
        :return: the size of the ego network
        :rtype: int
        """
        return len(self.get_alter_freq_list(contact_type=contact_type, active=active))

    @memoize
    def get_mean_freq(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__", active: bool = False) -> float:
        """Get the mean contact frequency for the alters in the ego network.

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :param active: if True, count only alters with active contact frequency (i.e. greater or equal to 1)
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :type active: bool
        :return: the mean contact frequency of the ego network
        :rtype: float
        """
        if self.get_egonet_size(contact_type=contact_type, active=active) > 0:
            return float(np.mean(self.__get_freq_list(contact_type=contact_type, active=active)))
        else:
            return 0

    @memoize
    def get_total_freq(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__", active: bool = False) -> float:
        """Get the total frequency of contact for the ego in messages per year.

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :param active: if True, count only alters with active contact frequency (i.e. greater or equal to 1)
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :type active: bool
        :return: the total frequency of contact
        :rtype: float
        """
        if self.get_egonet_size(contact_type=contact_type, active=active) > 0:
            return float(np.sum(self.__get_freq_list(contact_type=contact_type, active=active)))
        else:
            return 0

    @memoize
    def get_entropy(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__", active: bool = False,
                    global_norm: bool = True, base: float = 2) -> float:
        """Returns the entropy associated to the ego network, considering the frequencies of contact with the alters.

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :param active: if True, count only alters with active contact frequency (i.e. greater or equal to 1)
        :param global_norm: normalize frequencies considering all the types of contacts in the ego network and not only
            those selected through the *contact_type* argument
        :param base: the base of the logarithm used to calculate the entropy
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :type active: bool
        :type global_norm: bool
        :type base: float
        :return: the entropy of the ego network
        :rtype: float
        """
        freqs = self.__get_freq_list(contact_type=contact_type, active=active)
        entropy = 0
        max_freq = -1
        if global_norm:
            glob_freqs = self.__get_freq_list(contact_type="__all__", active=active)
            if glob_freqs:
                max_freq = np.max(glob_freqs)
        else:
            if freqs:
                max_freq = np.max(freqs)
        if max_freq > 0:
            try:
                entropy = - sum([(freq / max_freq) * math.log(freq / max_freq, base) for freq in freqs])
            except ValueError:
                entropy = 0
        return entropy

    @memoize
    def get_lognorm_fit_frequencies(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__",
                                    active: bool=False) -> Tuple[float, float, float]:
        """Get the parameters shape, location, and scale of a lognorm fitted on the distribution of the contact
        frequency of the ego network.

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :param active: if True, count only alters with active contact frequency (i.e. greater or equal to 1)
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :type active: bool
        :return: a three-valued tuple with the fitted parameters shape, location, and scale
        :rtype: Tuple[float, float, float]
        """
        if self.get_egonet_size(contact_type=contact_type, active=active) > 0:
            return lognorm.fit(self.__get_freq_list(contact_type=contact_type, active=active), floc=0)
        else:
            return 0, 0, 0

    @memoize
    def get_optimal_num_circles(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__") -> int:
        """Calculate the optimal number of circles for the ego network.

        Mean Shift algorithm is used to determine the optimal number of circles for the ego network, based on the
        distribution of the frequency of contact of the network.

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :return: the optimal number of circles of the ego network
        :rtype: int
        """
        if self.get_egonet_size(contact_type=contact_type, active=True) > 0:
            x = np.sort(self.__get_freq_list(contact_type=contact_type, active=True))
            if len(set(x)) > 1:
                x = x.reshape(-1, 1)
                # bandwidth = estimate_bandwidth(x, quantile=0.1)
                ms = MeanShift(bin_seeding=True)
                try:
                    ms.fit(x)
                    labels = ms.labels_
                    labels_unique = np.unique(labels)
                    return len(labels_unique)
                except ValueError:
                    return 1
            else:
                return len(set(x))
        else:
            return 0

    @memoize
    def get_circles_properties(self, n_circles: int, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__",
                               active: bool = False) -> CirclesProperties:
        """Get the properties of ego network circles given the number of circles.

        Social relationships are divided into ego network circles using Jenks breaks algorithm applied to the values
        of contact frequencies, and then statistics about the properties of relationships in the circles are returned.
        Note that the sizes of the ego network circles returned are inclusive. For example, the outermost circle
        contains all the other internal circles and its size is cumulative.

        :param n_circles: the number of circles. Note that the results could contain a different number of circles in
            case the ego network cannot be divided into n_circles layers
        :type n_circles: int
        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :param active: whether to consider only active alters or not
        :type active: bool
        :return: the properties of the ego network circles. In case the ego network cannot be divided into n_circles
            circles, the returned properties will be all set to None
        :rtype: CirclesProperties
        :raises NumberOfCirclesOutOfRangeException: if the number of circles is not between 1 and 99
        """
        if 100 < n_circles <= 0:
            raise NumberOfCirclesOutOfRangeException("the number of circles must be between 1 and 99")
        sizes = []
        min_freqs = []
        total_freqs = []
        membership = {}
        if self.get_egonet_size(contact_type=contact_type, active=active) > 0:
            alter_id_freq_vec = self.get_alter_freq_list(contact_type=contact_type, active=active)
            x = self.__get_freq_list(contact_type=contact_type, active=active)
            breaks = jenks(x, n_circles)
            if breaks is not None:
                breaks = sorted(set([float(bk) for bk in breaks]))
                breaks[0] = 0
                tot_size = 0
                if len(breaks) < n_circles:
                    breaks.append(max(x) + 1)
                for k in sorted(range(len(breaks) - 1), reverse=True):
                    # inserted or condition in list comprehension to cope with approx errors of jenks
                    members = [alter_freq for alter_freq in alter_id_freq_vec if
                               float(breaks[k]) < alter_freq.frequency <= float(breaks[k + 1])]
                    freqs = []
                    for alter_freq in members:
                        membership[alter_freq.alter_id] = len(breaks) - 1 - k
                        freqs.append(alter_freq.frequency)
                    tot_size += len(freqs)
                    sizes.append(tot_size)
                    if len(members) > 0:
                        total_freqs.append(np.sum(freqs))
                        min_freqs.append(np.min(freqs))
                    else:
                        total_freqs.append(0)
                        min_freqs.append(breaks[k])
                if len(sizes) > 0:
                    last_size = sizes[-1]
                else:
                    last_size = 0
                if len(min_freqs) > 0:
                    last_min_freq = min_freqs[-1]
                else:
                    last_min_freq = 0
                for i in range(n_circles - (len(breaks))):
                    sizes.append(last_size)
                    min_freqs.append(last_min_freq)
                    total_freqs.append(0)
                if len(sizes) == n_circles:
                    return CirclesProperties(n_circles, sizes, min_freqs, total_freqs, membership)
        return CirclesProperties(num_circles=n_circles, sizes=[0] * n_circles, min_freqs=[0] * n_circles,
                                 total_freqs=[0] * n_circles, membership=membership)

    @staticmethod
    def get_egonet_similarity(egonet1, egonet2, sim_type: str = None, limit: int = 0,
                              contact_type: Union[Hashable, Tuple[Hashable]] = "__all__",
                              n_circles=1) -> List[float]:
        """Calculate the similarity between two ego networks.

        :param egonet1: the first ego network
        :param egonet2: the second ego network
        :param sim_type: the type of similarity measure to calculate. Possible values are "jaccard" for the Jaccard
            index, "norm_spearman" for a similarity value calculated as 1 - the normalized version of Spearman's
            footrule index, which is an index of the variability in [0,1], "ring_member" for a similarity value
            calculated as 1 - the normalized version of the level changes index, "ring_member_unstable" for a similarity
            index similar to the previous one but calculated only for unstable relationships, "ring_member_rel_var" for
            a measure of dissimilarity based on the level changes index with relative normalization, and
            "ring_member_var_raw" for the raw level changes dissimilarity measure (not normalized). Also the last two
            measures can have the "_unstable" suffix to consider only unstable relationships
        :param limit: consider only the top k elements with highest frequency in the slices
        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :param n_circles: divide the alters in the ego network of each time window into the specified number of circles
            and calculate the stability for the different rings of the network
        :type egonet1: EgoNetwork
        :type egonet2: EgoNetwork
        :type sim_type: str
        :type limit: int
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :type n_circles: int
        :return: the list of similarity values between the two ego networks, one value per ego network circle
        :rtype: List[float]
        :raise InvalidIntValueException: if limit is smaller than 0
        """
        if limit < 0:
            raise InvalidIntValueException("limit must be grater than 0")
        if sim_type.startswith("ring_member") and n_circles < 2:
            raise InvalidSimilarityTypeException(
                "cannot calculate ring_member similarity for the specified number of circles")

        if egonet1 is None and egonet2 is None:
            return [0]
        elif egonet1 is None or egonet2 is None:
            return [1]

        set1 = egonet1.get_alter_freq_list(contact_type=contact_type, limit=limit)
        set2 = egonet2.get_alter_freq_list(contact_type=contact_type, limit=limit)
        sim_vec = []
        if n_circles > 1:
            egonet1_circles_prop = egonet1.get_circles_properties(n_circles, contact_type)
            egonet2_circles_prop = egonet2.get_circles_properties(n_circles, contact_type)
            # if both the ego networks can be divided into n_circles circles, else return empty result
            if egonet1_circles_prop.num_circles == egonet2_circles_prop.num_circles == n_circles:
                rings_membership1 = egonet1_circles_prop.membership
                rings_membership2 = egonet2_circles_prop.membership
                for ring_idx in range(1, n_circles + 1):
                    ring_list1 = [alter_freq.alter_id for alter_freq in set1 if
                                  alter_freq.alter_id in rings_membership1 and
                                  rings_membership1[alter_freq.alter_id] == ring_idx]
                    ring_list2 = [alter_freq.alter_id for alter_freq in set2 if
                                  alter_freq.alter_id in rings_membership2 and
                                  rings_membership2[alter_freq.alter_id] == ring_idx]
                    if sim_type.startswith("ring_member"):
                        # for this special similarity measure, we need the membership of all the alter_ids for both the
                        # ego networks. If an alter is not in one of the two networks, its membership is set to
                        # n_circles + 1, which is the value that indicates the outer part of the net
                        ids_only_in_list2 = set(ring_list2).difference(set(ring_list1))
                        ids_only_in_list1 = set(ring_list1).difference(set(ring_list2))
                        ring_list1 = [(alter_id, ring_idx) for alter_id in ring_list1]
                        ring_list1.extend([(alter_id, rings_membership1.get(alter_id, n_circles + 1)) for alter_id in
                                           ids_only_in_list2])
                        ring_list2 = [(alter_id, ring_idx) for alter_id in ring_list2]
                        ring_list2.extend([(alter_id, rings_membership2.get(alter_id, n_circles + 1)) for alter_id in
                                           ids_only_in_list1])
                    sim_vec.append(lists.get_list_similarity(ring_list1, ring_list2, n_levels=n_circles,
                                                             sim_type=sim_type))
            else:
                sim_vec = [0] * n_circles
        else:
            sim_vec.append(lists.get_list_similarity([alter_freq.alter_id for alter_freq in set1],
                                                     [alter_freq.alter_id for alter_freq in set2],
                                                     n_levels=n_circles, sim_type=sim_type))
        return sim_vec


# namedtuples used in class ContactAlter
BasicContact = namedtuple("Contact", ["timestamp", "text", "num_contacted_alters", "additional_info"])
FullContact = namedtuple("FullContact", BasicContact._fields + ("alter_id", "contact_type"))
FullContact.__new__.__defaults__ = (None,) * len(FullContact._fields)


class ContactAdditionalInfo(metaclass=ABCMeta):
    """This abstract class defines a generic object to be added to a contact as additional information"""

    @abstractmethod
    def to_json_encoded_obj(self):
        """Transform the object into a json object"""
        pass

    @staticmethod
    @abstractmethod
    def from_json(obj: Dict):
        """Create a new instance from a json object

        :param obj: the json object from which to create a new instance
        :type obj: Dict"""
        pass


class ContactAlter(Memoizable):
    """A ContactAlter object.

    The representation of an alter in an ego network based on a set of social contacts.
    """
    def __init__(self, min_rel_duration: float = ONE_MONTH, alter_id: Hashable = None):
        """Create a new alter.

        An Alter object represents a social relationship of an ego. This object contains all the contacts from the ego
        to the related alter and the methods to calculate statistics related to the social relationship.

        :return: the new alter
        :rtype: ContactAlter
        """
        # sets prevents contacts with the same properties to be inserted multiple times
        self.contacts = defaultdict(set)
        self.min_rel_duration = min_rel_duration
        self.alter_id = alter_id
        super().__init__()

    def get_sorted_contacts(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__") -> List[FullContact]:
        """Get the list of contacts sorted by timestamp.

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :return: the list of contacts sorted by timestamp
        :rtype: List[FullContact]
        """
        return self.get_contacts(contact_type=contact_type, sort=True)

    @memoize
    def get_contacts(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__",
                     sort: bool = False) -> List[FullContact]:
        """Get the list of contacts.

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :param sort: whether to return contacts sorted by timestamp
        :type sort: bool
        :return: the list of contacts
        :rtype: List[FullContact]
        """
        contact_type = standardize_contact_type(contact_type)
        contacts = set()
        if "__all__" in contact_type:
            for c_type, contact_set in self.contacts.items():
                if not str(c_type).startswith("__no_contact"):
                    contacts.update(contact_set)
        else:
            for ct in contact_type:
                contacts.update(self.contacts[ct])
        if not sort:
            return list(contacts)
        else:
            return sorted(list(contacts), key=lambda x: x.timestamp)

    @memoize
    def get_inter_contact_times(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__") -> List[float]:
        """Get the list of inter contact times

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :return: the list of inter contact times
        :rtype: List[float]
        """
        sorted_contacts = self.get_sorted_contacts(contact_type=contact_type)
        return [c.timestamp - sorted_contacts[i - 1].timestamp for i, c in enumerate(sorted_contacts)][1:]

    def get_burstiness_index(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__") -> float:
        """Calculates the burstiness index for the alter

        The index of burstiness of the distribution of the inter contact times of the alter

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :return: the index of burstiness of the communication with the alter
        :rtype: float
        """
        return calculate_burstiness(self.get_inter_contact_times(contact_type=contact_type))

    def get_dispersion_index(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__") -> float:
        """Calculates the dispersion index for the alter

        The index of dispersion of the distribution of the inter contact times of the alter

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :return: the index of dispersion of the communication with the alter
        :rtype: float
        """
        return calculate_dispersion(self.get_inter_contact_times(contact_type=contact_type))

    def has_contact_type(self, contact_type: Hashable):
        """Check whether the alter has been contacted with the specified contact type.

        :param contact_type: the type of contact
        :type contact_type: Hashable
        :return: whether the alter has been contacted with the contact type
        :rtype: bool
        """
        if contact_type == "__all__" or contact_type in self.contacts:
            return True
        else:
            return False

    def get_num_contacts(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__"):
        """Get the total number of contacts for the specified contact type.

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :return: the number of contacts (normalized by the number of alters in each contact)
        :rtype: float
        """
        contact_type = standardize_contact_type(contact_type)
        if "__all__" in contact_type or any([self.has_contact_type(ct) for ct in contact_type]):
            contacts = self.get_contacts(contact_type=contact_type)
            num_contacts = len(contacts)
            return num_contacts
        else:
            return 0

    def get_first_contact_time(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__"):
        """Get the timestamp of the first contact for the specified contact type.

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :return: the timestamp of the first contact
        :rtype: int
        """
        contact_type = standardize_contact_type(contact_type)
        if "__all__" in contact_type or any([self.has_contact_type(ct) for ct in contact_type]):
            contacts = self.get_sorted_contacts(contact_type=contact_type)
            if len(contacts) > 0:
                return contacts[0].timestamp
        return None

    def get_last_contact_time(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__"):
        """Get the timestamp of the last contact for the specified contact type.

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :return: the timestamp of the last contact
        :rtype: int
        """
        contact_type = standardize_contact_type(contact_type)
        if "__all__" in contact_type or any([self.has_contact_type(ct) for ct in contact_type]):
            contacts = self.get_sorted_contacts(contact_type=contact_type)
            if len(contacts) > 0:
                return contacts[-1].timestamp
        return None

    def get_duration(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__", last_contact_time: float = None):
        """Get the duration of the relationship considering the specified contact types, in years.

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :param last_contact_time: the last contact time of the relationship to consider for the duration
        :type last_contact_time: float
        :return: the duration of the relationship
        :rtype: float
        """
        first_contact_time = self.get_first_contact_time(contact_type=contact_type)
        if last_contact_time is None:
            last_contact_time = self.get_last_contact_time(contact_type=contact_type)
        if first_contact_time and last_contact_time and first_contact_time <= last_contact_time:
            return (last_contact_time - first_contact_time) / ONE_YEAR
        else:
            return None

    def get_frequency(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__", first_contact_time: float = None,
                      last_contact_time: float = None) -> float:
        """Get the contact frequency for the alter.

        The contact frequency is defined as the number of contacts from ego to the alter in question divided by the
        duration of their relationship, calculated as the time span between their first contact and the time of the
        download. The frequency is calculated in number of contacts in one year.

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :param first_contact_time: the timestamp at which the relationship begun. If None, the time of the first conatac
            of the relationship for the given contact_type is used
        :type first_contact_time: float
        :param last_contact_time: the timestamp at which the relationship ended. If None, the time of the last contact
            of the relationship for the given contact_type is used
        :type last_contact_time: float
        :return: the contact frequency of the alter
        :rtype: float
        """
        contact_type = standardize_contact_type(contact_type=contact_type)
        num_contacts = self.get_num_contacts(contact_type=contact_type)
        if first_contact_time is None:
            first_contact_time = self.get_first_contact_time(contact_type=contact_type)
        if last_contact_time is None:
            last_contact_time = self.get_last_contact_time(contact_type=contact_type)
        if num_contacts > 0 and first_contact_time is not None and last_contact_time is not None \
                and first_contact_time < last_contact_time:
            if num_contacts / ((last_contact_time - first_contact_time) / ONE_YEAR) > 0:
                return num_contacts / ((last_contact_time - first_contact_time) / ONE_YEAR)
            else:
                return 0
        else:
            return 0

    @memoize
    def is_stable(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__",
                  first_contact_time: float = None, last_contact_time: float = None) -> bool:
        """Determine whether the relationship with the alter is stable or not.

        A stable relationship has a duration longer than the minimum duration specified at creation time.

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :param first_contact_time: the time to condider as the beginning of the relationship. If None, the time of the
            first contact with the alter is considered
        :param last_contact_time: the time to consider as the end of the relationship. If None, the time of the last
            contact with the alter is considered
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :type first_contact_time: float
        :type last_contact_time: float
        :return: True if the alter is stable, False otherwise
        :rtype: bool
        """
        num_contacts = self.get_num_contacts(contact_type=contact_type)
        if first_contact_time is None:
            first_contact_time = self.get_first_contact_time(contact_type=contact_type)
        if last_contact_time is None:
            last_contact_time = self.get_last_contact_time(contact_type=contact_type)
        return num_contacts > 0 and first_contact_time is not None and last_contact_time is not None and \
            (last_contact_time - first_contact_time) >= self.min_rel_duration

    def add_contact(self, alter_id: Hashable, timestamp: int, contact_type: Hashable = "__default__", text: str = None,
                    num_contacted_alters: int = 1, additional_info: ContactAdditionalInfo = None) ->FullContact:
        """Add a new contact for the alter.

        A contact is composed of a timestamp of a direct message sent by ego to the alter and the text of the message.

        :param alter_id: the id of the alter
        :type alter_id: Hashable
        :param contact_type: a value indicating the type of contact. Used to support ego networks with multiple types
            of social relationships. Contact types starting with "__no_contact" will be ignored for the calculation of
            properties of the whole ego network ("__all__" contacts). This special contact type can be used to store
            contacts that do not have a specific recipient and need to be analysed separately from other contacts
        :type contact_type: Hashable
        :param timestamp: the Unix timestamp at which the contact has been sent
        :type timestamp: int
        :param text: the text associated to the contact
        :type text: str
        :param num_contacted_alters: the number of alters in the same contact
        :type num_contacted_alters: int
        :param additional_info: additional information to be attached to the contact
        :type additional_info: ContactAdditionalInfo
        :return the contact object added to the alter
        :rtype FullContact
        """
        self.clear_cache()
        new_contact = FullContact(timestamp=timestamp, text=text, num_contacted_alters=num_contacted_alters,
                                  additional_info=additional_info, contact_type=contact_type, alter_id=alter_id)
        self.contacts[contact_type].add(new_contact)
        return new_contact


MinContactsTimeWin = namedtuple("MinContactsTimeWin", ["min_num_contacts", "win_size", "function"])
TopicIntensity = namedtuple("TopicIntensity", ["topic_id", "intensity"])
RelationshipsProperties = namedtuple("RelationshipsProperties", ["frequency", "ring_changes_index", "ring_membership"])


class ContactEgoNetwork(EgoNetwork):
    """Egocentric Network Graph based on explicit contacts between users.

    This class defines an ego network from a set of explicit direct contacts (messages or other types of contacts)
    between the ego and its alters, for which information about their timestamp and the id of the alters contacted is
    available.
    """

    def __init__(self, ego_id: int = None, last_time: float = None, first_time: float = None,
                 window_mode: bool = False, rel_min_duration: float = ONE_YEAR):
        """Create a new contact ego network.

        :param ego_id: the id of the ego
        :type ego_id: int
        :param first_time: the first time to consider for the calculation of the contact frequencies with the alters.
            If None, the first contact time of each relationship is used to calculate its frequency
        :type first_time: float
        :param last_time: the last time to consider for the ego network (usually the time when the ego network has been
            downloaded. If None, the last time will be calculated as the last contact of the ego. The last time is used
            to calculate the duration of the relationships with the alters and their contact frequencies
        :type last_time: float
        :param window_mode: whether to consider the ego network in window mode. This is a special condition where the
            duration of social relationships in the ego networks are calculated as the difference between the
            *first_time* and the *last_time* values of the ego networks, or as the result of get_first_time() and
            get_last_time() in case the parameters are None. This is opposed to the normal behaviour of the ego network
            functions, that consider the duration of a relationship as the difference between the first contact time on
            the relationship and the last time of the ego network.
        :type window_mode: bool
        :return: a new ego network
        :rtype: ContactEgoNetwork
        """
        super().__init__(ego_id)

        self.__last_time = last_time
        self.__first_time = first_time
        self.__window_mode = window_mode
        self._saved_freqs = set()
        self.alters = {}
        self._contacts_sort_keys = defaultdict(list)
        self._contacts_sorted = False
        self._contacts_by_type = defaultdict(list)
        self._alters_by_type = defaultdict(set)
        self.ego_id = ego_id
        self._rel_min_duration = rel_min_duration

    @property
    def last_time(self):
        return self.__last_time

    @last_time.setter
    def last_time(self, last_time):
        """Set the last time of the ego network.

        The last time is used to calculate the duration of the ego network and that of the relationships. If last time
        is not set, it is calculated automatically as the time of the last contact of the ego network for the ego and
        as the last contact of the relationship for relationships.
        """
        self.clear_cache()
        self._freq_vector = defaultdict(list)
        self._saved_freqs = set()
        self.__last_time = last_time

    @property
    def first_time(self):
        return self.__first_time

    @first_time.setter
    def first_time(self, first_time):
        """Set the first time of the ego network.

        The first time is used to calculate the duration of the ego network and that of the relationships. If first time
        is not set, it is calculated automatically as the time of the first contact of the ego network for the ego and
        as the first contact of the relationship for relationships.
        """
        self.clear_cache()
        self._freq_vector = defaultdict(list)
        self._saved_freqs = set()
        self.__first_time = first_time

    @property
    def window_mode(self):
        return self.__window_mode

    @window_mode.setter
    def window_mode(self, window_mode):
        """Set the window_mode of the ego network

        This is a special parameter that controls how the duration of the relationhsips of the ego network and their
        respective frequencies are calculated. In windowed_mode, the durations are calculated as the difference between
        the *first_time* and the *last_time* values of the ego networks, or as the result of get_first_time() and
        get_last_time() in case the parameters are None. This is opposed to the normal behaviour of the ego network
        functions, that consider the duration of a relationship as the difference between the first contact time on the
        relationship and the last time of the ego network.
        """
        self.clear_cache()
        self._freq_vector = defaultdict(list)
        self._saved_freqs = set()
        self.__window_mode = window_mode

    def _get_sorted_contacts(self, from_timestamp: float = None, to_timestamp: float = None,
                             contact_type: Union[Hashable, Tuple[Hashable]] = "__all__"):
        """Sort the data structure indexing all the contacts of the ego network

        :param from_timestamp: get only contacts created after the specified timestamp (included)
        :type from_timestamp: float
        :param to_timestamp: get only contacts created before the specified timestamp (excluded)
        :type to_timestamp: float
        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :return the contacts of the ego networks sorted by timestamp
        :rtype: List[FullContact]
        """
        contact_type = standardize_contact_type(contact_type)
        if not self._contacts_sorted:
            for ct, contact_list in self._contacts_by_type.items():
                contact_list.sort(key=lambda x: x.timestamp)
                self._contacts_sort_keys[ct] = [contact.timestamp for contact in contact_list]
            self._contacts_sorted = True
        merged_contact_list = []
        for ct in contact_type:
            from_idx = 0
            to_idx = len(self._contacts_by_type[ct])
            if from_timestamp is not None:
                from_idx = bisect_left(self._contacts_sort_keys[ct], from_timestamp)
            if to_timestamp is not None:
                to_idx = bisect_left(self._contacts_sort_keys[ct], to_timestamp)
            merged_contact_list.extend(self._contacts_by_type[ct][from_idx:to_idx])
        if len(contact_type) > 1:
            # a normal sort should be enough efficient for relatively short lists (< 1M elements), as timsort is used
            merged_contact_list.sort(key=lambda x: x.timestamp)
        return merged_contact_list

    def to_json(self, compact=True, egonet_type="contact_egonet"):
        """Convert the ego network into JSON format

        :param compact: use a compact JSON format, deleting spaces
        :type compact: bool
        :param egonet_type: type of ego network JSON object
        :type egonet_type: str
        :return the JSON representation of the ego network
        :rtype: str
        """
        dict_egonet = {"last_time": self.last_time,
                       "contacts": defaultdict(list)}
        dict_egonet.update({k: v for k, v in self.__dict__.items() if not k.startswith("_") and k != "alters"})
        dict_egonet["egonet_type"] = egonet_type
        for alter_id, alter_info in self.alters.items():
            for contact_type, contact_arr in alter_info.contacts.items():
                for contact in contact_arr:
                    # noinspection PyProtectedMember
                    dict_contact = contact._asdict()
                    dict_contact["alter_id"] = alter_id
                    if dict_contact["additional_info"] is not None:
                        dict_contact["additional_info"] = dict_contact["additional_info"].to_json_encoded_obj()
                    dict_egonet["contacts"][contact_type].append(dict_contact)
        if compact:
            separators = (',', ':')
        else:
            separators = (', ', ': ')
        return json.dumps(dict_egonet, separators=separators)

    @staticmethod
    def create_from_json(egonet: str):
        """Create a new ego network from a JSON object.

        :param egonet: the JSON string representing the ego network
        :type egonet: str
        :return: a new ego network built from json data
        """
        dict_egonet = json.loads(egonet)
        ego = ContactEgoNetwork()
        ego.__dict__.update({k: v for k, v in dict_egonet.items() if
                             k in ego.__dict__ and not k.startswith("_") and k != "contacts" and k != "last_time"})
        ego.last_time = dict_egonet["last_time"]
        for contact_type, contact_arr in dict_egonet["contacts"].items():
            for contact in contact_arr:
                ego.add_contact(**contact)
        return ego

    def copy_egonet(self, exclude_contacts: bool = False):
        """Create a deepcopy of the ego network

        :param exclude_contacts: copy only profile data excluding contacts
        :type exclude_contacts: bool
        :return: a copy of the ego network
        :rtype: ContactEgoNetwork
        """
        egonet = type(self)()
        egonet.__dict__.update({k: v for k, v in self.__dict__.items() if
                                not k.startswith("_") and k != "alters" and k != "last_time"})
        egonet.last_time = self.last_time
        if not exclude_contacts:
            for contact in self._contacts_by_type["__all__"]:
                egonet.add_contact(timestamp=contact.timestamp, alter_id=contact.alter_id,
                                   contact_type=contact.contact_type, text=contact.text,
                                   num_contacted_alters=contact.num_contacted_alters,
                                   additional_info=contact.additional_info)
        return egonet

    def update_contacts(self, ego_network, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__"):
        """Update the contacts with those of another ego network.

        :param ego_network: the ego network from which contacts are taken
        :type ego_network: ContactEgoNetwork
        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        """
        contact_type = standardize_contact_type(contact_type)
        for ct in contact_type:
            for contact in ego_network._contacts_by_type[ct]:
                self.add_contact(**contact._asdict())

    def get_num_contacts(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__", from_timestamp: float = None,
                         to_timestamp: float = None) -> float:
        """The number of contacts for the ego network.

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :param from_timestamp: consider the contacts from the given timestamp (included)
        :type from_timestamp: float
        :param to_timestamp: consider the contacts up to the specified timestamp (excluded)
        :type to_timestamp; float
        :return: the number of contacts
        :rtype: float
        """
        contact_type = standardize_contact_type(contact_type)
        if from_timestamp is None and to_timestamp is None:
            return sum([len(self._contacts_by_type[ct]) for ct in contact_type])
        else:
            return len(self._get_sorted_contacts(from_timestamp=from_timestamp, to_timestamp=to_timestamp,
                                                 contact_type=contact_type))

    @memoize
    def get_first_contact_time(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__",
                               use_first_time: bool = False) -> float:
        """The timestamp of the first contact for the given contact type.

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :param use_first_time: whether to use the *first_time* parameter of the ego network for the calculation of the
            duration. If **False**, the time of the first contact is used
        :type use_first_time: bool
        :return: the timestamp of the first contact
        :rtype: float
        """
        if self.first_time is not None and (use_first_time or self.window_mode):
            return self.first_time
        contacts = self._get_sorted_contacts(contact_type=contact_type)
        if len(contacts) > 0:
            return contacts[0].timestamp
        else:
            return 0

    @memoize
    def get_last_contact_time(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__") -> float:
        """The timestamp of the last contact of the ego for the given contact type or the value stored for the class
        variable **last_time** if set.

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :return: the timestamp of the last contact or the value of the class variable **last_time** if set
        :rtype: float
        """
        if self.last_time is not None:
            return self.last_time
        contacts = self._get_sorted_contacts(contact_type=contact_type)
        if len(contacts) > 0:
            return contacts[-1].timestamp
        else:
            return 0

    @memoize
    def get_duration(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__",
                     use_first_time: bool = False) -> int:
        """Get the duration of the ego network, in number of years.

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :param use_first_time: whether to consider the *first_time* parameter of the ego network for the calculation of
            its duration. If **False**, the time of the first contact is used
        :type use_first_time: bool
        :return: the duration of the ego network
        :rtype: float
        """
        first_contact_time = self.get_first_contact_time(contact_type=contact_type, use_first_time=use_first_time)
        last_contact_time = self.get_last_contact_time(contact_type=contact_type)
        if first_contact_time and last_contact_time and first_contact_time <= last_contact_time:
            return (last_contact_time - first_contact_time) / ONE_YEAR
        else:
            return 0

    def get_alters_with_types(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__"):
        """Get the alters that have the specified contact types

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :return the list of alters with the specified contact types
        :rtype: list
        """
        contact_type = standardize_contact_type(contact_type=contact_type)
        return [alter for ct in contact_type for alter in self._alters_by_type[ct]]

    @memoize
    def get_avg_relationships_duration(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__",
                                       exclude_one_contact_relationships: bool = False) -> float:
        """Get the average duration of the relationships of the ego network, in years.

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :param exclude_one_contact_relationships: whether to exclude relationships with a single contact from the
            calculation
        :type exclude_one_contact_relationships: bool
        :return: the average duration of the relationships of the ego network
        :rtype: float
        """
        last_contact_time = self.get_last_contact_time(contact_type=contact_type)
        return float(np.nanmean([alter.get_duration(contact_type=contact_type, last_contact_time=last_contact_time)
                                 for alter in self.get_alters_with_types(contact_type=contact_type) if
                                 alter.get_num_contacts(contact_type=contact_type) > int(
                                     exclude_one_contact_relationships)]))

    def is_stable(self, min_duration: float = ONE_YEAR, min_total_contacts: int = 10,
                  check_dyn_stability: bool = False,
                  min_cts_win: MinContactsTimeWin = MinContactsTimeWin(10, ONE_MONTH, np.median),
                  contact_type: Union[Hashable, Tuple[Hashable]] = "__all__") -> bool:
        """Determine whether the ego network is stable or not.

        A stable ego network must satisfy all the requirements passed as arguments.

        :param min_duration: the minimum duration to consider the ego network as stable
        :type min_duration: float
        :param min_total_contacts: the minimum number of contacts to consider the ego network as stable
        :type min_total_contacts: int
        :param check_dyn_stability: whether to consider the stability of the ego network over time or not
        :type check_dyn_stability: bool
        :param min_cts_win: a tuple containing the minimum number of contacts in the time windows of the
            ego network, the temporal length of the window to consider the ego network as stable (in seconds), and a
            function to be applied to the list of the number of contacts over the time windows before checking if the
            output is greater or equal to the provided number of contacts in the time windows. These values are
            considered only in case **check_dyn_stability** is *True*
        :type min_cts_win: MinContactTimeWin
        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :return: True if the ego network is stable, False otherwise
        :rtype: bool
        :raise InvalidIntValueException: if one or more of the provided parameters are < 0
        """
        if min_duration < 0 or min_total_contacts < 0 or min_cts_win is not None \
                and (min_cts_win.min_num_contacts < 0 or min_cts_win.win_size < 0):
            raise InvalidIntValueException("negative thresholds are not allowed")

        first_contact_time = self.get_first_contact_time(contact_type=contact_type)
        last_contact_time = self.get_last_contact_time(contact_type=contact_type)
        total_num_contacts = self.get_num_contacts(contact_type=contact_type)
        static_stability, dyn_stability = False, True
        if first_contact_time is not None and last_contact_time is not None and first_contact_time < last_contact_time:
            static_stability = last_contact_time - first_contact_time >= min_duration \
                               and total_num_contacts >= min_total_contacts
            if static_stability and check_dyn_stability:
                n_windows = math.floor((last_contact_time - first_contact_time) / min_cts_win.win_size)
                # do not count the contacts in the last window if it is shorter than a whole window size
                dyn_counts = [self.get_num_contacts(contact_type=contact_type,
                                                    from_timestamp=int(first_contact_time + min_cts_win.win_size * i),
                                                    to_timestamp=int(
                                                        first_contact_time + min_cts_win.win_size * (i + 1)))
                              for i in range(n_windows)]
                dyn_stability = min_cts_win.function(dyn_counts) > min_cts_win.min_num_contacts
        return static_stability and dyn_stability

    @synchronized_method
    def add_contact(self, timestamp: int, alter_id: Hashable, contact_type: Hashable = "__default__", text: str = None,
                    num_contacted_alters: int = 1, additional_info: ContactAdditionalInfo = None):
        """Add a contact to the ego network.

        Add a new direct contact for the ego by specifying the timestamp of the contact and the id of the alter
        contacted. A new Alter object is added to the ego network if the contacted alter id is not yet in the ego
        network.

        :param timestamp: the timestamp of the contact
        :type timestamp: int
        :param alter_id: the id of the contacted alter
        :type alter_id: Hashable
        :param contact_type: a value indicating the type of contact. Used to support ego networks with multiple types
            of social relationships. The special value "__all__" can be used to consider the complete network
            containing all types of contacts. Contact types starting with "__no_contact" will be ignored for the
            calculation of properties of the whole ego network ("__all__" contacts). This special contact type can be
            used to store contacts that do not have a specific recipient and need to be analysed separately from other
            contacts
        :type contact_type: Hashable
        :param text: the text associated with the contact
        :type text: str
        :param num_contacted_alters: the number of alters in the same contact
        :type num_contacted_alters: int
        :param additional_info: additional information to be attached to the contact
        :type additional_info: ContactAdditionalInfo
        """
        if alter_id != self.ego_id:
            self.clear_cache()
            self._freq_vector = defaultdict(list)
            self._saved_freqs = set()
            if alter_id not in self.alters:
                self.alters[alter_id] = ContactAlter(min_rel_duration=self._rel_min_duration, alter_id=alter_id)
            new_contact = self.alters[alter_id].add_contact(timestamp=timestamp, contact_type=contact_type, text=text,
                                                            num_contacted_alters=num_contacted_alters,
                                                            additional_info=additional_info, alter_id=alter_id)
            if not str(contact_type).startswith("__no_contact"):
                self._contacts_by_type["__all__"].append(new_contact)
            self._contacts_sorted = False
            self._contacts_sort_keys = defaultdict(list)
            self._contacts_by_type[contact_type].append(new_contact)
            self._alters_by_type[contact_type].add(self.alters[alter_id])
            self._alters_by_type["__all__"].add(self.alters[alter_id])

    def add_contacts(self, contacts: List[FullContact]):
        """Add a list of contacts to the ego network.

        :param contacts: a list of contacts
        :type contacts: List[FullContact]
        """
        for contact in contacts:
            # noinspection PyProtectedMember
            self.add_contact(**(contact._asdict()))

    def get_all_contacts(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__") -> Iterator[FullContact]:
        """Get all the contacts of the ego network

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :return: an iterator for the contacts
        :rtype: Iterator[FullContact]
        """
        contact_type = standardize_contact_type(contact_type=contact_type)
        return [contact for ct in contact_type for contact in self._contacts_by_type[ct]]

    def _calculate_frequencies(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__",):
        """Calculate the frequencies of contact between the ego and its alters and store them

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        """
        contact_type = standardize_contact_type(contact_type)
        if "__all__" in contact_type:
            contact_type = "__all__"
        if contact_type not in self._saved_freqs:
            if self.window_mode:
                    first_contact_time = self.first_time
            else:
                first_contact_time = None
            last_contact_time = self.get_last_contact_time()
            for alter in self.get_alters_with_types(contact_type=contact_type):
                freq = alter.get_frequency(first_contact_time=first_contact_time, last_contact_time=last_contact_time,
                                           contact_type=contact_type)
                if freq > 0:
                    self._freq_vector[contact_type].append(AlterFreq(alter.alter_id, freq))
            self._saved_freqs.add(contact_type)

    def get_alter_freq_list(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__", active: bool = False,
                            limit: int = 0, stable: bool = True) -> List[AlterFreq]:
        """Get the vector of contact frequencies of the network with their alter_id.

        Calculate the frequencies of contact of the alters. If from_timestamp or to_timestamp are provided, stable is
        automatically set to False

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :param active: if True, return only alters with active contact frequency (i.e. greater or equal to 1)
        :type active: bool
        :param limit: return only the specified maximum number of alters, taking those with highest contact frequency
        :type limit: int
        :param stable: whether to consider only stable relationships
        :type stable: bool
        :return: the vector containing AlterFreq tuples
        :rtype: List[AlterFreq]
        """
        self._calculate_frequencies(contact_type=contact_type)
        if self.window_mode:
            first_contact_time = self.first_time
        else:
            first_contact_time = None
        last_contact_time = self.get_last_contact_time()
        alter_freq_list = super().get_alter_freq_list(contact_type=contact_type, active=active, limit=limit)
        filtered_alter_freq_list = []
        for alter_freq in alter_freq_list:
            # check stability only if it is required and a global first time for the ego network is not set
            if not stable or self.window_mode or self.alters[alter_freq.alter_id].is_stable(
                    contact_type=contact_type, first_contact_time=first_contact_time,
                    last_contact_time=last_contact_time):
                filtered_alter_freq_list.append(alter_freq)
        return filtered_alter_freq_list

    @memoize
    def get_alter_count_list(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__") -> List[AlterCount]:
        """Get the vector of contact intensities of the network with their alter_id.

        Calculate the intensity of contact (number of exchanged messages) between the ego and the alters.

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :return: the vector containing AlterCount tuples
        :rtype: List[AlterCount]
        """
        return [AlterCount(alter.alter_id, alter.get_num_contacts(contact_type=contact_type)) for
                alter in self.get_alters_with_types(contact_type=contact_type)]

    def get_egonet_size(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__", active: bool = False,
                        stable: bool = True, exclude_one_contact_relationships: bool = False) -> int:
        """Get the total size of the ego network as the total number of alters.

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :param active: if True, count only alters with active contact frequency (i.e. greater or equal to 1)
        :type active: bool
        :param stable: whether to consider only stable relationships
        :type stable: bool
        :param exclude_one_contact_relationships: whether to exclude relationships with a single contact from the
            calculation
        :type exclude_one_contact_relationships: bool
        :return: the size of the ego network
        :rtype: int
        """
        return len([alter_id for alter_id, freq in self.get_alter_freq_list(
            contact_type=contact_type, active=active, stable=stable) if not exclude_one_contact_relationships or
                    self.alters[alter_id].get_num_contacts(contact_type=contact_type) >
                    int(exclude_one_contact_relationships)])

    def get_subnetwork(self, from_timestamp: float = None, to_timestamp: float = None,
                       contact_type: Union[Hashable, Tuple[Hashable]] = "__all__", set_window_mode: bool = False):
        """Return a copy of the ego network containing only contacts within a given period of time.

        This portion or slice of ego network represents the contacts of the ego within a given time interval.

        :param from_timestamp: first time considered for the contacts in the sliced data structure (included)
        :param to_timestamp: last time considered for the contacts in the sliced data structure (excluded)
        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :param set_window_mode: whether to set the window_mode parameter of the sub ego network to True
        :type from_timestamp: float
        :type to_timestamp: float
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :type set_window_mode: bool
        :return: a copy of the data structure containing the alters with contacts in the specified interval
        :rtype: ContactEgoNetwork
        :raise InvalidTimestampException: if from_timestamp > to_timestamp
        """
        contact_type = standardize_contact_type(contact_type)
        sub_egonet = self.copy_egonet(exclude_contacts=True)
        if not from_timestamp:
            from_timestamp = self.get_first_contact_time()
        if not to_timestamp:
            to_timestamp = self.get_last_contact_time()
        # if the size of the time window is less than one year, the duration of the relationships are calculated as the
        # size of the windows instead of the time between the first contact for each relationship and the end of the
        # window
        if set_window_mode:
            sub_egonet.window_mode = True
        else:
            sub_egonet.window_mode = False
        sub_egonet.first_time = from_timestamp
        sub_egonet.last_time = to_timestamp
        if from_timestamp and to_timestamp and from_timestamp < to_timestamp:
            sub_contacts = self._get_sorted_contacts(from_timestamp=from_timestamp, to_timestamp=to_timestamp,
                                                     contact_type=contact_type)
            for sub_contact in sub_contacts:
                sub_egonet.add_contact(timestamp=sub_contact.timestamp, alter_id=sub_contact.alter_id,
                                       contact_type=sub_contact.contact_type, text=sub_contact.text,
                                       num_contacted_alters=sub_contact.num_contacted_alters,
                                       additional_info=sub_contact.additional_info)
        return sub_egonet

    def get_stability(self, window_size: float = ONE_YEAR, sim_types=None, limits=None,
                      step: float = ONE_YEAR, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__",
                      n_circles=1) -> Dict:
        """Get a structure containing the stability values for the ego network over time.

        Each similarity value represents the stability over time of the ego network. Each value is calculated as the
        average stability calculated between pairs of adjacent time slices of the network.

        The returned structure is a dictionary with sim_types as key and, as value, dictionaries with limit as key and
        the actual stability values as value.

        :param window_size: the size of the time window, in seconds. The default value is one year
        :type window_size: float
        :param sim_types: the list of types of stability measure to calculate, "jaccard" for Jaccard coefficient,
            "norm_spearman" for a stability measure based on the normalized version of Spearman's footrule index,
            "ring_member" for a stability measure based on the normalized level_changes index between rings,
            "ring_member_unstable" for a similarity measure based on the normalized level_changes index, but which does
            not consider stable relationships, thus measuring the dynamics of unstable relationships,
            "ring_member_rel_var" for a measure of variability (i.e. the opposite of stability) based on layer_changes
            index with relative normalization, and "ring_member_var_raw" for the raw value of layer_changes index (not
            normalized). The last two measures of variability can have the "_unstable" suffix to consider only
            unstable relationships. See :any:`egonetworks.similarity` module for more information about these indices.
        :type sim_types: Iterable[str]
        :param limits: a list of limits as maximum number of top alters (in terms of contact frequency) to consider
            in the calculation
        :type limits: Iterable[int]
        :param step: distance between the starting time of adjacent temporal windows. Windows are overlapped if the
            step is less than *window_size*
        :type step: float
        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :param n_circles: divide the alters in the ego network of each time window into the specified number of circles
            and calculate the stability for the different rings of the network
        :type n_circles: int
        :return: the similarity values for the given parameters
        :rtype: Dict
        """
        if sim_types is None:
            sim_types = ["jaccard"]
        if limits is None:
            limits = [0]
        sim_struc = defaultdict(lambda: defaultdict(list))
        first_contact_time = self.get_first_contact_time(contact_type=contact_type)
        last_contact_time = self.get_last_contact_time(contact_type=contact_type)
        if first_contact_time is not None and last_contact_time is not None:
            n_windows = math.floor((last_contact_time - first_contact_time - window_size) / step + 1)
            if n_windows > 1:
                for i in range(n_windows - 1):
                    # this part, in particular when circles are required is easier and more readable with sub_networks,
                    # and the overhead of creating new objects is negligible compared to the computational cost of the
                    # functions
                    sub_egonet1 = self.get_subnetwork(int(first_contact_time + i * step),
                                                      int(first_contact_time + i * step + window_size),
                                                      contact_type, set_window_mode=True)
                    sub_egonet2 = self.get_subnetwork(int(first_contact_time + (i + 1) * step),
                                                      int(first_contact_time + (i + 1) * step + window_size),
                                                      contact_type, set_window_mode=True)
                    for sim_type in sim_types:
                        for limit in limits:
                            sim_struc[sim_type][limit].append(self.get_egonet_similarity(sub_egonet1, sub_egonet2,
                                                                                         sim_type, limit, contact_type,
                                                                                         n_circles))
                for sim_type in sim_types:
                    for limit in limits:
                        sim_struc[sim_type][limit] = [sum(item) / len(item) for item in
                                                      zip(*sim_struc[sim_type][limit])]
        return sim_struc

    def get_size_time_series(self, window_size: float = ONE_YEAR, step: float = ONE_YEAR,
                             contact_type: Union[Hashable, Tuple[Hashable]] = "__all__", active: bool = False,
                             n_circles: int = 1, count_contacts: bool = False) -> List[List[int]]:
        """Calculate the size of the ego network over time.

        :param window_size: the size of the time window, in seconds. The default value is one year
        :type window_size: float
        :param step: distance between the starting time of adjacent temporal windows. Windows are overlapped if the
            step is less than *window_size*
        :type step: float
        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :param active: whether to consider only the active network size of the ego network
        :type active: bool
        :param n_circles: divide the alters in the ego network of each time window into the specified number of circles
            and calculate the stability for the different rings of the network
        :type n_circles: int
        :param count_contacts: whether to count raw contacts instead of social relationships. Valid only when
            *n_circles* is equal to 1.
        :type count_contacts: bool
        :return: the time series of the sizes of the ego network, one list for each circle
        :rtype: List[List[int]]
        """
        first_contact_time = self.get_first_contact_time(contact_type=contact_type)
        last_contact_time = self.get_last_contact_time(contact_type=contact_type)
        sizes = [[] for _ in range(n_circles)]
        if first_contact_time > 0 and last_contact_time > 0:
            n_windows = math.floor((last_contact_time - first_contact_time - window_size) / step + 1)
            if n_windows > 1:
                for i in range(n_windows - 1):
                    # a sub ego network is created only in case circles are required
                    sub_egonet = self.get_subnetwork(int(first_contact_time + i * step),
                                                     int(first_contact_time + i * step + window_size), contact_type,
                                                     set_window_mode=True)
                    if n_circles > 1:
                        circles_properties = sub_egonet.get_circles_properties(n_circles=n_circles,
                                                                               contact_type=contact_type, active=active)
                        if circles_properties.sizes is not None:
                            for circle, size in enumerate(circles_properties.sizes):
                                sizes[circle].append(size)
                        else:
                            for i in range(n_circles):
                                sizes[i].append(0)
                    else:
                        if count_contacts:
                            sizes[0].append(sub_egonet.get_num_contacts())
                        else:
                            sizes[0].append(sub_egonet.get_egonet_size(active=active))
        return sizes

    def get_relationships_burstiness(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__") -> List[float]:
        """Get a list of burstiness indices, one for each relationship in the ego network

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :return: a list of bursiness indices, one for each relationship
        :rtype: List[float]
        """
        return [alter.get_burstiness_index(contact_type=contact_type) for alter in
                self.get_alters_with_types(contact_type=contact_type)]

    def get_relationships_dispersion(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__") -> List[float]:
        """Get a list of dispersion indices, one for each relationship in the ego network

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :return: a list of dispersion indices, one for each relationship
        :rtype: List[float]
        """
        return [alter.get_dispersion_index(contact_type=contact_type) for alter in
                self.get_alters_with_types(contact_type=contact_type)]

    def get_ego_frequency_contact_time_series(self, window_size: float = ONE_MONTH,
                                              contact_type: Union[Hashable, Tuple[Hashable]] = "__all__",
                                              first_time: float = None, last_time: float = None) -> List:
        """Generate a time series with the number of contacts sent by ego in each time window of the ego network.

        :param window_size: the size of the time window, in seconds. The default value is one month
        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :param first_time: the first time to consider
        :param last_time: the last time to consider
        :type window_size: float
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :return: the time series of the number of contacts created by ego over time
        :rtype: List
        """
        contact_series = []
        if first_time is None:
            first_time = self.get_first_contact_time(contact_type=contact_type)
        if last_time is None:
            last_time = self.get_last_contact_time(contact_type=contact_type)
        if first_time is not None and last_time is not None:
            n_windows = math.floor((last_time - first_time) / window_size)
            if n_windows > 0:
                for i in range(n_windows):
                    contact_series.append(self.get_num_contacts(contact_type=contact_type,
                                                                from_timestamp=int(first_time + i * window_size),
                                                                to_timestamp=int(
                                                                    first_time + i * window_size + window_size)))
        return contact_series

    def get_growth_rate(self, window_size: float = ONE_MONTH,
                        contact_type: Union[Hashable, Tuple[Hashable]] = "__all__"):
        """This method calculates the growth rate of the ego network in terms of number of new alters contacted over
        time.

        The growth rate is calculated as the number of new distinct alters contacted over time by the ego. This is
        defined for each pair of consecutive time windows of the specified length and is equal to the number of the
        additional alters in each window with respect to all the previous windows. The method returns the sequence of
        growth rates for all the windows in the ego network.

        :param window_size: the window size to consider
        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type window_size: float
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :return: the vector of growth values of the ego network
        :rtype: List[int]
        """
        growth_vec = []
        first_contact_time = self.get_first_contact_time(contact_type=contact_type)
        last_contact_time = self.get_last_contact_time(contact_type=contact_type)
        if first_contact_time is not None and last_contact_time is not None:
            n_windows = math.floor((last_contact_time - first_contact_time) / window_size)
            prev_contacted_alters = set()
            for i in range(n_windows):
                sub_egonet = self.get_subnetwork(from_timestamp=int(first_contact_time + i * window_size),
                                                 to_timestamp=int(first_contact_time + (i + 1) * window_size),
                                                 contact_type=contact_type)
                contacted_alters = sub_egonet.get_alter_ids()
                growth_vec.append(len(set(contacted_alters).difference(prev_contacted_alters)))
                prev_contacted_alters.update(set(contacted_alters))
        return growth_vec

    def generate_topics_network(self, gcube_token: str, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__",
                                cluster_topics: bool = False, num_threads: int = 500, rho_thr: float = 0.1,
                                min_edge_weight: float = 0):
        """Get the network of topics extracted from the text of the contacts that the ego sent to her alters and create
        a new ego network containing topics as alters.

        This method uses TagMe API (see https://tagme.d4science.org/tagme) API to extract a set of topics (in the form
        of Tagme topic ids - the same ids used by Wikipedia - or clusters of topics numbered from 0 to n) from the text
        of each contact sent by ego and then creates a new ContactEgoNetwork object containing the topics as alters.
        Contacts with the topics coincide with the messages where they appear.

        :param gcube_token: the key to access Tagme API
        :type gcube_token: str
        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :param cluster_topics: whether to group together similar topics and consider them as one single topic. The name
            of the most frequently used topic is used as name of the group. To get the clusters of topics, the
            similarity network of topics is used
        :type cluster_topics: bool
        :param num_threads: the number of parallel requests to be sent to TagMe API
        :type num_threads: int
        :param rho_thr: the minimum rho value of the topic (see https://tagme.d4science.org/tagme/tagme_help.html)
        :type rho_thr: float
        :param min_edge_weight: minimum edge weight to be considered for clustering the relatedness network of topics
        :type min_edge_weight: float
        :return: a new ego network containing the topics as alters, a mapping of the node ids in case they are
            clustered, and the modularity score (goodness) of the clusterization
        :rtype: Tuple[ContactEgoNetwork, ContactEgoNetwork, Dict, float]
        """
        contact_type = standardize_contact_type(contact_type)

        def fetch_topics(text: str, gcube_token: str, topic_egonet: ContactEgoNetwork, topics_list: deque):
            topics = get_topics_from_text(text=text, gcube_token=gcube_token, rho_th=rho_thr).items()
            for topic_id, topic_count in topics:
                topic_egonet.add_contact(timestamp=contact.timestamp, alter_id="__info_topic_" + str(topic_id),
                                         contact_type="info_topic",
                                         num_contacted_alters=len(topics))
                if cluster_topics:
                    topics_list.append(topic_id)

        topic_egonet = ContactEgoNetwork(ego_id=self.ego_id, last_time=self.last_time)
        topics_list = deque()
        cluster_map = {}
        modularity = 0
        logging.debug("extracting topics from contacts")
        try:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                contacts = [contact for contact in self.get_all_contacts(contact_type=contact_type)]
                if "__all__" in contact_type:
                    contacts.extend([no_contact for no_contact in self.get_all_contacts(contact_type="__no_contact__")])
                for contact in contacts:
                    executor.submit(fetch_topics, contact.text, gcube_token, topic_egonet, topics_list)
        except Exception as e:
            print(str(e))

        logging.debug("topic network has " + str(topic_egonet.get_egonet_size()) + " alters")
        topic_egonet_clustered = None
        if cluster_topics:
            logging.debug("calculating topics relatedness")
            topics_set = set(topics_list)
            topics_edgelist = get_topics_rel_network(topics=list(topics_set), gcube_token=gcube_token,
                                                     num_threads=num_threads)
            # remap nodes ids for igraph
            vertex_map = dict([(str(topic), idx) for idx, topic in enumerate(topics_set)])
            # create a graph with the same number of vertices of the topics
            graph = igraph.Graph()
            logging.debug("creating topics relatedness graph adding vertices first")
            graph.add_vertices(len(topics_set))
            # add edges between topics if they are above the chosen threshold and if they are not already there
            logging.debug("adding links to the graph")
            es = []
            es_set = set()
            weights = []
            for topic_pair in topics_edgelist:
                if float(topic_pair[2]) > min_edge_weight and (vertex_map[topic_pair[0]], vertex_map[topic_pair[1]]) \
                        not in es_set and (vertex_map[topic_pair[1]], vertex_map[topic_pair[0]]) not in es_set:
                    es.append((vertex_map[topic_pair[0]], vertex_map[topic_pair[1]]))
                    es_set.add((vertex_map[topic_pair[0]], vertex_map[topic_pair[1]]))
                    weights.append(float(topic_pair[2]))
            graph.add_edges(es)
            graph.es["weight"] = weights
            topic_egonet_clustered = ContactEgoNetwork(ego_id=self.ego_id, last_time=self.last_time)
            if graph.ecount() > 0:
                # there is at least one edge, so we apply community detection
                logging.debug("applying community detection on relatedness network")
                dendogram = graph.community_fastgreedy(weights="weight")
                clusters = dendogram.as_clustering()
                membership = clusters.membership
                modularity = clusters.modularity
                for contact in topic_egonet.get_all_contacts(contact_type="info_topic"):
                    topic_egonet_clustered.add_contact(timestamp=contact.timestamp,
                                                       alter_id="__clustered_topic__" +
                                                                str(membership[vertex_map[
                                                                    str(contact.alter_id.split("_")[-1])]]),
                                                       contact_type="info_clustered_topic",
                                                       num_contacted_alters=contact.num_contacted_alters)
                    # store the membership values in the map
                    cluster_map[str(contact.alter_id)] = membership[vertex_map[str(contact.alter_id)]]
                logging.debug("clustered network has " + str(topic_egonet_clustered.get_egonet_size()) + " alters")
            else:
                logging.debug("since there are no edges in the relatedness network, I skip community detection")
                # change contact_type to __clustered_topic__
                for contact in topic_egonet.get_all_contacts(contact_type="info_topic"):
                    topic_egonet_clustered.add_contact(timestamp=contact.timestamp,
                                                       alter_id=contact.alter_id,
                                                       contact_type="info_clustered_topic",
                                                       num_contacted_alters=contact.num_contacted_alters)
        return topic_egonet, topic_egonet_clustered, cluster_map, modularity

    def get_relationships_properties(self, window_size: float = ONE_YEAR,
                                     contact_type: Union[Hashable, Tuple[Hashable]] = "__all__",
                                     relationship_ids: List[Hashable] = None, active=False, n_circles: int = 5,
                                     step: float = ONE_MONTH,
                                     normalized: bool = False,
                                     stability: bool = False) -> Dict[Hashable, RelationshipsProperties]:
        """Get the properties of social relationships.

        :param window_size: the size of the time window for the calculation of dynamic properties
        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :param relationship_ids: get the properties only for the specified relationship ids. Get all the relationships
            if None
        :param active: if True, count only alters with active contact frequency (i.e. greater or equal to 1)
        :param n_circles: number of circles to consider for the classification of relationships into ego network rings
        :param step: distance between the starting time of adjacent temporal windows. Windows are overlapped if the
            step is less than one year. This is used for the calculation of dynamic properties
        :param normalized: if True, the normalized version of the indices is calculated, where applicable
        :param stability: if True, stability indices are calculated, where applicable
        :type window_size: float
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :type relationship_ids: List[Hashable]
        :type active: bool
        :type n_circles: int
        :type step: float
        :type normalized: bool
        :type stability: bool
        :return: the properties of the relationships
        :rtype: Dict[Hashable, RelationshipsProperties]
        """
        rel_alter_freq_vec = self.get_alter_freq_list(contact_type=contact_type, active=active)
        rel_changes_index = self.get_relationships_ring_change_index(window_size=window_size,
                                                                     contact_type=contact_type,
                                                                     n_circles=n_circles, step=step,
                                                                     normalized=normalized, stability=stability)
        rel_membership = self.get_circles_properties(contact_type=contact_type, n_circles=n_circles,
                                                     active=active).membership
        rel_properties = {}
        if rel_membership is not None:
            if relationship_ids is not None:
                relationship_ids = set(relationship_ids)
            for rel_alter_freq in rel_alter_freq_vec:
                if rel_alter_freq.alter_id in rel_changes_index and rel_alter_freq.alter_id in rel_membership and (
                                relationship_ids is None or rel_alter_freq.alter_id in relationship_ids):
                    rel_properties[rel_alter_freq.alter_id] = RelationshipsProperties(
                        frequency=rel_alter_freq.frequency,
                        ring_changes_index=rel_changes_index[rel_alter_freq.alter_id],
                        ring_membership=rel_membership[rel_alter_freq.alter_id])
        return rel_properties

    def get_relationships_ring_change_index(self, window_size: float = ONE_YEAR,
                                            contact_type: Union[Hashable, Tuple[Hashable]] = "__all__",
                                            n_circles: int = 5, step: float = ONE_MONTH,
                                            normalized: int = False, stability: bool = False,
                                            remove_stable: bool=False) -> Dict[Hashable, float]:
        """Calculate an index that measures the average number of changes between rings for each relationship :math:`r`
        of the ego network over time.

        The index is derived from the :any:`egonetworks.similarity.level_changes` similarity measure and it is similar
        to the stability index of the ego network calculated through
        :any:`egonetworks.core.ContactEgoNetwork.get_stability` with **sim_type** argument set to "ring_member".
        However, the present method returns the average *level_change* index for each alter in the ego network and is
        useful for obtaining the properties of single social relationships. The index is calculated as follows:

        .. math::
           C(r) = \\frac{\\sum_{i=max(\\sigma(w^f(r)) - 1, 1)}^{min(\\sigma(w^l(r)), N - 1)}{c(r, w_i, w_{i+1})}}
               {min(\\sigma(w^l(r)), N - 1) - max(\\sigma(w^f(r)) - 1, 1)},

        where :math:`w_j` is the j :superscript:`th` time window, considering the list of all time windows in which the
        time series of the contacts of the ego network can be divided (according to the specified arguments) ordered by
        time, :math:`\\sigma(w)` is the position (index) of window :math:`w` in such list, :math:`N` is the number of
        windows in the lis, :math:`w^f(r)` and :math:`w^l(r)` are respectively the first and the last time window in
        which there is a contact for relationship :math:`r`, :math:`c(r, w_i, w_j)` is the number of changes between
        rings for :math:`r` between :math:`w_i` and :math:`w_j`, :math:`k` is the number of circles in the ego network.

        The windows considered for the calculation of the index start from the one before the first window containing
        contacts for the relationship and last with the one after the last window containing contacts for the
        relationship. Nonetheless, these are bounded by the first and the last available windows. In this way, the
        changes related to a relationship entering or leaving the ego network are counted, but, at the same time,
        relationships which are maintained from the first to the last window of the ego network are considered as they
        never joined or left the network and they were present also before and after the first and the last available
        windows.

        The number of changes for :math:`r` between :math:`w_i` and :math:`w_j` (i.e. :math:`c(r, w_i, w_j)`) is the
        absolute difference between the position of the relationship in the ego network layers at time :math:`w_i`
        (:math:`w_i(r)`) and the position of the same at time :math:`w_j` (i.e. :math:`w_j(r)`):

        .. math::
           c(r, w_i, w_j) = |w_i(r) - w_j(r)|

        In the normalized version of the index, the number of changes in the ring membership at each time step is
        divided by maximum number of possible changes, which is equal to :math:`k`:

        .. math::
           \\overline{C}(r) = \\frac{1}{k} C(r).

        In the relative normalization, the number of changes of the relationship between each pair of windows is
        divided by the maximum number of changes that the object could do from the starting layer:

        .. math::
           \\hat{C}(r) = \\frac{\\sum_{i=max(\\sigma(w^f(r)) - 1, 1)}^{min(\\sigma(w^l(r)), N - 1)}{\\frac{
               c(r, w_i, w_{i+1})}{max(k - (w_i(r) - 1), w_i(r) - 1)}}}{min(\\sigma(w^l(r)), N - 1) - max(
               \\sigma(w^f(r)) - 1, 1)}.

        The stability index based on this change index (see the documentation for the argument **stability**) is
        calculated as follows:

        .. math::
           S_c(r) = 1 - \\overline{C}(r).

        or

        .. math::
           S_c(r) = 1 - \\hat{C}(r)

        in case of relative normalization.

        :param window_size: the size of the time window to consider, in seconds. The default value is one year
        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :param n_circles: number of ego network circles to consider
        :param step: distance between the starting time of adjacent temporal windows. Windows are overlapped if the
            step is less than one year
        :param normalized: if 1, the normalized version of the index is calculated. If equal to 2, the relative
            normalization is used
        :param stability: if True, calculates the stability index :math:`S_c(r)` instead of the change index
        :param remove_stable: if True, stable objects (i.e. objects which do not change level in the two lists) are not
            counted
        :type window_size: float
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :type n_circles: int
        :type step: float
        :type normalized: int
        :type stability: bool
        :type remove_stable: bool
        :return: a dictionary containing the values of the index calculated for each relationship
        :rtype: Dict[Hashable, float]
        :raise InvalidIntValueException: if window_size, step or n_circles are not greater than 0
        """
        if window_size <= 0 or step <= 0:
            raise InvalidIntValueException("window size and step must be greater than 0")
        if n_circles <= 0:
            raise InvalidIntValueException("n_circles must be greater than 0")

        # noinspection PyPep8Naming
        C_dict = {}
        first_contact_time = self.get_first_contact_time(contact_type=contact_type)
        last_contact_time = self.get_last_contact_time(contact_type=contact_type)
        # we need at least two windows (window_size + step)
        if first_contact_time is not None and last_contact_time is not None \
                and last_contact_time - first_contact_time >= window_size + step:
            membership_arr = []
            alter_ids = self.get_alter_ids(contact_type=contact_type)
            # these structures will contain the first and the last windows for each relationship
            alters_first_window = dict(zip(alter_ids, [-1] * len(alter_ids)))
            alters_last_window = dict(zip(alter_ids, [-1] * len(alter_ids)))
            n_windows = math.floor((last_contact_time - first_contact_time - window_size) / step + 1)
            for i in range(n_windows):
                sub_egonet = self.get_subnetwork(first_contact_time + i * step,
                                                 first_contact_time + i * step + window_size, contact_type,
                                                 set_window_mode=True)
                win_member = sub_egonet.get_circles_properties(n_circles=n_circles,
                                                               contact_type=contact_type, active=False).membership
                if win_member is None:
                    win_member = {}
                membership_arr.append(win_member)
                for alter_id in win_member.keys():
                    if alters_first_window[alter_id] == -1:
                        alters_first_window[alter_id] = i
                    alters_last_window[alter_id] = i

            for alter_id in alter_ids:
                c_sum = 0
                first_win_idx = max(alters_first_window[alter_id] - 1, 0)
                # considering an additional window since the slicing excludes the last index
                last_win_idx = min(alters_last_window[alter_id] + 2, len(membership_arr))
                count = last_win_idx - first_win_idx - 1
                # if count is equal to 0 it means that the relationship has not been evaluated. This may happen when the
                # relationship is present only in the very last part of the time series of the network, and this has
                # been excluded since it is not multiple of the step
                if count > 0:
                    # for relationships existing from the first win this is the first ring is the membership of first
                    # win, for the other relationships, this is equal to the number of rings + 1 (they are outside the
                    # egonet)
                    prev_ring = membership_arr[first_win_idx].get(alter_id, n_circles + 1)
                    # skip the very first window since we counted it with prev_ring variable
                    for win_member in membership_arr[first_win_idx + 1:last_win_idx]:
                        # get actual ring. If the rel is not present in this window, set the actual ring to n_circles
                        # + 1 (it's outside the ego network)
                        actual_ring = win_member.get(alter_id, n_circles + 1)
                        c_sum += abs(prev_ring - actual_ring)
                        if normalized == 2:
                            c_sum /= max(n_circles - (actual_ring - 1), actual_ring - 1)
                            if remove_stable and c_sum == 0:
                                count -= 1
                        prev_ring = actual_ring
                    if count > 0:
                        # noinspection PyPep8Naming
                        C = c_sum / count
                    else:
                        # noinspection PyPep8Naming
                        C = 0
                    if normalized == 1 or not normalized and stability:
                        C /= n_circles
                    if stability:
                        # noinspection PyPep8Naming
                        C = 1 - C
                    C_dict[alter_id] = C
                else:
                    C_dict[alter_id] = None
        return C_dict

    def get_avg_relationships_persistence(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__",
                                          contiguous: bool = False, window_size: float = ONE_MONTH,
                                          normalized: bool=False) -> float:
        """Calculate the average persistence of relationships in the ego network over time.

        This is a stability index that tells the degree to which relationships persist in the ego network over time. It
        is calculated as the average number of time windows in which the alter related to a relationship has been. If
        normalized, the index is calculated in percentage with respect to the total number of time windows in the ego
        network.

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :param contiguous: consider only the maximum number of contiguous windows in which the alters have been
            contacted
        :param window_size: the size of the time windows in seconds
        :param normalized: calculate the index in percentage with respect to the total number of time windows of the ego
            network
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :type contiguous: bool
        :type window_size: float
        :type normalized: bool
        :return: the average persistence index of the relationships in the ego network
        :rtype: float
        """
        alter_ids = self.get_alter_ids(contact_type=contact_type)
        pers_dict = dict(zip(alter_ids, [0] * len(alter_ids)))
        cumulative_pers_dict = None
        prev_pers_dict = None
        if contiguous:
            cumulative_pers_dict = dict(zip(alter_ids, [0] * len(alter_ids)))
            prev_pers_dict = dict(zip(alter_ids, [0] * len(alter_ids)))
        avg_pers = None
        first_time = self.get_first_contact_time(contact_type=contact_type)
        last_time = self.get_last_contact_time(contact_type=contact_type)
        if first_time is not None and last_time is not None:
            n_windows = math.floor((last_time - first_time) / window_size)
            if n_windows > 0:
                for i in range(n_windows):
                    sub_egonet = self.get_subnetwork(from_timestamp=int(first_time + i * window_size),
                                                     to_timestamp=int(first_time + i * window_size + window_size),
                                                     contact_type=contact_type)
                    sub_alter_ids = sub_egonet.get_alter_ids()
                    for sub_alter_id in sub_alter_ids:
                        if contiguous:
                            cumulative_pers_dict[sub_alter_id] += 1
                            if cumulative_pers_dict[sub_alter_id] > pers_dict[sub_alter_id]:
                                pers_dict[sub_alter_id] = cumulative_pers_dict[sub_alter_id]
                        else:
                            pers_dict[sub_alter_id] += 1
                    if contiguous:
                        not_present_alter_ids = set(alter_ids).difference(set(sub_alter_ids))
                        for not_present_alter_id in not_present_alter_ids:
                            prev_pers_dict[not_present_alter_id] = 0
                            cumulative_pers_dict[not_present_alter_id] = 0
                        for sub_alter_id in sub_alter_ids:
                            prev_pers_dict[sub_alter_id] = 1
                if len(pers_dict) > 0:
                    if normalized:
                        avg_pers = sum([act_win_n / n_windows for act_win_n in pers_dict.values()]) / len(pers_dict)
                    else:
                        avg_pers = sum([act_win_n for act_win_n in pers_dict.values()]) / len(pers_dict)
        return avg_pers

    def get_static_dynamic_strength(self, window_size: float = ONE_YEAR,
                                    contact_type: Union[Hashable, Tuple[Hashable]] = "__all__",
                                    n_circles: int = 5, step: float = ONE_MONTH) -> List[Tuple[int, Counter]]:
        """compute the ring in which the relationship is contained considering the static view of the ego network and
           count the number of times the relationships is in each circle in its dynamic view

        :param window_size: the size of the time windows in seconds
        :type window_size: float
        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :param n_circles: the number of ego network circles into which the ego network must be divided
        :type n_circles: int
        :param step: distance between the starting time of adjacent temporal windows. Windows are overlapped if the
            step is less than one year
        :type step: float
        :return: a list of elements, one for each relationship in the ego network, containing two sub-elements: the tie
            strength for the static view of the relationship and a counter of the visits in each ring in its dynamic
            view
        :rtype: List[Tuple[int, Counter]]
        """
        first_time = self.get_first_contact_time(contact_type=contact_type)
        last_time = self.get_last_contact_time(contact_type=contact_type)
        ts_counters = []
        alter_ids = self.get_alter_ids(contact_type=contact_type)
        if first_time is not None and last_time is not None:
            n_windows = math.floor((last_time - first_time - window_size) / step + 1)
            # initialize data structure and set static strength to n_circles + 1 for all alters - basic case is that
            # alters are outside the ego network
            alter_ts_stat_dyn = dict([(alter_id, [n_circles + 1, [n_circles + 1] * n_windows]) for alter_id in
                                      alter_ids])
            # static tie strength (ring number)
            static_properties = self.get_circles_properties(n_circles=n_circles, contact_type=contact_type, active=True)
            for alter_id, ring in static_properties.membership.items():
                alter_ts_stat_dyn[alter_id][0] = ring
            if n_windows > 0:
                for i in range(n_windows):
                    sub_egonet = self.get_subnetwork(int(first_time + i * step),
                                                     int(first_time + i * step + window_size), contact_type,
                                                     set_window_mode=True)
                    # consider all the relationships in the sub_network if the duration is <= ONE_YEAR, otherwise
                    # consider only the active ones
                    subnet_prop = sub_egonet.get_circles_properties(n_circles=n_circles, contact_type=contact_type,
                                                                    active=False)
                    for alter_id, ring in subnet_prop.membership.items():
                        alter_ts_stat_dyn[alter_id][1][i] = ring
                for ts_prop in alter_ts_stat_dyn.values():
                    ts_counters.append((ts_prop[0], Counter(ts_prop[1])))
        return ts_counters

    def get_total_activity_freq(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__",
                                count_no_contacts: bool = True) -> float:
        """compute the total activity frequency for the ego (messages/year), considering all possible types of
        communications, either direct or not

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :param count_no_contacts: count also non-direct contacts, whose type is __no_contact__
        :type count_no_contacts: bool
        :return: the total activity frequency for the ego in msg/year
        :rtype: float
        """
        contact_type = standardize_contact_type(contact_type=contact_type)
        freq = None
        first_time = self.get_first_contact_time()
        last_time = self.get_last_contact_time()
        n_tot = self.get_num_contacts(contact_type=contact_type)
        if count_no_contacts:
            n_tot += self.get_num_contacts(contact_type="__no_contact__")
        if first_time < last_time and n_tot > 0:
            freq = n_tot / ((last_time - first_time) / ONE_YEAR)
        return freq

    def get_alter_ids(self, contact_type: Union[Hashable, Tuple[Hashable]] = "__all__") -> List:
        """Get the list of alter ids for the given contact type.

        :param contact_type: a value or a list of values indicating the types of contacts to use. Used to support ego
            networks with multiple types of social relationships. The special value "__all__" can be used for the
            network containing all the types of contacts
        :type contact_type: Union[Hashable, Tuple[Hashable]]
        :return: the list of alter ids
        :rtype: List
        """
        return [alter.alter_id for alter in self.get_alters_with_types(contact_type=contact_type)]


class FrequencyEgoNetwork(EgoNetwork):
    """Class to represent an ego network by directly specifying contact frequencies with alters."""

    def __init__(self, ego_id=None):
        super().__init__(ego_id)

    def add_frequency(self, alter_id: Hashable, frequency: float, contact_type: Hashable = "__default__"):
        """Add a new social relationship to the ego network by specifying the id of the alter and the related contact
        frequency.

        :param alter_id: the unique id of the alter
        :param frequency: the contact frequency related to the specified alter
        :param contact_type: a value indicating the type of contacts to use. Used to support ego networks with multiple
            types of social relationships
        :type alter_id: Hashable
        :type frequency: float
        :type contact_type: Hashable
        """
        self._freq_vector[contact_type].append(AlterFreq(alter_id=alter_id, frequency=frequency))
        self.clear_cache()

    def add_frequencies(self, frequencies: List[Tuple[AlterFreq, Hashable]]):
        """Add a list of social relationships to the ego network.

        Each relationship must be a tuple (alter_id, frequency, contact_type).

        :param frequencies: the list of social relationships to add
        :type frequencies: List[Tuple[AlterFreq, Hashable]]
        """
        for alter_freq, contact_type in frequencies:
            self.add_frequency(alter_freq.alter_id, alter_freq.frequency, contact_type)
