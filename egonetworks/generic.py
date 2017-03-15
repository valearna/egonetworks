"""this module contains a collection of generic functions used by the other modules"""

from functools import partial
from abc import ABCMeta
import urllib.parse
import urllib.request
import json
from collections import defaultdict
from typing import List, Tuple, Union
import itertools
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
import numpy as np


__author__ = "Valerio Arnaboldi"
__license__ = "MIT"
__version__ = "1.0.1"


tagme_url_tag = 'https://tagme.d4science.org/tagme/tag'
tagme_url_rel = 'https://tagme.d4science.org/tagme/rel'


def get_topics_from_text(text: str, gcube_token: str, lang: str = "en", rho_th: float = 0.1, tweet: bool = False):
    """Get a set of topics from a given text string.

    Topics are extracted through TagMe API (http://tagme.di.unipi.it/tagme_help.html).

    :param text: the text from which to extract the topics
    :param gcube_token: the key to access the Tagme API
    :param lang: the language of the text
    :param rho_th: the minimum rho value of the topic (see https://tagme.d4science.org/tagme/tagme_help.html)
    :type text: str
    :type gcube_token: str
    :type lang: str
    :type rho_th: float
    :return: the dictionary of topics associated to text string provided. Keys are tagme ids and values
        are the number of times the topics appear in the text
    :rtype: Dict[str, int]
    """
    topics = defaultdict(int)
    values = {'text': text, 'gcube-token': gcube_token, 'tweet': 'true', 'lang': lang}
    data = urllib.parse.urlencode(values)
    data = data.encode('ascii')  # data should be bytes
    retries = 0
    while retries < 20:
        try:
            req = urllib.request.Request(tagme_url_tag, data)
            response = urllib.request.urlopen(req)
            tags_resp = response.read()
            tags_json = json.loads(tags_resp.decode())
            for tag in tags_json["annotations"]:
                if float(tag["rho"]) > rho_th:
                    topics[tag["id"]] += 1
            break
        except Exception as e:
            retries += 1
    else:
        logging.error("after 20 attempts to contact TagMe, I'm skipping the extraction of topics from text")
    return topics


def get_topics_rel_network(topics: List[int], gcube_token: str, lang: str = "en", num_threads: int = 500):
    """Get the relatedness network of topics from a list of topic ids.

    :param topics: a list of topic ids
    :param gcube_token: the key to access Tagme API
    :param lang: the language of the text to be analyzed
    :param num_threads: the number of parallel requests to be sent to TagMe API
    :type topics: List
    :type gcube_token: str
    :type lang: str
    :type num_threads: int
    :return: the network of topics extracted from the ego network
    :rtype: List[Tuple[str, str, float]]
    """
    def fetch_rel(gcube_token: str, pairs: List, lang: str, topics_net: List):
        data = 'gcube-token=' + gcube_token + '&id=' + '&id='.join(
            ['+'.join(pair) for pair in pairs_block]) + "&lang=" + lang
        data = data.encode('ascii')  # data should be bytes
        req = urllib.request.Request(tagme_url_rel, data)
        response = urllib.request.urlopen(req)
        rel_resp = response.read()
        rel_json = json.loads(rel_resp.decode())
        for couple_rel in rel_json["result"]:
            couple_arr = couple_rel["couple"].split(" ")
            rel = couple_rel["rel"]
            topics_net.append((couple_arr[0], couple_arr[1], rel))

    topics_net = []
    pairs_blocks = []
    for i, pair in enumerate(itertools.combinations([str(id) for id in topics], 2)):
        if i % 100 == 0:
            pairs_blocks.append([])
        pairs_blocks[-1].append(pair)
    logging.debug("sending " + str(len(pairs_blocks)) +
                  " relatedness requests to TagMe, for a total of " + str(len(topics)) + " topics")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for pairs_block in pairs_blocks:
            retries = 0
            while retries < 20:
                try:
                    executor.submit(fetch_rel, gcube_token, pairs_block, lang, topics_net)
                    break
                except Exception as e:
                    retries += 1
            else:
                logging.error("after 20 attempts to contact TagMe, I'm skipping a block of topic relatedness links")

    logging.debug("finished retrieving relatedness network")
    return topics_net


def calculate_dispersion(l: List[Union[int, float]]):
    """calculate the index of dispersion of a list of numeric elements

    The index of dispersion is a measure of burstiness of the distribution of the elements and it is calculated as the
    variance of the distribution divided by its mean:

    .. math::
       D_X = \\frac{{\\sigma^2}_X}{\\mu_X}.

    :param l: the list of elements
    :type l: List[Union[int,float]]
    :return: the index of dispersion of the distibution of the elements
    :rtype: float
    """
    return np.var(l) / np.mean(l)


def calculate_burstiness(l: List[Union[int, float]]):
    """Calculate the burstiness parameter for a distribution of inter contact times

    The burstiness parameter is calculated as follows:

    .. math::
       B_X = \\frac{{\\sigma^2}_X - \\mu_X}{{\\sigma^2}_X + \\mu_X}.

    The index ranges between -1 (regular distribution, no burstiness) and 1 (maximum burstiness)
    """
    return (np.var(l) - np.mean(l)) / (np.var(l) + np.mean(l))


def count_bursts(l: list, max_val: int, normalize: bool = False, count_decrease: bool = False):
    n_bursts = 0
    if not count_decrease and l[0] < max_val:
        n_bursts += 1
    for i, elem in enumerate(l[1:], 1):
        if (count_decrease and l[i - 1] < max_val <= elem) or \
                (not count_decrease and elem < max_val <= l[i - 1]):
            n_bursts += 1
    if normalize:
        n_bursts /= len(l)
    return n_bursts


def get_mean_persistence(l: list, val: int):
    bursts_pers = []
    if l[0] == val:
        bursts_pers.append(1)
    for i, elem in enumerate(l[1:], 1):
        if elem == val and l[i - 1] == val:
            bursts_pers[-1] += 1
        elif elem == val and l[i - 1] != val:
            bursts_pers.append(1)
    if len(bursts_pers) > 0:
        return np.nanmean(bursts_pers)
    else:
        return 0


def get_exit_jumps(l: list, val: int, normalize: bool = True):
    exit_jumps = defaultdict(int)
    for i, elem in enumerate(l[1:], 1):
        if elem != val and l[i - 1] == val:
            exit_jumps[elem - val] += 1
    if normalize:
        for k, v in exit_jumps.items():
            exit_jumps[k] = v / sum(exit_jumps.values())
    return exit_jumps


class memoize(object):
    """cache the return value of a method.

    This class is meant to be used as a decorator of methods. The return value
    from a given method invocation will be cached on the instance whose method
    was invoked. All arguments passed to a method decorated with memoize must
    be hashable.

    If a memoized method is invoked directly on its class the result will not
    be cached. Instead the method will be invoked like a static method.
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return partial(self, obj)

    def __call__(self, *args, **kw):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res


class Memoizable(metaclass=ABCMeta):
    """This is a metaclass that defines a memoizable object."""

    def __init__(self):
        """Initialize new memoizable object."""
        self._memoize__cache = {}

    def clear_cache(self):
        """Reset the memoize cache."""
        self._memoize__cache = {}


class DictReaderWithHeader:
    """Read files with header and map each line to a dictionary where keys are tokens extracted from the header line."""
    def __init__(self, reader, separator: str=" "):
        """Create a new reader from an existing stream reader (a file object).

        :param reader: a file object
        :param separator: the field separator
        :type separator: str
        """
        self.reader = reader
        self.separator = separator.encode("utf-8").decode("unicode_escape")
        self.header_arr = self.reader.__next__().strip().split(separator)

    def __iter__(self):
        return self

    def __next__(self):
        line = self.reader.__next__()
        line_arr = line.strip().split(self.separator)
        return dict(zip(self.header_arr, line_arr))


def synchronized_method(method):
    outer_lock = threading.Lock()
    lock_name = "__" + method.__name__ + "_lock" + "__"

    def sync_method(self, *args, **kws):
        with outer_lock:
            if not hasattr(self, lock_name): setattr(self, lock_name, threading.Lock())
            lock = getattr(self, lock_name)
            with lock:
                return method(self, *args, **kws)
    return sync_method


def merge_and_sort_sorted_lists(list_a: list, list_b: list, key = None) -> list:
    """Merge two sorted lists and obtain a single sorted list

    :param list_a: the first list
    :type list_a: list
    :param list_b: the second list
    :type list_b: list

    """