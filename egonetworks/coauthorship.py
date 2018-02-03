from egonetworks.core import ContactEgoNetwork
from typing import List, Tuple, Any


class CoauthorshipNamedEgoNetwork(ContactEgoNetwork):
    """This class extends :class:`egonetworks.core.ContactEgoNetwork` and represents a coauthorship ego network where
    each alter is specified by name and not by id.

    A coauthorship named ego network represents the network of coauthors of an author, where coauthors are not
    identified by unique uids, but by their names. Each link indicates that the author (ego) and the coauthor (alter)
    have coauthored at least one publication. The weight of the links is calculated as the number of publications
    coauthored together divided by the duration of their collaboration, normalized by the number of coauthors in each
    publication."""

    @property
    def add_contact(self, timestamp: int, alter_id, contact_type="__all__", text: str = None,
                    num_contacted_alters: int=1, additional_info=None):
        raise AttributeError("this method is not accessible for this class")

    @property
    def add_contacts(self, contacts: List[Tuple[int, int, Any, str, int]]):
        raise AttributeError("this method is not accessible for this class")

    def __init__(self, ego_name: str, ego_id: int = None, last_time: int = 0,
                 min_pub_date: int = None):
        """Create a new CoauthorshipEgoNetwork by specifying the id and the name of the ego."""
        self.ego_name = ego_name
        self.publications = {}
        self.min_pub_date = min_pub_date
        super(CoauthorshipNamedEgoNetwork, self).__init__(ego_id, last_time)

    @staticmethod
    def __are_names_of_same_author(name1: str, name2: str):
        name1_arr = [part.strip() for part in name1.split()]
        name2_arr = [part.strip() for part in name2.split()]
        if len(name1_arr) == 1 or len(name2_arr) == 1:
            if name1_arr[-1].lower() == name2_arr[-1].lower():
                return True
        elif len(name1_arr) > 1 and len(name2_arr) > 1:
            if (name1_arr[0].lower() == name1_arr[0].lower() or name1_arr[0][0].lower() == name2_arr[0][0].lower()) \
                    and name1_arr[-1].lower() == name2_arr[-1].lower():
                return True
        return False

    @staticmethod
    def get_std_author_name(name: str):
        name_arr = [part.strip() for part in name.split()]
        if len(name_arr) == 1:
            return name_arr[0].lower()
        else:
            return name_arr[0][0].lower() + " " + name_arr[-1].lower()

    def add_publication(self, pub_id: int, timestamp: int, title: str, coauthors: List[str], contact_type="__all__"):
        """Add a contact representing a publication coauthored by the ego to the ego network.

        :param pub_id: the id of the paper
        :param timestamp: the publication date in unix timestamp format
        :param title: the title of the publication
        :param coauthors: the list of coauthors' names
        :param contact_type: a value indicating the type of contact. Used to support ego networks with multiple types
            of social relationships
        :type pub_id: int
        :type timestamp: int
        :type title: str
        :type coauthors: List[str]
        """
        if self.min_pub_date is None or timestamp >= self.min_pub_date:
            # standardize names, remove possible duplicates and wrong names, and remove ego if present
            std_coauth_names = set()
            for coauthor_name in coauthors:
                if len(coauthor_name) > 1:
                    std_coauth_names.add(self.get_std_author_name(coauthor_name))
            std_coauth_names.discard(self.get_std_author_name(self.ego_name))

            for coauthor_name in std_coauth_names:
                if self.last_time is None or timestamp > self.last_time:
                    self.last_time = timestamp
                super(CoauthorshipNamedEgoNetwork, self).add_contact(timestamp=timestamp, alter_id=coauthor_name,
                                                                     contact_type=contact_type, text=title,
                                                                     num_contacted_alters=len(std_coauth_names))
            self.publications[pub_id] = (title, std_coauth_names)
