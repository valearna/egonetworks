import json
import unittest
from egonetworks.core import ContactEgoNetwork
from egonetworks.core import ONE_MONTH, ONE_YEAR
import numpy as np


class TestContactEgoNetwork(unittest.TestCase):

    def setUp(self):
        self.first_time = 1420070400
        self.last_time = 1454284800
        self.n_direct_contacts = 7
        self.n_no_contacts = 1
        self.egonet = ContactEgoNetwork(ego_id=12, last_time=1454284800, first_time=1420000000)
        self.egonet.add_contact(timestamp=1420070400, alter_id=1,
                                text="This is a test message, but should contain some topics")
        self.egonet.add_contact(timestamp=1423094400, alter_id=1)
        self.egonet.add_contact(timestamp=1451692800, alter_id=1,
                                text="This is another test, with other topics, e.g., sport")
        self.egonet.add_contact(timestamp=1439337600, alter_id=3,
                                text="This is another test, with other topics, e.g., ski")
        self.egonet.add_contact(timestamp=1454284800, alter_id=1)
        self.egonet.add_contact(timestamp=1422403200, alter_id=1)
        self.egonet.add_contact(timestamp=1454284799, alter_id=4)
        self.egonet.add_contact(timestamp=1439337600, alter_id=0,
                                text="This is a message to nobody", contact_type="__no_contact__")

    def test_update_contacts(self):
        ego2 = ContactEgoNetwork()
        ego2.add_contact(timestamp=3, alter_id=1)
        self.egonet.update_contacts(ego2)
        self.assertEqual(len(self.egonet.alters[1].contacts["__default__"]), 6, "wrong number of alters after update")

    def test_copy_egonet(self):
        copy_egonet = self.egonet.copy_egonet()
        self.assertTrue(self.egonet.ego_id, copy_egonet.ego_id)
        self.assertCountEqual(self.egonet.get_all_contacts(), copy_egonet.get_all_contacts())

    def test_get_all_contacts(self):
        contacts = self.egonet.get_all_contacts()
        self.assertTrue(len(contacts) == 7)

    def test_to_json(self):
        json_egonet = self.egonet.to_json(compact=False)
        self.assertTrue(json.loads(json_egonet)["ego_id"] == self.egonet.ego_id)
        self.assertTrue("contacts" in json.loads(json_egonet))

    def test_create_from_json(self):
        json_egonet = self.egonet.to_json(compact=False)
        egonet_from_json = self.egonet.create_from_json(json_egonet)
        self.assertTrue(egonet_from_json.ego_id == self.egonet.ego_id)

    def test_get_egonet_size(self):
        # complete net size
        self.assertTrue(self.egonet.get_egonet_size(active=False, stable=False), 3)
        # stable net size
        self.assertTrue(self.egonet.get_egonet_size(active=False, stable=True), 2)
        # active (and stable) net size
        self.assertTrue(self.egonet.get_egonet_size(active=True, stable=True), 2)
        self.assertTrue(self.egonet.get_egonet_size(active=False, stable=False,
                                                    exclude_one_contact_relationships=True), 1)

    def test_get_avg_relationships_duration(self):
        self.assertTrue(self.egonet.get_avg_relationships_duration() == np.mean(
            [(1454284800 - 1420070400) / (86400 * 365.25), (1454284800 - 1439337600) / (86400 * 365.25),
             (1454284800 - 1454284799) / (86400 * 365.25)]))

        egonet = self.egonet.copy_egonet()
        egonet.add_contact(timestamp=1420000000, alter_id=3, contact_type="reply", text="test")
        self.assertTrue(egonet.get_avg_relationships_duration(contact_type="__all__",
                                                              exclude_one_contact_relationships=False),
                        np.mean([(1454284800 - 1420070400) / (86400 * 365.25),
                                 (1454284800 - 1420000000) / (86400 * 365.25),
                                 (1454284800 - 1454284799) / (86400 * 365.25)]))
        self.assertTrue(egonet.get_avg_relationships_duration(contact_type="__default__",
                                                              exclude_one_contact_relationships=False),
                        np.mean([(1454284800 - 1420070400) / (86400 * 365.25),
                                 (1454284800 - 1439337600) / (86400 * 365.25),
                                 (1454284800 - 1454284799) / (86400 * 365.25)]))
        self.assertTrue(egonet.get_avg_relationships_duration(contact_type="__all__",
                                                              exclude_one_contact_relationships=True),
                        np.mean([(1454284800 - 1420070400) / (86400 * 365.25),
                                 (1454284800 - 1439337600) / (86400 * 365.25)]))

    def test_get_duration(self):
        # considering the first contact time of the ego network
        self.assertTrue(self.egonet.get_duration(use_first_time=False), (1454284800 - 1420070400) / ONE_YEAR)
        # considering the first_time parameter of the ego network
        self.assertTrue(self.egonet.get_duration(use_first_time=True), (1454284800 - 1420000000) / ONE_YEAR)

    def test_is_stable(self):
        egonet = self.egonet.copy_egonet()
        egonet.add_contact(timestamp=1454284800, alter_id=4)
        egonet.add_contact(timestamp=1454284801, alter_id=4)
        egonet.add_contact(timestamp=1454284802, alter_id=4)
        self.assertFalse(egonet.is_stable(check_dyn_stability=True))
        self.assertTrue(egonet.is_stable())

    def test_get_entropy(self):
        self.assertGreaterEqual(self.egonet.get_entropy(), 0, "wrong entropy value")

    def test_get_size_time_series(self):
        self.assertEqual(np.var(self.egonet.get_size_time_series(
            window_size=ONE_MONTH, step=ONE_MONTH)), 0.1875)

    def test_get_stability_rings(self):
        stab = self.egonet.get_stability(window_size=ONE_MONTH, step=ONE_MONTH, n_circles=2)
        self.assertTrue("jaccard" in stab)
        self.assertTrue(len(stab["jaccard"][0]) == 2)

    def test_get_relationships_burstiness(self):
        self.assertTrue(len(self.egonet.get_relationships_burstiness()), self.egonet.get_egonet_size(stable=False))

    def test_get_growth_rate(self):
        self.assertListEqual(self.egonet.get_growth_rate(), [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                             "wrong growth rate")

    def test_get_total_activity_freq(self):
        self.assertTrue(self.egonet.get_total_activity_freq() == (self.n_direct_contacts + self.n_no_contacts) /
                        ((self.last_time - self.first_time) / ONE_YEAR))


if __name__ == "__main__":
    unittest.main()
