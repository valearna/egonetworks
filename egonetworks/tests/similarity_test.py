import unittest
import egonetworks.similarity


class TestSimilarity(unittest.TestCase):

    def setUp(self):
        self.list1 = [1, 3, 7, 8, 10]
        self.list2 = [1, 10, 12, 3]
        self.level_list1 = [("obj1", 1), ("obj2", 1), ("obj3", 3)]
        self.level_list2 = [("obj3", 1), ("obj1", 1), ("obj2", 2)]

    def test_jaccard(self):
        ji = egonetworks.similarity.jaccard(self.list1, self.list2)
        self.assertEqual(egonetworks.similarity.jaccard([], []), 1, "wrong jaccard index with empty lists")
        self.assertEqual(ji, 0.5, "wrong Jaccard index")

    def test_spearman_footrule(self):
        sp = egonetworks.similarity.spearman_footrule(self.list1, self.list2, normalized=False)
        sp_n = egonetworks.similarity.spearman_footrule(self.list1, self.list2, normalized=True)
        self.assertEqual(egonetworks.similarity.spearman_footrule([], []), 0,
                         "wrong spearman footrule index with empty lists")
        self.assertEqual(sp, 20/6, "wrong spearman footrule index")
        self.assertEqual(sp_n, 2/3, "wrong normalized spearman footrule index")

    def test_level_index(self):
        li = egonetworks.similarity.level_changes(self.level_list1, self.level_list2, normalized=False,
                                                  remove_stable=False)
        li_n = egonetworks.similarity.level_changes(self.level_list1, self.level_list2, 3, normalized=True,
                                                    remove_stable=False)
        li_n_r = egonetworks.similarity.level_changes(self.level_list1, self.level_list2, 3, normalized=True,
                                                      remove_stable=True)
        li_rn = egonetworks.similarity.level_changes(self.level_list1, self.level_list2, 3, normalized=2)
        li_rn_r = egonetworks.similarity.level_changes(self.level_list1, self.level_list2, 3, normalized=2,
                                                       remove_stable=True)
        self.assertEqual(egonetworks.similarity.level_changes([], []), 0, "wrong level changes index with empty lists")
        self.assertEqual(li, 1, "wrong level changes index")
        self.assertEqual(li_n, 1/3, "wrong normalized level changes index")
        self.assertEqual(li_n_r, 1/2, "wrong normalized level changes index with stable objects removed")
        self.assertEqual(li_rn, 4/9, "wrong relative normalization")
        self.assertEqual(li_rn_r, 4/6, "wrong relative normalization with stable objects removed")

if __name__ == "__main__":
    unittest.main()
