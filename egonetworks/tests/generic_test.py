import unittest
import egonetworks.generic
import random


class TestGeneric(unittest.TestCase):

    def setUp(self):
        self.topic_list = [random.randint(1, 50000) for _ in range(30)]
        self.text = "This is a test. Text must contain a topic, I know, something like Madonna, the singer and actress"


if __name__ == "__main__":
    unittest.main()
