import unittest

from evaluation.evaluate_retrieval import score_retrieval


class RetrievalScoringTests(unittest.TestCase):
    def test_alternatives_and_required_documents(self):
        self.assertEqual(score_retrieval(["a", "b"], ["b", "c"], "any"), (True, 1, 0.5))
        self.assertEqual(score_retrieval(["a", "b"], ["b", "c"], "all"), (False, 1, 0.5))
        self.assertEqual(score_retrieval(["a", "b"], ["b", "a"], "all"), (True, 1, 1.0))

    def test_unanswerable_and_invalid_reference(self):
        self.assertEqual(score_retrieval([], ["a"], "unanswerable"), (None, None, None))
        with self.assertRaises(ValueError):
            score_retrieval([], ["a"], "any")
        with self.assertRaises(ValueError):
            score_retrieval(["a"], ["a"], "unanswerable")


if __name__ == "__main__":
    unittest.main()
