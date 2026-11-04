import unittest

from rag_chroma import rerank_faq


class FakeReranker:
    def predict(self, pairs, show_progress_bar=False):
        assert pairs == [("پرسش", "متن الف"), ("پرسش", "متن ب"), ("پرسش", "متن پ")]
        return [0.2, 0.9, 0.5]


class RerankTests(unittest.TestCase):
    def test_reorders_all_fields_and_leaves_input_unchanged(self):
        original = {
            "ids": [["a", "b", "c"]],
            "documents": [["متن الف", "متن ب", "متن پ"]],
            "metadatas": [[{"n": 1}, {"n": 2}, {"n": 3}]],
            "distances": [[0.1, 0.2, 0.3]],
        }
        result = rerank_faq("پرسش", original, FakeReranker(), top_k=2)
        self.assertEqual(result["ids"], [["b", "c"]])
        self.assertEqual(result["documents"], [["متن ب", "متن پ"]])
        self.assertEqual(result["metadatas"], [[{"n": 2}, {"n": 3}]])
        self.assertEqual(result["distances"], [[0.2, 0.3]])
        self.assertEqual(result["retrieval_ranks"], [[2, 3]])
        self.assertEqual(result["rerank_scores"], [[0.9, 0.5]])
        self.assertEqual(original["ids"], [["a", "b", "c"]])

    def test_empty_results_and_invalid_top_k(self):
        empty = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        self.assertEqual(rerank_faq("پرسش", empty, FakeReranker())["rerank_scores"], [[]])
        with self.assertRaises(ValueError):
            rerank_faq("پرسش", empty, FakeReranker(), top_k=0)


if __name__ == "__main__":
    unittest.main()
