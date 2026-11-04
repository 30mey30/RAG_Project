import unittest
from unittest.mock import patch

from rag_chroma import hybrid_search, rerank_faq


class FakeBM25:
    def get_scores(self, tokens):
        return [0.0, 4.0, 0.0]


class FakeReranker:
    def predict(self, pairs, show_progress_bar=False):
        return list(range(len(pairs)))


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            {"id": "a", "text": "متن دیگر", "metadata": {}},
            {"id": "b", "text": "ودیعه رهن", "metadata": {}},
            {"id": "c", "text": "سند دیگر", "metadata": {}},
        ]
        self.semantic = {"ids": [["a", "c"]], "distances": [[0.1, 0.2]]}

    def test_lexical_only_candidate_is_preserved_through_reranking(self):
        with patch('rag_chroma.search_faq', return_value=self.semantic):
            result = hybrid_search('ودیعه رهن', None, None, self.docs, FakeBM25(), top_k=3)
        self.assertIn('b', result['ids'][0])
        self.assertIsNone(result['distances'][0][result['ids'][0].index('b')])
        reranked = rerank_faq('ودیعه رهن', result, FakeReranker(), top_k=3)
        self.assertEqual(len(reranked['rrf_scores'][0]), 3)
        self.assertEqual(set(reranked['ids'][0]), {'a', 'b', 'c'})

    def test_no_keyword_match_uses_semantic_results(self):
        with patch('rag_chroma.search_faq', return_value=self.semantic):
            result = hybrid_search('واژه ناشناخته', None, None, self.docs, FakeBM25(), top_k=3)
        self.assertEqual(result['ids'][0], ['a', 'c'])


if __name__ == '__main__':
    unittest.main()
