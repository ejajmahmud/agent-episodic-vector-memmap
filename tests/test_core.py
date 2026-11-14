"""Unit tests for agent-episodic-vector-memmap."""
import unittest
from agent_episodic_vector_memmap.core import VectorMemMapIndex, CosineRetriever

class TestVectorMemMap(unittest.TestCase):
    def setUp(self):
        self.idx = VectorMemMapIndex(dimension=3)

    def test_exact_match(self):
        self.idx.add("v1", [1.0, 0.0, 0.0])
        self.idx.add("v2", [0.0, 1.0, 0.0])
        res = self.idx.search([1.0, 0.0, 0.0], top_k=1)
        self.assertEqual(res[0][0], "v1")
        self.assertAlmostEqual(res[0][1], 1.0, places=4)

    def test_orthogonal_zero_similarity(self):
        sim = CosineRetriever.cosine_similarity([1.0, 0.0], [0.0, 1.0])
        self.assertAlmostEqual(sim, 0.0, places=4)

    def test_dimension_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.idx.add("bad", [1.0, 2.0])
