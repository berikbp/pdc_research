import unittest
from pdc_research import analysis

class AnalysisTests(unittest.TestCase):
    def test_group_pairing_and_degenerate_intervals(self):
        result = analysis.intervals({'no_interaction': [0.0] * 19,
                                     'constant_effect': [3.0] * 19}, samples=100)
        self.assertEqual(result['no_interaction']['simultaneous95_low'], 0)
        self.assertEqual(result['constant_effect']['simultaneous95_high'], 3)

    def test_reproducibility_and_simultaneous_family(self):
        vectors = {'a': list(range(19)), 'b': list(range(0, 38, 2))}
        a = analysis.intervals(vectors, samples=500)
        b = analysis.intervals(vectors, samples=500)
        self.assertEqual(a, b)
        self.assertEqual(a['a']['estimate'], 9)
        self.assertAlmostEqual(a['b']['simultaneous95_low'], 2*a['a']['simultaneous95_low'])
        self.assertLess(a['a']['simultaneous95_low'], 9)
        self.assertGreater(a['a']['simultaneous95_high'], 9)
