"""Execution must respect the caller's GPU allocation."""
import os
import unittest
from unittest.mock import patch

from rsisys.labs.weight_engine import WeightEngine


class WeightEngineTests(unittest.TestCase):
    def test_gpu_selection_cannot_escape_inherited_allocation(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2,3'}):
            self.assertEqual(WeightEngine({'gpu_ids': [2, 3]})._gpu_ids(), [2, 3])
            with self.assertRaisesRegex(ValueError, 'inherited CUDA_VISIBLE_DEVICES'):
                WeightEngine({'gpu_ids': [0]})._gpu_ids()
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            with self.assertRaises(ValueError):
                WeightEngine({})._gpu_ids()
