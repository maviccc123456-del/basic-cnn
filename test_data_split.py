"""Regression checks for content leakage and checkpoint/split provenance."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from data_split import (CLASS_TO_IDX, SPLIT_NAMES, digest_json, image_digest,
                        load_split_manifest, scan_images, validate_manifest)
from experiments import validate_checkpoint


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="split_test_", dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name) / "images"
        self.data.mkdir()
        self.manifest_path = Path(self.temp.name) / "split.json"
        for class_name, label in CLASS_TO_IDX.items():
            for index in range(20):
                Image.new("L", (8, 8), label * 20 + index).save(self.data / f"{class_name}_{index:02d}.bmp")
        # Same model-visible pixels, different filename AND file encoding.
        Image.new("L", (8, 8), 0).save(self.data / "Cr_copy.bmp", format="PNG")

    def load(self):
        return load_split_manifest(self.data, self.manifest_path)[1]

    def test_deduplicates_pixels_and_preserves_original_files(self):
        manifest = self.load()
        assigned = [r for rows in manifest["splits"].values() for r in rows]
        self.assertEqual(len(assigned), 120)
        self.assertEqual(len(list(self.data.glob("*.bmp"))), 121)
        self.assertEqual(len({image_digest(self.data / r["path"]) for r in assigned}), 120)
        self.assertEqual(manifest["excluded_duplicates"][0]["path"], "Cr_copy.bmp")
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            self.assertFalse({r["pixel_sha256"] for r in manifest["splits"][a]} &
                             {r["pixel_sha256"] for r in manifest["splits"][b]})

    def test_saved_split_survives_directory_enumeration_order(self):
        expected = self.load()
        paths = list(reversed(list(self.data.glob("*.bmp"))))
        before = self.manifest_path.read_bytes()
        with patch("data_split.Path.glob", return_value=iter(paths)):
            self.assertEqual(expected, self.load())
        self.assertEqual(before, self.manifest_path.read_bytes())

    def test_changed_image_is_rejected_without_overwriting_manifest(self):
        self.load()
        before = self.manifest_path.read_bytes()
        Image.new("L", (8, 8), 250).save(self.data / "In_00.bmp")
        with self.assertRaisesRegex(ValueError, "Dataset changed"):
            self.load()
        self.assertEqual(before, self.manifest_path.read_bytes())

    def test_added_image_is_rejected(self):
        self.load()
        Image.new("L", (8, 8), 250).save(self.data / "In_new.bmp")
        with self.assertRaisesRegex(ValueError, "Dataset changed"):
            self.load()

    def test_identical_images_with_conflicting_labels_are_rejected(self):
        Image.new("L", (8, 8), 0).save(self.data / "In_conflict.bmp")
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            self.load()

    def test_corrupted_manifest_is_rejected(self):
        manifest = self.load()
        manifest["seed"] = 123
        with self.assertRaisesRegex(ValueError, "edited or corrupted"):
            validate_manifest(manifest, scan_images(self.data))

    def test_repeated_image_assignment_is_rejected_even_with_valid_digest(self):
        manifest = copy.deepcopy(self.load())
        manifest["splits"]["validation"][0] = manifest["splits"]["train"][0]
        manifest["split_id"] = digest_json({k: v for k, v in manifest.items() if k != "split_id"})
        with self.assertRaisesRegex(ValueError, "exactly once"):
            validate_manifest(manifest, scan_images(self.data))

    def test_baseline_uses_saved_split_and_eval_is_deterministic(self):
        import dataset
        expected = self.load()
        for module in (dataset,):
            datasets = module.create_datasets(self.data, self.manifest_path)
            for name, ds in zip(SPLIT_NAMES, datasets):
                self.assertEqual(ds.split_id, expected["split_id"])
                self.assertEqual([p.name for p in ds.image_paths],
                                 [r["path"] for r in expected["splits"][name]])
            for ds in datasets[1:]:
                self.assertTrue(torch.equal(ds[0][0], ds[0][0]))


class CheckpointTests(unittest.TestCase):
    def test_legacy_or_mismatched_weights_cannot_be_evaluated(self):
        with self.assertRaisesRegex(ValueError, "Legacy"):
            validate_checkpoint({"fc.weight": torch.zeros(6, 512)}, "simple_cnn", "new-split")
        checkpoint = {"format_version": 1, "model_state_dict": {},
                      "model_name": "simple_cnn", "split_id": "old-split"}
        with self.assertRaisesRegex(ValueError, "another split"):
            validate_checkpoint(checkpoint, "simple_cnn", "new-split")
        with self.assertRaisesRegex(ValueError, "model does not match"):
            validate_checkpoint(checkpoint, "different_model", "old-split")
        validate_checkpoint(checkpoint, "simple_cnn", "old-split")


if __name__ == "__main__":
    unittest.main()
