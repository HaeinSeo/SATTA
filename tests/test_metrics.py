"""Regression cases for matching, scale assignment and reported FP denominators."""
import importlib.util
import unittest

import numpy as np

from satta.metrics import (EMPTY, coco_ap, coco_evaluate, evaluate_scale_resolved,
                           match_image, taxonomy, transitions)


def dets(rows):
    return np.asarray(rows, dtype=np.float32).reshape(-1, 6)


def ground_truth(rows):
    return {"width": 200, "height": 200, "boxes": rows}


class MatchingTests(unittest.TestCase):
    def test_match_best_unmatched_gt_not_only_best_gt(self):
        gt = np.asarray([[0, 0, 10, 10, 0], [2, 0, 12, 10, 0]], np.float32)
        pred = dets([[0, 0, 10, 10, .8, 0], [0, 0, 10, 10, .9, 0],
                     [0, 0, 10, 10, .7, 0], [0, 0, 10, 10, .99, 1]])
        matches, _ = match_image(pred, gt)
        np.testing.assert_array_equal(matches, [1, 0, -1, -1])

    def test_confidence_ties_are_stable(self):
        gt = np.asarray([[0, 0, 10, 10, 0]], np.float32)
        matches, _ = match_image(dets([[0, 0, 10, 10, .9, 0]] * 2), gt)
        np.testing.assert_array_equal(matches, [0, -1])

    def test_invalid_iou_threshold_rejected_even_if_empty(self):
        for threshold in [0, -.1, 1.1, float("nan")]:
            with self.assertRaises(ValueError):
                match_image(EMPTY, np.empty((0, 5)), threshold)


class ScaleAndErrorTests(unittest.TestCase):
    def test_false_discovery_rate_is_separate_from_fp_per_image(self):
        gts = {"a": ground_truth([[0, 0, 10, 10, 0], [30, 0, 50, 20, 0]]),
               "b": ground_truth([])}
        pred = {"a": dets([[0, 0, 10, 10, .9, 0], [30, 0, 50, 20, .999, 0],
                           [60, 0, 70, 10, .99, 0], [90, 0, 110, 20, .98, 0]])}
        result = evaluate_scale_resolved(pred, gts)
        self.assertEqual((result["TP"], result["FP"], result["FN"]), (2, 2, 0))
        self.assertEqual(result["FP_per_img"], 1)
        self.assertEqual(result["false_discovery_rate"], .5)
        self.assertEqual(result["XS_precision"], .5)
        self.assertAlmostEqual(result["XS_precision_allFP"], 1 / 3)
        self.assertAlmostEqual(result["XS_AP50"], .5)
        self.assertAlmostEqual(result["XS_AP50_allFP"], 1 / 3)
        self.assertEqual(result["n_images"], 2)
        self.assertEqual(result["_per_image"]["a"]["tp_xs"], 1)

    def test_scale_boundaries_and_matched_detections_inherit_gt_scale(self):
        sizes = [8, 12, 16, 20, 24, 32, 48, 64, 96, 100]
        gt = [[0, 0, side, side, i] for i, side in enumerate(sizes)]
        # A 20px predicted square can match the 16px GT at IoU=.64 and is an XS TP.
        pred = [[0, 0, 20 if i == 2 else side, 20 if i == 2 else side, .9, i]
                for i, side in enumerate(sizes)]
        result = evaluate_scale_resolved({"a": dets(pred)}, {"a": ground_truth(gt)})
        self.assertEqual(list(result["n_by_band"].values()), [1] * 10)
        self.assertEqual(result["TP_XS"], 3)
        self.assertEqual(result["XS_recall"], 1)
        self.assertEqual(result["XS_precision"], 1)

    def test_missing_prediction_image_counts_false_negatives(self):
        result = evaluate_scale_resolved({}, {"missing": ground_truth([[0, 0, 10, 10, 0]])})
        self.assertEqual((result["TP"], result["FP"], result["FN"]), (0, 0, 1))
        self.assertEqual(result["XS_recall"], 0)

    def test_empty_images_with_false_positives_are_kept(self):
        result = evaluate_scale_resolved({"empty": dets([[0, 0, 10, 10, .9, 0]])},
                                         {"empty": ground_truth([])})
        self.assertEqual(result["FP"], 1)
        self.assertEqual(result["FP_per_img"], 1)

    def test_empty_or_unknown_image_universe_is_an_error(self):
        for fn in [evaluate_scale_resolved, coco_ap]:
            with self.assertRaises(ValueError):
                fn({}, {})
            with self.assertRaises(ValueError):
                fn({"typo": EMPTY}, {"real": ground_truth([])})

    def test_invalid_predictions_fail_instead_of_entering_metrics(self):
        for row in [[1, 0, 0, 10, .9, 0], [0, 0, 10, 10, float("nan"), 0],
                    [0, 0, 10, 10, .9, .5], [0, 0, 10, 10, -.1, 0]]:
            with self.assertRaises(ValueError):
                evaluate_scale_resolved({"a": dets([row])}, {"a": ground_truth([])})

    def test_zero_area_predictions_remain_false_positives(self):
        pred = {"a": dets([[0, 0, 0, 10, .99, 0], [0, 0, 10, 0, .98, 0],
                           [0, 0, 10, 10, .9, 0]])}
        result = evaluate_scale_resolved(pred, {"a": ground_truth([[0, 0, 10, 10, 0]])})
        self.assertEqual((result["TP"], result["FP"], result["FN"]), (1, 2, 0))
        self.assertEqual(result["FP_XS"], 2)
        self.assertAlmostEqual(result["XS_precision"], 1 / 3)

    def test_taxonomy_totals_and_transition_identity(self):
        gts = {"a": ground_truth([[0, 0, 10, 10, 0], [30, 0, 40, 10, 0]])}
        a = {"a": dets([[0, 0, 10, 10, .9, 0], [0, 0, 10, 10, .8, 0],
                        [30, 0, 40, 10, .9, 1], [100, 100, 110, 110, .9, 0]])}
        b = {"a": dets([[0, 0, 10, 10, .9, 0], [30, 0, 40, 10, .9, 0]])}
        result = evaluate_scale_resolved(a, gts)
        fp, fn = taxonomy(a, gts)
        self.assertEqual(fp["total"], result["FP"])
        self.assertEqual(fn["total"], result["FN"])
        self.assertEqual((fp["duplicate"], fp["class_confusion"], fp["background"]), (1, 1, 1))
        move = transitions(a, b, gts)
        self.assertEqual(move["recovered_TP"] - move["lost_TP"],
                         evaluate_scale_resolved(b, gts)["TP"] - result["TP"])

    def test_all_taxonomy_categories_have_explicit_precedence(self):
        gts = {"a": ground_truth([[0, 0, 10, 10, 0], [40, 0, 50, 10, 0],
                                   [80, 0, 90, 10, 0], [120, 0, 130, 10, 0],
                                   [160, 0, 170, 10, 0]])}
        pred = {"a": dets([[0, 0, 10, 10, .9, 0], [0, 0, 10, 10, .8, 0],
                           [40, 0, 50, 10, .9, 1], [86, 0, 96, 10, .9, 0],
                           [126, 0, 136, 10, .9, 1], [190, 80, 200, 90, .9, 0]])}
        fp, fn = taxonomy(pred, gts)
        self.assertEqual(fp, {"background": 1, "localization": 1, "class_confusion": 1,
                              "near_object": 1, "duplicate": 1, "total": 5})
        self.assertEqual(fn, {"missed": 1, "localization": 1, "class_confusion": 2, "total": 4})

    def test_legacy_ap_preserves_its_documented_trapezoidal_convention(self):
        ap50, ap5095 = coco_ap({"a": dets([[0, 0, 10, 10, .9, 0]])},
                               {"a": ground_truth([[0, 0, 10, 10, 0]])})
        self.assertAlmostEqual(ap50, .995)
        self.assertAlmostEqual(ap5095, .995)


@unittest.skipUnless(importlib.util.find_spec("pycocotools"), "optional pycocotools unavailable")
class CocoEvalTests(unittest.TestCase):
    def test_actual_coco_ap_is_one_for_perfect_detection(self):
        result = coco_evaluate({"a": dets([[0, 0, 10, 10, .9, 0]])},
                               {"a": ground_truth([[0, 0, 10, 10, 0]])})
        self.assertAlmostEqual(result["AP50"], 1)
        self.assertAlmostEqual(result["AP50_95"], 1)
        self.assertIsNone(result["AP_large"])
        self.assertEqual(result["protocol"]["max_dets"], [1, 10, 100])

    def test_actual_coco_accepts_no_detections(self):
        result = coco_evaluate({}, {"a": ground_truth([[0, 0, 10, 10, 0]])})
        self.assertEqual(result["AP50"], 0)

    def test_nondefault_maxdet_is_used_for_overall_ap(self):
        result = coco_evaluate({"a": dets([[0, 0, 10, 10, .9, 0]])},
                               {"a": ground_truth([[0, 0, 10, 10, 0]])}, max_dets=500)
        self.assertAlmostEqual(result["AP50_95"], 1)


if __name__ == "__main__":
    unittest.main()
