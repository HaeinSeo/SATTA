"""Public-interface regressions: split completeness, true labels, and safe JSON."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import evaluate
from satta.io import load_candidates, load_gt, load_predictions
from satta.satta import SattaConfig, bucket

ROOT = Path(__file__).resolve().parents[1]


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/"input.json"
        self.gt = {"a": {"boxes": [[0, 0, 10, 10, 0]]}, "empty": {"boxes": []}}

    def write(self, value, encoding="utf-8"):
        self.path.write_text(json.dumps(value), encoding=encoding)
        return self.path

    def test_unknown_image_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown image"):
            load_predictions(self.write([{"image": "typo", "bbox": [0,0,10,10], "score": .9, "category_id": 0}]), self.gt)

    def test_negative_or_invalid_predictions_rejected(self):
        for field, value in [("bbox",[0,0,-1,10]), ("score",float("nan")), ("category_id",.5)]:
            row = {"image":"a", "bbox":[0,0,10,10], "score":.9, "category_id":0}
            row[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                load_predictions(self.write([row]), self.gt)

    def test_missing_images_and_zero_area_predictions_preserved(self):
        pred = load_predictions(self.write([{"image":"empty", "bbox":[0,0,0,10], "score":.8, "category_id":0}]), self.gt)
        self.assertEqual(pred["a"].shape, (0,6))
        self.assertEqual(pred["empty"].shape, (1,6))

    def test_utf8_bom_is_supported(self):
        self.assertEqual(load_gt(self.write(self.gt, "utf-8-sig")), self.gt)

    def test_gt_shape_must_be_rows_not_implicitly_reshaped(self):
        with self.assertRaises(ValueError):
            load_gt(self.write({"a":{"boxes":[0,0,10,10,0]}}))

    def candidates(self):
        return {"meta":{"complete":True,"conf":.03},
                "preds":{name:{"g":[],"t":[],"tl":[]} for name in self.gt}}

    def test_partial_candidate_split_rejected(self):
        data = self.candidates()
        del data["preds"]["empty"]
        with self.assertRaisesRegex(ValueError,"exactly cover"):
            load_candidates(self.write(data), self.gt, .03)

    def test_cannot_recover_candidates_below_recorded_floor(self):
        data = self.candidates()
        data["meta"]["conf"] = .1
        with self.assertRaisesRegex(ValueError,"below"):
            load_candidates(self.write(data), self.gt, .03)

    def test_tile_center_alignment_and_domain(self):
        for centers in [[], [[.5]], [[1.1,.5]], [[float("nan"),.5]]]:
            data = self.candidates()
            data["preds"]["a"].update(t=[[0,0,10,10,.8,0]],tl=centers)
            with self.subTest(centers=centers), self.assertRaises(ValueError):
                load_candidates(self.write(data), self.gt, .03)


class CommandTests(unittest.TestCase):
    def arguments(self, *extra):
        return evaluate.build_parser().parse_args([
            "--pred",str(ROOT/"examples/predictions.json"),
            "--gt",str(ROOT/"examples/gt.json"),*extra])

    def test_example_counts_and_taxonomy_conserve(self):
        report = evaluate.run(self.arguments("--taxonomy"))
        m = report["methods"]["predictions"]
        self.assertEqual((m["TP"],m["FP"],m["FN"]),(2,2,1))
        self.assertEqual(m["XS_recall"],.5)
        self.assertEqual(m["XS_precision"],.5)
        self.assertAlmostEqual(m["FP_per_img"],2/3)
        self.assertEqual(m["taxonomy"]["FP"]["total"],m["FP"])
        self.assertEqual(m["taxonomy"]["FN"]["total"],m["FN"])
        self.assertTrue(all(v is None or np.isfinite(v) for v in m["recall_by_band"].values()))
        json.dumps(report,allow_nan=False)

    def test_nonpointfive_ap_not_mislabelled_ap50(self):
        m = evaluate.run(self.arguments("--iou","0.7"))["methods"]["predictions"]
        self.assertNotIn("XS_AP50",m)
        self.assertNotIn("XS_AP50_allFP",m)
        self.assertIn("XS_AP_at_match_iou",m)
        self.assertIn("legacy_AP50",m)

    def test_explicit_class_universe_rejects_outside_classes(self):
        with self.assertRaisesRegex(ValueError,"num-classes"):
            evaluate.run(self.arguments("--num-classes","1"))

    def test_bad_iou_rejected(self):
        for value in ["0","1.1","nan"]:
            with self.assertRaises(ValueError):
                evaluate.run(self.arguments("--iou",value))

    def test_numpy_only_import_and_run(self):
        # Block optional packages even when the developer machine has them.
        script = '''import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch','torchvision','sahi','ultralytics','pycocotools'}:
        raise ImportError('Optional package blocked in NumPy-only check: '+name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = ['evaluate.py','--pred','examples/predictions.json','--gt','examples/gt.json','--taxonomy']
runpy.run_path('evaluate.py',run_name='__main__')
'''
        result = subprocess.run([sys.executable,"-c",script],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn("3 images",result.stdout)

    def test_cli_creates_portable_json_output(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/"nested"/"result.json"
            result = subprocess.run([sys.executable,str(ROOT/"evaluate.py"),"--pred",str(ROOT/"examples/predictions.json"),
                                     "--gt",str(ROOT/"examples/gt.json"),"--out",str(out)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertFalse(out.read_bytes().startswith(b"\xef\xbb\xbf"))
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["protocol"]["n_images"],3)


class ConfigurationTests(unittest.TestCase):
    def test_shipped_config_is_complete_and_matches_defaults(self):
        from dataclasses import asdict
        config = json.loads((ROOT/"configs/default.json").read_text())
        self.assertEqual(config,asdict(SattaConfig()))

    def test_routing_boundaries_differ_from_evaluation_bands(self):
        np.testing.assert_array_equal(bucket(np.array([15.9,16,32,96]),SattaConfig()),[0,1,2,3])

    def test_invalid_configuration_rejected(self):
        for kwargs in [{"margin":0},{"gamma":1.1},{"xs_max":32},{"nms_iou":0},{"use_boundary":"false"}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SattaConfig(**kwargs).validate()

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"),
                         "optional merge packages unavailable")
    def test_boundary_penalty_changes_selection_without_ground_truth(self):
        from satta.satta import satta_select
        record = {"g":np.empty((0,6),np.float32),
                  "t":np.array([[0,0,10,10,.07,0],[30,0,40,10,.07,0]],np.float32),
                  "tl":np.array([[0,.5],[.5,.5]])}
        selected = satta_select(record,SattaConfig())
        self.assertEqual(len(selected),1)
        self.assertEqual(selected[0,0],30)
        self.assertEqual(len(satta_select(record,SattaConfig(use_boundary=False))),2)


if __name__ == "__main__":
    unittest.main()
