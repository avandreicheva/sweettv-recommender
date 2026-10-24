"""Unit test for the MAP@5 evaluator (08_evaluator.py) against two
hand-worked examples: relevant = {A, B, C}.
  pred [A,B,D,C,E] -> AP@5 = (1/1 + 2/2 + 3/4) / 3 = 0.91667
  pred [D,A,E,B,C] -> AP@5 = (1/2 + 2/4 + 3/5) / 3 = 0.53333
Run directly: python3 test_evaluator.py
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("evaluator", Path(__file__).parent / "08_evaluator.py")
evaluator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluator)


def test_ap_at_5_example_1():
    pred = ["A", "B", "D", "C", "E"]
    relevant = {"A", "B", "C"}
    ap = evaluator.average_precision_at_k(pred, relevant, k=5)
    assert abs(ap - 0.9167) < 1e-3, f"expected 0.9167, got {ap}"
    return ap


def test_ap_at_5_example_2():
    pred = ["D", "A", "E", "B", "C"]
    relevant = {"A", "B", "C"}
    ap = evaluator.average_precision_at_k(pred, relevant, k=5)
    assert abs(ap - 0.5333) < 1e-3, f"expected 0.5333, got {ap}"
    return ap


def test_empty_relevant_excluded():
    assert evaluator.average_precision_at_k(["A", "B"], set(), k=5) is None


def test_map_at_k_aggregation():
    preds = {"u1": ["A", "B", "D", "C", "E"], "u2": ["D", "A", "E", "B", "C"], "u3": ["X"]}
    gts = {"u1": {"A", "B", "C"}, "u2": {"A", "B", "C"}, "u3": set()}
    res = evaluator.map_at_k(preds, gts, k=5)
    expected_mean = (0.9167 + 0.5333) / 2
    assert abs(res["map@k"] - expected_mean) < 1e-3, res
    assert res["n_scored"] == 2
    assert res["n_empty_Ru"] == 1
    assert res["n_total"] == 3


if __name__ == "__main__":
    ap1 = test_ap_at_5_example_1()
    print(f"Example 1  pred=[A,B,D,C,E]  relevant={{A,B,C}}  AP@5={ap1:.4f}  (expected 0.9167)  PASS")
    ap2 = test_ap_at_5_example_2()
    print(f"Example 2  pred=[D,A,E,B,C]  relevant={{A,B,C}}  AP@5={ap2:.4f}  (expected 0.5333)  PASS")
    test_empty_relevant_excluded()
    print("Empty-R_u exclusion: PASS")
    test_map_at_k_aggregation()
    print("map_at_k aggregation (mean + empty-R_u bookkeeping): PASS")
    print("\nALL TESTS PASSED")
