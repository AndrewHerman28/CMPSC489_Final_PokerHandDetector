"""
test_hand_eval.py - Phase 5
Checks the evaluator with one hard-coded hand per hand type (no images needed).
Run: python3 test_hand_eval.py
"""

from src.hand_eval import evaluate_hand

CASES = [
    (["AH", "KH", "QH", "JH", "10H"], "Royal Flush"),
    (["9S", "8S", "7S", "6S", "5S"], "Straight Flush"),
    (["AD", "2D", "3D", "4D", "5D"], "Straight Flush"),     # ace-low straight flush
    (["KC", "KD", "KH", "KS", "2C"], "Four of a Kind"),
    (["QC", "QD", "QH", "7S", "7C"], "Full House"),
    (["2H", "9H", "JH", "4H", "KH"], "Flush"),
    (["10C", "9D", "8H", "7S", "6C"], "Straight"),
    (["AC", "2D", "3H", "4S", "5C"], "Straight"),            # wheel: A-2-3-4-5
    (["10C", "JD", "QH", "KS", "AC"], "Straight"),           # broadway
    (["8C", "8D", "8H", "KS", "2C"], "Three of a Kind"),
    (["JC", "JD", "4H", "4S", "AC"], "Two Pair"),
    (["10C", "10D", "3H", "7S", "AC"], "One Pair"),
    (["AC", "JD", "8H", "4S", "2C"], "High Card"),
    (["QC", "KD", "AH", "2S", "3C"], "High Card"),           # no wrap-around straight
]


def main():
    passed = 0
    for cards, expected in CASES:
        got = evaluate_hand(cards)
        ok = got == expected
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {' '.join(cards):<22} -> {got:<16} (expected {expected})")
    print(f"\n{passed}/{len(CASES)} passed")


if __name__ == "__main__":
    main()