"""
hand_eval.py - Phase 5
Rule-based poker hand evaluator. Input: 5 card labels like ["AH", "KH", "QH", "JH", "10H"].
"""

from collections import Counter

RANK_VALUES = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
               "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
SUIT_NAMES = {"C": "Clubs", "D": "Diamonds", "H": "Hearts", "S": "Spades"}


def parse_card(label):
    """'10H' -> (10, 'H'),  'AS' -> (14, 'S')"""
    rank, suit = label[:-1], label[-1]
    if rank not in RANK_VALUES or suit not in SUIT_NAMES:
        raise ValueError(f"Unknown card label: {label}")
    return RANK_VALUES[rank], suit


def is_straight(values):
    """values = sorted unique ranks. Returns True for 5 consecutive ranks, incl. A-2-3-4-5."""
    if len(values) != 5:
        return False
    if values[-1] - values[0] == 4:
        return True
    return values == [2, 3, 4, 5, 14]          # ace-low straight ("wheel")


def evaluate_hand(labels):
    """Returns the poker hand name for exactly 5 unique card labels."""
    if len(labels) != 5:
        raise ValueError(f"Need exactly 5 cards, got {len(labels)}")
    if len(set(labels)) != 5:
        raise ValueError(f"Duplicate cards: {labels}")

    cards = [parse_card(lb) for lb in labels]
    values = sorted(v for v, _ in cards)
    suits = [s for _, s in cards]

    flush = len(set(suits)) == 1
    straight = is_straight(sorted(set(values)))
    counts = sorted(Counter(values).values(), reverse=True)   # e.g. [3, 2] = full house

    if flush and straight and values == [10, 11, 12, 13, 14]:
        return "Royal Flush"
    if flush and straight:
        return "Straight Flush"
    if counts[0] == 4:
        return "Four of a Kind"
    if counts == [3, 2]:
        return "Full House"
    if flush:
        return "Flush"
    if straight:
        return "Straight"
    if counts[0] == 3:
        return "Three of a Kind"
    if counts == [2, 2, 1]:
        return "Two Pair"
    if counts[0] == 2:
        return "One Pair"
    return "High Card"


if __name__ == "__main__":
    print(evaluate_hand(["AH", "KH", "QH", "JH", "10H"]))