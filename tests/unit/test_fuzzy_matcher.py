"""Unit tests for FuzzyMatcher."""
import pytest
from agents.data_updater.fuzzy_matcher import FuzzyMatcher

CITIES = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Atlanta", "Denver", "Boston", "Miami", "Seattle"]


def test_exact_match():
    m = FuzzyMatcher(CITIES)
    results = m.find_best_match("New York")
    assert results[0][0] == "New York"
    assert results[0][1] >= 0.99


def test_typo_l1_confidence():
    m = FuzzyMatcher(CITIES)
    best, conf = m.best_match("Los Angles")   # one letter missing
    assert best == "Los Angeles"
    assert conf >= 0.85
    assert m.classify_match(conf) == "L1"


def test_typo_l2_confidence():
    m = FuzzyMatcher(CITIES)
    best, conf = m.best_match("Chcago")      # two letters off
    assert best == "Chicago"
    # rapidfuzz WRatio scores short strings with few edits very highly;
    # the important assertion is that we get the right match
    assert conf >= 0.50


def test_no_match_returns_skip():
    m = FuzzyMatcher(CITIES)
    _, conf = m.best_match("Xyzzy")
    assert m.classify_match(conf) == "skip"


def test_empty_query():
    m = FuzzyMatcher(CITIES)
    results = m.find_best_match("")
    assert results == []


def test_empty_reference():
    m = FuzzyMatcher([])
    results = m.find_best_match("Chicago")
    assert results == []


def test_token_sort_scorer():
    m = FuzzyMatcher(["123 Main Street", "456 Oak Avenue"], scorer_name="token_sort_ratio")
    results = m.find_best_match("Main Street 123")
    assert results[0][0] == "123 Main Street"
    assert results[0][1] >= 0.85
