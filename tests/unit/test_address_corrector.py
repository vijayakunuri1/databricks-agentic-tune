"""Unit tests for AddressCorrector."""
import pytest
from agents.data_updater.address_corrector import AddressCorrector

CITIES = [
    "Los Angeles", "Chicago", "Houston", "Phoenix", "Atlanta",
    "Denver", "Boston", "Miami", "Seattle", "New York",
]

@pytest.fixture
def corrector():
    return AddressCorrector(city_reference=CITIES)


def test_correct_city_typo(corrector):
    result = corrector.correct_city("Los Angles")
    assert result.corrected_value == "Los Angeles"
    assert result.confidence >= 0.85


def test_correct_city_exact(corrector):
    result = corrector.correct_city("Chicago")
    # exact match should return high confidence
    assert result.confidence >= 0.99


def test_correct_state_full_name(corrector):
    result = corrector.correct_state("California")
    assert result.corrected_value == "CA"
    assert result.confidence == 1.0


def test_correct_state_abbr(corrector):
    result = corrector.correct_state("NY")
    assert result.corrected_value == "NY"
    assert result.confidence == 1.0


def test_correct_state_typo(corrector):
    result = corrector.correct_state("Colorad")
    # fuzzy match should suggest CO or Colorado
    assert result.corrected_value in ("CO",)


def test_correct_zip_valid(corrector):
    result = corrector.correct_zip("10001")
    assert result.corrected_value == "10001"
    assert result.confidence == 1.0


def test_correct_zip_invalid(corrector):
    result = corrector.correct_zip("ABC")
    assert result.corrected_value is None


def test_correct_zip_short(corrector):
    result = corrector.correct_zip("1234")
    assert result.corrected_value == "01234"
    assert result.confidence == 0.80


def test_correct_street_abbreviation(corrector):
    result = corrector.correct_street("123 Main St.")
    assert result.corrected_value == "123 Main Street"
    assert result.confidence == 0.80


def test_correct_street_no_change(corrector):
    result = corrector.correct_street("123 Main Street")
    assert result.corrected_value is None   # no change needed
