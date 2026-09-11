import pytest

from parser import parse_row, total


def test_parse_row_ok():
    assert parse_row("apple,3,120.5\n") == {"name": "apple", "qty": 3, "price": 120.5}


def test_parse_row_wrong_field_count():
    with pytest.raises(ValueError):
        parse_row("apple,3\n")


def test_total_accounts_for_quantity():
    rows = [
        {"name": "apple", "qty": 3, "price": 100.0},
        {"name": "pear", "qty": 2, "price": 50.0},
    ]
    # 3*100 + 2*50 = 400
    assert total(rows) == 400.0
