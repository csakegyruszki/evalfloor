"""Mini CSV totaller. Deliberately contains ONE bug (see the tests)."""


def parse_row(line):
    parts = line.rstrip("\n").split(",")
    if len(parts) != 3:
        raise ValueError(f"expected 3 fields, got {len(parts)}: {line!r}")
    name, qty, price = parts
    return {"name": name.strip(), "qty": int(qty), "price": float(price)}


def total(rows):
    # BUG: ignores the quantity and only sums the prices.
    return sum(r["price"] for r in rows)


def load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        header = fh.readline()
        if not header:
            return rows
        for line in fh:
            if line.strip():
                rows.append(parse_row(line))
    return rows
