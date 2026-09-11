"""Billing helpers for orders."""

from decimal import Decimal

VAT_RATE = Decimal("0.27")
FREE_SHIPPING_LIMIT = Decimal("15000")
SHIPPING_FEE = Decimal("1490")


def line_total(item):
    """Gross total of a single line item."""
    net = Decimal(str(item["unit_price"])) * item["quantity"]
    if item.get("discount_pct"):
        net = net * (1 - Decimal(str(item["discount_pct"])) / 100)
    return net * (1 + VAT_RATE)


def order_total(items, coupon=None):
    """Order total including shipping and coupon."""
    subtotal = Decimal("0")
    for item in items:
        subtotal += line_total(item)

    if coupon:
        subtotal -= coupon["amount"]

    if subtotal < FREE_SHIPPING_LIMIT:
        subtotal += SHIPPING_FEE

    return subtotal.quantize(Decimal("1"))


def split_by_vat(items):
    """Group line items by VAT rate for the invoice."""
    groups = {}
    for item in items:
        rate = item.get("vat_rate", VAT_RATE)
        if rate not in groups:
            groups[rate] = []
        groups[rate].append(item)
    return groups


def apply_refund(order, amount):
    """Record a partial refund against the order."""
    order["refunded"] = order.get("refunded", Decimal("0")) + amount
    order["balance"] = order["total"] - order["refunded"]
    return order
