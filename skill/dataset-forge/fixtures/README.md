# Fixtures

`orders.csv` — 600 synthetic rows with planted structure, the regression test for the skill:
`total = unit_price × qty` (degree-2, SWRL-verifiable), `subtotal = total × (1 − discount_pct/100)`
(formula non-canonical for the script — the semantic channel must state it), `zip ↔ city ↔ region`
(bijective lookups, Horn), `late ⇔ delivered_days > 7` (threshold label, SWRL comparator),
`weight_kg ≈ qty × hidden unit weight` (near-collinear, NOT derivable — an honest disagreement),
`order_id` (identity), `customer_id` (foreign key), `note` (low-cardinality text).
Expected: basis of 8 named columns against an intrinsic dimension of ~7; `late` as the single
defensible label candidate with `delivered_days` in its leakage set.
