"""
tickets — customer support tickets and the tags an agent gave them (lane L03, multilabel with an
FT-Transformer over mixed numeric and nominal inputs).

    frame = make(seed, rows)        # ticket_id, length_chars, hour, attachments, tenure_days,
                                    # product, channel, priority, tags

`tags` is one delimited column (BUILD-PLAN D1 Q3): the tags a ticket carries, joined with `|`
in the lane's class order — billing, bug, account, shipping. Every ticket carries at least one.

The signal: each tag is an independent logistic function of what the ticket looked like when
it was opened, so a tag's rate moves with the features and the tags co-occur only through
them:

- billing: subscriptions, e-mail, office hours, long-standing customers;
- bug: the software products, attachments (screenshots, logs), long first messages;
- account: new customers (under a month of tenure), chat, the web portal;
- shipping: the hardware product (the router) and the telephone.

A ticket the draw leaves untagged takes its most likely tag, as an agent would file it
somewhere. Deterministic in (seed, rows).

The recipe goes one step beyond the standard pattern (D6 §4): the length of a first message is
log-normal and acts on the tags through its log, so the generator plants the finding F7 (the
column is right-skewed; lanekit.planted) and the recipe answers it with a `log1p` transform of
`length_chars` before the scale.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import ColumnSpec
from ..planted import skew_finding, with_transform
from ..recipes import standard_recipe

__all__ = ["SCHEMA", "RECIPE", "VOCAB", "VOCAB_WITHHELD", "FINDINGS", "TAGS", "SEP", "make"]

TAGS = ("billing", "bug", "account", "shipping")
SEP = "|"
PRODUCTS = ("mobile_app", "web_portal", "public_api", "router", "subscription")
PRODUCT_SHARES = (0.26, 0.24, 0.12, 0.16, 0.22)
CHANNELS = ("email", "chat", "phone", "web_form")
CHANNEL_SHARES = (0.35, 0.30, 0.15, 0.20)
PRIORITIES = ("low", "normal", "urgent")
SOFTWARE = ("mobile_app", "web_portal", "public_api")

SCHEMA = (
    ColumnSpec("ticket_id", "str", "key", comment="The ticket's number in the help desk."),
    ColumnSpec("length_chars", "int", "numeric", "characters", "The length of the ticket's first message."),
    ColumnSpec("hour", "int", "numeric", "h", "The hour of the day the ticket was opened, 0 to 23."),
    ColumnSpec("attachments", "int", "numeric", None, "How many files the first message carried."),
    ColumnSpec("tenure_days", "int", "numeric", "days", "How long the customer has held an account."),
    ColumnSpec("product", "str", "nominal", comment="The product the ticket is about."),
    ColumnSpec("channel", "str", "nominal", comment="How the customer reached the help desk."),
    ColumnSpec("priority", "str", "nominal", comment="The priority the customer chose."),
    ColumnSpec("tags", "str", "label", comment="The tags the support agent gave the ticket, joined with |."),
)

# the words this dataset brings to a page (D6 §4: column names, class names, entity, row noun),
# read by the literal lint and the page leak check (lint_templates.py, D6 §10.6)
VOCAB = frozenset({"ticket", "tickets", "ticket_id", "billing", "tenure_days", "tenure", "length_chars",
                   "priority", "urgent", "mobile_app", "web_portal", "public_api", "router", "subscription",
                   "email", "chat", "web_form", "helpdesk"})
# D6 §4's words VOCAB leaves out: a shared template uses each in a sense of its own, so in VOCAB
# each would fail the literal gate there — `channel` an input channel (the autoregressive
# channel, 42 lines), `product` a dot or matrix product, `tags` the page fixtures' multilabel
# column, `attachments` a graph's query attachments, `account` "account for", `bug` a bug in the
# code, `shipping` shipping a bundle, `phone` a phone-width card. Each joins VOCAB with a
# sense-scoped lint-allow.json entry per line (T04's file); until then the ticket's other words
# carry the lane's leak check.
VOCAB_WITHHELD = frozenset({"channel", "product", "tags", "attachments", "account", "bug", "shipping", "phone"})
COMPONENT = "support"
ENTITY_COMMENT = "One support ticket as the help desk recorded it when it was opened."
OVERVIEW = """# tickets

A support ticket arrives at the help desk and an agent tags it: billing, bug, account,
shipping — any of them, at least one. `tags` is the attribute the chain partitions on; every
other property describes the ticket at the moment it was opened, before an agent read it.

The domain has one class, Ticket. Its number names it and says nothing about it, so the chain
drops it before anything is fitted.

The product, the channel and the priority are categories, and the network reads each through
its own token; the length of the first message, the hour, the attachments and the customer's
tenure are quantities.
"""


def _sigmoid(x: Any) -> Any:
    import numpy as np

    return 1.0 / (1.0 + np.exp(-x))


def make(seed: int, rows: int):
    """`rows` tickets, deterministic in (seed, rows): the ticket as opened, then its tags."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    n = int(rows)
    product = rng.choice(PRODUCTS, size=n, p=PRODUCT_SHARES)
    channel = rng.choice(CHANNELS, size=n, p=CHANNEL_SHARES)
    priority = rng.choice(PRIORITIES, size=n, p=(0.30, 0.55, 0.15))
    new = rng.uniform(size=n) < 0.18
    tenure = np.where(new, rng.integers(0, 30, n), np.round(rng.lognormal(6.2, 0.8, n))).astype(int)
    office = rng.uniform(size=n) < 0.72
    hour = np.where(office, rng.integers(8, 18, n), rng.integers(0, 24, n)).astype(int)
    software = np.isin(product, SOFTWARE)
    attachments = rng.poisson(np.where(software, 0.9, 0.3)).astype(int)
    length = np.clip(np.round(rng.lognormal(5.6 + 0.35 * software + 0.2 * (priority == "urgent"), 0.6, n)),
                     12, 6000).astype(int)
    z_len = (np.log(length) - 5.8) / 0.6
    logits = np.stack([
        -1.6 + 2.4 * (product == "subscription") + 0.9 * (channel == "email") + 0.5 * office
        + 0.4 * (tenure > 365),
        -1.4 + 2.0 * software + 0.7 * attachments + 0.5 * z_len - 1.0 * (product == "subscription"),
        -1.7 + 2.0 * new + 1.1 * (channel == "chat") + 0.8 * (product == "web_portal"),
        -2.8 + 3.6 * (product == "router") + 1.0 * (channel == "phone") + 0.4 * (priority == "urgent"),
    ], axis=1)
    prob = _sigmoid(logits)
    tagged = rng.uniform(size=prob.shape) < prob
    empty = ~tagged.any(axis=1)
    tagged[empty, np.argmax(prob[empty], axis=1)] = True
    tags = [SEP.join(t for t, on in zip(TAGS, row) if on) for row in tagged]
    frame = pd.DataFrame({"ticket_id": [f"TK{i + 100000:07d}" for i in range(n)], "length_chars": length,
                          "hour": hour, "attachments": attachments, "tenure_days": tenure,
                          "product": product, "channel": channel, "priority": priority, "tags": tags})
    return frame[[c.name for c in SCHEMA]]


# the findings this generator plants beyond lanekit.recipes.findings_for (lanekit.planted)
FINDINGS = {"F7": skew_finding(
    "F7", "T7", "length_chars",
    "a first message's length is log-normal: most tickets say a few hundred characters, a few paste "
    "a whole log, so the column's right tail is long",
    "the length acts on the tags through its log (a long message reads as a bug report whatever its "
    "exact length): the log brings the pasted logs in before the scale")}


def RECIPE(raw_path: Path, shaped_dir: Path, lane: Any) -> dict:  # noqa: N802 (the protocol's name)
    """The standard recipe (lanekit.recipes): the ticket number dropped, `tags` selected, a
    random split, the three categories encoded ordinal (the feature tokenizer reads every
    nominal), the quantities scaled on train — with `length_chars` log1p-transformed before the
    scale, citing F7/T7 (a length of zero stays finite at serving, which a plain log would not)."""
    return with_transform(standard_recipe(SCHEMA, raw_path, shaped_dir, lane), FINDINGS["F7"], "log1p")
