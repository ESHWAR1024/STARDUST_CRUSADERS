"""
Phase 5 - Fraud injection engine.

Takes the fraud-role accounts produced by entities.generate_fraud_population
(mules, collectors, fresh_layering_node shells) and organizes them into
config.N_FRAUD_RINGS independent fraud networks, each implementing one of
four typologies:

  - layering_chain        : sequential hop-to-hop movement with skim/decay
  - fan_out_mule_network   : one inbound lump sum split across many mule legs
  - fan_in_collector       : many small/medium inbound legs converge on one
                             collector who then exits/forwards the pool
  - smurfing_structuring   : a large sum broken into sub-threshold legs over
                             multiple days to avoid scrutiny thresholds

Every fraud event is tagged is_fraud=True / fraud_pattern / fraud_ring_id.
These tags live ONLY in the ground-truth export (ground_truth.py) - they
are stripped before the data reaches the "investigation system" input, so
detection genuinely has to be discovered, not read off a label column.

Money origin: each ring's first hop receives a seed amount framed as
proceeds already inside the banking system (e.g. a phished/OTP-fraud
victim's transfer) - modeled as a single inbound credit with no
counterparty account in our population (the victim's own account isn't
part of this dataset, mirroring how real cases only seize the
launderers' accounts, not every victim's). Money exits the ring via cash
withdrawal, international remittance, or crypto-exchange transfer.
"""

import random
from datetime import timedelta
from faker import Faker

import config
from behavior_engine import _event, _ts

fake = Faker("en_IN")


def _fraud_hour():
    if random.random() < config.FRAUD_ODD_HOUR_BIAS:
        return random.randint(0, 5)
    return random.randint(6, 23)


def _fraud_ts(d, base_dt=None, delay=None):
    if base_dt is not None and delay is not None:
        return base_dt + delay
    return _ts(d, hour=_fraud_hour())


def _exit_event(acct, ts, amount, ring_id):
    channel = random.choices(config.EXIT_CHANNELS, weights=config.EXIT_CHANNEL_WEIGHTS, k=1)[0]
    label = {
        "ATM_CASH_WITHDRAWAL": f"ATM WDL-{acct.bank_code} ATM {fake.city().upper()}",
        "INTERNATIONAL_REMITTANCE": f"SWIFT OUT-{fake.country().upper()}-FOREIGN REMITTANCE",
        "CRYPTO_EXCHANGE_TRANSFER": f"UPI-{random.choice(['WAZIRX','COINSWITCH','BINANCE P2P'])}-CRYPTO PURCHASE",
    }[channel]
    ev = _event(acct.account_id, ts, label, channel, debit=amount)
    ev["is_fraud"] = True
    ev["fraud_pattern"] = "exit_" + channel.lower()
    ev["fraud_ring_id"] = ring_id
    return ev


def _seed_credit_event(acct, ts, amount, ring_id):
    victim_label = random.choice([
        "IMPS-UNKNOWN-FUNDS TRANSFER", "UPI-CR-PAYMENT RECEIVED",
        "NEFT CR-THIRD PARTY-TRANSFER",
    ])
    ev = _event(acct.account_id, ts, victim_label, "IMPS", credit=amount)
    ev["is_fraud"] = True
    ev["fraud_pattern"] = "ring_seed"
    ev["fraud_ring_id"] = ring_id
    return ev


def _fraud_transfer(sender, receiver, ts, amount, ring_id, pattern, channel="UPI"):
    utr = f"{channel}{random.randint(10**11, 10**12 - 1)}"
    debit_leg = _event(sender.account_id, ts,
                        f"{channel}-{receiver.holder_name.upper()}-TRANSFER",
                        channel, debit=amount,
                        counterparty_account_id=receiver.account_id,
                        counterparty_name=receiver.holder_name, utr_ref=utr)
    credit_leg = _event(receiver.account_id, ts,
                         f"{channel}-{sender.holder_name.upper()}-TRANSFER",
                         channel, credit=amount,
                         counterparty_account_id=sender.account_id,
                         counterparty_name=sender.holder_name, utr_ref=utr)
    for leg in (debit_leg, credit_leg):
        leg["is_fraud"] = True
        leg["fraud_pattern"] = pattern
        leg["fraud_ring_id"] = ring_id
    return [debit_leg, credit_leg]


def _random_day_in_window(margin_days=10):
    span = (config.SIM_END_DATE - config.SIM_START_DATE).days - margin_days
    offset = random.randint(0, max(span, 1))
    return config.SIM_START_DATE + timedelta(days=offset)


# ---------------------------------------------------------------------------
# Typology 1: Layering chain
# ---------------------------------------------------------------------------
def build_layering_chain(ring_id, nodes):
    """nodes: ordered list of Account objects (the chain hops), exact length
    already decided by plan_fraud_rings()."""
    events = []

    seed_day = _random_day_in_window()
    amount = random.uniform(*config.LAYERING_SEED_AMOUNT_RANGE)
    ts = _fraud_ts(seed_day)
    events.append(_seed_credit_event(nodes[0], ts, amount, ring_id))

    current_amount = amount
    current_ts = ts
    for i in range(len(nodes) - 1):
        skim = random.uniform(*config.LAYERING_SKIM_RATIO_RANGE)
        forward_amount = current_amount * (1 - skim)
        delay = timedelta(hours=random.uniform(*config.LAYERING_HOP_DELAY_HOURS_RANGE))
        current_ts = current_ts + delay
        events += _fraud_transfer(nodes[i], nodes[i + 1], current_ts,
                                   forward_amount, ring_id, "layering_hop")
        current_amount = forward_amount

    # final node exits the funds
    exit_delay = timedelta(hours=random.uniform(*config.LAYERING_HOP_DELAY_HOURS_RANGE))
    current_ts = current_ts + exit_delay
    events.append(_exit_event(nodes[-1], current_ts, current_amount * random.uniform(0.9, 1.0), ring_id))
    return events


# ---------------------------------------------------------------------------
# Typology 2: Fan-out mule network
# ---------------------------------------------------------------------------
def build_fan_out_network(ring_id, hub, mules):
    events = []

    seed_day = _random_day_in_window()
    total_amount = random.uniform(*config.FAN_OUT_SEED_AMOUNT_RANGE)
    seed_ts = _fraud_ts(seed_day)
    events.append(_seed_credit_event(hub, seed_ts, total_amount, ring_id))

    weights = [random.uniform(0.5, 1.5) for _ in mules]
    wsum = sum(weights)
    shares = [w / wsum for w in weights]

    for mule, share in zip(mules, shares):
        delay = timedelta(minutes=random.uniform(*config.FAN_OUT_DELAY_MINUTES_RANGE))
        leg_ts = seed_ts + delay
        leg_amount = total_amount * share * random.uniform(0.95, 1.0)
        events += _fraud_transfer(hub, mule, leg_ts, leg_amount, ring_id, "fan_out_leg")

        # mule quickly cashes out most of what it received (classic mule signature)
        cashout_delay = timedelta(hours=random.uniform(0.5, 30))
        cashout_amount = leg_amount * random.uniform(0.75, 0.97)
        events.append(_exit_event(mule, leg_ts + cashout_delay, cashout_amount, ring_id))
    return events


# ---------------------------------------------------------------------------
# Typology 3: Fan-in collector
# ---------------------------------------------------------------------------
def build_fan_in_network(ring_id, sources, collector):
    events = []

    window_start_day = _random_day_in_window()
    window_hours = random.uniform(*config.FAN_IN_WINDOW_HOURS_RANGE)
    window_start_ts = _fraud_ts(window_start_day)

    total_collected = 0.0
    last_ts = window_start_ts
    for src in sources:
        # each source first receives its own small "proceeds" seed, then
        # forwards into the collector - mirrors many small scam victims'
        # money being funneled by individual mules toward one pool account
        amount = random.uniform(*config.FAN_IN_LEG_AMOUNT_RANGE)
        seed_ts = window_start_ts + timedelta(hours=random.uniform(0, window_hours * 0.3))
        events.append(_seed_credit_event(src, seed_ts, amount, ring_id))

        forward_delay = timedelta(hours=random.uniform(0.2, window_hours))
        forward_ts = seed_ts + forward_delay
        events += _fraud_transfer(src, collector, forward_ts, amount * random.uniform(0.9, 1.0),
                                   ring_id, "fan_in_leg")
        total_collected += amount
        last_ts = max(last_ts, forward_ts)

    # collector exits the pooled funds shortly after the window closes
    exit_delay = timedelta(hours=random.uniform(1, 24))
    events.append(_exit_event(collector, last_ts + exit_delay,
                               total_collected * random.uniform(0.85, 0.98), ring_id))
    return events


# ---------------------------------------------------------------------------
# Typology 4: Smurfing / structuring
# ---------------------------------------------------------------------------
def build_smurfing_network(ring_id, sources, beneficiary):
    events = []
    n_legs = random.randint(*config.SMURF_LEG_COUNT_RANGE)
    spread_days = random.randint(*config.SMURF_SPREAD_DAYS_RANGE)
    start_day = _random_day_in_window(margin_days=spread_days + 2)

    total = 0.0
    for i in range(n_legs):
        src = sources[i % len(sources)]
        amount = min(random.uniform(*config.SMURF_LEG_AMOUNT_RANGE),
                     config.CTR_LIKE_THRESHOLD * random.uniform(0.6, 0.97))
        day_offset = random.randint(0, spread_days)
        d = start_day + timedelta(days=day_offset)
        ts = _fraud_ts(d)

        # each leg's own small seed before being relayed onward
        events.append(_seed_credit_event(src, ts - timedelta(hours=random.uniform(1, 6)), amount, ring_id))
        events += _fraud_transfer(src, beneficiary, ts, amount, ring_id, "smurf_leg")
        total += amount

    exit_ts = _fraud_ts(start_day + timedelta(days=spread_days)) + timedelta(hours=12)
    events.append(_exit_event(beneficiary, exit_ts, total * random.uniform(0.85, 0.98), ring_id))
    return events


# ---------------------------------------------------------------------------
# Orchestration: plan exact account requirements, then build each ring
# ---------------------------------------------------------------------------
def plan_fraud_rings():
    """
    Decide each ring's typology AND concrete size parameters up front, so
    the exact number of accounts-per-role needed is known before any
    accounts are generated. This is what guarantees every planned ring
    actually gets built (no silent failures from running out of a role
    partway through, which is what a pop-as-you-go allocator risks).

    Returns: (ring_plans, role_totals)
      ring_plans: list of dicts {ring_id, typology, n_main, n_secondary}
      role_totals: dict {"mule": x, "collector": y, "fresh_layering_node": z}
    """
    typologies = list(config.FRAUD_TYPOLOGY_WEIGHTS.keys())
    weights = list(config.FRAUD_TYPOLOGY_WEIGHTS.values())

    # Guarantee each typology appears at least once (when there are enough
    # rings to do so) - makes for a far better demo than risking 4/4 rings
    # randomly landing on the same pattern by chance.
    if config.N_FRAUD_RINGS >= len(typologies):
        forced = typologies.copy()
        random.shuffle(forced)
        extra = random.choices(typologies, weights=weights, k=config.N_FRAUD_RINGS - len(typologies))
        typology_sequence = forced + extra
        random.shuffle(typology_sequence)
    else:
        typology_sequence = random.choices(typologies, weights=weights, k=config.N_FRAUD_RINGS)

    role_totals = {"mule": 0, "collector": 0, "fresh_layering_node": 0}
    ring_plans = []

    for ring_idx in range(config.N_FRAUD_RINGS):
        ring_id = f"RING{ring_idx:03d}"
        typology = typology_sequence[ring_idx]

        if typology == "layering_chain":
            n_nodes = random.randint(*config.LAYERING_CHAIN_LENGTH_RANGE)
            role_totals["fresh_layering_node"] += n_nodes
            ring_plans.append({"ring_id": ring_id, "typology": typology, "n_nodes": n_nodes})

        elif typology == "fan_out_mule_network":
            n_legs = random.randint(*config.FAN_OUT_LEG_COUNT_RANGE)
            role_totals["collector"] += 1
            role_totals["mule"] += n_legs
            ring_plans.append({"ring_id": ring_id, "typology": typology, "n_legs": n_legs})

        elif typology == "fan_in_collector":
            n_sources = random.randint(*config.FAN_IN_SOURCE_COUNT_RANGE)
            role_totals["collector"] += 1
            role_totals["mule"] += n_sources
            ring_plans.append({"ring_id": ring_id, "typology": typology, "n_sources": n_sources})

        elif typology == "smurfing_structuring":
            n_sources = random.randint(4, 8)
            role_totals["collector"] += 1
            role_totals["mule"] += n_sources
            ring_plans.append({"ring_id": ring_id, "typology": typology, "n_sources": n_sources})

    return ring_plans, role_totals


def build_fraud_rings(fraud_accounts, ring_plans):
    """
    fraud_accounts: accounts generated by entities.generate_fraud_population
    using the exact role_totals returned by plan_fraud_rings() - so popping
    by role here is guaranteed to have enough supply for every ring.
    """
    random.shuffle(fraud_accounts)
    by_role = {"mule": [], "collector": [], "fresh_layering_node": []}
    for a in fraud_accounts:
        by_role[a.fraud_role].append(a)

    events = []
    ring_summaries = []

    for plan in ring_plans:
        ring_id = plan["ring_id"]
        typology = plan["typology"]

        if typology == "layering_chain":
            nodes = [by_role["fresh_layering_node"].pop() for _ in range(plan["n_nodes"])]
            for n in nodes:
                n.fraud_ring_id = ring_id
            events += build_layering_chain(ring_id, nodes)
            ring_summaries.append({"ring_id": ring_id, "typology": typology,
                                    "accounts": [n.account_id for n in nodes]})

        elif typology == "fan_out_mule_network":
            hub = by_role["collector"].pop()
            mules = [by_role["mule"].pop() for _ in range(plan["n_legs"])]
            hub.fraud_ring_id = ring_id
            for m in mules:
                m.fraud_ring_id = ring_id
            events += build_fan_out_network(ring_id, hub, mules)
            ring_summaries.append({"ring_id": ring_id, "typology": typology,
                                    "accounts": [hub.account_id] + [m.account_id for m in mules]})

        elif typology == "fan_in_collector":
            collector = by_role["collector"].pop()
            sources = [by_role["mule"].pop() for _ in range(plan["n_sources"])]
            collector.fraud_ring_id = ring_id
            for s in sources:
                s.fraud_ring_id = ring_id
            events += build_fan_in_network(ring_id, sources, collector)
            ring_summaries.append({"ring_id": ring_id, "typology": typology,
                                    "accounts": [collector.account_id] + [s.account_id for s in sources]})

        elif typology == "smurfing_structuring":
            beneficiary = by_role["collector"].pop()
            sources = [by_role["mule"].pop() for _ in range(plan["n_sources"])]
            beneficiary.fraud_ring_id = ring_id
            for s in sources:
                s.fraud_ring_id = ring_id
            events += build_smurfing_network(ring_id, sources, beneficiary)
            ring_summaries.append({"ring_id": ring_id, "typology": typology,
                                    "accounts": [beneficiary.account_id] + [s.account_id for s in sources]})

    return events, ring_summaries
