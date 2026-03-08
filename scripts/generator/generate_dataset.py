"""
Pantau Synthetic Dataset Generator v2
======================================
Generates 2,000,000 QRIS transactions with realistic class overlap.
Fraud rate: configurable 1-5% (default 3%).

Key v2 changes over v1:
- 9 normal merchant categories + 3 judol types + hybrid merchants
- 9 normal user profiles + 4 judol profiles with behavioral params
- Judol users also make normal purchases (40-60% of their transactions)
- Judol happens 24/7 (peak at night, not restricted to night)
- Normal nighttime traffic (24h stores, shift workers, drivers)
- Round amounts in normal (fuel, pulsa, parking, bills) ~25%
- Transaction-level labeling (not user/merchant level)
- Amount tails: clean hundreds (90-95%), rare random (5-10%)
- Minimum judol deposit: Rp 25,000

Output: data/generated/parametric/pantau_dataset.csv
"""

import argparse
import csv
import os
import random
import string
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

# ============================================================
# GEOLOCATION
# ============================================================

MAJOR_CITIES = {
    "Jakarta Selatan", "Jakarta Pusat", "Jakarta Barat", "Jakarta Timur", "Jakarta Utara",
    "Surabaya", "Bandung", "Medan", "Semarang", "Makassar",
    "Palembang", "Tangerang", "Tangerang Selatan", "Depok", "Bekasi",
    "Yogyakarta", "Denpasar", "Balikpapan", "Malang", "Bogor",
}


def load_wilayah():
    provinces = {}
    with open(os.path.join(DATA_DIR, "geolocation", "provinsi.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            provinces[row["id"].strip('"')] = row["name"].strip('"')
    cities = []
    with open(os.path.join(DATA_DIR, "geolocation", "kabupaten_kota.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            kid = row["id"].strip('"')
            kname = row["name"].strip('"')
            pname = provinces.get(kid.split(".")[0], "Unknown")
            cities.append((kname, pname))
    return cities


def build_city_weights(cities):
    return [5 if c[0] in MAJOR_CITIES else 1 for c in cities]


# ============================================================
# ID GENERATORS
# ============================================================


def gen_txn_id(dt, seq):
    return f"TXN-{dt.strftime('%Y%m%d')}-{seq:07d}"


def gen_user_id():
    return "".join(str(random.randint(0, 9)) for _ in range(random.randint(10, 15)))


def gen_nmid():
    return "ID" + str(random.randint(10, 99)) + "".join(
        random.choices(string.digits + string.ascii_uppercase, k=11)
    )


def gen_device_id():
    return "DEV-" + "".join(random.choices("0123456789ABCDEF", k=10))


# ============================================================
# AMOUNT HELPERS
# ============================================================

# Round amounts that appear in QRIS (deposit denominations for judol,
# but also fuel, pulsa, parking, bills for normal)
ROUND_AMOUNTS_NORMAL = [5000, 10000, 15000, 20000, 25000, 50000, 75000, 100000, 150000, 200000, 300000, 500000]
ROUND_AMOUNTS_JUDOL = [25000, 30000, 50000, 75000, 100000, 150000, 200000, 300000, 500000]


def snap_to_clean_hundred(amount):
    """Snap amount to nearest 100. ~90-95% of amounts end in clean hundreds,
    ~5-10% keep random tails. All clean hundreds (000-900) are equally likely."""
    if random.random() < 0.93:
        return round(amount / 100) * 100
    return amount


def gen_normal_amount(profile):
    """Generate amount based on normal user's merchant context."""
    cat = profile.get("_current_merchant_cat", "regular_retail")

    if cat == "fuel_station":
        return random.choice([40000, 50000, 75000, 100000, 150000, 200000, 250000])
    elif cat == "parking_toll":
        return random.choice([3000, 5000, 7000, 10000, 12000, 15000])
    elif cat == "cash_out":
        return random.choice([100000, 200000, 300000, 500000, 750000])
    elif cat == "food_beverage":
        raw = int(np.random.lognormal(mean=10.2, sigma=0.5))
        return snap_to_clean_hundred(max(5000, min(raw, 200000)))
    elif cat == "small_warung":
        raw = int(np.random.lognormal(mean=9.8, sigma=0.5))
        return snap_to_clean_hundred(max(3000, min(raw, 100000)))
    elif cat == "online_ecommerce":
        raw = int(np.random.lognormal(mean=11.5, sigma=1.0))
        return snap_to_clean_hundred(max(10000, min(raw, 5000000)))
    elif cat == "minimarket_24h":
        raw = int(np.random.lognormal(mean=10.4, sigma=0.7))
        return snap_to_clean_hundred(max(3000, min(raw, 500000)))
    else:  # regular_retail, event_seasonal
        raw = int(np.random.lognormal(mean=11.2, sigma=0.9))
        return snap_to_clean_hundred(max(5000, min(raw, 5000000)))

    # Some profiles add round-amount purchases (pulsa, bills)


def gen_normal_round_purchase():
    """Round-amount normal purchases: pulsa, bills, subscriptions."""
    return random.choice([10000, 25000, 50000, 100000, 150000, 200000])


def gen_judol_amount(user_profile, tx_index, total_tx):
    """Judol deposit: free denomination ≥ 25K, gravitates toward round numbers."""
    profile_type = user_profile["judol_type"]

    # Base amount depends on profile — means lowered for overlap with normal
    if profile_type == "casual":
        base = int(np.random.lognormal(mean=10.3, sigma=0.5))
        base = max(25000, min(base, 200000))
    elif profile_type == "regular":
        base = int(np.random.lognormal(mean=10.7, sigma=0.6))
        base = max(25000, min(base, 500000))
        if user_profile.get("escalation") and total_tx > 1:
            escalation = 1.0 + (tx_index / total_tx) * 0.6
            base = int(base * escalation)
    elif profile_type == "heavy":
        base = int(np.random.lognormal(mean=11.1, sigma=0.7))
        base = max(25000, min(base, 2000000))
        if user_profile.get("escalation") and total_tx > 1:
            escalation = 1.0 + (tx_index / total_tx) * 1.5
            base = int(base * escalation)
    else:  # smurfer — lognormal, not uniform
        base = int(np.random.lognormal(mean=10.5, sigma=0.7))
        base = max(25000, min(base, 150000))

    base = max(25000, base)

    # ~35% exact round, ~30% near-round, ~35% free denomination
    roll = random.random()
    if roll < 0.35:
        candidates = [r for r in ROUND_AMOUNTS_JUDOL if r >= 25000]
        closest = min(candidates, key=lambda x: abs(x - base))
        base = closest
    elif roll < 0.65:
        candidates = [r for r in ROUND_AMOUNTS_JUDOL if r >= 25000]
        closest = min(candidates, key=lambda x: abs(x - base))
        offset = random.choice([-1, 1]) * random.randint(500, 3000)
        base = max(25000, closest + offset)
    else:
        base = snap_to_clean_hundred(base)
        base = max(25000, base)

    return base


def is_round_amount(amount):
    """Round = divisible by 5000. Catches 5K, 10K, 25K, 50K, 100K, etc."""
    return amount >= 5000 and amount % 5000 == 0


# ============================================================
# TIMING
# ============================================================

GAJIAN_DATES = {1, 25, 26, 27, 28}
TOGEL_DRAW_HOURS = [13, 16, 19, 22]

# Normal hour weights by merchant category
HOUR_WEIGHTS = {
    "regular_retail":  [1, 1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 9, 10, 9, 7, 6, 5, 7, 8, 6, 4, 3, 2, 1],
    "minimarket_24h":  [3, 3, 2, 2, 2, 2, 3, 5, 7, 7, 7, 8, 9, 8, 7, 6, 6, 7, 8, 8, 7, 6, 5, 4],
    "fuel_station":    [2, 1, 1, 1, 1, 2, 4, 6, 7, 7, 6, 6, 7, 7, 6, 5, 5, 6, 7, 6, 5, 4, 3, 2],
    "food_beverage":   [1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 8, 10, 8, 5, 4, 4, 5, 8, 10, 8, 5, 3, 2],
    "small_warung":    [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 9, 10, 10, 9, 7, 5, 4, 5, 6, 4, 2, 1, 1, 1],
    "parking_toll":    [1, 1, 1, 1, 1, 1, 3, 8, 10, 7, 5, 5, 6, 5, 4, 5, 6, 9, 10, 5, 2, 1, 1, 1],
    "online_ecommerce":[4, 3, 2, 2, 2, 2, 2, 3, 4, 5, 5, 6, 7, 6, 5, 5, 5, 5, 6, 7, 8, 9, 8, 6],
    "cash_out":        [1, 1, 1, 1, 1, 1, 1, 2, 5, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1],
    "event_seasonal":  [1, 1, 1, 1, 1, 1, 1, 2, 3, 5, 6, 7, 8, 7, 6, 5, 5, 6, 8, 10, 9, 7, 4, 2],
}

# Shift worker hours (night shift: 22:00-06:00 concentrated)
HOUR_WEIGHTS_SHIFT = [6, 5, 5, 4, 3, 3, 7, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 5, 7]
# Online shopper: midnight bursts
HOUR_WEIGHTS_ONLINE = [5, 4, 3, 2, 2, 2, 2, 3, 4, 5, 5, 6, 7, 6, 5, 5, 5, 5, 6, 7, 8, 9, 8, 6]

# Judol: 24/7 with night peak (40-50% between 20:00-04:00)
JUDOL_HOUR_WEIGHTS = [7, 6, 5, 4, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 4, 4, 4, 5, 6, 8, 9, 10, 9]
#                     Mon Tue Wed Thu Fri Sat Sun
NORMAL_DAY_WEIGHTS = [15, 15, 15, 15, 15, 13, 12]
JUDOL_DAY_WEIGHTS  = [10, 10, 10, 12, 15, 20, 18]


def gen_timestamp(start, end, hour_weights, day_weights, gajian_boost=1.0):
    """Generate a timestamp with hour/day/gajian weighting."""
    days_range = (end - start).days
    if days_range <= 0:
        return start.replace(hour=random.randint(0, 23), minute=random.randint(0, 59),
                             second=random.randint(0, 59))

    # Pick a date with day-of-week and gajian weighting
    for _ in range(50):
        day_offset = random.randint(0, days_range)
        candidate = start + timedelta(days=day_offset)
        dow = candidate.weekday()
        weight = day_weights[dow]
        if candidate.day in GAJIAN_DATES:
            weight *= gajian_boost
        if random.random() < weight / (max(day_weights) * max(gajian_boost, 1.0)):
            break

    hour = random.choices(range(24), weights=hour_weights, k=1)[0]
    return candidate.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59))


def gen_togel_timestamp(start, end):
    """Togel: deposits 10-30 min before draw times (13, 16, 19, 22)."""
    days_range = max(1, (end - start).days)
    day_offset = random.randint(0, days_range)
    base = start + timedelta(days=day_offset)
    draw_hour = random.choice(TOGEL_DRAW_HOURS)
    minutes_before = random.randint(10, 30)
    ts = base.replace(hour=draw_hour, minute=0, second=0) - timedelta(minutes=minutes_before)
    return ts.replace(second=random.randint(0, 59))


# ============================================================
# MERCHANT POOL (9 normal + 3 judol + hybrid)
# ============================================================

NORMAL_MERCHANT_CATS = {
    "regular_retail":   0.30,
    "minimarket_24h":   0.20,
    "fuel_station":     0.08,
    "food_beverage":    0.20,
    "small_warung":     0.10,
    "parking_toll":     0.04,
    "online_ecommerce": 0.05,
    "cash_out":         0.01,
    "event_seasonal":   0.02,
}

JUDOL_MERCHANT_CATS = {
    "slot_casino":    0.60,
    "togel":          0.30,
    "diversified":    0.10,
}


def create_merchant_pool(cities, weights, n_normal=20000, n_judol=600):
    """Create categorized merchant pools + hybrid subset."""
    rng = np.random.default_rng(SEED)

    # Normal merchants by category
    normal_merchants = []
    for cat, ratio in NORMAL_MERCHANT_CATS.items():
        count = int(n_normal * ratio)
        for _ in range(count):
            city, prov = random.choices(cities, weights=weights, k=1)[0]
            normal_merchants.append({
                "merchant_id": gen_nmid(), "city": city, "province": prov,
                "category": cat, "is_hybrid": False,
            })

    # Pad to exact count
    while len(normal_merchants) < n_normal:
        city, prov = random.choices(cities, weights=weights, k=1)[0]
        normal_merchants.append({
            "merchant_id": gen_nmid(), "city": city, "province": prov,
            "category": "regular_retail", "is_hybrid": False,
        })

    # Judol merchants by category
    judol_merchants = []
    for cat, ratio in JUDOL_MERCHANT_CATS.items():
        count = int(n_judol * ratio)
        for _ in range(count):
            city, prov = random.choices(cities, weights=weights, k=1)[0]
            judol_merchants.append({
                "merchant_id": gen_nmid(), "city": city, "province": prov,
                "category": cat,
            })

    while len(judol_merchants) < n_judol:
        city, prov = random.choices(cities, weights=weights, k=1)[0]
        judol_merchants.append({
            "merchant_id": gen_nmid(), "city": city, "province": prov,
            "category": "slot_casino",
        })

    # Hybrid merchants: ~10% of total, sourced from normal (24h, warung, F&B)
    hybrid_eligible_cats = {"minimarket_24h", "small_warung", "food_beverage"}
    eligible = [m for m in normal_merchants if m["category"] in hybrid_eligible_cats]
    n_hybrid = int(len(normal_merchants) * 0.10)
    n_hybrid = min(n_hybrid, len(eligible))
    hybrid_merchants = random.sample(eligible, n_hybrid)
    for m in hybrid_merchants:
        m["is_hybrid"] = True

    print(f"  Normal merchants: {len(normal_merchants):,} (hybrid: {n_hybrid:,})")
    print(f"  Judol merchants:  {len(judol_merchants):,}")

    return normal_merchants, judol_merchants, hybrid_merchants


# ============================================================
# USER PROFILES (9 normal + 4 judol)
# ============================================================

NORMAL_USER_PROFILES = {
    "regular_worker":   0.35,
    "shift_worker":     0.08,
    "driver_commuter":  0.10,
    "student":          0.10,
    "online_shopper":   0.12,
    "payday_spender":   0.08,
    "family_account":   0.07,
    "business_traveler": 0.04,
    "retiree":          0.03,
    "power_user":       0.03,
}

JUDOL_USER_PROFILES = {
    "casual":  0.30,
    "regular": 0.40,
    "heavy":   0.20,
    "smurfer": 0.10,
}


def create_user_pool(cities, weights, n_normal=200000, n_judol=2400):
    """Create profiled user pools."""
    normal_users = []
    for profile, ratio in NORMAL_USER_PROFILES.items():
        count = int(n_normal * ratio)
        for _ in range(count):
            city, prov = random.choices(cities, weights=weights, k=1)[0]

            # Profile-specific params
            params = {"profile": profile, "city": city, "province": prov,
                      "user_id": gen_user_id()}

            if profile == "regular_worker":
                params["n_favorites"] = random.randint(3, 6)
                params["round_purchase_rate"] = random.uniform(0.08, 0.18)
            elif profile == "shift_worker":
                params["n_favorites"] = random.randint(2, 4)
                params["shift_nights_per_week"] = random.randint(3, 5)
            elif profile == "driver_commuter":
                params["n_favorites"] = random.randint(2, 3)
                params["n_route_merchants"] = random.randint(10, 25)
                params["fuel_per_month"] = random.randint(2, 6)
            elif profile == "student":
                params["n_favorites"] = random.randint(3, 5)
                params["allowance_day"] = 1
            elif profile == "online_shopper":
                params["n_favorites"] = random.randint(2, 4)
                params["midnight_burst_rate"] = random.uniform(0.03, 0.08)
            elif profile == "payday_spender":
                params["n_favorites"] = random.randint(4, 8)
            elif profile == "family_account":
                params["n_favorites"] = random.randint(4, 8)
                # Two time clusters
                params["second_city"] = random.choices(cities, weights=weights, k=1)[0]
            elif profile == "business_traveler":
                params["n_favorites"] = random.randint(2, 3)
                params["n_cities"] = random.randint(2, 4)
                params["travel_cities"] = random.choices(cities, weights=weights,
                                                         k=random.randint(2, 4))
            elif profile == "retiree":
                params["n_favorites"] = random.randint(2, 3)
            elif profile == "power_user":
                params["n_favorites"] = random.randint(8, 15)
                params["round_purchase_rate"] = random.uniform(0.08, 0.18)

            normal_users.append(params)

    while len(normal_users) < n_normal:
        city, prov = random.choices(cities, weights=weights, k=1)[0]
        normal_users.append({
            "profile": "regular_worker", "city": city, "province": prov,
            "user_id": gen_user_id(), "n_favorites": 3, "round_purchase_rate": 0.08,
        })

    # Judol users — ALL also have normal transaction behavior
    judol_users = []
    for profile, ratio in JUDOL_USER_PROFILES.items():
        count = int(n_judol * ratio)
        for _ in range(count):
            city, prov = random.choices(cities, weights=weights, k=1)[0]
            params = {
                "user_id": gen_user_id(), "city": city, "province": prov,
                "judol_type": profile,
                "n_favorites": random.randint(3, 6),
            }

            if profile == "casual":
                params["judol_tx_count"] = random.randint(10, 25)
                params["normal_ratio"] = random.uniform(0.55, 0.75)
                params["escalation"] = False
            elif profile == "regular":
                params["judol_tx_count"] = random.randint(25, 55)
                params["normal_ratio"] = random.uniform(0.45, 0.65)
                params["escalation"] = random.random() < 0.4
            elif profile == "heavy":
                params["judol_tx_count"] = random.randint(50, 120)
                params["normal_ratio"] = random.uniform(0.35, 0.55)
                params["escalation"] = random.random() < 0.5
            elif profile == "smurfer":
                params["judol_tx_count"] = random.randint(25, 80)
                params["normal_ratio"] = random.uniform(0.50, 0.70)
                params["escalation"] = False
                params["smurfing_merchants"] = random.randint(3, 5)

            judol_users.append(params)

    while len(judol_users) < n_judol:
        city, prov = random.choices(cities, weights=weights, k=1)[0]
        judol_users.append({
            "user_id": gen_user_id(), "city": city, "province": prov,
            "judol_type": "casual", "n_favorites": 4,
            "judol_tx_count": 10, "normal_ratio": 0.75, "escalation": False,
        })

    print(f"  Normal users: {len(normal_users):,}")
    print(f"  Judol users:  {len(judol_users):,}")

    return normal_users, judol_users


# ============================================================
# TRANSACTION RECORD BUILDER
# ============================================================

_seq_counter = [0]


def build_record(user, merchant, amount, ts, label):
    _seq_counter[0] += 1
    return {
        "transaction_id": gen_txn_id(ts, _seq_counter[0] % 10000000),
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": user["user_id"],
        "merchant_id": merchant["merchant_id"],
        "amount": int(amount),
        "user_city": user.get("city", ""),
        "user_province": user.get("province", ""),
        "merchant_city": merchant["city"],
        "merchant_province": merchant["province"],
        "transaction_type": "QRIS",
        "device_id": gen_device_id(),
        "is_round_amount": is_round_amount(int(amount)),
        "tx_hour": ts.hour,
        "tx_day_of_week": ts.weekday(),
        "label": label,
    }


# ============================================================
# NORMAL TRANSACTION GENERATION
# ============================================================


def generate_normal_transactions(n_total, normal_merchants, normal_users, hybrid_merchants,
                                 start, end):
    """Generate normal transactions from profiled users at categorized merchants."""
    print(f"  Generating {n_total:,} normal transactions...")
    records = []

    # Index merchants by category
    merchants_by_cat = {}
    for m in normal_merchants:
        merchants_by_cat.setdefault(m["category"], []).append(m)

    # Index merchants by city for repeat customer behavior
    merchants_by_city = {}
    for m in normal_merchants:
        merchants_by_city.setdefault(m["city"], []).append(m)

    # Pre-assign favorite merchants per user
    user_favorites = {}
    for u in normal_users:
        n_favs = u.get("n_favorites", 3)
        same_city = merchants_by_city.get(u["city"], [])
        if len(same_city) >= n_favs:
            user_favorites[u["user_id"]] = random.sample(same_city, n_favs)
        else:
            user_favorites[u["user_id"]] = random.sample(
                normal_merchants, min(n_favs, len(normal_merchants)))

    # Event merchants get burst days
    event_merchants = merchants_by_cat.get("event_seasonal", [])
    event_burst_days = {}
    for m in event_merchants:
        n_events = random.randint(1, 3)
        days_range = (end - start).days
        for _ in range(n_events):
            burst_start = start + timedelta(days=random.randint(0, days_range))
            event_burst_days.setdefault(m["merchant_id"], []).append(burst_start)

    # Seasonal merchants: some are inactive then active
    seasonal_inactive = {}
    for m in merchants_by_cat.get("event_seasonal", []):
        if random.random() < 0.6:
            inactive_days = random.randint(30, 60)
            seasonal_inactive[m["merchant_id"]] = start + timedelta(days=inactive_days)

    # Pre-compute user frequency weights so some normal users transact heavily
    user_freq_weights = []
    for u in normal_users:
        p = u["profile"]
        if p == "power_user":
            w = max(0.5, np.random.gamma(6.0, 2.0))
        elif p == "online_shopper":
            w = max(0.3, np.random.gamma(3.0, 1.5))
        elif p in ("regular_worker", "payday_spender", "family_account"):
            w = max(0.2, np.random.gamma(2.0, 1.0))
        elif p == "driver_commuter":
            w = max(0.3, np.random.gamma(2.5, 1.2))
        else:
            w = max(0.1, np.random.gamma(1.5, 1.0))
        user_freq_weights.append(w)

    # Pre-sample all user indices at once (much faster than per-tx sampling)
    total_w = sum(user_freq_weights)
    probs = np.array(user_freq_weights) / total_w
    user_indices = np.random.choice(len(normal_users), size=n_total, p=probs)

    # Generate transactions — weighted user sampling
    for i in range(n_total):
        user = normal_users[user_indices[i]]
        profile = user["profile"]

        # Pick merchant based on profile
        if random.random() < 0.55:
            # Favorite merchant (repeat customer)
            merchant = random.choice(user_favorites.get(user["user_id"],
                                                         [random.choice(normal_merchants)]))
        else:
            # Random merchant selection weighted by profile
            if profile == "driver_commuter":
                cat = random.choices(
                    ["fuel_station", "food_beverage", "minimarket_24h", "regular_retail", "parking_toll"],
                    weights=[25, 30, 20, 15, 10], k=1)[0]
            elif profile == "student":
                cat = random.choices(
                    ["food_beverage", "small_warung", "minimarket_24h", "regular_retail", "online_ecommerce"],
                    weights=[30, 25, 20, 15, 10], k=1)[0]
            elif profile == "online_shopper":
                cat = random.choices(
                    ["online_ecommerce", "regular_retail", "food_beverage", "minimarket_24h"],
                    weights=[40, 25, 20, 15], k=1)[0]
            elif profile == "payday_spender":
                cat = random.choices(
                    ["regular_retail", "food_beverage", "minimarket_24h", "online_ecommerce", "fuel_station"],
                    weights=[30, 25, 20, 15, 10], k=1)[0]
            elif profile == "shift_worker":
                cat = random.choices(
                    ["minimarket_24h", "food_beverage", "fuel_station", "regular_retail"],
                    weights=[35, 30, 15, 20], k=1)[0]
            elif profile == "business_traveler":
                cat = random.choices(
                    ["food_beverage", "regular_retail", "fuel_station", "parking_toll", "online_ecommerce"],
                    weights=[30, 25, 20, 15, 10], k=1)[0]
            elif profile == "retiree":
                cat = random.choices(
                    ["small_warung", "food_beverage", "regular_retail", "minimarket_24h"],
                    weights=[30, 30, 25, 15], k=1)[0]
            else:  # regular_worker, family_account, power_user
                cat = random.choices(
                    list(NORMAL_MERCHANT_CATS.keys()),
                    weights=list(NORMAL_MERCHANT_CATS.values()), k=1)[0]

            pool = merchants_by_cat.get(cat, normal_merchants)
            merchant = random.choice(pool) if pool else random.choice(normal_merchants)

        # Amount
        user["_current_merchant_cat"] = merchant["category"]

        # Profile-specific round purchase injection
        if profile in ("regular_worker", "payday_spender", "family_account", "power_user"):
            if random.random() < user.get("round_purchase_rate", 0.08):
                amount = gen_normal_round_purchase()
            else:
                amount = gen_normal_amount(user)
        elif profile == "student" and random.random() < 0.15:
            # Pulsa top-up
            amount = random.choice([10000, 25000, 50000, 100000])
        elif profile == "driver_commuter" and merchant["category"] == "fuel_station":
            amount = random.choice([50000, 100000, 150000, 200000])
        else:
            amount = gen_normal_amount(user)

        # Timestamp based on profile
        if profile == "shift_worker":
            hour_w = HOUR_WEIGHTS_SHIFT
        elif profile == "online_shopper":
            hour_w = HOUR_WEIGHTS_ONLINE
        elif profile == "family_account":
            # Dual pattern: 50% daytime (one person), 50% evening (other)
            if random.random() < 0.5:
                hour_w = HOUR_WEIGHTS["regular_retail"]
            else:
                hour_w = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 3, 5, 8, 10, 9, 6, 4, 2]
        else:
            hour_w = HOUR_WEIGHTS.get(merchant["category"], HOUR_WEIGHTS["regular_retail"])

        # Gajian boost for normal: +80-100%
        gajian_boost = 1.9 if profile == "payday_spender" else 1.8

        # Student allowance pattern: heavy first 5 days of month
        if profile == "student":
            ts = gen_timestamp(start, end, hour_w, NORMAL_DAY_WEIGHTS, gajian_boost=1.3)
            if random.random() < 0.4:
                # Force to first 5 days of a month
                month_start = ts.replace(day=1)
                ts = month_start + timedelta(days=random.randint(0, 4),
                                              hours=random.randint(8, 20),
                                              minutes=random.randint(0, 59))
                ts = max(start, min(ts, end))
        else:
            ts = gen_timestamp(start, end, hour_w, NORMAL_DAY_WEIGHTS, gajian_boost)

        records.append(build_record(user, merchant, amount, ts, label=0))

        if (i + 1) % 500000 == 0:
            print(f"    ... {i+1:,} normal done")

    return records


# ============================================================
# JUDOL TRANSACTION GENERATION
# ============================================================


def generate_judol_transactions(n_judol_label1, judol_users, judol_merchants,
                                 hybrid_merchants, normal_merchants, start, end):
    """Generate judol + normal transactions for judol users.
    Returns (judol_records, normal_records_from_judol_users)."""
    print(f"  Target judol (label=1) transactions: {n_judol_label1:,}")
    judol_records = []
    normal_records = []

    # Index normal merchants for cross-visiting
    norm_by_city = {}
    for m in normal_merchants:
        norm_by_city.setdefault(m["city"], []).append(m)

    # Assign favorite normal merchants to each judol user
    judol_user_norm_favs = {}
    for u in judol_users:
        n_favs = u.get("n_favorites", 4)
        same_city = norm_by_city.get(u["city"], [])
        if len(same_city) >= n_favs:
            judol_user_norm_favs[u["user_id"]] = random.sample(same_city, n_favs)
        else:
            judol_user_norm_favs[u["user_id"]] = random.sample(
                normal_merchants, min(n_favs, len(normal_merchants)))

    # Distribute judol label=1 transactions across users proportionally
    total_planned = sum(u["judol_tx_count"] for u in judol_users)
    scale = n_judol_label1 / max(total_planned, 1)

    for user in judol_users:
        planned_judol = max(1, int(user["judol_tx_count"] * scale))
        normal_ratio = user.get("normal_ratio", 0.50)
        planned_normal = int(planned_judol * normal_ratio / (1.0 - normal_ratio + 1e-9))

        jtype = user["judol_type"]

        # Pick judol merchants for this user
        if jtype == "smurfer":
            n_jm = user.get("smurfing_merchants", 4)
            user_judol_merchants = random.sample(
                judol_merchants, min(n_jm, len(judol_merchants)))
            # Also use some hybrid merchants
            if hybrid_merchants:
                n_hm = random.randint(1, 2)
                user_judol_merchants += random.sample(
                    hybrid_merchants, min(n_hm, len(hybrid_merchants)))
        else:
            n_jm = random.randint(1, 4)
            user_judol_merchants = random.sample(
                judol_merchants, min(n_jm, len(judol_merchants)))
            # Some users also deposit via hybrid merchants
            if hybrid_merchants and random.random() < 0.45:
                user_judol_merchants.append(random.choice(hybrid_merchants))

        # Generate judol transactions (label=1)
        for j in range(planned_judol):
            merchant = random.choice(user_judol_merchants)
            amount = gen_judol_amount(user, j, planned_judol)

            # Timing: 24/7 with night peak
            if merchant.get("category") == "togel":
                if random.random() < 0.5:
                    ts = gen_togel_timestamp(start, end)
                else:
                    ts = gen_timestamp(start, end, JUDOL_HOUR_WEIGHTS,
                                       JUDOL_DAY_WEIGHTS, gajian_boost=1.7)
            else:
                ts = gen_timestamp(start, end, JUDOL_HOUR_WEIGHTS,
                                   JUDOL_DAY_WEIGHTS, gajian_boost=1.7)

            # Smurfing: cluster transactions in short windows
            if jtype == "smurfer" and j > 0 and random.random() < 0.6:
                prev_ts = judol_records[-1]["timestamp"] if judol_records else None
                if prev_ts:
                    prev_dt = datetime.strptime(prev_ts, "%Y-%m-%d %H:%M:%S")
                    ts = prev_dt + timedelta(minutes=random.randint(2, 45))
                    ts = min(ts, end)
                    merchant = random.choice(user_judol_merchants)

            judol_records.append(build_record(user, merchant, amount, ts, label=1))

        # Generate normal transactions for this judol user (label=0)
        user_norm_merchants = judol_user_norm_favs.get(user["user_id"], [])
        for _ in range(planned_normal):
            if user_norm_merchants and random.random() < 0.6:
                merchant = random.choice(user_norm_merchants)
            else:
                merchant = random.choice(normal_merchants)

            user["_current_merchant_cat"] = merchant["category"]
            amount = gen_normal_amount(user)
            hour_w = HOUR_WEIGHTS.get(merchant["category"], HOUR_WEIGHTS["regular_retail"])
            ts = gen_timestamp(start, end, hour_w, NORMAL_DAY_WEIGHTS, gajian_boost=1.8)

            normal_records.append(build_record(user, merchant, amount, ts, label=0))

    print(f"  Generated {len(judol_records):,} judol (label=1)")
    print(f"  Generated {len(normal_records):,} normal from judol users (label=0)")

    return judol_records, normal_records


# ============================================================
# MAIN
# ============================================================


def generate_full_dataset(total_rows=2000000, fraud_rate=0.03):
    print("=" * 60)
    print("Pantau Synthetic Dataset Generator v2")
    print("=" * 60)
    print(f"  Total rows: {total_rows:,}")
    print(f"  Fraud rate: {fraud_rate*100:.1f}%")

    n_judol_label1 = int(total_rows * fraud_rate)
    n_normal_total = total_rows - n_judol_label1
    print(f"  Target label=1: {n_judol_label1:,}")
    print(f"  Target label=0: {n_normal_total:,}")

    # Scale pools proportionally
    n_normal_merchants = max(5000, int(20000 * (total_rows / 2000000)))
    n_judol_merchants = max(100, int(600 * (total_rows / 2000000)))
    n_normal_users = max(50000, int(200000 * (total_rows / 2000000)))
    n_judol_users = max(200, int(2400 * (total_rows / 2000000)))

    # [1] Load geolocation
    print("\n[1/5] Loading wilayah data...")
    cities = load_wilayah()
    weights = build_city_weights(cities)
    print(f"  {len(cities)} kab/kota, {len(set(p for _, p in cities))} provinces")

    end_date = datetime(2026, 3, 1)
    start_date = end_date - timedelta(days=90)
    print(f"  Date range: {start_date.date()} → {end_date.date()}")

    # [2] Create pools
    print("\n[2/5] Creating entity pools...")
    normal_merchants, judol_merchants, hybrid_merchants = create_merchant_pool(
        cities, weights, n_normal=n_normal_merchants, n_judol=n_judol_merchants)
    normal_users, judol_users = create_user_pool(
        cities, weights, n_normal=n_normal_users, n_judol=n_judol_users)

    # [3] Generate judol transactions (+ normal from judol users)
    print("\n[3/5] Generating judol user transactions...")
    judol_recs, judol_normal_recs = generate_judol_transactions(
        n_judol_label1, judol_users, judol_merchants, hybrid_merchants,
        normal_merchants, start_date, end_date)

    # [4] Generate remaining normal transactions
    n_remaining_normal = n_normal_total - len(judol_normal_recs)
    n_remaining_normal = max(0, n_remaining_normal)
    print(f"\n[4/5] Generating {n_remaining_normal:,} normal transactions from normal users...")
    normal_recs = generate_normal_transactions(
        n_remaining_normal, normal_merchants, normal_users, hybrid_merchants,
        start_date, end_date)

    # [5] Combine and shuffle
    print("\n[5/5] Combining and shuffling...")
    all_recs = normal_recs + judol_normal_recs + judol_recs
    df = pd.DataFrame(all_recs)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    n0 = (df["label"] == 0).sum()
    n1 = (df["label"] == 1).sum()
    print(f"Total rows:       {len(df):,}")
    print(f"Normal (label=0): {n0:,} ({n0/len(df)*100:.1f}%)")
    print(f"Judol  (label=1): {n1:,} ({n1/len(df)*100:.1f}%)")
    print(f"Unique users:     {df['user_id'].nunique():,}")
    print(f"Unique merchants: {df['merchant_id'].nunique():,}")
    print(f"Amount range:     Rp{df['amount'].min():,.0f} — Rp{df['amount'].max():,.0f}")

    # Overlap stats
    judol_user_ids = set(u["user_id"] for u in judol_users)
    judol_users_with_normal = df[(df["user_id"].isin(judol_user_ids)) & (df["label"] == 0)]
    judol_users_with_judol = df[(df["user_id"].isin(judol_user_ids)) & (df["label"] == 1)]
    print(f"\n--- Overlap Statistics ---")
    print(f"  Judol users with label=0 tx: {judol_users_with_normal['user_id'].nunique():,}")
    print(f"  Judol users label=0 tx count: {len(judol_users_with_normal):,}")
    print(f"  Judol users label=1 tx count: {len(judol_users_with_judol):,}")
    if len(judol_users_with_normal) + len(judol_users_with_judol) > 0:
        mix_pct = len(judol_users_with_normal) / (len(judol_users_with_normal) + len(judol_users_with_judol)) * 100
        print(f"  Judol user normal tx ratio:   {mix_pct:.1f}%")

    hybrid_ids = set(m["merchant_id"] for m in hybrid_merchants)
    hybrid_txs = df[df["merchant_id"].isin(hybrid_ids)]
    if len(hybrid_txs) > 0:
        h0 = (hybrid_txs["label"] == 0).sum()
        h1 = (hybrid_txs["label"] == 1).sum()
        print(f"  Hybrid merchant txs: {len(hybrid_txs):,} (normal={h0:,}, judol={h1:,})")

    # Temporal
    print(f"\n--- Temporal Distribution ---")
    df_ts = pd.to_datetime(df["timestamp"])
    judol_df = df[df["label"] == 1]
    judol_ts = pd.to_datetime(judol_df["timestamp"])
    prime = judol_ts.dt.hour.isin([20, 21, 22, 23, 0, 1, 2, 3]).sum()
    print(f"  Judol prime time (20-03): {prime:,}/{len(judol_df):,} ({prime/max(len(judol_df),1)*100:.1f}%)")
    normal_df = df[df["label"] == 0]
    normal_ts = pd.to_datetime(normal_df["timestamp"])
    normal_night = normal_ts.dt.hour.isin([20, 21, 22, 23, 0, 1, 2, 3, 4, 5]).sum()
    print(f"  Normal nighttime (20-05): {normal_night:,}/{len(normal_df):,} ({normal_night/max(len(normal_df),1)*100:.1f}%)")

    # Round amount stats
    print(f"\n--- Round Amount Distribution ---")
    normal_round = df[(df["label"] == 0) & (df["is_round_amount"] == True)]
    judol_round = df[(df["label"] == 1) & (df["is_round_amount"] == True)]
    print(f"  Normal round rate: {len(normal_round)/max(n0,1)*100:.1f}%")
    print(f"  Judol round rate:  {len(judol_round)/max(n1,1)*100:.1f}%")

    # Hour distribution check: verify no hour has 0 judol
    hour_counts = judol_df["tx_hour"].value_counts().sort_index()
    zero_hours = [h for h in range(24) if h not in hour_counts.index]
    if zero_hours:
        print(f"  ⚠️  Hours with zero judol: {zero_hours}")
    else:
        print(f"  ✓ Judol present in all 24 hours")

    # Save
    output_dir = os.path.join(DATA_DIR, "generated", "parametric")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pantau_dataset.csv")
    df.to_csv(output_path, index=False)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nSaved: {output_path} ({size_mb:.1f} MB)")

    # Samples
    print("\n--- Sample normal (label=0) ---")
    print(df[df["label"] == 0].iloc[0].to_string())
    print("\n--- Sample judol (label=1) ---")
    print(df[df["label"] == 1].iloc[0].to_string())

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pantau Dataset Generator v2")
    parser.add_argument("--rows", type=int, default=2000000, help="Total transactions")
    parser.add_argument("--fraud-rate", type=float, default=0.03, help="Fraud rate (0.01-0.05)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    SEED = args.seed

    generate_full_dataset(total_rows=args.rows, fraud_rate=args.fraud_rate)
