"""
Pantau Synthetic Dataset Generator
===================================
Generates 500,000 transactions (85% normal, 15% judol) calibrated from:
- IBM AML dataset (fraud rate, amount distribution, layering patterns)
- PaySim dataset (hourly distribution, repeat rate, geo radius)
- Bustabit dataset (gambling session patterns, deposit cycles)
- PPATK statistics (judol hours, round amounts, geo spread)
- Real Indonesian wilayah data (38 provinces, 514 kab/kota)

Output: data/generated/pantau_dataset.csv
"""

import pandas as pd
import numpy as np
import csv
import random
import string
from datetime import datetime, timedelta
import os

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# LOAD REAL INDONESIAN WILAYAH DATA (38 provinces, 514 kab/kota)
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_wilayah() -> list:
    """
    Load provinces and kab/kota from CSV files.
    Returns list of (kota_name, province_name) tuples.
    Links via ID prefix: kota "12.71" → province "12".
    """
    provinces = {}
    with open(os.path.join(DATA_DIR, "geolocation", "provinsi.csv"), encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            provinces[row["id"].strip('"')] = row["name"].strip('"')

    cities = []
    with open(os.path.join(DATA_DIR, "geolocation", "kabupaten_kota.csv"), encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kota_id = row["id"].strip('"')
            kota_name = row["name"].strip('"')
            prov_id = kota_id.split(".")[0]
            prov_name = provinces.get(prov_id, "Unknown")
            cities.append((kota_name, prov_name))

    return cities


# Major urban centers get higher transaction weight
# Based on approximate population ranking of Indonesian cities
MAJOR_CITIES = {
    "Jakarta Selatan", "Jakarta Pusat", "Jakarta Barat", "Jakarta Timur", "Jakarta Utara",
    "Surabaya", "Bandung", "Medan", "Semarang", "Makassar",
    "Palembang", "Tangerang", "Tangerang Selatan", "Depok", "Bekasi",
    "Yogyakarta", "Denpasar", "Balikpapan", "Malang", "Bogor",
}


def build_city_weights(cities: list) -> list:
    """Assign weights: major cities get 5x, others get 1."""
    weights = []
    for city_name, _ in cities:
        if city_name in MAJOR_CITIES:
            weights.append(5)
        else:
            weights.append(1)
    return weights


# ============================================================
# CALIBRATED PARAMETERS (from base datasets + PPATK)
# ============================================================

# Normal regular merchants: business hours peak, weekday-heavy
NORMAL_HOUR_WEIGHTS_REGULAR = [1, 1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 9, 10, 9, 7, 6, 5, 7, 8, 6, 4, 3, 2, 1]
# 24-hour merchants (Indomaret, Alfamart, etc.): always open, slight night dip
NORMAL_HOUR_WEIGHTS_24H = [3, 2, 2, 2, 2, 2, 3, 5, 7, 7, 7, 8, 9, 8, 7, 6, 6, 7, 8, 8, 7, 6, 5, 4]
# 30% of normal merchants are 24-hour type
NORMAL_24H_MERCHANT_RATIO = 0.30
#                  Mon Tue Wed Thu Fri Sat Sun
NORMAL_DAY_WEIGHTS = [15, 15, 15, 15, 15, 13, 12]

# Judol: prime time 20:00-02:00, weekend-heavy, with togel draw spikes
JUDOL_HOUR_WEIGHTS_SLOT = [8, 7, 6, 4, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 4, 5, 6, 7, 9, 10]
# Togel draw times: 13:00, 16:00, 19:00, 22:00 — deposits 10-30 min before
TOGEL_DRAW_HOURS = [13, 16, 19, 22]
#                  Mon Tue Wed Thu Fri Sat Sun
JUDOL_DAY_WEIGHTS = [10, 10, 10, 12, 18, 22, 18]

# Gajian dates: salary spike on these days of month (PNS + swasta)
GAJIAN_DATES = {1, 25, 26, 27, 28}

# ============================================================
# E-WALLET & PHONE NUMBER CONFIG
# ============================================================

EWALLET_PROVIDERS = ["DANA", "OVO", "GOPAY", "LINKAJA"]

# Indonesian mobile operator prefixes
PHONE_PREFIXES = [
    # Telkomsel
    "0852", "0853", "0811", "0812", "0813", "0821", "0822", "0851",
    # Indosat Ooredoo
    "0857", "0856",
    # Tri
    "0896", "0895", "0897", "0898", "0899",
    # XL
    "0817", "0818", "0819", "0859", "0877", "0878",
    # AXIS
    "0832", "0833", "0838",
    # Smartfren
    "0881", "0882", "0883", "0884", "0885", "0886", "0887", "0888", "0889",
]

# Transaction type distribution
# Normal: 70% QRIS, 30% e-wallet
NORMAL_EWALLET_RATIO = 0.30
# Judol: 60% QRIS, 40% e-wallet (e-wallet popular for judol deposits too)
JUDOL_EWALLET_RATIO = 0.40

# Normal transaction params (from PaySim + IBM AML)
NORMAL_PARAMS = {
    "amount_mean": 125000,
    "amount_std": 180000,
    "amount_min": 1000,
    "amount_max": 5000000,
    "round_amount_rate": 0.12,
    "repeat_merchant_rate": 0.62,
}

# Judol merchant params (from PPATK + Bustabit)
JUDOL_MERCHANT_PARAMS = {
    "amount_choices": [10000, 20000, 25000, 50000, 100000, 200000, 500000],
    "amount_weights": [0.10, 0.10, 0.05, 0.35, 0.25, 0.10, 0.05],
    "round_amount_rate": 0.91,
    "repeat_sender_rate": 0.02,
    "txs_per_merchant_range": (200, 2000),
    "togel_ratio": 0.30,  # 30% of judol merchant txs are togel-style
}

# Judol user params (from Bustabit dataset)
JUDOL_USER_PARAMS = {
    "escalation_rate": 0.68,
    "late_night_rate": 0.72,
    "round_amount_rate": 0.88,
    "deposits_per_session": (2, 8),
    "txs_per_user_range": (20, 200),
    "togel_ratio": 0.30,
}

# ============================================================
# ID GENERATORS
# ============================================================


def generate_transaction_id(date: datetime, seq: int) -> str:
    """Format: TXN-YYYYMMDD-NNNNN"""
    return f"TXN-{date.strftime('%Y%m%d')}-{seq:05d}"


def generate_user_account() -> str:
    """Indonesian bank account number: 10-15 digits."""
    length = random.randint(10, 15)
    return "".join([str(random.randint(0, 9)) for _ in range(length)])


def generate_nmid() -> str:
    """
    NMID (National Merchant ID): 15-character alphanumeric.
    Format: ID + 2-digit acquirer code + 11 alphanumeric chars.
    Example: ID1022003150001
    """
    prefix = "ID"
    acquirer = str(random.randint(10, 99))
    rest = "".join(random.choices(string.digits + string.ascii_uppercase, k=11))
    return prefix + acquirer + rest


def generate_device_id() -> str:
    """Device fingerprint: DEV-10 hex chars."""
    hex_chars = "".join(random.choices("0123456789ABCDEF", k=10))
    return f"DEV-{hex_chars}"


def generate_phone_number() -> str:
    """Indonesian phone number with real operator prefix. Total 11-13 digits."""
    prefix = random.choice(PHONE_PREFIXES)
    remaining = random.randint(7, 9)  # total length 11-13 digits
    suffix = "".join([str(random.randint(0, 9)) for _ in range(remaining)])
    return prefix + suffix


def generate_ewallet_id() -> str:
    """E-wallet receiver format: PROVIDER-08XXXXXXXXXX"""
    provider = random.choice(EWALLET_PROVIDERS)
    phone = generate_phone_number()
    return f"{provider}-{phone}"


def pick_transaction_type(ewallet_ratio: float) -> tuple:
    """
    Returns (transaction_type, merchant_id_override).
    For QRIS: use NMID. For e-wallet: use PROVIDER-phone format.
    """
    if random.random() < ewallet_ratio:
        ewallet_id = generate_ewallet_id()
        provider = ewallet_id.split("-")[0]
        return f"EWALLET_{provider}", ewallet_id
    else:
        return "QRIS", None


# ============================================================
# TIMESTAMP GENERATORS
# ============================================================


def generate_normal_timestamp(start_date: datetime, end_date: datetime, is_24h_merchant: bool = False) -> datetime:
    """Normal transaction: business hours or 24h pattern, weekday-heavy, gajian spike."""
    days_range = (end_date - start_date).days
    base = start_date + timedelta(days=random.randint(0, days_range))

    # Day-of-week weighting + gajian boost for normal spending too
    for _ in range(100):
        candidate = base + timedelta(days=random.randint(-3, 3))
        candidate = max(start_date, min(candidate, end_date))
        dow = candidate.weekday()
        weight = NORMAL_DAY_WEIGHTS[dow]
        # Gajian boost: +30% normal spending on salary dates (bills, shopping)
        if candidate.day in GAJIAN_DATES:
            weight = int(weight * 1.3)
        if random.random() < weight / (max(NORMAL_DAY_WEIGHTS) * 1.3):
            base = candidate
            break

    hour_weights = NORMAL_HOUR_WEIGHTS_24H if is_24h_merchant else NORMAL_HOUR_WEIGHTS_REGULAR
    hour = random.choices(range(24), weights=hour_weights, k=1)[0]
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return base.replace(hour=hour, minute=minute, second=second)


def generate_judol_timestamp_slot(start_date: datetime, end_date: datetime) -> datetime:
    """Slot/casino judol: prime time 20:00-02:00, weekend-heavy, gajian spike."""
    days_range = (end_date - start_date).days
    base = start_date + timedelta(days=random.randint(0, days_range))

    # Day-of-week weighting (weekend-heavy)
    for _ in range(100):
        candidate = base + timedelta(days=random.randint(-3, 3))
        candidate = max(start_date, min(candidate, end_date))
        dow = candidate.weekday()
        weight = JUDOL_DAY_WEIGHTS[dow]
        # Gajian boost: +50% volume on salary dates
        if candidate.day in GAJIAN_DATES:
            weight = int(weight * 1.5)
        if random.random() < weight / (max(JUDOL_DAY_WEIGHTS) * 1.5):
            base = candidate
            break

    hour = random.choices(range(24), weights=JUDOL_HOUR_WEIGHTS_SLOT, k=1)[0]
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return base.replace(hour=hour, minute=minute, second=second)


def generate_judol_timestamp_togel(start_date: datetime, end_date: datetime) -> datetime:
    """Togel judol: deposits 10-30 min before draw times (13, 16, 19, 22)."""
    days_range = (end_date - start_date).days
    base = start_date + timedelta(days=random.randint(0, days_range))

    # Day-of-week weighting
    for _ in range(100):
        candidate = base + timedelta(days=random.randint(-3, 3))
        candidate = max(start_date, min(candidate, end_date))
        dow = candidate.weekday()
        if random.random() < JUDOL_DAY_WEIGHTS[dow] / max(JUDOL_DAY_WEIGHTS):
            base = candidate
            break

    # Pick a draw time, deposit 10-30 min before
    draw_hour = random.choice(TOGEL_DRAW_HOURS)
    minutes_before = random.randint(10, 30)
    ts = base.replace(hour=draw_hour, minute=0, second=0) - timedelta(minutes=minutes_before)
    ts = ts.replace(second=random.randint(0, 59))
    return ts


def generate_judol_timestamp(start_date: datetime, end_date: datetime, togel_ratio: float = 0.30) -> datetime:
    """Combined judol timestamp: mix of slot/casino and togel patterns."""
    if random.random() < togel_ratio:
        return generate_judol_timestamp_togel(start_date, end_date)
    else:
        return generate_judol_timestamp_slot(start_date, end_date)


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def pick_city_weighted(cities: list, weights: list) -> tuple:
    """Pick a city weighted by population."""
    return random.choices(cities, weights=weights, k=1)[0]


def pick_city_random(cities: list) -> tuple:
    """Pick a city uniformly (for judol geo spread)."""
    return random.choice(cities)


def generate_normal_amount() -> int:
    """
    Normal transaction amount in Rupiah.
    Log-normal distribution calibrated from PaySim + IBM AML.
    """
    amount = int(np.random.lognormal(mean=11.0, sigma=0.8))
    amount = max(NORMAL_PARAMS["amount_min"], min(amount, NORMAL_PARAMS["amount_max"]))

    # 12% chance of rounding to nearest 10k/50k
    if random.random() < NORMAL_PARAMS["round_amount_rate"]:
        round_to = random.choice([10000, 50000])
        amount = max(round_to, round(amount / round_to) * round_to)

    return amount


def generate_judol_amount() -> int:
    """Judol deposit amount: weighted choice of common denominations."""
    return random.choices(
        JUDOL_MERCHANT_PARAMS["amount_choices"],
        weights=JUDOL_MERCHANT_PARAMS["amount_weights"],
        k=1,
    )[0]


def is_round_amount(amount: int) -> bool:
    return amount % 10000 == 0


# ============================================================
# POOL GENERATORS
# ============================================================


def create_merchant_pools(cities, weights, n_normal=5000, n_judol=500) -> tuple:
    """Pre-generate merchant pools with fixed city assignments. All QRIS."""
    normal_merchants = []
    for _ in range(n_normal):
        city, province = pick_city_weighted(cities, weights)
        is_24h = random.random() < NORMAL_24H_MERCHANT_RATIO
        normal_merchants.append(
            {"merchant_id": generate_nmid(), "city": city, "province": province,
             "is_24h": is_24h, "tx_type": "QRIS"}
        )

    judol_merchants = []
    for _ in range(n_judol):
        city, province = pick_city_weighted(cities, weights)
        judol_merchants.append(
            {"merchant_id": generate_nmid(), "city": city, "province": province,
             "tx_type": "QRIS"}
        )

    return normal_merchants, judol_merchants


def create_user_pools(cities, weights, n_normal=50000, n_judol=2000) -> tuple:
    """Pre-generate user pools with fixed city assignments."""
    normal_users = []
    for _ in range(n_normal):
        city, province = pick_city_weighted(cities, weights)
        normal_users.append(
            {"user_id": generate_user_account(), "city": city, "province": province}
        )

    judol_users = []
    for _ in range(n_judol):
        city, province = pick_city_random(cities)
        judol_users.append(
            {"user_id": generate_user_account(), "city": city, "province": province}
        )

    return normal_users, judol_users


# ============================================================
# NORMAL TRANSACTION GENERATOR (425,000 rows)
# ============================================================


def generate_normal_transactions(
    n: int,
    normal_merchants: list,
    normal_users: list,
    start_date: datetime,
    end_date: datetime,
) -> list:
    print(f"  Generating {n:,} normal transactions...")
    records = []

    # Build user->merchant affinity (repeat customer behavior)
    # Each user has 2-5 "favorite" merchants they visit regularly
    # Pre-index merchants by city for fast lookup
    merchants_by_city = {}
    for m in normal_merchants:
        merchants_by_city.setdefault(m["city"], []).append(m)

    user_favorites = {}
    for user in normal_users:
        n_favs = random.randint(2, 5)
        same_city = merchants_by_city.get(user["city"], [])
        if len(same_city) >= n_favs:
            user_favorites[user["user_id"]] = random.sample(same_city, n_favs)
        else:
            user_favorites[user["user_id"]] = random.sample(
                normal_merchants, min(n_favs, len(normal_merchants))
            )

    seq = 0
    for _ in range(n):
        user = random.choice(normal_users)

        # 62% chance: go to a favorite merchant (repeat customer)
        if random.random() < NORMAL_PARAMS["repeat_merchant_rate"]:
            merchant = random.choice(user_favorites[user["user_id"]])
        else:
            merchant = random.choice(normal_merchants)

        amount = generate_normal_amount()
        ts = generate_normal_timestamp(start_date, end_date, is_24h_merchant=merchant.get("is_24h", False))
        tx_type = merchant.get("tx_type", "QRIS")
        seq += 1

        records.append(
            {
                "transaction_id": generate_transaction_id(ts, seq % 100000),
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "user_id": user["user_id"],
                "merchant_id": merchant["merchant_id"],
                "amount": amount,
                "user_city": user["city"],
                "user_province": user["province"],
                "merchant_city": merchant["city"],
                "merchant_province": merchant["province"],
                "transaction_type": tx_type,
                "device_id": generate_device_id(),
                "is_round_amount": is_round_amount(amount),
                "tx_hour": ts.hour,
                "tx_day_of_week": ts.weekday(),
                "label": 0,
            }
        )

    return records


# ============================================================
# JUDOL MERCHANT TRANSACTION GENERATOR (50,000 rows)
# Simulates: many unique users depositing to judol payment gateways
# ============================================================


def generate_judol_merchant_transactions(
    n: int,
    judol_merchants: list,
    cities: list,
    start_date: datetime,
    end_date: datetime,
) -> list:
    print(f"  Generating {n:,} judol merchant transactions...")
    records = []
    seq = 0

    # Each judol merchant gets a burst of transactions
    txs_per_merchant = []
    remaining = n
    for i, merchant in enumerate(judol_merchants):
        if i == len(judol_merchants) - 1:
            count = remaining
        else:
            lo, hi = JUDOL_MERCHANT_PARAMS["txs_per_merchant_range"]
            count = min(random.randint(lo, hi), remaining)
        txs_per_merchant.append(count)
        remaining -= count
        if remaining <= 0:
            break

    for i, merchant in enumerate(judol_merchants):
        if i >= len(txs_per_merchant):
            break
        count = txs_per_merchant[i]

        # Judol merchants have a short active window (dormant → spike)
        dormant_end = start_date + timedelta(days=random.randint(1, 60))
        active_days = random.randint(1, 14)
        active_start = dormant_end
        active_end = min(active_start + timedelta(days=active_days), end_date)

        # Generate a pool of unique senders (near-zero repeat)
        sender_pool_size = max(count, int(count * 0.98))
        sender_pool = []
        for _ in range(sender_pool_size):
            city, province = pick_city_random(cities)  # nationwide geo spread
            sender_pool.append(
                {
                    "user_id": generate_user_account(),
                    "city": city,
                    "province": province,
                }
            )

        for _ in range(count):
            # 2% repeat sender, 98% unique
            if random.random() < JUDOL_MERCHANT_PARAMS["repeat_sender_rate"] and len(sender_pool) > 1:
                sender = random.choice(sender_pool[:10])
            else:
                sender = random.choice(sender_pool)

            amount = generate_judol_amount()
            ts = generate_judol_timestamp(
                active_start, active_end,
                togel_ratio=JUDOL_MERCHANT_PARAMS["togel_ratio"],
            )
            tx_type = merchant.get("tx_type", "QRIS")
            seq += 1

            records.append(
                {
                    "transaction_id": generate_transaction_id(ts, seq % 100000),
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "user_id": sender["user_id"],
                    "merchant_id": merchant["merchant_id"],
                    "amount": amount,
                    "user_city": sender["city"],
                    "user_province": sender["province"],
                    "merchant_city": merchant["city"],
                    "merchant_province": merchant["province"],
                    "transaction_type": tx_type,
                    "device_id": generate_device_id(),
                    "is_round_amount": is_round_amount(amount),
                    "tx_hour": ts.hour,
                    "tx_day_of_week": ts.weekday(),
                    "label": 1,
                }
            )

    return records


# ============================================================
# JUDOL USER TRANSACTION GENERATOR (25,000 rows)
# Simulates: gambling users with deposit cycles & escalation
# ============================================================


def generate_judol_user_transactions(
    n: int,
    judol_users: list,
    normal_merchants: list,
    judol_merchants: list,
    start_date: datetime,
    end_date: datetime,
) -> list:
    print(f"  Generating {n:,} judol user transactions...")
    records = []
    seq = 0

    # Distribute transactions across judol users
    txs_per_user = []
    remaining = n
    for i, user in enumerate(judol_users):
        if i == len(judol_users) - 1:
            count = remaining
        else:
            lo, hi = JUDOL_USER_PARAMS["txs_per_user_range"]
            count = min(random.randint(lo, hi), remaining)
        txs_per_user.append(count)
        remaining -= count
        if remaining <= 0:
            break

    for i, user in enumerate(judol_users):
        if i >= len(txs_per_user):
            break
        count = txs_per_user[i]

        # Each user has gambling sessions
        base_amount = random.choice([10000, 20000, 50000, 100000])
        session_merchants = random.sample(
            judol_merchants, min(random.randint(3, 10), len(judol_merchants))
        )

        for j in range(count):
            merchant = random.choice(session_merchants)

            # Escalation pattern: amounts tend to increase over time
            if random.random() < JUDOL_USER_PARAMS["escalation_rate"]:
                escalation = 1.0 + (j / max(count, 1)) * 2.0
                amount = int(base_amount * escalation)
                amount = max(10000, round(amount / 10000) * 10000)
            else:
                amount = generate_judol_amount()

            ts = generate_judol_timestamp(
                start_date, end_date,
                togel_ratio=JUDOL_USER_PARAMS["togel_ratio"],
            )
            tx_type = merchant.get("tx_type", "QRIS")
            seq += 1

            records.append(
                {
                    "transaction_id": generate_transaction_id(ts, seq % 100000),
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "user_id": user["user_id"],
                    "merchant_id": merchant["merchant_id"],
                    "amount": amount,
                    "user_city": user["city"],
                    "user_province": user["province"],
                    "merchant_city": merchant["city"],
                    "merchant_province": merchant["province"],
                    "transaction_type": tx_type,
                    "device_id": generate_device_id(),
                    "is_round_amount": is_round_amount(amount),
                    "tx_hour": ts.hour,
                    "tx_day_of_week": ts.weekday(),
                    "label": 1,
                }
            )

    return records


# ============================================================
# MAIN: GENERATE FULL DATASET
# ============================================================


def generate_full_dataset():
    print("=" * 60)
    print("Pantau Synthetic Dataset Generator")
    print("=" * 60)

    # Load real Indonesian wilayah data
    print("\n[1/6] Loading wilayah data...")
    cities = load_wilayah()
    weights = build_city_weights(cities)
    print(f"  Loaded {len(cities)} kab/kota across {len(set(p for _, p in cities))} provinces")

    # Time range: 90 days
    end_date = datetime(2026, 3, 1)
    start_date = end_date - timedelta(days=90)
    print(f"  Date range: {start_date.date()} → {end_date.date()}")

    # Create entity pools
    print("\n[2/6] Creating entity pools...")
    normal_merchants, judol_merchants = create_merchant_pools(
        cities, weights, n_normal=5000, n_judol=500
    )
    normal_users, judol_users = create_user_pools(
        cities, weights, n_normal=50000, n_judol=2000
    )
    print(f"  Normal merchants: {len(normal_merchants):,}")
    print(f"  Judol merchants:  {len(judol_merchants):,}")
    print(f"  Normal users:     {len(normal_users):,}")
    print(f"  Judol users:      {len(judol_users):,}")

    # Generate transactions
    print("\n[3/6] Generating normal transactions...")
    normal_txs = generate_normal_transactions(
        425000, normal_merchants, normal_users, start_date, end_date
    )

    print("\n[4/6] Generating judol merchant transactions...")
    judol_merchant_txs = generate_judol_merchant_transactions(
        50000, judol_merchants, cities, start_date, end_date
    )

    print("\n[5/6] Generating judol user transactions...")
    judol_user_txs = generate_judol_user_transactions(
        25000, judol_users, normal_merchants, judol_merchants, start_date, end_date
    )

    # Combine and shuffle
    print("\n[6/6] Combining and shuffling...")
    all_records = normal_txs + judol_merchant_txs + judol_user_txs
    df = pd.DataFrame(all_records)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Summary stats
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total rows:          {len(df):,}")
    print(f"Normal (label=0):    {len(df[df['label']==0]):,} ({len(df[df['label']==0])/len(df)*100:.1f}%)")
    print(f"Judol  (label=1):    {len(df[df['label']==1]):,} ({len(df[df['label']==1])/len(df)*100:.1f}%)")
    print(f"Unique users:        {df['user_id'].nunique():,}")
    print(f"Unique merchants:    {df['merchant_id'].nunique():,}")
    print(f"Unique cities:       {df['user_city'].nunique():,}")
    print(f"Unique provinces:    {df['user_province'].nunique():,}")
    print(f"Amount range:        Rp{df['amount'].min():,.0f} - Rp{df['amount'].max():,.0f}")
    print(f"Date range:          {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"Columns:             {list(df.columns)}")

    # Temporal stats
    print(f"\n--- Temporal Distribution ---")
    df_ts = pd.to_datetime(df["timestamp"])
    print(f"  Weekday vs Weekend: {(df_ts.dt.weekday < 5).sum():,} / {(df_ts.dt.weekday >= 5).sum():,}")
    judol_df = df[df["label"] == 1]
    judol_ts = pd.to_datetime(judol_df["timestamp"])
    prime_time = judol_ts.dt.hour.isin([20, 21, 22, 23, 0, 1, 2]).sum()
    print(f"  Judol prime time (20-02): {prime_time:,} / {len(judol_df):,} ({prime_time/len(judol_df)*100:.1f}%)")
    gajian = judol_ts.dt.day.isin(GAJIAN_DATES).sum()
    print(f"  Judol on gajian dates:    {gajian:,} / {len(judol_df):,} ({gajian/len(judol_df)*100:.1f}%)")

    # Transaction type stats
    print(f"\n--- Transaction Type Distribution ---")
    tx_type_counts = df["transaction_type"].value_counts()
    for tx_type, count in tx_type_counts.items():
        pct = count / len(df) * 100
        label_1_count = df[(df["transaction_type"] == tx_type) & (df["label"] == 1)].shape[0]
        print(f"  {tx_type}: {count:,} ({pct:.1f}%) — {label_1_count:,} judol")

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "generated")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pantau_dataset.csv")
    df.to_csv(output_path, index=False)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nSaved: {output_path} ({file_size_mb:.1f} MB)")

    # Show sample rows
    print("\n" + "=" * 60)
    print("SAMPLE ROWS")
    print("=" * 60)
    print("\n--- Normal transaction (label=0) ---")
    print(df[df["label"] == 0].iloc[0].to_string())
    print("\n--- Judol transaction (label=1) ---")
    print(df[df["label"] == 1].iloc[0].to_string())

    return df


if __name__ == "__main__":
    df = generate_full_dataset()
