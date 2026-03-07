"""
Pantau Synthetic Dataset Generator
===================================
Generates 500,000 transactions (85% normal, 15% judol) calibrated from:
- IBM AML dataset (fraud rate, amount distribution, layering patterns)
- PaySim dataset (hourly distribution, repeat rate, geo radius)
- Bustabit dataset (gambling session patterns, deposit cycles)
- PPATK statistics (judol hours, round amounts, geo spread)

Output: data/generated/pantau_dataset.csv
"""

import pandas as pd
import numpy as np
import uuid
import random
import string
from datetime import datetime, timedelta
import os

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# INDONESIAN CITIES & PROVINCES
# ============================================================

INDONESIAN_CITIES = [
    ("Jakarta", "DKI Jakarta"),
    ("Surabaya", "Jawa Timur"),
    ("Bandung", "Jawa Barat"),
    ("Medan", "Sumatera Utara"),
    ("Semarang", "Jawa Tengah"),
    ("Makassar", "Sulawesi Selatan"),
    ("Palembang", "Sumatera Selatan"),
    ("Tangerang", "Banten"),
    ("Depok", "Jawa Barat"),
    ("Bekasi", "Jawa Barat"),
    ("Yogyakarta", "DI Yogyakarta"),
    ("Denpasar", "Bali"),
    ("Balikpapan", "Kalimantan Timur"),
    ("Manado", "Sulawesi Utara"),
    ("Padang", "Sumatera Barat"),
    ("Pekanbaru", "Riau"),
    ("Banjarmasin", "Kalimantan Selatan"),
    ("Pontianak", "Kalimantan Barat"),
    ("Malang", "Jawa Timur"),
    ("Bogor", "Jawa Barat"),
    ("Cirebon", "Jawa Barat"),
    ("Samarinda", "Kalimantan Timur"),
    ("Mataram", "Nusa Tenggara Barat"),
    ("Kupang", "Nusa Tenggara Timur"),
    ("Jayapura", "Papua"),
    ("Ambon", "Maluku"),
    ("Kendari", "Sulawesi Tenggara"),
    ("Palu", "Sulawesi Tengah"),
    ("Jambi", "Jambi"),
    ("Bengkulu", "Bengkulu"),
    ("Gorontalo", "Gorontalo"),
    ("Ternate", "Maluku Utara"),
    ("Mamuju", "Sulawesi Barat"),
    ("Pangkal Pinang", "Bangka Belitung"),
]

# City weights: larger cities = more transactions
CITY_WEIGHTS = [
    15, 8, 7, 6, 5, 4, 4, 5, 4, 5,  # Jakarta-Bekasi
    3, 3, 2, 2, 2, 2, 2, 2, 3, 3,    # Yogya-Bogor
    2, 2, 1, 1, 1, 1, 1, 1, 1, 1,    # Cirebon-Bengkulu
    1, 1, 1, 1,                        # Gorontalo-Pangkal Pinang
]

# ============================================================
# CALIBRATED PARAMETERS (from base datasets + PPATK)
# ============================================================

# From PaySim: hourly distribution of normal mobile payments
NORMAL_HOUR_WEIGHTS = [1, 1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 9, 10, 9, 7, 6, 5, 7, 8, 6, 4, 3, 2, 1]

# From Bustabit + PPATK: judol peaks at late night
JUDOL_HOUR_WEIGHTS = [8, 7, 6, 4, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 4, 5, 6, 7, 9, 10]

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
}

# Judol user params (from Bustabit dataset)
JUDOL_USER_PARAMS = {
    "escalation_rate": 0.68,
    "late_night_rate": 0.72,
    "round_amount_rate": 0.88,
    "deposits_per_session": (2, 8),
    "txs_per_user_range": (20, 200),
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
    NMID (National Merchant ID): 15-digit alphanumeric.
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


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def generate_timestamp(hour_weights: list, start_date: datetime, end_date: datetime) -> datetime:
    """Generate a random timestamp within date range, weighted by hour."""
    days_range = (end_date - start_date).days
    base = start_date + timedelta(days=random.randint(0, days_range))
    hour = random.choices(range(24), weights=hour_weights, k=1)[0]
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return base.replace(hour=hour, minute=minute, second=second)


def pick_city_weighted() -> tuple:
    """Pick a city weighted by population."""
    return random.choices(INDONESIAN_CITIES, weights=CITY_WEIGHTS, k=1)[0]


def pick_city_random() -> tuple:
    """Pick a city uniformly (for geo spread)."""
    return random.choice(INDONESIAN_CITIES)


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


def create_merchant_pools(n_normal: int = 5000, n_judol: int = 500) -> tuple:
    """Pre-generate merchant pools with fixed city assignments."""
    normal_merchants = []
    for _ in range(n_normal):
        city, province = pick_city_weighted()
        normal_merchants.append(
            {"merchant_id": generate_nmid(), "city": city, "province": province}
        )

    judol_merchants = []
    for _ in range(n_judol):
        city, province = pick_city_weighted()
        judol_merchants.append(
            {"merchant_id": generate_nmid(), "city": city, "province": province}
        )

    return normal_merchants, judol_merchants


def create_user_pools(n_normal: int = 50000, n_judol: int = 2000) -> tuple:
    """Pre-generate user pools with fixed city assignments."""
    normal_users = []
    for _ in range(n_normal):
        city, province = pick_city_weighted()
        normal_users.append(
            {"user_id": generate_user_account(), "city": city, "province": province}
        )

    judol_users = []
    for _ in range(n_judol):
        city, province = pick_city_random()
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
    user_favorites = {}
    for user in normal_users:
        n_favs = random.randint(2, 5)
        # Prefer merchants in same city
        same_city = [m for m in normal_merchants if m["city"] == user["city"]]
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
        ts = generate_timestamp(NORMAL_HOUR_WEIGHTS, start_date, end_date)
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
                "transaction_type": "QRIS",
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
            city, province = pick_city_random()  # nationwide geo spread
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
                sender = random.choice(sender_pool[:10])  # repeat from small subset
            else:
                sender = random.choice(sender_pool)

            amount = generate_judol_amount()
            ts = generate_timestamp(JUDOL_HOUR_WEIGHTS, active_start, active_end)
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
                    "transaction_type": "QRIS",
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
            # Pick a judol merchant (they cycle through multiple)
            merchant = random.choice(session_merchants)

            # Escalation pattern: amounts tend to increase over time
            if random.random() < JUDOL_USER_PARAMS["escalation_rate"]:
                escalation = 1.0 + (j / max(count, 1)) * 2.0
                amount = int(base_amount * escalation)
                amount = max(10000, round(amount / 10000) * 10000)
            else:
                amount = generate_judol_amount()

            # 72% late night
            if random.random() < JUDOL_USER_PARAMS["late_night_rate"]:
                hour_weights = JUDOL_HOUR_WEIGHTS
            else:
                hour_weights = NORMAL_HOUR_WEIGHTS

            ts = generate_timestamp(hour_weights, start_date, end_date)
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
                    "transaction_type": "QRIS",
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

    # Time range: 90 days
    end_date = datetime(2026, 3, 1)
    start_date = end_date - timedelta(days=90)
    print(f"\nDate range: {start_date.date()} → {end_date.date()}")

    # Create entity pools
    print("\n[1/5] Creating entity pools...")
    normal_merchants, judol_merchants = create_merchant_pools(
        n_normal=5000, n_judol=500
    )
    normal_users, judol_users = create_user_pools(n_normal=50000, n_judol=2000)
    print(f"  Normal merchants: {len(normal_merchants):,}")
    print(f"  Judol merchants:  {len(judol_merchants):,}")
    print(f"  Normal users:     {len(normal_users):,}")
    print(f"  Judol users:      {len(judol_users):,}")

    # Generate transactions
    print("\n[2/5] Generating normal transactions...")
    normal_txs = generate_normal_transactions(
        425000, normal_merchants, normal_users, start_date, end_date
    )

    print("\n[3/5] Generating judol merchant transactions...")
    judol_merchant_txs = generate_judol_merchant_transactions(
        50000, judol_merchants, start_date, end_date
    )

    print("\n[4/5] Generating judol user transactions...")
    judol_user_txs = generate_judol_user_transactions(
        25000, judol_users, normal_merchants, judol_merchants, start_date, end_date
    )

    # Combine and shuffle
    print("\n[5/5] Combining and shuffling...")
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
    print(f"Amount range:        Rp{df['amount'].min():,.0f} - Rp{df['amount'].max():,.0f}")
    print(f"Date range:          {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"Columns:             {list(df.columns)}")

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
