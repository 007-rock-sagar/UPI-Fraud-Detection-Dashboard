"""
Synthetic UPI Transaction Data Generator for Fraud Detection ETL Pipeline.

Generates realistic UPI payment logs containing both:
1. Fraud anomalies:
   - Nocturnal high-value transfers (02:00 AM - 04:30 AM)
   - Velocity bursts (rapid consecutive transfers from a single VPA within 3 minutes)
   - Collect request scams (high-value COLLECT_REQUEST transactions)
   - Dormant / Mule account spikes (sudden large credits to newly active VPAs)
2. Data engineering flaws (for ETL cleaning practice):
   - Duplicate webhook retries (identical txn_id emitted twice)
   - Mixed datetime formats (ISO-8601 vs DD-MM-YYYY HH:MM:SS vs slash formats)
   - Messy VPA strings (leading/trailing whitespace, inconsistent casing)
   - Missing beneficiary or device metadata (NULL / empty values)
"""

import argparse
import os
import random
import uuid
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


BANK_HANDLES = ["oksbi", "okaxis", "okhdfcbank", "okicici", "ybl", "paytm", "ibl", "axl"]
MERCHANT_PREFIXES = ["swiggy", "zomato", "amazon", "flipkart", "irctc", "uber", "blinkit", "zepto", "bigbasket", "makemytrip"]
FIRST_NAMES = [
    "aarav", "vivaan", "aditya", "vihaan", "arjun", "sai", "reyansh", "ayaan",
    "krishna", "ishaan", "ananya", "diya", "priya", "sneha", "kavya", "neha",
    "rohit", "rahul", "amit", "sumit", "pooja", "ankit", "vikram", "divya",
    "nikhil", "shreya", "manish", "deepak", "varun", "megha"
]
CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Ahmedabad", "Chennai",
    "Kolkata", "Pune", "Jaipur", "Lucknow", "Patna", "Bhopal", "Indore", "Kochi"
]


def generate_vpa_pool(num_users: int = 500, num_merchants: int = 80, num_mules: int = 25):
    """Creates pools of user, merchant, and mule Virtual Payment Addresses (VPAs)."""
    users = []
    for i in range(num_users):
        name = random.choice(FIRST_NAMES)
        handle = random.choice(BANK_HANDLES)
        vpa = f"{name}{random.randint(100, 9999)}@{handle}"
        users.append({
            "vpa": vpa,
            "device_id": f"DEV-{uuid.uuid4().hex[:10].upper()}",
            "city": random.choice(CITIES),
            "avg_spend": round(random.uniform(150.0, 2500.0), 2),
            "account_age_days": random.randint(45, 1800)
        })

    merchants = []
    for i in range(num_merchants):
        prefix = random.choice(MERCHANT_PREFIXES)
        handle = random.choice(["ybl", "paytm", "okaxis", "okhdfcbank"])
        merchants.append(f"{prefix}.merchant{i}@{handle}")

    mules = []
    for i in range(num_mules):
        handle = random.choice(BANK_HANDLES)
        mules.append(f"verify.refund{random.randint(1000, 9999)}@{handle}")

    return users, merchants, mules


def format_messy_timestamp(dt: datetime, inject_flaw: bool = False) -> str:
    """Returns standard ISO timestamp or deliberately inconsistent date formats."""
    if not inject_flaw:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    fmt = random.choice([
        "%d-%m-%Y %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%Y-%m-%dT%H:%M:%SZ"
    ])
    return dt.strftime(fmt)


def build_dataset(num_records: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generates the synthetic UPI dataset with fraud anomalies and ETL flaws."""
    random.seed(seed)
    np.random.seed(seed)

    users, merchants, mules = generate_vpa_pool()
    base_time = datetime(2026, 10, 1, 8, 0, 0)
    records = []

    i = 0
    while i < num_records:
        user = random.choice(users)
        # Normal daytime hours weighted higher (07:00 to 23:00)
        day_offset = random.randint(0, 30)
        hour = random.choices(
            population=list(range(24)),
            weights=[1, 1, 1, 1, 1, 2, 4, 7, 9, 10, 10, 10, 11, 10, 9, 9, 10, 11, 12, 12, 10, 8, 4, 2],
            k=1
        )[0]
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        txn_time = base_time + timedelta(days=day_offset, hours=hour - 8, minutes=minute, seconds=second)

        roll = random.random()

        # 1. Fraud Scenario A: Rapid Velocity Burst (~2.5% of rows)
        if roll < 0.025 and (num_records - i) >= 6:
            burst_count = random.randint(4, 6)
            new_device = f"DEV-{uuid.uuid4().hex[:10].upper()}"
            for b in range(burst_count):
                b_time = txn_time + timedelta(seconds=b * random.randint(15, 35))
                records.append({
                    "txn_id": f"UPI{uuid.uuid4().hex[:14].upper()}",
                    "timestamp": format_messy_timestamp(b_time),
                    "sender_vpa": user["vpa"],
                    "receiver_vpa": random.choice(mules),
                    "amount": round(random.uniform(9500.0, 48000.0), 2),
                    "txn_type": "P2P",
                    "status": "SUCCESS",
                    "device_id": new_device,
                    "sender_city": user["city"],
                    "account_age_days": user["account_age_days"],
                    "is_fraud": 1,
                    "fraud_reason": "VELOCITY_BURST_NEW_DEVICE"
                })
                i += 1
            continue

        # 2. Fraud Scenario B: Nocturnal High-Value Drain (02:00 AM - 04:30 AM) (~1.8% of rows)
        elif roll < 0.043:
            odd_time = (base_time + timedelta(days=day_offset)).replace(
                hour=random.choice([2, 3, 4]),
                minute=random.randint(0, 59),
                second=random.randint(0, 59)
            )
            records.append({
                "txn_id": f"UPI{uuid.uuid4().hex[:14].upper()}",
                "timestamp": format_messy_timestamp(odd_time),
                "sender_vpa": user["vpa"],
                "receiver_vpa": random.choice(mules),
                "amount": round(random.uniform(25000.0, 99500.0), 2),
                "txn_type": "P2P",
                "status": "SUCCESS",
                "device_id": user["device_id"],
                "sender_city": user["city"],
                "account_age_days": user["account_age_days"],
                "is_fraud": 1,
                "fraud_reason": "NOCTURNAL_HIGH_VALUE"
            })
            i += 1
            continue

        # 3. Fraud Scenario C: Social Engineering Collect Request Scam (~1.5% of rows)
        elif roll < 0.058:
            records.append({
                "txn_id": f"UPI{uuid.uuid4().hex[:14].upper()}",
                "timestamp": format_messy_timestamp(txn_time),
                "sender_vpa": user["vpa"],
                "receiver_vpa": random.choice(mules),
                "amount": round(random.uniform(4999.0, 45000.0), 2),
                "txn_type": "COLLECT_REQUEST",
                "status": "SUCCESS",
                "device_id": user["device_id"],
                "sender_city": user["city"],
                "account_age_days": user["account_age_days"],
                "is_fraud": 1,
                "fraud_reason": "SUSPICIOUS_COLLECT_REQUEST"
            })
            i += 1
            continue

        # 4. Fraud Scenario D: Newly Created / Dormant Account Sudden High Transfer (~1.0% of rows)
        elif roll < 0.068:
            records.append({
                "txn_id": f"UPI{uuid.uuid4().hex[:14].upper()}",
                "timestamp": format_messy_timestamp(txn_time),
                "sender_vpa": user["vpa"],
                "receiver_vpa": random.choice(users)["vpa"],
                "amount": round(random.uniform(50000.0, 98000.0), 2),
                "txn_type": "P2P",
                "status": "SUCCESS",
                "device_id": user["device_id"],
                "sender_city": user["city"],
                "account_age_days": random.randint(1, 5),
                "is_fraud": 1,
                "fraud_reason": "NEW_ACCOUNT_HIGH_VALUE_SPIKE"
            })
            i += 1
            continue

        # 5. Legitimate Transactions (~93.2% of rows)
        is_merchant = random.random() < 0.62
        if is_merchant:
            receiver = random.choice(merchants)
            txn_type = "P2M"
            amount = round(max(15.0, np.random.lognormal(mean=5.5, sigma=0.85)), 2)
        else:
            receiver = random.choice(users)["vpa"]
            while receiver == user["vpa"]:
                receiver = random.choice(users)["vpa"]
            txn_type = random.choices(["P2P", "COLLECT_REQUEST"], weights=[0.92, 0.08], k=1)[0]
            amount = round(max(50.0, np.random.normal(loc=user["avg_spend"], scale=user["avg_spend"] * 0.35)), 2)

        status = random.choices(["SUCCESS", "FAILED", "PENDING"], weights=[0.94, 0.04, 0.02], k=1)[0]

        records.append({
            "txn_id": f"UPI{uuid.uuid4().hex[:14].upper()}",
            "timestamp": format_messy_timestamp(txn_time),
            "sender_vpa": user["vpa"],
            "receiver_vpa": receiver,
            "amount": min(amount, 100000.0),
            "txn_type": txn_type,
            "status": status,
            "device_id": user["device_id"],
            "sender_city": user["city"],
            "account_age_days": user["account_age_days"],
            "is_fraud": 0,
            "fraud_reason": "NONE"
        })
        i += 1

    df = pd.DataFrame(records)
    num_rows = len(df)

    # Flaw 1: Here we need Mixed date formats (~4% of rows)
    messy_date_idx = np.random.choice(df.index, size=int(num_rows * 0.04), replace=False)
    for idx in messy_date_idx:
        try:
            dt_obj = datetime.strptime(df.at[idx, "timestamp"], "%Y-%m-%d %H:%M:%S")
            df.at[idx, "timestamp"] = format_messy_timestamp(dt_obj, inject_flaw=True)
        except ValueError:
            pass

    # Flaw 2: include Whitespace & uppercase noise in VPA strings (~3% of rows)
    noisy_vpa_idx = np.random.choice(df.index, size=int(num_rows * 0.03), replace=False)
    for idx in noisy_vpa_idx:
        df.at[idx, "sender_vpa"] = f"  {df.at[idx, 'sender_vpa'].upper()} "

    # Flaw 3: Missing device_id or receiver_vpa on failed/pending logs (~1.5% of rows)
    null_device_idx = np.random.choice(df.index, size=int(num_rows * 0.015), replace=False)
    df.loc[null_device_idx, "device_id"] = None

    # Flaw 4:This i to Duplicate webhook retry events (~1.5% of rows duplicated)
    dup_idx = np.random.choice(df.index, size=int(num_rows * 0.015), replace=False)
    duplicates = df.loc[dup_idx].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic UPI transaction logs.")
    parser.add_argument("--rows", type=int, default=10000, help="Number of base transactions to generate.")
    parser.add_argument("--output", type=str, default="data/raw/upi_transactions_raw.csv", help="Output CSV path.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"Generating {args.rows} synthetic UPI transactions...")
    df = build_dataset(num_records=args.rows)
    df.to_csv(args.output, index=False)

    total_rows = len(df)
    fraud_rows = int(df["is_fraud"].sum())
    dup_rows = int(df.duplicated(subset=["txn_id"]).sum())

    print(f"Saved dataset to: {args.output}")
    print(f"Total rows (including webhook duplicates): {total_rows}")
    print(f"Duplicate txn_id rows injected:            {dup_rows}")
    print(f"Fraudulent transactions injected:          {fraud_rows} ({fraud_rows / total_rows:.2%})")
    print("\nBreakdown by Fraud Reason:")
    print(df[df["is_fraud"] == 1]["fraud_reason"].value_counts().to_string())


if __name__ == "__main__":
    main()
