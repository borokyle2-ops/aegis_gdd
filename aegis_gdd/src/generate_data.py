import numpy as np
import pandas as pd
from pathlib import Path

def generate_ajocard_standin(n_samples: int = 6000, seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic stand-in dataset for AjoCard micro-merchant transactions.
    Injects a structural business-size confound to test downstream debiasing logic.
    """
    np.random.seed(seed)
    
    # 1. Demographic & Business Profile
    # Gender split: ~65% female micro-merchants (reflective of market baseline)
    is_female = np.random.binomial(1, 0.65, n_samples)
    
    # Business Size Confound: Females slightly skewed toward informal/micro scale
    # Scale index: 0 = Micro/Street Vendor, 1 = Small Kiosk, 2 = Medium Retailer
    size_logits = np.where(is_female == 1, -0.6, 0.4)
    business_size = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.35, 0.15])
    
    # 2. Transactional Metadata (AjoCard operational signals)
    # Micro-businesses have higher daily deposit frequency but lower average transaction values
    daily_thrift_freq = np.random.poisson(lam=np.where(business_size == 0, 4.2, 1.8), size=n_samples)
    avg_txn_value = np.random.gamma(shape=2.0, scale=np.where(business_size == 0, 1500, 4500), size=n_samples)
    
    # Cash-out velocity (high ratio means low retained liquidity)
    cash_out_freq = daily_thrift_freq * np.random.uniform(0.4, 0.95, n_samples)
    
    # Account age in months
    account_age_months = np.random.randint(1, 36, n_samples)
    
    # 3. Target Variable: Repayment Outcome (0 = Default, 1 = Repaid)
    # Logit construction: Base repayment driven by deposit frequency and account age
    logit = (
        -0.8 
        + 0.35 * daily_thrift_freq 
        + 0.04 * account_age_months 
        - 0.0001 * avg_txn_value
        + 0.30 * (1 - business_size) # Confound: Micro-scale exhibits tight repayment discipline
        + 0.25 * is_female           # Direct effect
    )
    
    prob_repay = 1 / (1 + np.exp(-logit))
    repaid = np.random.binomial(1, prob_repay)
    
    # Build DataFrame
    df = pd.DataFrame({
        "merchant_id": [f"MERCH_{10000 + i}" for i in range(n_samples)],
        "gender": np.where(is_female == 1, "F", "M"),
        "business_size_idx": business_size,
        "daily_thrift_freq": daily_thrift_freq,
        "cash_out_freq": np.round(cash_out_freq, 2),
        "avg_txn_value": np.round(avg_txn_value, 2),
        "account_age_months": account_age_months,
        "repaid": repaid
    })
    
    return df

if __name__ == "__main__":
    print("Executing Step 1: Generating AjoCard Stand-in Data...")
    df = generate_ajocard_standin(n_samples=6000, seed=42)
    
    # Ensure data directory exists
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "fake_ajocard_data.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Step 1 Complete. Dataset saved to: {output_path}")
    print(f"Total Rows: {len(df)}")
    
    # Output baseline stats to observe the injected gap
    female_repay = df[df["gender"] == "F"]["repaid"].mean()
    male_repay = df[df["gender"] == "M"]["repaid"].mean()
    print(f"📊 Raw Repayment Rate -> Female: {female_repay:.1%} | Male: {male_repay:.1%} (Gap: {(female_repay - male_repay)*100:.1f} pts)")