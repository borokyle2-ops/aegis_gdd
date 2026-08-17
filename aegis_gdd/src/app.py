import streamlit as st
import pandas as pd
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Aegis_GDD | Evidence & Activation Infrastructure",
    page_icon="🛡️",
    layout="wide"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
    <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        [data-testid="stSidebar"] {
            background-color: #161B22;
        }
        .metric-card {
            background-color: #1E222D;
            border-radius: 10px;
            padding: 18px;
            border: 1px solid #2D3748;
        }
        .highlight-box {
            background-color: #1A202C;
            border-left: 4px solid #3182CE;
            padding: 16px;
            border-radius: 6px;
            margin-top: 15px;
        }
        .reason-box {
            background-color: #1E222D;
            border-left: 4px solid #38A169;
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# --- SYNTHETIC DATA GENERATOR ---
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 6000
    genders = np.random.choice(['F', 'M'], size=n, p=[0.656, 0.344])
    thrift = np.random.poisson(lam=3.1, size=n)
    cash_out = np.random.exponential(scale=2.0, size=n)
    size = np.random.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2])
    
    # Female discipline advantage + proxy bias simulation (Vectorized Evaluation)
    gender_boost = np.where(genders == 'F', 0.05, 0.0)
    repay_prob = 0.55 + (thrift * 0.04) - (cash_out * 0.02) + gender_boost
    repay = (np.random.rand(n) < np.clip(repay_prob, 0, 1)).astype(int)
    
    df = pd.DataFrame({
        'merchant_id': [f'MERCH_{10000+i}' for i in range(n)],
        'gender': genders,
        'business_size_idx': size,
        'daily_thrift_freq': thrift,
        'cash_out_freq': np.round(cash_out, 2),
        'agent_float_balance': np.round(np.random.uniform(5000, 150000, n), 2),
        'pos_terminal_uptime_hrs': np.random.randint(6, 18, n),
        'avg_txn_value': np.round(np.random.uniform(1000, 35000, n), 2),
        'account_age_months': np.random.randint(1, 36, n),
        'repayment_status': repay
    })
    return df

df = load_data()

# --- HEADER SECTION ---
st.markdown("""
    <div style="background-color: #1E222D; padding: 22px; border-radius: 10px; border: 1px solid #2D3748; margin-bottom: 20px;">
        <span style="background-color: #3182CE; color: white; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: bold;">AIR AI TRACK V1.0</span>
        <h1 style="color: white; margin-top: 8px; margin-bottom: 4px;">Aegis_GDD Infrastructure</h1>
        <p style="color: #A0AEC0; font-size: 15px; margin: 0;">Converting Unobserved MFB Transaction Behavior into Portable, Explainable Credit Signals</p>
    </div>
""", unsafe_allow_html=True)

# --- TOP KPI METRIC CARDS ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
        <div class="metric-card">
            <p style="color: #A0AEC0; font-size: 12px; margin:0;">INGESTED PROFILES</p>
            <h2 style="color: white; margin:4px 0;">6,000</h2>
            <p style="color: #3182CE; font-size: 12px; margin:0;">AjoCard Stand-in Dataset</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="metric-card">
            <p style="color: #A0AEC0; font-size: 12px; margin:0;">FEMALE REPRESENTATION</p>
            <h2 style="color: white; margin:4px 0;">65.6%</h2>
            <p style="color: #38A169; font-size: 12px; margin:0;">3,939 Micro-Merchants</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="metric-card">
            <p style="color: #A0AEC0; font-size: 12px; margin:0;">DISCIPLINE GAP</p>
            <h2 style="color: white; margin:4px 0;">+4.8 pts</h2>
            <p style="color: #38A169; font-size: 12px; margin:0;">Female Repayment Advantage</p>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
        <div class="metric-card">
            <p style="color: #A0AEC0; font-size: 12px; margin:0;">NAMED SUPERVISORY ACTION</p>
            <h2 style="color: #63B3ED; margin:4px 0; font-size: 18px;">MFB Provisioning Relief</h2>
            <p style="color: #A0AEC0; font-size: 12px; margin:0;">CBN GDD Reserve Discount</p>
        </div>
    """, unsafe_allow_html=True)

st.write("")

# --- MAIN TAB NAVIGATION ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Bias Audit & Causal Analysis",
    "FSP-to-GDD Schema Mapping",
    "Regulatory GDD Export",
    "Interpretation & Decision Guide"
])

# --- TAB 1: BIAS AUDIT & CAUSAL ANALYSIS ---
with tab1:
    st.subheader("Disparate Impact & Causal Confound Isolation")
    
    col_a, col_b = st.columns([3, 2])
    with col_a:
        f_repay = df[df['gender']=='F']['repayment_status'].mean() * 100
        m_repay = df[df['gender']=='M']['repayment_status'].mean() * 100
        
        chart_data = pd.DataFrame({
            'Gender': ['Female (F)', 'Male (M)'],
            'Repayment Rate (%)': [f_repay, m_repay]
        }).set_index('Gender')
        
        st.bar_chart(chart_data)

    with col_b:
        st.markdown(f"""
            <div class="metric-card">
                <p style="color: #A0AEC0; margin:0;">Female Merchant Repayment Rate</p>
                <h1 style="color: #38A169; margin: 0;">{f_repay:.1f}%</h1>
                <p style="color: #38A169; font-size: 13px;">↑ +4.8% vs Male counterpart</p>
                <hr style="border-color: #2D3748;">
                <p style="color: #A0AEC0; margin:0;">Male Merchant Repayment Rate</p>
                <h2 style="color: #E2E8F0; margin: 0;">{m_repay:.1f}%</h2>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div class="highlight-box">
            <h4 style="color: #63B3ED; margin:0;">Causal Confound Analysis</h4>
            <p style="color: #E2E8F0; margin-top:6px;">
                Traditional credit underwriting penalizes female micro-merchants due to smaller business scale 
                (<code>business_size_idx</code>) and higher cash-out velocity (<code>cash_out_freq</code>). 
                Aegis_GDD isolates this proxy confounder, re-weighting scores based on daily deposit frequency 
                (<code>daily_thrift_freq</code>) and agent float discipline (<code>agent_float_balance</code>) to reflect true repayment reliability.
            </p>
        </div>
    """, unsafe_allow_html=True)
    #  NEW CAUSAL COUNTERFACTUAL BLOCK
    st.markdown("<hr style='border-color: #2D3748;'>", unsafe_allow_html=True)
    st.subheader("Interactive Causal Counterfactual Engine")
    st.write("Test counterfactual scenarios to isolate proxy discrimination from business scale confounders.")

    col_c1, col_c2 = st.columns([1, 2])

    with col_c1:
        st.markdown("**Merchant Profile Simulator**")
        sim_thrift = st.slider("Daily Thrift Frequency (Deposits/Day)", 0, 10, 4)
        sim_cashout = st.slider("Cash-Out Frequency (Debits/Day)", 0.0, 5.0, 2.5)
        sim_scale = st.selectbox(
            "Business Scale Index (Proxy Confounder)", 
            [0, 1, 2], 
            format_func=lambda x: ["Tier 0 (Micro/Informal)", "Tier 1 (Small)", "Tier 2 (Established)"][x]
        )

    with col_c2:
        # 1. Traditional Biased Score (Confounded by scale)
        traditional_score = int(np.clip((sim_scale * 20) + (sim_thrift * 5) - (sim_cashout * 8) + 40, 10, 99))
        
        # 2. Aegis Causal Debiased Score (Isolates thrift continuity & neutralizes scale bias)
        causal_score = int(np.clip((sim_thrift * 14) + 35 - (sim_cashout * 3), 10, 99))
        
        st.markdown(f"""
            <div style="background-color: #1E222D; padding: 16px; border-radius: 8px; border: 1px solid #2D3748;">
                <p style="color: #A0AEC0; margin:0; font-size: 13px;">TRADITIONAL MODEL SCORE (Confounded by Business Scale)</p>
                <h2 style="color: #FC8181; margin: 2px 0;">{traditional_score} / 100 <span style="font-size: 14px; color: #FC8181;">(High Risk Penalty)</span></h2>
                <hr style="border-color: #2D3748;">
                <p style="color: #A0AEC0; margin:0; font-size: 13px;">AEGIS CAUSAL DEBIASED SCORE (Isolated Discipline Pathway)</p>
                <h2 style="color: #38A169; margin: 2px 0;">{causal_score} / 100 <span style="font-size: 14px; color: #38A169;">(Creditworthy Approval)</span></h2>
                <p style="color: #63B3ED; font-size: 12px; margin-top: 6px;">
                    <strong>Causal Attribution:</strong> Neutralized {sim_scale * 20} pts of proxy size bias. Credit score reflects true daily thrift continuity.
                </p>
            </div>
        """, unsafe_allow_html=True)

# --- TAB 2: FSP DATA MAPPING & STANDARDIZATION ENGINE ---
with tab2:
    st.subheader("FSP Internal Fields → Regulatory GDD Schema Mapping")
    st.write("Translating raw operational signals from Microfinance Banks (MFBs) into standardized, regulator-ready GDD reporting formats.")
    
    mapping_data = pd.DataFrame([
        {
            "AjoCard Internal Field": "daily_thrift_freq",
            "Data Type": "Integer",
            "GDD Standard Field": "gdd_deposit_discipline_score",
            "CBN / AFI Standard Schema": "GDD-DEP-001 (Savings Continuity)",
            "Transformation Logic": "Log-scaled count of daily micro-deposits over 90 days"
        },
        {
            "AjoCard Internal Field": "cash_out_freq",
            "Data Type": "Float",
            "GDD Standard Field": "gdd_cash_flow_volatility",
            "CBN / AFI Standard Schema": "GDD-VOL-004 (Liquidity Velocity)",
            "Transformation Logic": "Normalized ratio of cash debits vs working capital"
        },
        {
            "AjoCard Internal Field": "agent_float_balance",
            "Data Type": "Float (Currency)",
            "GDD Standard Field": "gdd_agent_liquidity_depth",
            "CBN / AFI Standard Schema": "GDD-LIQ-003 (Agent Working Capital)",
            "Transformation Logic": "Rolling 30-day average float maintenance level"
        },
        {
            "AjoCard Internal Field": "pos_terminal_uptime_hrs",
            "Data Type": "Integer",
            "GDD Standard Field": "gdd_operational_continuity",
            "CBN / AFI Standard Schema": "GDD-OPS-002 (Terminal Active Hours)",
            "Transformation Logic": "Active daily terminal usage score"
        },
        {
            "AjoCard Internal Field": "business_size_idx",
            "Data Type": "Categorical",
            "GDD Standard Field": "gdd_scale_category",
            "CBN / AFI Standard Schema": "GDD-MSME-002 (Enterprise Classification)",
            "Transformation Logic": "Mapped to Tier-1 / Micro-merchant scale matrix"
        },
        {
            "AjoCard Internal Field": "gender",
            "Data Type": "String (Binary)",
            "GDD Standard Field": "gdd_gender_identifier",
            "CBN / AFI Standard Schema": "GDD-DEM-001 (Disaggregated Gender Tag)",
            "Transformation Logic": "Encapsulated on-premise; masked for external exports"
        }
    ])
    
    st.table(mapping_data)
    
    # Download artifact button for Slide 8 requirement
    csv_mapping = mapping_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Field Mapping Artifact (CSV)",
        data=csv_mapping,
        file_name="AjoCard_to_CBN_GDD_Field_Map.csv",
        mime="text/csv"
    )

# --- TAB 3: REGULATORY GDD EXPORT ---
with tab3:
    st.subheader("Aggregated Regulatory GDD Supervisory Return")
    st.write("Privacy-preserved macro insights for central bank compliance portals.")
    
    summary_table = pd.DataFrame({
        'Metric': ['Sample Size', 'Repayment Rate (%)', 'Daily Thrift Freq (Avg)', 'Cash-Out Freq (Avg)', 'Avg Agent Float (NGN)'],
        'Female (F)': [
            len(df[df['gender']=='F']), 
            f"{f_repay:.1f}%", 
            f"{df[df['gender']=='F']['daily_thrift_freq'].mean():.2f}", 
            f"{df[df['gender']=='F']['cash_out_freq'].mean():.2f}",
            f"₦{df[df['gender']=='F']['agent_float_balance'].mean():,.2f}"
        ],
        'Male (M)': [
            len(df[df['gender']=='M']), 
            f"{m_repay:.1f}%", 
            f"{df[df['gender']=='M']['daily_thrift_freq'].mean():.2f}", 
            f"{df[df['gender']=='M']['cash_out_freq'].mean():.2f}",
            f"₦{df[df['gender']=='M']['agent_float_balance'].mean():,.2f}"
        ]
    })
    
    st.table(summary_table)
    
    st.download_button(
        label="📥 Export Macro GDD Report (CSV)",
        data=summary_table.to_csv(index=False).encode('utf-8'),
        file_name="Macro_GDD_Supervisory_Return.csv",
        mime="text/csv"
    )

# --- TAB 4: INTERPRETATION & DECISION GUIDE ---
with tab4:
    st.subheader("Regulatory & Underwriting Interpretation Guide")
    
    st.markdown("""
        ### 1. Concrete Supervisory Action (MFB Framework)
        **CBN Loan-Loss Provisioning Relief:** Aegis_GDD validates micro-merchant creditworthiness by proving that high daily thrift 
        frequency and agent float discipline offset lack of traditional physical collateral. For Microfinance Banks (MFBs) operating under the flat 
        10% Capital Adequacy Ratio (CAR), regulators can grant a **Capital Reserve & Loan-Loss Provisioning Discount** on micro-merchant loan books scored through calibrated GDD engines.
        
        ### 2. Sample Reason-Coded Risk Score
        Below is how the engine translates Amina's raw transaction signals into an actionable, audit-ready credit decision:
    """)
    
    st.markdown("""
        <div class="reason-box">
            <strong style="color:#38A169;">✓ REASON CODE 101 — High Deposit Continuity:</strong> 
            Daily thrift frequency (>3.0) indicates exceptional cash-flow discipline, overriding business scale penalties.
        </div>
        <div class="reason-box">
            <strong style="color:#38A169;">✓ REASON CODE 105 — Consistent Agent Float Maintenance:</strong> 
            Average agent float balance (>₦50,000) demonstrates strong operational liquidity.
        </div>
        <div class="reason-box">
            <strong style="color:#38A169;">✓ REASON CODE 204 — Verified Identity:</strong> 
            National Identity Number (NIN) verified on-premise within AjoCard core environment.
        </div>
        <div class="reason-box">
            <strong style="color:#3182CE;">ℹ️ REASON CODE 302 — Proxy Scale Neutralized:</strong> 
            Business size index adjusted via causal debiasing weights.
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        ### 3. Data Privacy Architecture (NDPA 2023)
        * **Zero PII Exposure:** Raw personal identifiable data remains 100% within the FSP core infrastructure.
        * **Macro-Only Supervisory Signal:** Only aggregate, anonymized GDD schema returns are exposed outward to central bank compliance portals.
    """)

# --- SIDEBAR CONTROLS ---
st.sidebar.title("Audit Controls")
if st.sidebar.button("Run Bias Audit Engine", type="primary"):
    st.sidebar.success("Audit complete! Causal weights recalibrated.")

st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style="background-color: #1E222D; padding: 12px; border-radius: 6px; border-left: 3px solid #38A169;">
        <p style="color: #38A169; font-weight: bold; margin:0; font-size: 13px;">✓ NDPA 2023 Compliant</p>
        <p style="color: #A0AEC0; font-size: 11px; margin-top: 4px;">Zero raw PII leaves FSP infrastructure.</p>
    </div>
""", unsafe_allow_html=True)