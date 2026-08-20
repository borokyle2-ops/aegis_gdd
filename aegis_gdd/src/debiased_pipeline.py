import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


class AjoCausalCreditPipeline:

    def __init__(
        self,
        protected_col="gender",
        direct_cols=None,
        alt_cols=None,
        target_col="repayment_status",
        ref_val=1,
    ):
        """
        Structural Causal Engine for Ajo Card Credit Decisioning.
        - protected_col: Sensitive attribute (A) e.g., 'gender'
        - direct_cols: Core discipline signals (X_W) e.g., daily_thrift_freq, agent_float_balance
        - alt_cols: Proxy business scale / behavioral signals (X_Z) e.g., business_size_idx, cash_out_freq
        - target_col: Repayment outcome (Y) e.g., 'repayment_status'
        - ref_val: Neutral baseline group for counterfactual do-operator A = a' (1 = Female, 0 = Male)
        """
        self.protected_col = protected_col
        self.direct_cols = (
            direct_cols
            if direct_cols
            else [
                "daily_thrift_freq",
                "agent_float_balance",
                "pos_terminal_uptime_hrs",
            ]
        )
        self.alt_cols = (
            alt_cols
            if alt_cols
            else [
                "business_size_idx",
                "cash_out_freq",
                "avg_txn_value",
                "account_age_months",
            ]
        )
        self.target_col = target_col
        self.ref_val = ref_val

        self.model = None
        self.retained_alt_cols = []
        self.scrubbed_cols = []

    def preprocess_and_fit(self, df: pd.DataFrame, corr_threshold: float = 0.80):
        """Step 1: Feature Scrubbing (De-Confounding) & Causal Model Training"""
        df_encoded = df.copy()

        # Encode categorical gender strings ('F'/'M') to numeric indicators (1/0)
        if df_encoded[self.protected_col].dtype == "object":
            df_encoded["gender_num"] = (
                df_encoded[self.protected_col] == "F"
            ).astype(int)
            enc_target_col = "gender_num"
        else:
            enc_target_col = self.protected_col

        # Scrub proxy features that exceed correlation limits with protected attribute
        self.retained_alt_cols = []
        self.scrubbed_cols = []
        for col in self.alt_cols:
            corr = abs(df_encoded[enc_target_col].corr(df_encoded[col]))
            if corr < corr_threshold:
                self.retained_alt_cols.append(col)
            else:
                self.scrubbed_cols.append(col)

        # Assemble full feature set including gender_num to block backdoor paths
        feature_cols = (
            [enc_target_col] + self.direct_cols + self.retained_alt_cols
        )
        X = df_encoded[feature_cols]
        y = df_encoded[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train LightGBM Classifier
        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            verbose=-1,
        )
        self.model.fit(X_train, y_train)

        return X_test, y_test

    def predict_unconstrained(self, X_input: pd.DataFrame) -> np.ndarray:
        """Generates raw repayment probabilities using original applicant features"""
        return self.model.predict_proba(X_input)[:, 1]

    def predict_debiased_risk(
        self, X_input: pd.DataFrame, target_ref_num: int = 1
    ) -> np.ndarray:
        """Step 2: Neutral Prediction (Apply counterfactual do-operator A = a')"""
        X_counterfactual = X_input.copy()
        target_col_name = (
            "gender_num"
            if "gender_num" in X_counterfactual.columns
            else self.protected_col
        )
        X_counterfactual[target_col_name] = target_ref_num
        return self.model.predict_proba(X_counterfactual)[:, 1]

    def evaluate_metrics(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        probabilities: np.ndarray,
        threshold: float = 0.5,
    ) -> dict:
        """Step 3: Compute Performance and Fairness Parity KPIs"""
        target_col_name = (
            "gender_num"
            if "gender_num" in X_test.columns
            else self.protected_col
        )

        eval_df = pd.DataFrame(
            {
                "true_y": y_test,
                "repay_prob": probabilities,
                "group": X_test[target_col_name].values,
            }
        )

        auc = roc_auc_score(eval_df["true_y"], eval_df["repay_prob"])
        eval_df["approved"] = (eval_df["repay_prob"] >= threshold).astype(int)

        # Calculate approval rates per demographic group
        approval_rates = eval_df.groupby("group")["approved"].mean()
        f_rate = approval_rates.get(1, 0.0)
        m_rate = approval_rates.get(0, 0.0)

        rates = [f_rate, m_rate]
        disparate_impact = (
            min(rates) / max(rates) if max(rates) > 0 else 1.0
        )

        return {
            "auc": auc,
            "female_approval_rate": f_rate,
            "male_approval_rate": m_rate,
            "disparate_impact_ratio": disparate_impact,
        }