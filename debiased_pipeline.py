import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


class AjoCausalCreditPipeline:

    def __init__(
        self, protected_col, direct_cols, alt_cols, target_col, ref_val=0
    ):
        """
        protected_col : Sensitive attribute e.g., 'gender' (A)
        direct_cols    : Core financial signals e.g., 'ajo_contribution_consistency' (X_W)
        alt_cols       : Alternative digital/behavioral signals e.g., 'ajo_group_size' (X_Z)
        target_col     : Loan default outcome e.g., 'default_flag' (Y)
        ref_val        : Counterfactual neutral baseline (a')
        """
        self.protected_col = protected_col
        self.direct_cols = direct_cols
        self.alt_cols = alt_cols
        self.target_col = target_col
        self.ref_val = ref_val
        self.model = None
        self.retained_alt_cols = []

    def preprocess_and_fit(self, df, corr_threshold=0.80):
        """Step 1: Feature Scrubbing (De-Confounding) & Model Training"""
        # Scrub alternative features that heavily correlate with gender
        self.retained_alt_cols = []
        for col in self.alt_cols:
            corr = abs(df[self.protected_col].corr(df[col]))
            if corr < corr_threshold:
                self.retained_alt_cols.append(col)

        # Assemble full feature set including protected attribute to block backdoor path
        feature_cols = (
            [self.protected_col] + self.direct_cols + self.retained_alt_cols
        )
        X = df[feature_cols]
        y = df[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train LightGBM model conditioned on gender
        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            verbose=-1,
        )
        self.model.fit(X_train, y_train)

        return X_test, y_test

    def predict_unconstrained(self, X_input):
        """Generates raw predictions using unmodified applicant data (Baseline)"""
        return self.model.predict_proba(X_input)[:, 1]

    def predict_debiased_risk(self, X_input):
        """Step 2: Neutral Prediction (Apply do-operator A = a')"""
        X_neutral = X_input.copy()
        X_neutral[self.protected_col] = self.ref_val
        return self.model.predict_proba(X_neutral)[:, 1]

    def evaluate_metrics(self, X_test, y_test, probabilities, threshold=0.5):
        """Step 3: Compute Performance and Fairness Parity KPIs"""
        eval_df = pd.DataFrame(
            {
                "true_y": y_test,
                "risk_score": probabilities,
                "group": X_test[self.protected_col].values,
            }
        )

        auc = roc_auc_score(eval_df["true_y"], eval_df["risk_score"])
        eval_df["approved"] = (eval_df["risk_score"] < threshold).astype(int)

        rates = eval_df.groupby("group")["approved"].mean()
        disparate_impact = (
            rates.min() / rates.max() if rates.max() > 0 else 0.0
        )

        return {
            "auc": auc,
            "approval_rates": rates.to_dict(),
            "disparate_impact_ratio": disparate_impact,
        }