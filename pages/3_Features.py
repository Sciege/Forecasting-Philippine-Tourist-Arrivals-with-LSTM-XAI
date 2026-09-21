# pages/3_Features.py
import streamlit as st
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.title("3. Feature selection")

if "clean_df" not in st.session_state:
    st.warning("Run the Clean page first.")
    st.stop()

# Select all numeric candidate columns (excluding date, arrivals, year, month)
numeric_cols = st.session_state.clean_df.select_dtypes(include=[np.number]).columns.tolist()
CANDIDATES = [c for c in numeric_cols if c not in ["arrivals", "year", "month"]]

if st.button("Run Spearman + VIF"):
    df = st.session_state.clean_df

    # Step 1 — univariate filter
    results = []
    for col in CANDIDATES:
        rho, p_val = spearmanr(df[col], df["arrivals"], nan_policy="omit")
        results.append({"feature": col, "rho": float(rho), "p_value": float(p_val)})

    kept = [r["feature"] for r in results if abs(r["rho"]) > 0.10 and r["p_value"] < 0.05]

    if not kept:
        st.error("No features passed the Spearman filter (|rho| > 0.10, p < 0.05).")
        st.stop()

    # Step 2 — iterative VIF
    X = df[kept].dropna().copy()
    vif_log = []

    while X.shape[1] > 1:
        vifs = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        if not vifs:
            break
        max_vif = max(vifs)
        if max_vif < 5 or np.isnan(max_vif):
            break
        drop_col = X.columns[vifs.index(max_vif)]
        vif_log.append({"dropped": drop_col, "vif": max_vif})
        X = X.drop(columns=[drop_col])

    st.session_state.selected_features = list(X.columns)
    st.session_state.feature_report = {"results": results, "vif_log": vif_log}

if "feature_report" in st.session_state:
    report = st.session_state.feature_report
    st.dataframe(report["results"])
    st.write("VIF removals:", report["vif_log"])
    st.success(f"Selected features: {st.session_state.selected_features}")
else:
    st.info('Click "Run Spearman + VIF" to select features from the cleaned dataset.')