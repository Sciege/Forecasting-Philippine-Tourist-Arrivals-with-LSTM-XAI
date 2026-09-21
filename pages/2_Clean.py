# pages/2_Clean.py
import streamlit as st
import pandas as pd

st.title("2. Clean the data")

if "raw_df" not in st.session_state:
    st.warning("Run the Dataset page first.")
    st.stop()

if st.button("Run cleaning"):
    df = st.session_state.raw_df.copy()

    # 1. Convert arrivals from formatted string with commas to float
    if df["arrivals"].dtype == "object":
        df["arrivals"] = df["arrivals"].astype(str).str.replace(",", "").astype(float)

    # 2. Remove duplicate dates
    duplicates_removed = int(df.duplicated(subset="date").sum())
    df = df.drop_duplicates(subset="date", keep="first")

    # 3. Track missing values before filling them
    missing = df.isna().sum()
    missing = missing[missing > 0]

    # 4. One-hot encode categorical seasonal columns
    for col in ["season", "monsoon"]:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], drop_first=True, dtype=int)

    # 5. Impute missing values across numeric columns so NaNs don't reach the model
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    # 6. Flag IQR outliers on arrivals
    q1, q3 = df["arrivals"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flagged = df[(df["arrivals"] < lower) | (df["arrivals"] > upper)]

    # Save to session_state
    st.session_state.clean_df = df
    st.session_state.clean_report = {
        "duplicates_removed": duplicates_removed,
        "missing": missing,
        "flagged": flagged[["date", "arrivals"]],
    }

if "clean_report" in st.session_state:
    report = st.session_state.clean_report
    st.write(f"Duplicates removed: {report['duplicates_removed']}")
    st.write("Missing values detected (imputed with forward/backward fill):")
    st.dataframe(report["missing"])
    st.write("Flagged outliers:")
    st.dataframe(report["flagged"])
else:
    st.info('Click "Run cleaning" to process the dataset loaded on the Dataset page.')