# pages/5_Train.py
import streamlit as st
import itertools
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

st.title("5. Train and tune the LSTM")

if "X_train_seq" not in st.session_state:
    st.warning("Run the Prepare page first.")
    st.stop()

X_train = st.session_state.X_train_seq
y_train = st.session_state.y_train_seq

# 1. Quick sanity check on inputs
if len(X_train) == 0:
    st.error("X_train_seq has 0 sequences. Check your train split ratio or lookback window in Step 4.")
    st.stop()

if np.isnan(X_train).any() or np.isnan(y_train).any():
    st.error("Input data contains NaN values. Please re-run Clean and Prepare pages.")
    st.stop()

PARAM_GRID = {"units": [32, 64], "dropout": [0.1, 0.3], "batch_size": [16, 32]}

if st.button("Train & tune"):
    with st.spinner("Training… this can take a while"):
        lookback = X_train.shape[1]
        n_features = X_train.shape[2]
        best_val_loss, best_params, best_model, best_history = float("inf"), None, None, None

        for units, dropout, batch_size in itertools.product(*PARAM_GRID.values()):
            candidate = Sequential([
                Input(shape=(lookback, n_features)),
                LSTM(units),
                Dropout(dropout),
                Dense(1),
            ])
            candidate.compile(optimizer="adam", loss="mse")
            hist = candidate.fit(
                X_train, y_train,
                validation_split=0.15, epochs=100, batch_size=batch_size,
                callbacks=[EarlyStopping(patience=8, restore_best_weights=True)],
                verbose=0,
            )
            val_losses = [v for v in hist.history.get("val_loss", []) if not np.isnan(v)]
            if val_losses:
                val_loss = min(val_losses)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = {"units": units, "dropout": dropout, "batch_size": batch_size}
                    best_model, best_history = candidate, hist.history

        if best_model is None or best_history is None:
            st.error("Training failed to produce valid validation loss (models returned NaN). Ensure all columns in Clean.py are properly handled.")
            st.stop()

    st.session_state.model = best_model
    st.session_state.train_report = {
        "best_params": best_params,
        "val_loss": best_val_loss,
        "loss_history": best_history["loss"],
        "val_loss_history": best_history["val_loss"],
    }

if "train_report" in st.session_state:
    r = st.session_state.train_report
    st.success(f"Best hyperparameters: {r['best_params']} (val_loss = {r['val_loss']:.4f})")
    st.line_chart({"loss": r["loss_history"], "val_loss": r["val_loss_history"]})
else:
    st.info('Click "Train & tune" to fit the LSTM on the sequences from the Prepare page.')