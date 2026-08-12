import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Battery Cycle Life", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_parquet('models/app_data.parquet')
    dq = dict(np.load('models/dq_curves.npz'))
    fade = dict(np.load('models/fade_curves.npz'))
    vd = np.load('models/vdlin.npy')
    return df, dq, fade, vd


@st.cache_resource
def load_model():
    with open('models/LR_8feat_selected.pkl', 'rb') as f:
        return pickle.load(f)


app_df, dq_curves, fade_curves, vdlin = load_data()
bundle = load_model()
model, FEATS = bundle['pipeline'], bundle['features']

st.title("Li-ion Battery Cycle Life Prediction")
st.caption("Predicting total cycle life from the first 100 cycles — Severson et al., Nature Energy 2019")

cell_id = st.selectbox("Select a cell", app_df.index.tolist())
row = app_df.loc[cell_id]

pred_log = model.predict(app_df.loc[[cell_id], FEATS])[0]
pred_cycles = 10 ** pred_log
actual = row.cycle_life
err_pct = (pred_cycles - actual) / actual * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted", f"{pred_cycles:,.0f} cycles")
c2.metric("Actual", f"{actual:,.0f} cycles")
c3.metric("Error", f"{err_pct:+.1f}%")
c4.metric("Charging policy", row.policy)

import matplotlib.pyplot as plt

st.divider()
left, right = st.columns(2)

with left:
    st.subheader("Capacity fade")
    fade = fade_curves[cell_id]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(np.arange(1, len(fade) + 1), fade, lw=1)
    ax.axhline(0.88, ls='--', c='k', lw=0.8, label='end of life')
    ax.axvline(100, ls=':', c='r', lw=1.2, label='prediction made here')
    ax.set_xlabel('cycle'); ax.set_ylabel('discharge capacity / Ah')
    ax.legend(fontsize=8)
    st.pyplot(fig)
    st.caption("Nothing has visibly happened by cycle 100 — that is the problem this model solves.")

with right:
    st.subheader("ΔQ(V) = Q₁₀₀(V) − Q₁₀(V)")
    dq = dq_curves[cell_id]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(vdlin, dq, lw=1)
    ax.invert_xaxis()
    ax.set_xlabel('voltage / V'); ax.set_ylabel('ΔQ / Ah')
    st.pyplot(fig)
    st.caption(f"log₁₀ variance = {row.log_var_dQ:.3f} — the model's strongest input.")

    st.divider()
st.subheader("Prediction interval and reliability")

RESID_STD = 0.0532          # from the 8-feature model, batches 1-2
lo = 10 ** (pred_log - 1.96 * RESID_STD)
hi = 10 ** (pred_log + 1.96 * RESID_STD)

st.write(f"**95% interval:** {lo:,.0f} – {hi:,.0f} cycles  (±27%)")

train = app_df
outside = [f for f in FEATS
           if not (train[f].min() <= row[f] <= train[f].max())]

if outside:
    st.warning(
        f"This cell falls outside the model's training range on: **{', '.join(outside)}**. "
        "The ±27% interval was computed from cells inside that range and does not apply here. "
        "When this model was tested on an unseen production run, cells beyond the training "
        "range were systematically under-predicted — treat this number as a lower bound."
    )
    
else:
    st.success("All feature values fall within the training range.")