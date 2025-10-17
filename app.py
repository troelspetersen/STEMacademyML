import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Mini Streamlit Demo", layout="wide")

st.title("Mini Streamlit Demo App")
st.markdown("A compact demo showing common Streamlit patterns: widgets, file upload, plotting, caching and session state.")

# Sidebar controls
st.sidebar.header("Controls")
sample_size = st.sidebar.slider("Sample size for demo data", min_value=50, max_value=2000, value=300, step=50)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

# Session state example
if "visits" not in st.session_state:
    st.session_state.visits = 0
if st.button("Register visit"):
    st.session_state.visits += 1
st.sidebar.write("Visits this session:", st.session_state.visits)

# Caching example for generating data
@st.cache_data
def make_demo_df(n, random_seed=0):
    rng = np.random.default_rng(random_seed)
    x = rng.normal(loc=0.0, scale=1.0, size=n)
    y = 0.5 * x + rng.normal(scale=0.8, size=n)
    cat = rng.choice(["A", "B", "C"], size=n)
    return pd.DataFrame({"x": x, "y": y, "category": cat})

# Main area
st.header("1) Interactive demo data")
df = make_demo_df(sample_size, seed)
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Data preview")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)

with col2:
    st.subheader("Options")
    show_raw = st.checkbox("Show raw CSV upload area")
    color_by = st.selectbox("Color by", ["category", "none"], index=0)

# Plotting
st.header("2) Plot")
chart = alt.Chart(df).mark_circle(size=60).encode(
    x="x",
    y="y",
    tooltip=["x", "y", "category"],
    color=alt.Color("category") if color_by == "category" else alt.value("steelblue"),
).interactive()
st.altair_chart(chart, use_container_width=True)

# File upload example
st.header("3) Upload your CSV")
uploaded = st.file_uploader("Upload a CSV file to explore", type=["csv"])
if uploaded:
    try:
        user_df = pd.read_csv(uploaded)
        st.success("CSV loaded — showing first 5 rows")
        st.dataframe(user_df.head())
        if st.checkbox("Show dataset summary"):
            st.write(user_df.describe(include="all"))
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
elif show_raw:
    st.info("Upload a CSV file using the control above to preview it here.")

# Download button example
st.header("4) Download sample data")
if st.button("Download demo CSV"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Click to download", data=csv, file_name="demo_data.csv", mime="text/csv")

st.write("---")
st.caption("This app demonstrates some common Streamlit features. Run with: `streamlit run app.py`")
