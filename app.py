from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
MODEL_FILE = BASE / "artifacts" / "house_price_model.joblib"
DATA_FILE = BASE / "train.csv"

st.set_page_config(
    page_title="Kerala House Price Prediction",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Kerala House Price Prediction")
st.write("Machine Learning based house price estimation")

@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        return None
    return joblib.load(MODEL_FILE)

@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        return None
    return pd.read_csv(DATA_FILE)

bundle = load_model()

if bundle is None:
    st.error("Model not found. Please train the model first.")
    st.code("python train_model.py")
    st.stop()

df = load_data()
model = bundle["model"]
locations = bundle["categories"]["location"]

tab1, tab2, tab3 = st.tabs([
    "🏡 Predict Price",
    "📊 Dataset",
    "📈 Model Evaluation",
])

# Prediction
with tab1:
    st.subheader("Enter Property Details")

    with st.form("prediction_form"):
        area = st.number_input(
            "Area (square feet)",
            min_value=100,
            max_value=20000,
            value=1500,
            step=100,
        )

        bedrooms = st.number_input(
            "Number of Bedrooms",
            min_value=1,
            max_value=15,
            value=3,
        )

        bathrooms = st.number_input(
            "Number of Bathrooms",
            min_value=1,
            max_value=15,
            value=2,
        )

        location = st.selectbox(
            "Location in Kerala",
            locations,
        )

        submitted = st.form_submit_button(
            "Predict House Price",
            use_container_width=True,
        )

    if submitted:
        house = pd.DataFrame([{
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "location": location,
        }])

        price = float(model.predict(house)[0])

        if not np.isfinite(price) or price <= 0:
            st.warning("Invalid prediction. Please try different inputs.")
        else:
            st.success("Prediction completed!")

            col1, col2 = st.columns(2)

            col1.metric(
                "Estimated House Price",
                f"₹{price:,.0f}",
            )

            col2.metric(
                "Price in Lakhs",
                f"₹{price / 100000:.2f} lakh",
            )

            st.info(
                "This is an educational estimate based on "
                "15 sample records, not an actual market valuation."
            )

# Dataset
with tab2:
    st.subheader("Kerala House Price Dataset")

    if df is not None:
        col1, col2 = st.columns(2)
        col1.metric("Total Records", len(df))
        col2.metric("Total Columns", len(df.columns))

        st.dataframe(df, use_container_width=True)

        st.subheader("House Price Distribution")

        fig, ax = plt.subplots()
        ax.hist(df["price"], bins=8, color="skyblue", edgecolor="black")
        ax.set_xlabel("Price in Rupees")
        ax.set_ylabel("Number of Houses")
        ax.set_title("Price Distribution")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Average Price by Location")
        location_prices = df.groupby("location")["price"].mean()
        st.bar_chart(location_prices)

# Evaluation
with tab3:
    st.subheader("Model Evaluation")

    st.write("Selected Model:", bundle["best_model_name"])

    metrics = pd.DataFrame(bundle["metrics"]).T
    st.dataframe(metrics, use_container_width=True)

    st.caption(
        "MAE and RMSE are measured in rupees. "
        "Lower error is better. R² closer to 1 is generally better."
    )

    actual = np.asarray(bundle["y_test"])
    predicted = np.asarray(bundle["predictions"])

    fig, ax = plt.subplots()
    ax.scatter(actual, predicted, color="blue", alpha=0.7)

    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())
    ax.plot([low, high], [low, high], "r--")

    ax.set_xlabel("Actual Price (₹)")
    ax.set_ylabel("Predicted Price (₹)")
    ax.set_title("Actual vs Predicted House Prices")

    st.pyplot(fig)
    plt.close(fig)

st.divider()

st.caption(
    "College project demonstration. Predictions are not "
    "professional valuations or guaranteed market prices."
)