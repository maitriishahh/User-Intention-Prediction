import streamlit as st
import pandas as pd
import numpy as np

from user_intention_prediction.pipeline.prediction_pipeline import PredictionPipeline


# Page Config
st.set_page_config(
    page_title="User Purchase Prediction",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 User Purchase Intention Prediction")

st.info(
"""
Adjust the sliders to simulate user browsing behavior.
The model predicts whether the user is likely to make a purchase.
"""
)

# Initialize Pipeline
pipeline = PredictionPipeline()

st.subheader("User Browsing Behavior")

col1, col2 = st.columns(2)

# Left Column
with col1:

    administrative = st.slider(
        "Account / Login Pages Visited",
        0, 20, 3
    )

    informational = st.slider(
        "Information Pages Viewed",
        0, 10, 1
    )

    product_related = st.slider(
        "Products Browsed",
        0, 100, 10
    )

    bounce_rate = st.slider(
        "Quick Exit Level",
        0.0, 1.0, 0.2
    )

    exit_rate = st.slider(
        "Likelihood User Leaves",
        0.0, 1.0, 0.2
    )


# Right Column
with col2:

    page_values = st.slider(
        "Purchase Intent Score",
        0.0, 100.0, 5.0
    )

    month = st.selectbox(
        "Month",
        ["Jan", "Feb", "Mar", "Apr", "May", "June",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )

    visitor_type = st.selectbox(
        "Visitor Type",
        ["Returning_Visitor", "New_Visitor"]
    )

    browsing_day = st.selectbox(
        "Browsing Day",
        ["Weekday", "Weekend"]
    )

    weekend = 1 if browsing_day == "Weekend" else 0


# Try Example Button
if st.button("Try Example"):

    administrative = 3
    informational = 1
    product_related = 20
    bounce_rate = 0.02
    exit_rate = 0.05
    page_values = 25
    month = "Nov"
    visitor_type = "Returning_Visitor"
    weekend = 0

    st.success("Example data loaded!")


# Predict Button
if st.button("Predict Purchase"):

    # Auto-generate duration features
    administrative_duration = administrative * 60
    informational_duration = informational * 80
    product_related_duration = product_related * 120

    user_input = {
        "Administrative": administrative,
        "Administrative_Duration": administrative_duration,

        "Informational": informational,
        "Informational_Duration": informational_duration,

        "ProductRelated": product_related,
        "ProductRelated_Duration": product_related_duration,

        "BounceRates": bounce_rate,
        "ExitRates": exit_rate,
        "PageValues": page_values,
        "Month": month,
        "VisitorType": visitor_type,
        "Weekend": weekend
    }

    model = pipeline.load_model()

    prediction, probabilities = pipeline.predict(model, user_input)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        result = "Purchase"
    else:
        result = "Non-Purchase"

    st.divider()
    st.markdown(f"**Prediction:** {result}")
    st.markdown(f"**Purchase Probability:** {probabilities[0]*100:.2f}%")