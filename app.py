import streamlit as st

st.set_page_config(page_title="Sri Murugan Traders Analytics", layout="wide")

st.title("Sri Murugan Traders - Predictive Analytics Dashboard")
st.info("Dashboard UI will be connected to the ML outputs after the final dataset structure is confirmed.")

st.subheader("Planned Dashboard")
cols = st.columns(4)
cols[0].metric("Total Sales", "Run main.py")
cols[1].metric("Revenue", "Run main.py")
cols[2].metric("Forecast", "30 Days")
cols[3].metric("Inventory", "Demand Based")

st.write("The production dashboard will display sales trends, top products, model performance, forecasts and reorder recommendations.")
