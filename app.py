import streamlit as st
import pandas as pd
from io import BytesIO

# PAGE CONFIG
st.set_page_config(layout="wide")
st.title("DAILY DEMAND FORECAST")

# LOAD DATA
forecast = pd.read_parquet("outputs/final_planning.parquet")
forecast['type'] = 'Forecast'
forecast['trx_date'] = pd.to_datetime(forecast['trx_date'])
forecast['day_name'] = forecast['trx_date'].dt.day_name()

# MAP HARI KE INDO
hari_map = {
    "Monday": "Senin",
    "Tuesday": "Selasa",
    "Wednesday": "Rabu",
    "Thursday": "Kamis",
    "Friday": "Jumat",
    "Saturday": "Sabtu",
    "Sunday": "Minggu"}
forecast['day_name_id'] = forecast['day_name'].map(hari_map)

# SIDEBAR FILTER
st.sidebar.header("Filter Forecast")

# Store (SINGLE SELECT + ALL)
store_list = sorted(forecast['store'].unique())
store_options = ["All"] + store_list

selected_store = st.sidebar.selectbox(
    "Store",
    options=store_options,
    index=0   # default = All
)

# Material (CHECKBOX)
with st.sidebar.expander("Material", expanded=True):
    selected_material = []
    for m in sorted(forecast['material'].unique()):
        if st.checkbox(m, value=True, key=f"material_{m}"):
            selected_material.append(m)

# Type (CHECKBOX)
with st.sidebar.expander("Type", expanded=True):
    selected_type = []
    for t in sorted(forecast['type'].unique()):
        if st.checkbox(t, value=True, key=f"type_{t}"):
            selected_type.append(t)

# FILTER DATA
filtered = forecast.copy()

# Filter store hanya kalau bukan All
if selected_store != "All":
    filtered = filtered[filtered['store'] == selected_store]

filtered = filtered[
    filtered['material'].isin(selected_material) &
    filtered['type'].isin(selected_type)]

# LABEL TANGGAL + HARI (INDO)
filtered['trx_date_label'] = (
    filtered['trx_date'].dt.strftime('%Y-%m-%d')
    + ' ('
    + filtered['day_name_id']
    + ')')

# PIVOT TABLE 
table = filtered.pivot(
    index=['store', 'material'],
    columns='trx_date_label',
    values='forecast_qty')

st.subheader("Forecast Table")
st.dataframe(
    table.fillna(0).astype(int),
    use_container_width=True)

# DOWNLOAD EXCEL 
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Forecast')
    return output.getvalue()

st.download_button(
    label='Download Forecast Excel',
    data=to_excel(table.fillna(0).astype(int)),
    file_name='daily_forecast.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
