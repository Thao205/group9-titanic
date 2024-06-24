import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64
# Function to set a background image
def set_background(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the background image
set_background('/workspaces/group9-titanic/image copy.png')
# Load the Titanic dataset from the workspace directory
@st.cache_data  # Cache the dataset for improved performance
def load_data():
    Titanic = pd.read_csv('/workspaces/group9-titanic/Titanic.csv')
    return Titanic

def eda(df):
    explore_dataset_option = st.checkbox("Explore Dataset")

    if explore_dataset_option:
        with st.expander("Explore Dataset Options", expanded=True):
            show_dataset_summary_option = st.checkbox("Show Dataset Summary")
            if show_dataset_summary_option:
                st.write(df.describe())
            show_dataset = st.checkbox("Show Dataset")
            if show_dataset:
                number = st.number_input("Number of rows to view", min_value=1, value=5)
                st.dataframe(df.head(number))

            show_columns_option = st.checkbox("Show Columns Names")
            if show_columns_option:
                st.write(df.columns)

            show_shape_option = st.checkbox("Show Shape of Dataset")
            if show_shape_option:
                st.write(df.shape)
                data_dim = st.radio("Show Dimension by ", ("Rows", "Columns"))
                if data_dim == "Columns":
                    st.text("Number of Columns")
                    st.write(df.shape[1])
                elif data_dim == "Rows":
                    st.text("Number of Rows")
                    st.write(df.shape[0])
                else:
                    st.write(df.shape)

            select_columns_option = st.checkbox("Select Column to show")
            if select_columns_option:
                all_columns = df.columns.tolist()
                selected_columns = st.multiselect("Select Columns", all_columns)
                new_df = df[selected_columns]
                st.dataframe(new_df)

            show_value_counts_option = st.checkbox("Show Value Counts")
            if show_value_counts_option:
                all_columns = df.columns.tolist()
                selected_columns = st.selectbox("Select Column", all_columns)
                st.write(df[selected_columns].value_counts())

            show_data_types_option = st.checkbox("Show Data types")
            if show_data_types_option:
                st.text("Data Types")
                st.write(df.dtypes)

            show_summary_option = st.checkbox("Show Summary")
            if show_summary_option:
                st.text("Summary")
                st.write(df.describe().T)

            show_raw_data_option = st.checkbox('Show Raw Data')
            if show_raw_data_option:
                raw_data_rows = st.number_input("Number of Rows for Raw Data", min_value=1, value=5)
                raw_data_selection = df.head(raw_data_rows)
                selected_columns = st.multiselect("Select Columns", df.columns.tolist(), default=df.columns.tolist())
                new_df = raw_data_selection[selected_columns]
                st.dataframe(new_df)

def app():
    st.title("Data Explorer")
    st.subheader("Explore Dataset")
    
    # Load the dataset
    df = load_data()
    
    # Call the EDA function
    eda(df)

if __name__ == "__main__":
    app()
