import streamlit as st
import pandas as pd
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
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center center;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the background image
set_background('/workspaces/group9-titanic/image copy 4.png')

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'cover'

# Function to hide the sidebar
def hide_sidebar():
    hide_sidebar_style = """
        <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        </style>
    """
    st.markdown(hide_sidebar_style, unsafe_allow_html=True)

# Function to display the introduction page
def cover_page():
    st.markdown("<h1 style='text-align: center; color: white;'>TITANIC DATA</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: left; color: white;'>Introduction</h2>", unsafe_allow_html=True)
    st.markdown("""
        <p style='color:white'>
        Welcome to the company’s database. Here you will find all the necessary information you will need to study tourists’ behaviors, 
        ranging from expenditure of services to discovering the most popular destinations and exploring spending patterns among many others. 
        Hopefully, our data will help you understand what we are doing here as a travelling/tourism company.
        </p>
    """, unsafe_allow_html=True)
    if st.button("Let's get started"):
        st.session_state.page = 'main'
        st.experimental_rerun()

# Load the Titanic dataset from the workspace directory
@st.cache_data  # Cache the dataset for improved performance
def load_data():
    Titanic = pd.read_csv('/workspaces/group9-titanic/Titanic.csv')
    return Titanic

# Function to display the main app content
def main_app():
    st.title("Hi👋 we're from group 9 class Business Class𓍢ִ໋🌷͙֒")
    st.text('This is a web app to allow exploration of Titanic Survival🚢')

    # Load the dataset if it's not already loaded
    Titanic = load_data()

    # Sidebar setup
    st.sidebar.title('About This App✨')

    # Sidebar options
    options = st.sidebar.radio('Select what you want to display👇:', ['Home', 'Our Dataset', 'Data Summary', 'Data Header'])

    # Navigation options
    if options == 'Home':
        st.header('Introduction')
        st.balloons()
    elif options == 'Our Dataset':
        st.header("This is our Dataset📚")
        st.dataframe(Titanic)
    elif options == 'Data Summary':
        st.header('Statistics of Dataframe')
        st.write(Titanic.describe())
    elif options == 'Data Header':
        st.header('Header of Dataframe')
        st.write(Titanic.head())

# Conditional display based on session state
if st.session_state.page == 'cover':
    hide_sidebar()
    cover_page()
else:
    main_app()

# Additional styling if needed
st.markdown("""
<style>
body {
    font-family: 'Arial', sans-serif;
}
/* Sidebar background color */
[data-testid="stSidebar"] {
    background-color: #002147;  /* Dark navy background to match Titanic theme */
}
/* Sidebar text color */
[data-testid="stSidebar"] .css-1lcbmhc, [data-testid="stSidebar"] .css-145kmo2 {
    color: #FFFFFF;  /* Light text color */
}
/* Sidebar header color */
[data-testid="stSidebar"] .css-1a32fsj, [data-testid="stSidebar"] .css-1v3fvcr, [data-testid="stSidebar"] .css-1d391kg {
    color: #FFD700;  /* Gold color for headers */
}
</style>
""", unsafe_allow_html=True)

    color: #FFD700;  /* Gold color for headers */
}
</style>
""", unsafe_allow_html=True)

