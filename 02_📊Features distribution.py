import streamlit as st
import pandas as pd
import plotly.express as px
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
    return pd.read_csv('/workspaces/group9-titanic/Titanic.csv')

# Visualize the survival distribution
def visualize_survival_distribution(df):
    st.subheader('Survival Distribution')

    # Convert the columns to string type
    df['Survived'] = df['Survived'].map({0: 'Not Survived', 1: 'Survived'})
    
    # Map 'Embarked' values to readable names
    df['Embarked'] = df['Embarked'].fillna('Unknown')
    df['Embarked'] = df['Embarked'].map({'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown', 'Unknown': 'Unknown'})

    # Create the sunburst plot
    fig_survival = px.sunburst(df, path=['Survived', 'Sex', 'Embarked'],
                               color='Survived', color_discrete_map={'Survived': '#1f77b4', 'Not Survived': '#d62728'})
    
    st.plotly_chart(fig_survival)

# Function to visualize age distribution with enhanced interactivity
def visualize_age_distribution(df):
    st.subheader('Age Distribution')

    # Adding a range slider for selecting age range
    age_range = st.slider('Select Age Range', 
                          min_value=int(df['Age'].min()), 
                          max_value=int(df['Age'].max()), 
                          value=(int(df['Age'].min()), int(df['Age'].max())), step=20)

    # Plotting the histogram based on selected options
    fig_age = px.histogram(df, 
                           x='Age', 
                           range_x=age_range,
                           nbins=20,
                           title='Age Distribution',
                           labels={'Age': 'Age Count'},
                           color_discrete_sequence=['skyblue'])
    
    # Displaying the plotly chart
    st.plotly_chart(fig_age)

def visualize_family_size_distribution(df):
    st.subheader('Family Size Distribution')

    # Calculating family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Categorizing family size
    df['FamilyCategory'] = df['FamilySize'].apply(lambda x: 'Alone' if x == 1 else ('Small Family' if x <= 4 else 'Large Family'))

    # Sidebar checkboxes for filtering by family size categories
    st.sidebar.subheader('Filter by Family Size')
    alone_checkbox = st.sidebar.checkbox('Alone', value=True)
    small_family_checkbox = st.sidebar.checkbox('Small Family', value=True)
    large_family_checkbox = st.sidebar.checkbox('Large Family', value=True)

    # Filter the dataframe based on selected checkboxes
    selected_categories = []
    if alone_checkbox:
        selected_categories.append('Alone')
    if small_family_checkbox:
        selected_categories.append('Small Family')
    if large_family_checkbox:
        selected_categories.append('Large Family')

    filtered_df = df[df['FamilyCategory'].isin(selected_categories)]

    # Counting family sizes in the filtered data
    family_size_counts = filtered_df['FamilySize'].value_counts().sort_index().reset_index()
    family_size_counts.columns = ['FamilySize', 'Count']

    # Creating a bar plot
    fig_family_size = px.bar(family_size_counts, x='FamilySize', y='Count',
                             title='Family Size Distribution',
                             labels={'FamilySize': 'Family Size', 'Count': 'Count'},
                             color='FamilySize', color_continuous_scale=px.colors.sequential.Blues)

    # Displaying the plotly chart
    st.plotly_chart(fig_family_size)

def visualize_class_distribution(df):
    # Add a reset button
    if st.button("Reset Plot"):
        st.experimental_rerun()
    
    # Add checkboxes for filtering by Sex
    selected_sex = st.multiselect("Select Sex", df['Sex'].unique(), default=df['Sex'].unique())
    
    # Filter the dataframe based on selected Sex
    filtered_df = df[df['Sex'].isin(selected_sex)]
    
    # Mutate the 'Survived' and 'Pclass' columns
    filtered_df['Survived'] = filtered_df['Survived'].map({0: 'Death', 1: 'Survived'})
    filtered_df['Pclass'] = filtered_df['Pclass'].map({1: '1st', 2: '2nd', 3: '3rd'})
    
    # Ensure Pclass is ordered
    filtered_df['Pclass'] = pd.Categorical(filtered_df['Pclass'], categories=["1st", "2nd", "3rd"], ordered=True)
    
    # Create the interactive facet grid boxplot using Plotly
    fig = px.box(filtered_df, x='Pclass', y='Age', color='Pclass',
                 category_orders={"Pclass": ["1st", "2nd", "3rd"]},
                 labels={"Pclass": "Ticket Class", "Age": "Passenger Age"},
                 height=600, width=1000)
    
    fig.update_layout(
        title={
            'text': f'Age Distribution by Ticket Class',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        })
    
    st.plotly_chart(fig)
# Main function to run the Streamlit app
def main():
    st.title('Titanic Dataset Analysis')

    # Load data
    Titanic = load_data()

    # Define button labels for radio buttons
    button_labels = [
        'Survival Distribution', 
        'Age Distribution', 
        'Family Size Distribution',
        'Class Distribution'
    ]

    # Create radio buttons for selecting visualization
    selected_chart = st.radio('Select Chart', button_labels, index=0)

    # Display selected chart based on user selection
    if selected_chart == 'Survival Distribution':
        visualize_survival_distribution(Titanic)
    elif selected_chart == 'Age Distribution':
        visualize_age_distribution(Titanic)
    elif selected_chart == 'Family Size Distribution':
        visualize_family_size_distribution(Titanic)
    elif selected_chart == 'Class Distribution':
        visualize_class_distribution(Titanic)

# Run the main function
if __name__ == "__main__":
    main()
