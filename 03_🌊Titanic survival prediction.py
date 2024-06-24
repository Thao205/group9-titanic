import os
import pickle
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
train = pd.read_csv('Titanic.csv')
notebook_config = {  
    'random_state': 12345,  # for the GradientBoostingClassifier
    'n_jobs': 1 ,           # for the cross_val_score
    'cv': 10                # for the cross_val_score
}

# Data cleaning
train["Age"] = train["Age"].fillna(train["Age"].mean())
train["Embarked"] = train["Embarked"].fillna("S")

# Check for any remaining missing values
missing_values = train.isnull().sum()
print(missing_values)

train = train.drop(columns=['PassengerId', 'Name', 'Cabin', 'Ticket'])
train.loc[train['Fare'] > 300, 'Fare'] = 300
train['num_relatives'] = train['SibSp'] + train['Parch']

# Ensure consistent data types
train['Pclass'] = train['Pclass'].astype(int)
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1}).astype(int)
train['Embarked'] = train['Embarked'].map({'S': 'S', 'C': 'C', 'Q': 'Q'}).astype(str)

# Preprocessing pipelines
age_pipe = Pipeline(steps=[
    ('age_imp', SimpleImputer(strategy='median')),
    ('age_scale', MinMaxScaler())
])
fare_pipe = Pipeline(steps=[
    ('fare_imp', SimpleImputer(strategy='mean')),
    ('fare_scale', MinMaxScaler())
])
embarked_pipe = Pipeline(steps=[
    ('embarked_imp', SimpleImputer(strategy='most_frequent')),
    ('embarked_onehot', OneHotEncoder(drop=None))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('age_pipe', age_pipe, ['Age']),
        ('fare_pipe', fare_pipe, ['Fare']),
        ('embarked_pipe', embarked_pipe, ['Embarked']),
        ('minmax_scaler', MinMaxScaler(), ['SibSp', 'Parch', 'num_relatives']),
        ('pclass_onehot', OneHotEncoder(drop=None), ['Pclass']),
        ('sex_onehot', OneHotEncoder(drop='first'), ['Sex'])
    ]
)

# Gradient Boosting Classifier
y_train = train['Survived'].values
x_train = train.drop(columns=['Survived'])

grad_boost = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('grad_boost', GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=notebook_config['random_state']
    ))
])

grad_boost_acc = cross_val_score(
    estimator=grad_boost,
    X=x_train,
    y=y_train,
    scoring='accuracy',
    cv=notebook_config['cv'],
    n_jobs=notebook_config['n_jobs']
)

print('best GradientBoosting acc (mean) =', round(np.mean(grad_boost_acc), 2))
print('best GradientBoosting acc (std)  =', round(np.std(grad_boost_acc), 2))

grad_boost.fit(x_train, y_train)

# Save the model
with open("grad_boost.pkl", "wb") as f:
    pickle.dump(grad_boost, f)

# Define Streamlit functions
def get_user_data() -> pd.DataFrame:
    Titanic = {}

    Titanic['Age'] = st.slider('Enter Age:', min_value=0, max_value=80, value=20, step=1)
    Titanic['Fare'] = st.slider('How much did your ticket cost you? (in 1912$):', min_value=0, max_value=500, value=80, step=1)
    Titanic['SibSp'] = st.slider('Number of siblings and spouses aboard:', min_value=0, max_value=15, value=3, step=1)
    Titanic['Parch'] = st.slider('Number of parents and children aboard:', min_value=0, max_value=15, value=3, step=1)

    col1, col2, col3 = st.columns(3)
    Titanic['Pclass'] = col1.radio('Ticket class:', options=['1st', '2nd', '3rd'])
    Titanic['Sex'] = col2.radio('Sex:', options=['Man', 'Woman'])
    Titanic['Embarked'] = col3.radio('Port of Embarkation:', options=['Cherbourg', 'Queenstown', 'Southampton'], index=2)

    # Convert inputs to the same format as training data
    Titanic['Sex'] = 0 if Titanic['Sex'] == 'Man' else 1
    Titanic['Pclass'] = {'1st': 1, '2nd': 2, '3rd': 3}[Titanic['Pclass']]
    Titanic['Embarked'] = {'Cherbourg': 'C', 'Queenstown': 'Q', 'Southampton': 'S'}[Titanic['Embarked']]
    Titanic['num_relatives'] = Titanic['SibSp'] + Titanic['Parch']

    df = pd.DataFrame([Titanic])

    return df

@st.cache_resource
def load_model(model_file_path: str):
    with st.spinner("Loading model..."):
        with open(model_file_path, 'rb') as file:
            model = pickle.load(file)
    return model

def main():
    model_name = 'grad_boost.pkl'
    project_path = os.path.abspath(os.path.join(__file__, "../../"))

    st.header('Would you have survived the Titanic?🚢')
    df_user_data = get_user_data()

    model = load_model(model_file_path=os.path.join(project_path, model_name))
    prob = model.predict_proba(df_user_data)[0][1]
    prob = int(prob * 100)

    emojis = ["😕", "🙃", "🙂", "😀"]
    state = min(prob // 25, 3)

    st.write('')
    st.title(f'{prob}% chance to survive! {emojis[state]}')
    if state == 0:
        st.error("Good luck next time, you will be next Jack! ☠️")
        st.image('image_copy_7.png')
    elif state == 1:
        st.warning("Hey... I hope you know how to swim, maybe you have to do it! 🏊‍♂️")
    elif state == 2:
        st.info("Well done! You are on the right track, but don't get lost! 💪")
    else:
        st.success('Congratulations! You can rest assured, you will be fine! 🤩')
        st.image('image_copy_6.png')

    if st.button("Facts"):
        st.markdown("""
        # Insider Survival Facts Based on the dataset from Kaggle:
       
        - **Overall Survival Rate:**
            - Only about 38.4% of passengers survived in this accident.
       
        - **Survival Rate by Gender:**
            - **Females:** 74.2% survival rate
            - **Males:** 18.9% survival rate
       
        - **Survival Rate by Ticket Class:**
            - **1st Class:** 62.96% survival rate
            - **2nd Class:** 47.28% survival rate
            - **3rd Class:** 24.24% survival rate
       
        - **Survival Rate by Age Group:**
            - **Children (0-12 years):** Approximately 58.33% survival rate
            - **Teenagers (13-19 years):** Approximately 38.46% survival rate
            - **Adults (20-50 years):** Approximately 35.80% survival rate
            - **Seniors (50+ years):** Approximately 28.33% survival rate
       
        - **Survival Rate by Family Size:**
            - **Alone (0 relatives):** 30.13% survival rate
            - **Small Family (1-2 relatives):** 52.98% survival rate
            - **Large Family (3+ relatives):** 33.33% survival rate
        """)

if __name__ == '__main__':
    main()
