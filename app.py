# pip install streamlit
import streamlit as st
import pandas as pd
import os
import pickle
import json

from matplotlib import pyplot as plt
import seaborn as sns

st.title('Churn Prediction')
st.write('Hit Predict to score a new customer.')

data = os.path.join('..', 'data', 'train.csv')
df = pd.read_csv(data)

with open('schema.json', 'r') as f:
    schema = json.load(f)

options = {}
for column, column_properties in schema['column_info'].items():
    if column =='churn':
        pass
    elif column_properties['dtype'] == 'int64' or column_properties['dtype'] =='float64':
        feature_mean = ((column_properties['values'][0] + column_properties['values'][1]) / 2)
        if column_properties['dtype'] == 'int64':
            feature_mean = int(feature_mean)
        options[column] = st.sidebar.slider(column, column_properties['values'][0], column_properties['values'][1], value=feature_mean)
    elif column_properties['dtype'] == 'object':
        options[column] = st.sidebar.selectbox(column, column_properties['values'])

model_path = os.path.join('..', 'models', 'experiment_2', 'rf.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

if st.button('Predict'): 
    scoring_data = pd.Series(options).to_frame().T

    for col, values in schema['column_info'].items():
        if col in scoring_data.columns: 
            scoring_data[col] = scoring_data[col].astype(values['dtype'])

    scoring_data = pd.get_dummies(scoring_data)

    for col in schema['transformed_cols']['transformed_columns']:
        if col not in scoring_data.columns:
            scoring_data[col] = 0 

    # Reorder columns
    scoring_data = scoring_data[schema['transformed_cols']['transformed_columns']]

    prediction = model.predict(scoring_data)
    st.write('Predicted Outcome')
    st.write(prediction)

    st.write('Client Details')
    st.write(options)

# Save history
try:
    historical = pd.Series(options).to_frame().T
    historical['prediction'] = prediction 

    if os.path.isfile('historical_data.csv'):
        historical.to_csv('historical_data.csv', mode='a', header=False, index=False)
    else: 
        historical.to_csv('historical_data.csv', mode='a', header=True, index=False)

except: 
    pass

st.header('Historical Outcomes')
if os.path.isfile('historical_data.csv'):
    hist = pd.read_csv('historical_data.csv')
    st.dataframe(hist)
    fig, ax = plt.subplots()
    # ax.hist(hist['prediction'])
    sns.countplot(x='prediction', data=hist, ax=ax).set_title('Historical Predictions')
    st.pyplot(fig)
else:
    st.write('No historical data.')
