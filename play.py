import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



#reading the csv file
cfa = pd.read_csv("/Users/nayan/OneDrive - Temple University/SAP Interview Prep and Process/ChickfilA/chickfilaData.csv", index_col=0)

#cleaning the csv file to only use necessary columns
cols2drop = ['Playground', 'Breakfast Served', 'WiFi', 'location']
cfa.drop(columns=cols2drop, inplace=True)

#cleaning the data by removing the rows with nulls values in any column
cols2check = ['cost of chicken sandwich', 'state', 'Mobile Orders', 'Catering', 'Pickup', 'Delivery']
cfa.dropna(subset=cols2check, inplace=True)

#getting the average state price by grouping the price by state
avg_statePrice = cfa.groupby('state')['cost of chicken sandwich'].mean().reset_index()
avg_statePrice.columns = ['state','avg_statePrice']
cfa = pd.merge(cfa, avg_statePrice, on = 'state')

#training the model using linearRegression
x = cfa[['Mobile Orders', 'Catering', 'Pickup', 'Delivery', 'avg_statePrice']]
y = cfa['cost of chicken sandwich']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
mse = mean_squared_error(y_test, y_predict)


# Streamlit user interface
st.title("Average Chick-fil-A Chicken Sandwich Price according to the state")

st.write(f'Mean Squared Error: {mse:.2f}')
#getting user_inpt
state = st.selectbox('Select State', cfa['state'].unique())
mobile_orders = st.checkbox('Mobile Orders')
catering = st.checkbox('Catering')
pickup = st.checkbox('Pickup')
delivery = st.checkbox('Delivery')

#average price
if(st.button('Predict Price')):
    input_data = [[mobile_orders, catering, pickup, delivery, cfa.loc[cfa['state'] == state, 'avg_statePrice'].values[0]]]
    predicted_price = model.predict(input_data)[0]
    st.write(f'Average Price of Chicken Sandwich at Chick-Fil-A in {state}: ${predicted_price:.2f}')

