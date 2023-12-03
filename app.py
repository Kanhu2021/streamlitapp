import pandas as pd
import numpy as np
import streamlit as st
from prophet import Prophet
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import plot
from prophet.plot import plot_plotly,plot_components_plotly
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',160)



states = ['Uttar Pradesh', 'Uttarakhand', 'Jammu And Kashmir', 'Punjab',
       'Chhattisgarh', 'Gujarat', 'nan', 'Manipur', 'Himachal Pradesh',
       'Chandigarh', 'Haryana', 'Delhi', 'Telangana', 'Assam',
       'Madhya Pradesh', 'Puducherry', 'Arunachal Pradesh',
       'Dadra And Nagar Haveli', 'Mizoram', 'Jharkhand', 'Bihar',
       'Meghalaya', 'Tripura', 'Kerala']

df_final=pd.read_excel("ACCELERATED_HYPERTENSION - Copy.xlsx",usecols=['admission_dt_pat','proc_name_1','hospital_name_npt','hosp_state'])
df_final['month'] = df_final['admission_dt_pat'].dt.strftime('%m %B')
df_final['year'] = df_final['admission_dt_pat'].dt.year

st.title('Diseases prediction')
col1,col2 = st.columns(2)

with col1:
     st.selectbox('Duration from 2019-2023','2019-2023')
with col2:
    state_name = st.selectbox('States',states)
# Describibg data
st.subheader('Data Description')
st.write(df_final.describe())
## visualization

st.subheader('Sample Data')
df_state_name = df_final[df_final.hosp_state==state_name]
st.write(df_state_name.head(10))

st.subheader('Daily Trend Of The Data')
df_state_name = df_state_name.groupby('admission_dt_pat').size().reset_index(name='count')
fig =plt.figure(figsize=(20,7))
plt.plot('admission_dt_pat','count',data=df_state_name)
plt.xticks(fontsize=15,rotation=45)
plt.yticks(fontsize=15,rotation=45)
plt.show()
st.pyplot(fig)

st.subheader('Monthly Trend Of The Data')
df_state_name = df_final[df_final.hosp_state==state_name]
df_state_name = df_state_name.groupby('month').size().reset_index(name='count')
fig =plt.figure(figsize=(20,7))
plt.plot('month','count',data=df_state_name)
plt.xticks(fontsize=15,rotation=45)
plt.yticks(fontsize=15,rotation=45)
plt.show()
st.pyplot(fig)

st.subheader('Yearly Trend Of The Data')
df_state_name = df_final[df_final.hosp_state==state_name]
df_state_name = df_state_name.groupby('year').size().reset_index(name='count')
fig =plt.figure(figsize=(20,7))
plt.plot('year','count',data=df_state_name)
plt.xticks(fontsize=15,rotation=45)
plt.yticks(fontsize=15,rotation=45)
plt.show()
st.pyplot(fig)

st.subheader('Future forcasting--')
m1 = Prophet()
df_state_name = df_final[df_final.hosp_state==state_name]
df_state_name = df_state_name.groupby('admission_dt_pat').size().reset_index(name='count')
df_state_name.columns = ['ds','y']
pro1 = m1.fit(df_state_name)
    
futur = m1.make_future_dataframe(periods=1000,freq='D')
forcast = pro1.predict(futur)
#fig = plt.figure(figsize=(20,8))
#plot_plotly(pro1,forcast)
#st.pyplot(fig)
#fig = pro1.plot(forcast) 
#st.write(fig)
fig = plot_plotly(pro1,forcast)
st.plotly_chart(fig)

fig = plot_components_plotly(pro1,forcast)
st.plotly_chart(fig)