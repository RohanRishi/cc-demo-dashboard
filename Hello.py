# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import warnings
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit.logger import get_logger
import pydeck as pdk
import json
import requests


LOGGER = get_logger(__name__)


def run():

  st.set_page_config(layout='wide', initial_sidebar_state='collapsed')

  # warnings.filterwarnings('ignore')

  plt.style.use('fivethirtyeight')
  # sns.set()

  ##### Description: This is a dashboard for the loan tape data. It is a streamlit app that can be run locally or deployed to a server. #####


  # read Example Loan Tape.xlsx
  df = pd.read_excel('Example Loan Tape.xlsx')

  df = df[['Loan Type', 'Project Type', 'Unpaid Principal Balance at Purchase Date', 'Original Loan Amount', 'Loan Term (Months)', 'Interest Rate', 'Qualifying FICO', 'Initial Monthly Payment', 'State', 'Installer']]
  df.columns = ['LoanType', 'ProjectType', 'UPB', 'Bal', 'time', 'int', 'FICO', 'EMI', 'State', 'Installer']

  df['FICO bucket'] = pd.cut(df['FICO'], bins=[650, 699, 749, 799, 850], labels=['FICO 650-699', 'FICO 700-749', 'FICO 750-799', 'FICO 800+'])

  def weighted_avg(group, col, weights='Bal'):
      return sum(group[col] * group[weights]) / sum(group[weights])
  # Group by 'ProjectType'
  dfg_project = df.groupby('ProjectType')

  # 1. % total UPB by project type
  total_upb = dfg_project['UPB'].sum()
  pct_sum_upb = total_upb / total_upb.sum() * 100

  def calculate_metrics_by_tenor_weighted(start_year, end_year, df):
      """
      Filters the dataframe based on the provided tenor (in years) and calculates desired metrics
      for each FICO bucket using weighted averages.
      
      Args:
      - start_year (int): Starting year for the filter.
      - end_year (int): Ending year for the filter.
      - df (DataFrame): The dataframe containing the loan data.

      Returns:
      - result (DataFrame): A dataframe containing the calculated metrics.
      """
      
      # Convert years to months and filter the dataframe
      start_month = start_year * 12
      end_month = end_year * 12
      filtered_df = df[(df['time'] >= start_month) & (df['time'] < end_month)]
      
      # Group by 'FICO bucket' and calculate the required metrics
      dfg_ficob = filtered_df.groupby('FICO bucket')
      
      result = pd.DataFrame({
          'percentage_in_cohort_UPB': dfg_ficob['UPB'].sum() / filtered_df['UPB'].sum() * 100,
          'percentage_in_total_UPB': dfg_ficob['UPB'].sum() / total_upb.sum() * 100,
          'avg_original_balance': dfg_ficob['Bal'].mean(),
          'WA_tenor': dfg_ficob.apply(weighted_avg, col='time'),
          'WA_APR': dfg_ficob.apply(weighted_avg, col='int'),
          'WA_FICO': dfg_ficob.apply(weighted_avg, col='FICO'),
          'WA_Monthly_payment': dfg_ficob.apply(weighted_avg, col='EMI')
      })

      # Calculate distribution across loan product and project types
      loan_type_dist = dfg_ficob['LoanType'].value_counts(normalize=True).unstack().fillna(0)
      project_type_dist = dfg_ficob['ProjectType'].value_counts(normalize=True).unstack().fillna(0)
      
      # join the three dataframes
      result = result.join(loan_type_dist).join(project_type_dist)
      
      return result
  calculate_metrics_by_tenor_weighted(0, 10, df)
  
  # title
  st.title('Metrics by FICO Range')

  # Create a dropdown widget for selecting the date range
  tenure_options = {
      '5-10': (5, 10),
      '10-15': (10, 15),
      '15-20': (15, 20),
      '20-25': (20, 25),
      'aggregate': (0, 25)
  }

  selected_tenure = st.selectbox('Select Tenure Range:', list(tenure_options.keys()))

  start, end = tenure_options[selected_tenure]
  result_df = calculate_metrics_by_tenor_weighted(start, end, df).round(2)

  # Display the resulting DataFrame
  st.write(result_df)


  # groupby state and calculate metrics
  dfg_state = df.groupby('State')

  # 1. % total UPB by state
  total_upb = dfg_state['UPB'].sum()
  pct_sum_upb = total_upb / total_upb.sum() * 100

  # 2. % total Bal by state
  av_bal = dfg_state['Bal'].mean()

  # 3. WA time
  wa_time = dfg_state.apply(weighted_avg, 'time')

  # 4. WA int
  wa_int = dfg_state.apply(weighted_avg, 'int')

  # 5. WA FICO
  wa_fico = dfg_state.apply(weighted_avg, 'FICO')

  # 6. WA EMI
  wa_emi = dfg_state.apply(weighted_avg, 'EMI')

  df_state = pd.DataFrame({
      '% Total UPB': pct_sum_upb,
      'Average Bal': av_bal,
      'WA time': wa_time,
      'WA int': wa_int,
      'WA FICO': wa_fico,
      'WA EMI': wa_emi
  })

  df_state = df_state.drop('DC', axis=0)

  # Display the resulting DataFrame
  st.write(result_df)



  ############################################################ Plotting ############################################################




  # Create columns for side-by-side plots
  col1, col2 = st.columns(2)

  color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

  # Assign colors for bar1, line, and bar2
  color_bar = color_cycle[0]  # First color in the cycle
  color_line = color_cycle[1]  # Second color in the cycle
  color_bar2 = color_cycle[3]  # Third color in the cycle

  # First plot in the left column
  with col1:
      st.title('UPB plots')

      # Create a bar plot
      bar_fig = go.Figure(data=[
          go.Bar(name='Percentage in Total UPB', x=result_df.index, y=result_df['percentage_in_total_UPB'], marker_color=color_bar),
      ])
      
      # Create a line plot on the same axis
      bar_fig.add_trace(
          go.Scatter(name='Percentage in Cohort UPB', x=result_df.index, y=result_df['percentage_in_cohort_UPB'], mode='lines+markers', line=dict(color=color_line))
      )
      
      # Update layout
      bar_fig.update_layout(
          title='Metrics by FICO Range',
          xaxis_showgrid=True,
          yaxis_showgrid=True,
          xaxis_title="FICO Range"
      )
      
      # Display the plot
      st.plotly_chart(bar_fig)

  # Second plot in the right column
  with col2:
      st.title('Manual FICO plots')
      
      # Dropdown for selecting numerical column for bar plot
      numerical_columns = result_df.select_dtypes(include=['float64', 'int64']).columns
      selected_column = st.selectbox('Select Numerical Column for Bar Plot:', numerical_columns)
      
      bar_fig2 = px.bar(result_df, x=result_df.index, y=selected_column, color_discrete_sequence=[color_bar2], title=f'Bar Plot for {selected_column}', labels={'x': 'FICO Range', 'y': selected_column})
      st.plotly_chart(bar_fig2)


  ####################### US States Heatmap #######################



  # Function to get state code from state name
  def state_name_to_code(state_name):
      state_name_to_code_dict = {
          'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 
          'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA', 
          'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 
          'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 
          'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 
          'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 
          'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 
          'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 
          'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 
          'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
      }
      return state_name_to_code_dict.get(state_name)

  # Fetch the GeoJSON data for US states
  US_STATES_GEOJSON_URL = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
  geojson_data = json.loads(requests.get(US_STATES_GEOJSON_URL).text)

  # title
  st.title('US States Heatmap')

  # display the dataframe
  st.write(df_state)

  # Dropdown for selecting numeric column for choropleth map
  numeric_columns = df_state.select_dtypes(include=['float64', 'int64']).columns
  default_index = list(numeric_columns).index("WA EMI")  # Find the index of "WA EMI"
  selected_column = st.selectbox('Select Column for Choropleth Map:', numeric_columns, index=default_index)

  # Merge the GeoJSON data with your df_state DataFrame
  for feature in geojson_data['features']:
      state_name = feature['properties']['name']
      state_code = state_name_to_code(state_name)
      if state_code and state_code in df_state.index:  # Ensure state_code exists and is in df_state
          feature['properties']['value_to_display'] = df_state.loc[state_code, selected_column]
      else:
          feature['properties']['value_to_display'] = 0  # default value, can be adjusted as needed

  # Use Pydeck's GeoJsonLayer to visualize
  view_state = pdk.ViewState(latitude=37.7749, longitude=-122.4194, zoom=4, pitch=0)

  layer = pdk.Layer(
      "GeoJsonLayer",
      data=geojson_data,
      opacity=0.6,
      stroked=False,
      filled=True,
      extruded=True,
      wireframe=True,
      get_elevation=f"properties.value_to_display * 50",
      get_fill_color="[255, 255, properties.value_to_display]",
      get_line_color=[255, 255, 255],
      pickable=True
  )

  # Render the map
  st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))


if __name__ == "__main__":
    run()
