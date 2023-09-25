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
import os
import warnings
import ipywidgets as widgets
from IPython.display import display, HTML
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():

  st.set_page_config(layout='wide', initial_sidebar_state='collapsed')

  warnings.filterwarnings('ignore')

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




if __name__ == "__main__":
    run()
