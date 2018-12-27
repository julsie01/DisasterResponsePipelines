import pandas as pd
import plotly.graph_objs as go
import numpy as np


def clean_data(cols_to_drop, df):
    """Helper function to prepare data set for further wrangling
       by dropping unwanted columns and returning remaining dataframe
       Input: cols_to_drop: List of columns to remove
       df: Pandas dataframe to clean
       Output: dataframe with cols_to_drop removed
    """

    df_copy = df.drop(columns=cols_to_drop)
    return df_copy

def get_most_correlated_categories(df):
    """
    Function to return the 10 most correlated features
    Input: Pandas Dataframe
    Output: Multi Level Index Pandas Dataframe
    """
    #get correlation matrix
    corr_matrix = df.corr().abs()

    #thanks to:
    #https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/
    upper = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))).stack().sort_values(ascending=False).nlargest(10)
    #convert back to dataframe
    upper.to_frame()
    return upper

def data_wrangling(df):
    """
    function to extract information from a dataframe
    Input: Dataframe
    Output: List of Plotly Graph
    """
    cols_to_drop = ['genre', 'message', 'original', 'id']
    df_copy = df.drop(columns=cols_to_drop)
    df_by_category = pd.melt(df_copy)
   
    data = df_by_category.groupby(['variable','value']).size().reset_index(name='value count')
    #rename columns
    data.columns = ['category', 'value', 'value count']

    #get only selected categories
    selected_categories = data[(data['value'] ==1)]

    #find the top categories in order to display visuals
    sorted_selected_df = selected_categories.sort_values(by=['value count'], ascending=False)

    largest_categories = sorted_selected_df.nlargest(10, 'value count')
   #cleaned data for the second graph 
    smallest_categories = sorted_selected_df.nsmallest(10, 'value count')
    smallest_categories = smallest_categories.sort_values(by=['value count'],ascending=False)
   
    most_correlated_categories=get_most_correlated_categories(df_copy)
    #now prepare the data for display
    x_categories = most_correlated_categories.index.get_level_values(0)
    y_categories = most_correlated_categories.index.get_level_values(1)
    z_values = most_correlated_categories.values.tolist()

    graph_one = []

    graph_one.append(
             go.Bar(
              x=largest_categories['category'],
              y=largest_categories['value count']
             )
          )

    layout_one = dict(title = 'Most Common Categories',
                xaxis = dict(title = 'Category'),
                yaxis = dict(title = 'Count'),
               )

    graph_two = []

    graph_two.append(
             go.Bar(
              x=smallest_categories['category'],
              y=smallest_categories['value count']
           
          )
    )

    layout_two = dict(title = 'Least Common Categories',
                xaxis = dict(title = 'Category'),
                yaxis = dict(title = 'Count')
               )

    graph_three = []

    graph_three.append(
             go.Heatmap(
              z=z_values, 
              x=x_categories, 
              y=y_categories,
              colorscale='Viridis',
              connectgaps=False)
           
          )

    layout_three = dict(title = 'Most Related (correlated) categories',
                xaxis = dict(title = 'Category'),
                yaxis = dict(title = 'Category')
               )
    

    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))
   

    return figures

   