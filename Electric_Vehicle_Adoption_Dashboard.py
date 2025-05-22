# Streamlit Dashboard

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(
    page_title='EV Adoption Dashboard',
    layout='wide',
    initial_sidebar_state='expanded',
    page_icon='🚗',
)

# Title
st.title('🚗 Electric Vehicle Adoption Trends')
st.markdown('---')

# 2. Load Original Data (Uncleaned)
@st.cache_data
def load_original_data():
    df = pd.read_csv('sample_data.csv')
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    return df

df_orig = load_original_data()

# 3. Create Cleaned DataFrame
@st.cache_data
def clean_data(df):
    df_clean = df.copy()
    # Drop duplicates
    df_clean.drop_duplicates(inplace=True)
    # Drop rows missing critical values
    df_clean.dropna(inplace=True)
    # Drop irrelevant columns
    drop_cols = ['vin_(1-10)', 'postal_code', 'dol_vehicle_id', 'legislative_district']
    df_clean.drop(columns=drop_cols, errors='ignore', inplace=True)
    # Remove outliers via hierarchical median imputation
    for col in ['electric_range', 'base_msrp']:
        # Replace 0 with NaN
        df_clean[col] = df_clean[col].replace(0, np.nan)
        # Hierarchical imputation
        for cols in [
        ['electric_vehicle_type', 'make', 'model', 'model_year'],
        ['electric_vehicle_type', 'make', 'model'],
        ['electric_vehicle_type', 'make'],
        ['electric_vehicle_type']
        ]:
            df_clean[col] = df_clean[col].fillna(df_clean.groupby(cols)[col].transform('median'))
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    return df_clean

df_clean = clean_data(df_orig)

# 4. Sidebar: Page Navigation
st.sidebar.title('📊 Navigation')
st.sidebar.markdown('**Select a page to explore the data:**')
page = st.sidebar.radio(
    'Select Page', ['Data Overview', 'Cleaned Data', 'Charts']
)
st.sidebar.markdown('---')
# 5. Filters (for pages 2 & 3)
# Show filters only for Cleaned Data & Charts pages
def apply_filters(df_clean):
    # Sidebar - Professional Filters
    st.sidebar.title('🔎 Filters')
    # Multi-select: Make
    make_options = df_clean['make'].unique().tolist()
    selected_makes = st.sidebar.multiselect('Brand', make_options, default=make_options[:3])
    
    # Slider: Model Year
    year_min, year_max = int(df_clean['model_year'].min()), int(df_clean['model_year'].max())
    selected_year = st.sidebar.slider('Model Year', year_min, year_max, (year_min, year_max))

    # Multi-select: EV Type
    type_options = df_clean['electric_vehicle_type'].unique().tolist()
    selected_types = st.sidebar.multiselect('Electric Vehicle Type', type_options, default=type_options)

    # Slider: Range
    range_min, range_max = int(df_clean['electric_range'].min()), int(df_clean['electric_range'].max())
    selected_range = st.sidebar.slider('Electric Range (miles)', range_min, range_max, (range_min, range_max))

    # Multi-select: Electric Utility
    cafv_opts = [
        'Clean Alternative Fuel Vehicle Eligible',
        'Eligibility unknown as battery range has not been researched',
        'Not eligible due to low battery range'
    ]
    cafv_sel = st.sidebar.multiselect('CAFV Eligibility', cafv_opts, default=cafv_opts)

    # Apply Filters
    df_filtered = df_clean[
        df_clean['make'].isin(selected_makes) &
        df_clean['electric_vehicle_type'].isin(selected_types) &
        df_clean['model_year'].between(*selected_year) &
        (df_clean['electric_range'].between(*selected_range)) &
        df_clean['clean_alternative_fuel_vehicle_(cafv)_eligibility'].isin(cafv_sel)
    ]
    
    return df_filtered

# 6. Page Content
info_text = """
This dataset shows the Battery Electric Vehicles (BEVs) and Plug-in Hybrid Electric Vehicles (PHEVs) that are currently registered through Washington State Department of Licensing (DOL).
"""
notes_text = """
#### Notes

1. **Battery Electric Vehicle (BEV)**: An all-electric vehicle using one or more batteries to store the electrical energy that powers the motor and is charged by plugging the vehicle into an electric power source.

    **Plug-in Hybrid Electric Vehicle (PHEV)**: A vehicle that uses one or more batteries to power an electric motor, and also uses another fuel (e.g., gasoline or diesel) to power an internal combustion engine or other propulsion source; and is charged by plugging the vehicle into an electric power source.

2. **CAFV Eligibility**: Clean Alternative Fuel Vehicle (CAFV) Eligibility is based on the fuel requirement and electric-only range requirement as outlined in RCW 82.08.809 and RCW 82.12.809. Sales or leases must occur on or after 8/1/2019 and meet purchase price requirements to qualify for exemptions.

3. **Electric Range Note**: Range is no longer maintained for BEVs with range over 30 miles. Zero (0) indicates unknown or unresearched range.
"""

if page == 'Data Overview':
    st.header('📋 Data Overview')

    st.markdown(info_text)
    st.markdown(notes_text)
    
    st.subheader('Dataset Snapshot')
    st.write(df_orig.head(6))

    st.subheader('Columns Summary')
    col_defs = pd.DataFrame({
        'Column Name': [
            'VIN (1-10)', 
            'County . City . State',
            'Portal Code',
            'Model Year',
            'Make . Model',
            'Electric Vehicle Type',
            'CAFV Eligibility',
            'Electric Range',
            'Base MSRP',
            'Legislative District',
            'DOL Vehicle ID',
            'Vehicle Location',
            'Electric Utility',
            '2000 Census Track'
        ],
        'Type': [
            'object', 'object', 'float64', 'int64', 'object', 'object', 'object',
            'float64', 'float64', 'float64', 'int64', 'object', 'object', 'float64'
        ],
        'Description': [
            'Likely anonymized ID; good candidate for drop',
            'Location info for geographic trends',
            'Can help group by region might convert to int',
            'Useful for age analysis',
            'Brand and vehicle details',
            'BEV, PHEV, etc. - critical for filtering groups',
            'Whether vehicle qualifies for fuel programs',
            'Key metric - prediction target',
            'Price - strong predictor',
            'Possibly useful for policy insight',
            'Internal - drop candidate',
            '(lat, lon) as string - useful for geoplots',
            'Can be linked with infrastructure info',
            'Geographic demographic mapping'
        ]
    })

    st.dataframe(col_defs, 
                 column_order=['Column Name', 'Type', 'Description'],
                 hide_index=True,
                 use_container_width=True)


    st.subheader('Info')
    
    st.write(f'**Rows:** {df_orig.shape[0]}  \ **Columns:** {df_orig.shape[1]}  \ **Each row represents a:** Vehicle')
    st.write(f'**Memory Usage:** {df_orig.memory_usage().sum() / 1024 ** 2:.2f} MB')

    st.subheader('Statistical Summary')
    st.write(df_orig.describe())
    st.write(df_orig.describe(include='O'))

    st.subheader('Missing & Duplicate Values')
    st.write('Missing Values Of Each Column:')
    st.dataframe(pd.DataFrame(df_orig.isnull().sum()).T, hide_index=True, use_container_width=True)
    st.write(f'Duplicate Rows: {df_orig.duplicated().sum()}')

    fig = px.pie(df_orig, names='clean_alternative_fuel_vehicle_(cafv)_eligibility', 
                title='Distribution of Clean Alternative Fuel Vehicle (CAFV) Eligibility',
                color_discrete_sequence=px.colors.qualitative.Antique, width=800, height=400, hole=0.3)
    fig.update_traces(textinfo='percent+label').update_layout(title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)

elif page == 'Cleaned Data':
    cleaning_text = """
    #### Cleaning Steps:

    1. Drop duplicates.

    2. Drop rows missing critical values.

    3. Drop irrelevant columns.

    4. Remove outliers via hierarchical median imputation:
        
        4.1 Replace 0 with NaN.
        
        4.2 Hierarchical imputation.
    """
    st.header('🧹 Data Cleaning')
    st.markdown(cleaning_text)
    st.markdown('---')
    st.subheader('Filtered Data Snapshot')
    df_filtered = apply_filters(df_clean)
    st.write(df_filtered.head(10))
    st.subheader('Filtered Data Shape')
    st.write(df_filtered.shape)
    st.write(f'**Rows:** {df_filtered.shape[0]}  \ **Columns:** {df_filtered.shape[1]}  \ **Each row represents a:** Vehicle')

elif page == 'Charts':
    st.header('📊 Charts of Filtered and Cleaned Data')
    df_filtered = apply_filters(df_clean)
    tab1, tab2, tab3, tab4 = st.tabs(['Univariate Analysis', 'Bivariate Analysis', 'Geographical Distribution', 'Multivariate Analysis'])

    with tab1:
        st.subheader('Univariate Analysis')
        # Histogram
        for col in ['electric_range', 'base_msrp', 'model_year']:
            fig1 = px.histogram(df_filtered[col], x=col, title=f'Distribution of {col}', 
                                color_discrete_sequence= px.colors.qualitative.Antique,
                                text_auto=True, 
                                nbins=10, width=800, height=400)
            fig1.update_traces(marker=dict(line=dict(width=1, color='black')))
            fig1.update_layout(title_x=0.5, xaxis_title=col, yaxis_title='Count', xaxis_tickangle=-45)
            st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.histogram(df_filtered['make'].value_counts().reset_index(), x='count' ,y='make', title='Distribution of Make',
                            text_auto=True, color_discrete_sequence=px.colors.qualitative.Antique, 
                            width=800, height=400)
        fig2.update_traces(marker=dict(line=dict(width=1, color='black')))
        fig2.update_layout(title_x=0.5, yaxis_title='Make', xaxis_title='Count', xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

        # Pie
        fig3 = px.pie(df_filtered, names='electric_vehicle_type', title='Distribution of Electric Vehicle Types', 
                    color_discrete_sequence=px.colors.qualitative.Antique, 
                    width=800, height=400, hole=0.3)
        fig3.update_traces(textinfo='percent+label')
        fig3.update_layout(title_x=0.5)
        st.plotly_chart(fig3, use_container_width=True)


    with tab2:
        st.subheader('Bivariate Analysis')
        # Scatter
        fig4 = px.scatter(df_filtered, x='model_year', y='electric_range', title='Electric Rrange vs Model Year',
                 color_discrete_sequence=px.colors.qualitative.Antique,
                 width=800, height=400, marginal_x='histogram')
        fig4.update_traces(marker=dict(line=dict(width=1, color='black')))
        fig4.update_layout(title_x=0.5, xaxis_title='Model Year', yaxis_title='Electric Range', xaxis_tickangle=-45)
        st.plotly_chart(fig4, use_container_width=True)

        # Bar Chart
        make_range = df_filtered.groupby('make')['electric_range'].mean()\
                .reset_index().sort_values(by='electric_range', ascending=False)
        make_range = make_range.rename(columns={'electric_range': 'Average Electric Range'})

        fig5 = px.bar(make_range, x='make', y='Average Electric Range', title='Average Electric Range by Make',
                color_discrete_sequence=px.colors.qualitative.Antique, width=800, height=400, text_auto=True)
        fig5.update_traces(marker=dict(line=dict(width=1, color='black')))
        fig5.update_layout(title_x=0.5, xaxis_title='Make', yaxis_title='Average Electric Range')
        st.plotly_chart(fig5, use_container_width=True)

        EV_type_Range = df_filtered.groupby('electric_vehicle_type')['electric_range'].mean()\
                .reset_index().sort_values(by='electric_range', ascending=False)
        EV_type_Range = EV_type_Range.rename(columns={'electric_range': 'Average Electric Range'})

        fig6 = px.bar(EV_type_Range, x='electric_vehicle_type', y='Average Electric Range', title='Average Electric Range by Electric Vehicle Type',
                color_discrete_sequence=px.colors.qualitative.Antique, width=800, height=400, text_auto=True)
        fig6.update_traces(marker=dict(line=dict(width=1, color='black')))
        fig6.update_layout(title_x=0.5, xaxis_title='Electric Vehicle Type', yaxis_title='Average Electric Range')
        st.plotly_chart(fig6, use_container_width=True)

        EV_type_MSRP = df_filtered.groupby('electric_vehicle_type')['base_msrp'].mean().\
                    reset_index().sort_values(by='base_msrp', ascending=False)
        EV_type_MSRP = EV_type_MSRP.rename(columns={'base_msrp': 'Average Base MSRP'})

        fig7 = px.bar(EV_type_MSRP, x='electric_vehicle_type', y='Average Base MSRP', title='Average Base MSRP by Electric Vehicle Type',
                color_discrete_sequence=px.colors.qualitative.Antique, width=800, height=400, text_auto=True)
        fig7.update_traces(marker=dict(line=dict(width=1, color='black')))
        fig7.update_layout(title_x=0.5, xaxis_title='Electric Vehicle Type', yaxis_title='Average Base MSRP')
        st.plotly_chart(fig7, use_container_width=True)


    with tab3:
        st.subheader('Geographical Distribution')
        # Scatter Map
        df_filtered[['lon', 'lat']] = df_filtered['vehicle_location'].str.extract(r'\((-?\d+\.\d+) (-?\d+\.\d+)\)').astype(float)
        EV_geo_dist = df_filtered.groupby(['lon', 'lat','city', 'make'])['electric_vehicle_type']\
                        .value_counts().reset_index(name='count')\
                    .sort_values(by='count', ascending=False)

        fig8 = px.scatter_map(EV_geo_dist ,lat='lat', lon='lon', hover_name='city', hover_data=['make'],
                    color='electric_vehicle_type', size='count',
                    size_max=20, opacity=0.7,
                    title='Vehicle Locations', color_discrete_sequence=px.colors.qualitative.Antique, 
                    zoom=5, center=dict(lat=47.5, lon=-122.5))
        st.plotly_chart(fig8, use_container_width=True)

    with tab4:
        st.subheader('Multivariate Analysis')
        # Correlation Matrix
        corr = df_filtered[['electric_range', 'base_msrp', 'model_year']].corr()
        fig9 = px.imshow(corr, text_auto=True, color_continuous_scale=px.colors.sequential.Brwnyl, 
                title='Correlation Matrix',
                color_continuous_midpoint=0, width=800, height=500)
        st.plotly_chart(fig9, use_container_width=True)

        # Bar Chart
        eligibility_counts = df_filtered.groupby(['make','model_year','clean_alternative_fuel_vehicle_(cafv)_eligibility'])\
                                ['clean_alternative_fuel_vehicle_(cafv)_eligibility'].count().reset_index(name='count')\
                                .sort_values(by='count', ascending=False)
        fig10 = px.bar(eligibility_counts, x='make', y='count', color='clean_alternative_fuel_vehicle_(cafv)_eligibility',
             title='CAFV Eligibility by Make and Model Year', color_discrete_sequence=px.colors.qualitative.Antique,
             barmode='group', hover_data=['model_year'])
        fig10.update_traces(marker=dict(line=dict(width=1, color='black')))
        fig10.update_layout(title_x=0.5, xaxis_title='Make', yaxis_title='Count', xaxis_tickangle=-45)
        st.plotly_chart(fig10, use_container_width=True)

# 5. Footer
st.sidebar.markdown('---')
st.sidebar.markdown('**Data Source:** Washington State EV Registration Data')
st.sidebar.markdown('**Created by:** Amr Taghyan')
