import streamlit as st
import pandas as pd
import plotly.express as px

# Streamlit UI
st.title("Customer Churn Analysis Dashboard")

# File uploader
data_file = st.file_uploader("Upload your CSV file", type=["csv"])

if data_file is not None:
    # Load the uploaded CSV file
    data = pd.read_csv(data_file)

    # Convert date columns to datetime format
    data['created_date'] = pd.to_datetime(data['Created (UTC)'], errors='coerce')
    data['canceled_date'] = pd.to_datetime(data['Canceled At (UTC)'], errors='coerce')

    # Extract year-month for analysis
    data['created_month'] = data['created_date'].dt.to_period('M')
    data['canceled_month'] = data['canceled_date'].dt.to_period('M')

    # Ensure Customer ID uniqueness
    data = data.drop_duplicates(subset=['Customer ID'])

    # Calculate customers created per month
    created_per_month = data.groupby('created_month')['Customer ID'].count()

    # Calculate customers canceled per month
    canceled_per_month = data.groupby('canceled_month')['Customer ID'].count()

    # Calculate active customers per month correctly
    active_per_month = {}
    all_months = pd.period_range(start=data['created_month'].min(), end=pd.Timestamp.today().to_period('M'), freq='M')

    for month in all_months:
        active_customers = data[
            (data['created_month'] <= month) & 
            ((data['canceled_month'].isna()) | (data['canceled_month'] > month))  # Adjusted to exclude canceled customers
        ]
        active_per_month[month] = len(active_customers)

    active_per_month_series = pd.Series(active_per_month)

    # Correct churn rate calculation
    churn_rate = (canceled_per_month / created_per_month.replace(0, pd.NA)).fillna(0) * 100

    # Prepare final DataFrame
    output = pd.DataFrame({
        'Month': created_per_month.index.astype(str),
        'Created': created_per_month.values,
        'Canceled': canceled_per_month.reindex(created_per_month.index, fill_value=0).values,
        'Active': active_per_month_series.reindex(created_per_month.index, fill_value=0).values,
        'Churn Rate (%)': churn_rate.reindex(created_per_month.index, fill_value=0).values
    }).fillna(0)

    # Display results
    st.write("### Monthly Customer Analysis")
    st.dataframe(output)

    # Interactive Plot with Plotly
    st.write("### Interactive Customer Trends")
    fig = px.line(output, x='Month', y=['Active', 'Created', 'Canceled'], markers=True,
                  labels={'value': 'Number of Customers', 'variable': 'Customer Type'},
                  title='Monthly Customer Trends')
    fig.update_xaxes(type='category')
    st.plotly_chart(fig)

    # Export emails of canceled customers along with cancellation date
    st.write("### Canceled Customers Email List")
    canceled_customers = data.loc[data['canceled_date'].notna(), ['Customer Email', 'canceled_date']].drop_duplicates().sort_values(by='canceled_date')
    st.dataframe(canceled_customers)

    # Save to CSV
    canceled_customers.to_csv('canceled_customers_emails.csv', index=False)
    output.to_csv('monthly_customer_churn_analysis.csv', index=False)

    st.success("Analysis Completed!")
else:
    st.warning("Please upload a CSV file to begin analysis.")
