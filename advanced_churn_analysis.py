import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Streamlit UI
st.title("Advanced Customer Churn Analysis Dashboard")

# File uploader
data_file = st.file_uploader("Upload your CSV file", type=["csv"])

if data_file is not None:
    # Load the uploaded CSV file
    data = pd.read_csv(data_file)
    
    # Check if product column exists, if not create a default one
    if 'Product' not in data.columns:
        data['Product'] = 'Default Product'
    
    # Product selector
    products = data['Product'].unique()
    selected_products = st.multiselect(
        "Select products to analyze",
        options=products,
        default=products
    )
    
    # Filter data for selected products
    filtered_data = data[data['Product'].isin(selected_products)]

    # Convert date columns to datetime format
    filtered_data['created_date'] = pd.to_datetime(filtered_data['Created (UTC)'], errors='coerce')
    filtered_data['canceled_date'] = pd.to_datetime(filtered_data['Canceled At (UTC)'], errors='coerce')

    # Extract year-month for analysis
    filtered_data['created_month'] = filtered_data['created_date'].dt.to_period('M')
    filtered_data['canceled_month'] = filtered_data['canceled_date'].dt.to_period('M')

    # Ensure Customer ID uniqueness
    filtered_data = filtered_data.drop_duplicates(subset=['Customer ID', 'Product'])

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Monthly Analysis", "Product Comparison", "Customer Details", "Monthly Breakdown"])
    
    with tab1:
        # Calculate customers created per month
        created_per_month = filtered_data.groupby(['created_month', 'Product'])['Customer ID'].count().reset_index()
        created_per_month = created_per_month.pivot(index='created_month', columns='Product', values='Customer ID').fillna(0)

        # Calculate customers canceled per month
        canceled_per_month = filtered_data.groupby(['canceled_month', 'Product'])['Customer ID'].count().reset_index()
        canceled_per_month = canceled_per_month.pivot(index='canceled_month', columns='Product', values='Customer ID').fillna(0)

        # Calculate active customers per month correctly
        active_per_month = {}
        all_months = pd.period_range(start=filtered_data['created_month'].min(), end=pd.Timestamp.today().to_period('M'), freq='M')

        for month in all_months:
            active_customers = filtered_data[
                (filtered_data['created_month'] <= month) & 
                ((filtered_data['canceled_month'].isna()) | (filtered_data['canceled_month'] > month))
            ]
            active_per_month[month] = active_customers.groupby('Product')['Customer ID'].count()
        
        active_per_month_df = pd.DataFrame(active_per_month).T.fillna(0)

        # Calculate churn rate for each product
        churn_rates = {}
        for product in selected_products:
            if product in created_per_month.columns and product in canceled_per_month.columns:
                created = created_per_month[product].replace(0, np.nan)
                canceled = canceled_per_month[product].reindex(created.index, fill_value=0)
                churn_rates[product] = (canceled / created).fillna(0) * 100
        
        churn_rate_df = pd.DataFrame(churn_rates).fillna(0)
        
        # Calculate total churn rate - FIXED to ensure all arrays have the same length
        total_created = created_per_month.sum()
        total_canceled = canceled_per_month.sum()
        
        # Ensure all series have the same index
        all_products = sorted(set(total_created.index) | set(total_canceled.index))
        total_created = total_created.reindex(all_products, fill_value=0)
        total_canceled = total_canceled.reindex(all_products, fill_value=0)
        
        # Calculate churn rate with aligned data
        total_churn_rate = (total_canceled / total_created.replace(0, np.nan)).fillna(0) * 100
        
        # Display total churn rate
        st.write("### Overall Churn Rate by Product")
        total_churn_df = pd.DataFrame({
            'Product': all_products,
            'Total Created': total_created.values,
            'Total Canceled': total_canceled.values,
            'Total Churn Rate (%)': total_churn_rate.values
        })
        st.dataframe(total_churn_df)
        
        # Display monthly analysis
        st.write("### Monthly Customer Analysis")
        
        # Create a multi-product visualization
        fig = make_subplots(rows=3, cols=1, 
                            subplot_titles=("Active Customers", "Created vs Canceled", "Churn Rate"),
                            vertical_spacing=0.1)
        
        # Add traces for each product
        for product in selected_products:
            if product in active_per_month_df.columns:
                fig.add_trace(
                    go.Scatter(x=active_per_month_df.index.astype(str), y=active_per_month_df[product], 
                               name=f"{product} - Active", line=dict(width=2)),
                    row=1, col=1
                )
            
            if product in created_per_month.columns:
                fig.add_trace(
                    go.Scatter(x=created_per_month.index.astype(str), y=created_per_month[product], 
                               name=f"{product} - Created", line=dict(width=2)),
                    row=2, col=1
                )
            
            if product in canceled_per_month.columns:
                fig.add_trace(
                    go.Scatter(x=canceled_per_month.index.astype(str), y=canceled_per_month[product], 
                               name=f"{product} - Canceled", line=dict(width=2)),
                    row=2, col=1
                )
            
            if product in churn_rate_df.columns:
                fig.add_trace(
                    go.Scatter(x=churn_rate_df.index.astype(str), y=churn_rate_df[product], 
                               name=f"{product} - Churn Rate", line=dict(width=2)),
                    row=3, col=1
                )
        
        fig.update_layout(height=900, title_text="Multi-Product Customer Analysis", showlegend=True)
        st.plotly_chart(fig)
    
    with tab2:
        # Product comparison visualization
        st.write("### Product Performance Comparison")
        
        # Calculate key metrics for each product
        product_metrics = []
        for product in selected_products:
            product_data = filtered_data[filtered_data['Product'] == product]
            
            total_customers = len(product_data)
            total_canceled = product_data['canceled_date'].notna().sum()
            total_active = total_customers - total_canceled
            
            # Calculate average customer lifetime
            product_data['lifetime'] = (product_data['canceled_date'] - product_data['created_date']).dt.days
            avg_lifetime = product_data['lifetime'].mean()
            
            product_metrics.append({
                'Product': product,
                'Total Customers': total_customers,
                'Active Customers': total_active,
                'Canceled Customers': total_canceled,
                'Churn Rate (%)': (total_canceled / total_customers * 100) if total_customers > 0 else 0,
                'Avg Customer Lifetime (days)': avg_lifetime
            })
        
        product_metrics_df = pd.DataFrame(product_metrics)
        st.dataframe(product_metrics_df)
        
        # Visualize product comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=product_metrics_df['Product'],
            y=product_metrics_df['Active Customers'],
            name='Active Customers',
            marker_color='green'
        ))
        fig.add_trace(go.Bar(
            x=product_metrics_df['Product'],
            y=product_metrics_df['Canceled Customers'],
            name='Canceled Customers',
            marker_color='red'
        ))
        fig.update_layout(
            barmode='group',
            title='Active vs Canceled Customers by Product',
            xaxis_title='Product',
            yaxis_title='Number of Customers'
        )
        st.plotly_chart(fig)
        
        # Churn rate comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=product_metrics_df['Product'],
            y=product_metrics_df['Churn Rate (%)'],
            marker_color='purple'
        ))
        fig.update_layout(
            title='Churn Rate by Product',
            xaxis_title='Product',
            yaxis_title='Churn Rate (%)'
        )
        st.plotly_chart(fig)
    
    with tab3:
        # Customer details
        st.write("### Customer Details")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            status_filter = st.selectbox("Filter by status", ["All", "Active", "Canceled"])
        
        with col2:
            product_filter = st.selectbox("Filter by product", ["All"] + list(selected_products))
        
        # Apply filters
        customer_details = filtered_data.copy()
        if status_filter == "Active":
            customer_details = customer_details[customer_details['canceled_date'].isna()]
        elif status_filter == "Canceled":
            customer_details = customer_details[customer_details['canceled_date'].notna()]
        
        if product_filter != "All":
            customer_details = customer_details[customer_details['Product'] == product_filter]
        
        # Display customer details
        st.dataframe(customer_details[['Customer ID', 'Customer Email', 'Product', 'created_date', 'canceled_date']])
        
        # Export options
        if st.button("Export Customer Details"):
            customer_details.to_csv('customer_details.csv', index=False)
            st.success("Customer details exported to 'customer_details.csv'")
    
    with tab4:
        # Monthly breakdown analysis
        st.write("### Monthly Breakdown Analysis")
        
        # Create a monthly summary DataFrame
        monthly_summary = []
        
        for month in all_months:
            month_str = str(month)
            
            # Filter data for this month
            month_data = filtered_data[
                (filtered_data['created_month'] <= month) & 
                ((filtered_data['canceled_month'].isna()) | (filtered_data['canceled_month'] > month))
            ]
            
            # Calculate metrics for each product
            for product in selected_products:
                product_month_data = month_data[month_data['Product'] == product]
                
                # Count created in this month
                created_this_month = len(filtered_data[
                    (filtered_data['created_month'] == month) & 
                    (filtered_data['Product'] == product)
                ])
                
                # Count canceled in this month
                canceled_this_month = len(filtered_data[
                    (filtered_data['canceled_month'] == month) & 
                    (filtered_data['Product'] == product)
                ])
                
                # Count active at end of month
                active_at_month_end = len(product_month_data)
                
                # Calculate churn rate for this month
                churn_rate = (canceled_this_month / created_this_month * 100) if created_this_month > 0 else 0
                
                monthly_summary.append({
                    'Month': month_str,
                    'Product': product,
                    'Created': created_this_month,
                    'Canceled': canceled_this_month,
                    'Active': active_at_month_end,
                    'Churn Rate (%)': churn_rate
                })
        
        monthly_summary_df = pd.DataFrame(monthly_summary)
        
        # Display monthly summary
        st.write("#### Monthly Summary by Product")
        st.dataframe(monthly_summary_df)
        
        # Create a heatmap of churn rates
        st.write("#### Churn Rate Heatmap")
        
        # Pivot the data for the heatmap
        churn_heatmap_data = monthly_summary_df.pivot(
            index='Month', 
            columns='Product', 
            values='Churn Rate (%)'
        ).fillna(0)
        
        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=churn_heatmap_data.values,
            x=churn_heatmap_data.columns,
            y=churn_heatmap_data.index,
            colorscale='RdYlGn_r',  # Red for high churn, green for low churn
            colorbar=dict(title='Churn Rate (%)')
        ))
        
        fig.update_layout(
            title='Monthly Churn Rate Heatmap by Product',
            xaxis_title='Product',
            yaxis_title='Month',
            height=600
        )
        
        st.plotly_chart(fig)
        
        # Create a stacked area chart for customer growth
        st.write("#### Customer Growth Over Time")
        
        # Pivot the data for the stacked area chart
        growth_data = monthly_summary_df.pivot(
            index='Month', 
            columns='Product', 
            values='Active'
        ).fillna(0)
        
        # Create the stacked area chart
        fig = go.Figure()
        
        for product in growth_data.columns:
            fig.add_trace(go.Scatter(
                x=growth_data.index,
                y=growth_data[product],
                name=product,
                stackgroup='one',
                mode='lines'
            ))
        
        fig.update_layout(
            title='Customer Growth Over Time',
            xaxis_title='Month',
            yaxis_title='Number of Active Customers',
            height=500
        )
        
        st.plotly_chart(fig)
        
        # Export monthly summary
        if st.button("Export Monthly Summary"):
            monthly_summary_df.to_csv('monthly_summary.csv', index=False)
            st.success("Monthly summary exported to 'monthly_summary.csv'")

    # Save analysis results
    st.success("Analysis Completed!")
else:
    st.warning("Please upload a CSV file to begin analysis.") 