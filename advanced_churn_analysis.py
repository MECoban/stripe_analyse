import streamlit as st
import pandas as pd
import stripe
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load environment variables
load_dotenv()

# Password protection removed for now

st.title("Advanced Customer Churn Analysis Dashboard")

stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
if not stripe.api_key:
    st.error("Stripe API key not found. Please set the STRIPE_SECRET_KEY environment variable.")
    st.stop()

try:
    stripe.Customer.list(limit=1)
    st.success("Successfully connected to Stripe API!")
except Exception as e:
    st.error(f"Error connecting to Stripe: {str(e)}")
    st.stop()

# Date range selector
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
with col2:
    end_date = st.date_input("End Date", value=datetime.now())

# Product selector
product_selector = st.selectbox("Select Product", ["GOLD", "VIP"])

if st.button("Veriyi Getir ve Analizi Başlat"):
    start_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
    end_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())

    # Product IDs
    VIP_PRODUCT_ID = "prod_ReY9OvvKriHaxW"
    GOLD_PRODUCT_IDs = ["prod_QPlfKu5msYB4LD", "prod_QPl3bHPgziJbg7", "prod_R0TVW9SQX1HJXj"]

    # Fetch all customers
    st.info("Fetching customer data from Stripe...")
    customers = []
    has_more = True
    starting_after = None
    while has_more:
        params = {
            'limit': 100,
            'created': {'gte': start_timestamp, 'lte': end_timestamp}
        }
        if starting_after:
            params['starting_after'] = starting_after
        response = stripe.Customer.list(**params)
        customers.extend(response.data)
        has_more = response.has_more
        if has_more:
            starting_after = response.data[-1].id

    # Fetch all subscriptions
    st.info("Fetching subscription data from Stripe...")
    subscriptions = []
    has_more = True
    starting_after = None
    while has_more:
        params = {
            'limit': 100,
            'created': {'gte': start_timestamp, 'lte': end_timestamp},
            'status': 'all'
        }
        if starting_after:
            params['starting_after'] = starting_after
        response = stripe.Subscription.list(**params)
        subscriptions.extend(response.data)
        has_more = response.has_more
        if has_more:
            starting_after = response.data[-1].id

    # Process data into DataFrame
    customer_data = []
    for customer in customers:
        for sub in [s for s in subscriptions if s.customer == customer.id]:
            product_id = sub.plan.product if hasattr(sub, 'plan') and hasattr(sub.plan, 'product') else 'Unknown Product'
            # Map GOLD product IDs to 'GOLD', VIP to 'VIP', others unchanged
            if product_id in GOLD_PRODUCT_IDs:
                product_label = 'GOLD'
            elif product_id == VIP_PRODUCT_ID:
                product_label = 'VIP'
            else:
                product_label = product_id
            customer_data.append({
                'Customer ID': customer.id,
                'Customer Email': customer.email,
                'Product': product_label,
                'Created (UTC)': datetime.fromtimestamp(sub.created).strftime('%Y-%m-%d %H:%M:%S'),
                'Canceled At (UTC)': datetime.fromtimestamp(sub.canceled_at).strftime('%Y-%m-%d %H:%M:%S') if sub.canceled_at else None,
                'Status': sub.status
            })

    data = pd.DataFrame(customer_data)

    # Filter for selected product
    if product_selector == "GOLD":
        data = data[data['Product'] == 'GOLD']
        selected_products = ['GOLD']
        st.header("Analysis for Product: GOLD")
    else:
        data = data[data['Product'] == 'VIP']
        selected_products = ['VIP']
        st.header("Analysis for Product: VIP")

    if data.empty:
        st.warning(f"No data found for product: {product_selector}")
        st.stop()

    st.metric("Total Customers", len(data))
    st.metric("Active Customers", data['Status'].eq('active').sum())
    st.metric("Canceled Customers", data['Status'].eq('canceled').sum())
    st.dataframe(data)

    # Customer exclusion section
    st.sidebar.header("Customer Exclusion")
    exclusion_method = st.sidebar.radio(
        "Choose exclusion method",
        ["None", "Upload CSV", "Manual Entry"]
    )

    excluded_customers = []
    if exclusion_method == "Upload CSV":
        exclusion_file = st.sidebar.file_uploader("Upload CSV with customer emails to exclude", type=["csv"])
        if exclusion_file is not None:
            exclusion_df = pd.read_csv(exclusion_file)
            if 'Customer Email' in exclusion_df.columns:
                excluded_customers = exclusion_df['Customer Email'].tolist()
                st.sidebar.success(f"Loaded {len(excluded_customers)} customers to exclude")
            else:
                st.sidebar.error("CSV must contain a 'Customer Email' column")
    elif exclusion_method == "Manual Entry":
        customer_emails_text = st.sidebar.text_area(
            "Enter Customer Emails to exclude (one per line)",
            help="Enter each Customer Email on a new line"
        )
        if customer_emails_text:
            excluded_customers = [email.strip() for email in customer_emails_text.split('\n') if email.strip()]
            st.sidebar.success(f"Added {len(excluded_customers)} customers to exclude")

    # Filter out excluded customers
    if excluded_customers:
        original_count = len(data)
        data = data[~data['Customer Email'].isin(excluded_customers)]
        excluded_count = original_count - len(data)
        st.sidebar.info(f"Excluded {excluded_count} customer records from analysis")

    # Check if product column exists, if not create a default one
    if 'Product' not in data.columns:
        data['Product'] = 'Default Product'

    # Product selector (only VIP product, disabled)
    st.write("Selected product for analysis:", product_selector)

    # Filter data for selected products (redundant, but keeps code structure)
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Monthly Analysis", "Product Comparison", "Customer Details", "Monthly Breakdown", "Monthly Product Analysis"])

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

    with tab5:
        # Monthly Product Analysis
        st.write("### Monthly Product Analysis")
        
        # Product selector for detailed analysis
        selected_product = st.selectbox("Select a product for detailed monthly analysis", selected_products)
        
        # Filter data for selected product
        product_data = filtered_data[filtered_data['Product'] == selected_product]
        
        # Calculate monthly metrics for the selected product
        monthly_product_metrics = []
        
        for month in all_months:
            month_str = str(month)
            
            # Count created in this month
            created_this_month = len(product_data[product_data['created_month'] == month])
            
            # Count canceled in this month
            canceled_this_month = len(product_data[product_data['canceled_month'] == month])
            
            # Count active at end of month
            active_at_month_end = len(product_data[
                (product_data['created_month'] <= month) & 
                ((product_data['canceled_month'].isna()) | (product_data['canceled_month'] > month))
            ])
            
            # Calculate churn rate for this month
            churn_rate = (canceled_this_month / created_this_month * 100) if created_this_month > 0 else 0
            
            # Calculate net growth (created - canceled)
            net_growth = created_this_month - canceled_this_month
            
            # Calculate growth rate
            growth_rate = (net_growth / active_at_month_end * 100) if active_at_month_end > 0 else 0
            
            # Calculate retention rate (1 - churn rate)
            retention_rate = 100 - churn_rate
            
            monthly_product_metrics.append({
                'Month': month_str,
                'Created': created_this_month,
                'Canceled': canceled_this_month,
                'Active': active_at_month_end,
                'Net Growth': net_growth,
                'Churn Rate (%)': churn_rate,
                'Growth Rate (%)': growth_rate,
                'Retention Rate (%)': retention_rate
            })
        
        monthly_product_df = pd.DataFrame(monthly_product_metrics)
        
        # Display monthly product metrics
        st.write(f"#### Monthly Metrics for {selected_product}")
        st.dataframe(monthly_product_df)
        
        # Create visualizations for the selected product
        st.write(f"#### Monthly Trends for {selected_product}")
        
        # Create a multi-metric visualization
        fig = make_subplots(rows=3, cols=1, 
                            subplot_titles=("Customer Counts", "Growth Metrics", "Rates"),
                            vertical_spacing=0.1)
        
        # Add traces for customer counts
        fig.add_trace(
            go.Scatter(x=monthly_product_df['Month'], y=monthly_product_df['Created'], 
                       name="Created", line=dict(width=2, color='green')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=monthly_product_df['Month'], y=monthly_product_df['Canceled'], 
                       name="Canceled", line=dict(width=2, color='red')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=monthly_product_df['Month'], y=monthly_product_df['Active'], 
                       name="Active", line=dict(width=2, color='blue')),
            row=1, col=1
        )
        
        # Add traces for growth metrics
        fig.add_trace(
            go.Scatter(x=monthly_product_df['Month'], y=monthly_product_df['Net Growth'], 
                       name="Net Growth", line=dict(width=2, color='purple')),
            row=2, col=1
        )
        
        # Add traces for rates
        fig.add_trace(
            go.Scatter(x=monthly_product_df['Month'], y=monthly_product_df['Churn Rate (%)'], 
                       name="Churn Rate (%)", line=dict(width=2, color='red')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=monthly_product_df['Month'], y=monthly_product_df['Growth Rate (%)'], 
                       name="Growth Rate (%)", line=dict(width=2, color='green')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=monthly_product_df['Month'], y=monthly_product_df['Retention Rate (%)'], 
                       name="Retention Rate (%)", line=dict(width=2, color='blue')),
            row=3, col=1
        )
        
        fig.update_layout(height=900, title_text=f"Monthly Analysis for {selected_product}", showlegend=True)
        st.plotly_chart(fig)
        
        # Calculate and display key metrics
        st.write(f"#### Key Metrics for {selected_product}")
        
        # Calculate overall metrics
        total_created = monthly_product_df['Created'].sum()
        total_canceled = monthly_product_df['Canceled'].sum()
        current_active = monthly_product_df['Active'].iloc[-1] if not monthly_product_df.empty else 0
        overall_churn_rate = (total_canceled / total_created * 100) if total_created > 0 else 0
        overall_retention_rate = 100 - overall_churn_rate
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Created", total_created)
            st.metric("Total Canceled", total_canceled)
        with col2:
            st.metric("Current Active", current_active)
            st.metric("Net Growth", total_created - total_canceled)
        with col3:
            st.metric("Overall Churn Rate (%)", f"{overall_churn_rate:.2f}%")
            st.metric("Overall Retention Rate (%)", f"{overall_retention_rate:.2f}%")
        
        # Export monthly product analysis
        if st.button(f"Export {selected_product} Monthly Analysis"):
            monthly_product_df.to_csv(f'{selected_product}_monthly_analysis.csv', index=False)
            st.success(f"Monthly analysis for {selected_product} exported to '{selected_product}_monthly_analysis.csv'")

# Save analysis results
st.success("Analysis Completed!") 