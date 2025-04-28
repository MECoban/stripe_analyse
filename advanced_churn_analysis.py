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

# Initialize session state for data if it doesn't exist
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.app_data = pd.DataFrame() # Initialize with empty DataFrame

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
    
    # Store data in session state
    st.session_state.app_data = data
    st.session_state.data_loaded = True
    st.success("Veri başarıyla çekildi!")

# Only proceed with analysis if data is loaded
if st.session_state.data_loaded:
    data = st.session_state.app_data

    # Filter for selected product (apply this logic consistently)
    if product_selector == "GOLD":
        analysis_data = data[data['Product'] == 'GOLD'].copy() # Use .copy() to avoid SettingWithCopyWarning later
        selected_products = ['GOLD']
        st.header("Analysis for Product: GOLD")
    else:
        analysis_data = data[data['Product'] == 'VIP'].copy() # Use .copy() to avoid SettingWithCopyWarning later
        selected_products = ['VIP']
        st.header("Analysis for Product: VIP")

    if analysis_data.empty:
        st.warning(f"No data found for product: {product_selector}")
        st.stop()

    # Convert dates for analysis (do this only once after loading data)
    analysis_data.loc[:, 'created_date'] = pd.to_datetime(analysis_data['Created (UTC)'], errors='coerce')
    analysis_data.loc[:, 'canceled_date'] = pd.to_datetime(analysis_data['Canceled At (UTC)'], errors='coerce')
    analysis_data.loc[:, 'created_month'] = analysis_data['created_date'].dt.to_period('M')
    analysis_data.loc[:, 'canceled_month'] = analysis_data['canceled_date'].dt.to_period('M')

    # Define all_months here so it's available for all tabs
    if not analysis_data.empty and 'created_month' in analysis_data.columns:
        all_months = pd.period_range(start=analysis_data['created_month'].min(), end=pd.Timestamp.today().to_period('M'), freq='M')
    else:
        all_months = pd.PeriodIndex([]) # Empty index if no data

    # Basic metrics (using analysis_data)
    st.metric("Total Subscriptions", len(analysis_data))
    st.metric("Active Subscriptions", len(analysis_data[analysis_data['Status'] == 'active']))
    st.metric("Canceled Subscriptions", (analysis_data['Status'] == 'canceled').sum())
    st.dataframe(analysis_data)

    # Sidebar tab navigation in Turkish
    sidebar_tab = st.sidebar.radio(
        "Analiz Bölümü Seçin",
        [
            "Aylık Analiz",
            "Ürün Karşılaştırma",
            "Müşteri Detayları",
            "Aylık Dağılım",
            "Aylık Ürün Analizi"
        ]
    )

    # Analysis tabs use 'analysis_data' and 'all_months' from now on
    if sidebar_tab == "Aylık Analiz":
        # Calculate customers created per month
        created_per_month = analysis_data.groupby(['created_month', 'Product'])['Customer ID'].count().reset_index()
        created_per_month = created_per_month.pivot(index='created_month', columns='Product', values='Customer ID').fillna(0)

        # Calculate customers canceled per month
        canceled_per_month = analysis_data.groupby(['canceled_month', 'Product'])['Customer ID'].count().reset_index()
        canceled_per_month = canceled_per_month.pivot(index='canceled_month', columns='Product', values='Customer ID').fillna(0)

        # Calculate active customers per month correctly
        active_per_month = {}
        for month in all_months:
            active_customers = analysis_data[
                (analysis_data['created_month'] <= month) &
                ((analysis_data['canceled_month'].isna()) | (analysis_data['canceled_month'] > month))
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
        
        # Calculate total churn rate
        total_created = created_per_month.sum()
        total_canceled = canceled_per_month.sum()
        all_products = sorted(set(total_created.index) | set(total_canceled.index))
        total_created = total_created.reindex(all_products, fill_value=0)
        total_canceled = total_canceled.reindex(all_products, fill_value=0)
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
        
        # Display monthly analysis visualization
        st.write("### Monthly Customer Analysis")
        fig = make_subplots(rows=3, cols=1, subplot_titles=("Active Customers", "Created vs Canceled", "Churn Rate"), vertical_spacing=0.1)
        for product in selected_products:
            if product in active_per_month_df.columns:
                fig.add_trace(go.Scatter(x=active_per_month_df.index.astype(str), y=active_per_month_df[product], name=f"{product} - Active", line=dict(width=2)), row=1, col=1)
            if product in created_per_month.columns:
                fig.add_trace(go.Scatter(x=created_per_month.index.astype(str), y=created_per_month[product], name=f"{product} - Created", line=dict(width=2)), row=2, col=1)
            if product in canceled_per_month.columns:
                fig.add_trace(go.Scatter(x=canceled_per_month.index.astype(str), y=canceled_per_month[product], name=f"{product} - Canceled", line=dict(width=2)), row=2, col=1)
            if product in churn_rate_df.columns:
                fig.add_trace(go.Scatter(x=churn_rate_df.index.astype(str), y=churn_rate_df[product], name=f"{product} - Churn Rate", line=dict(width=2)), row=3, col=1)
        fig.update_layout(height=900, title_text="Multi-Product Customer Analysis", showlegend=True)
        st.plotly_chart(fig)
    elif sidebar_tab == "Ürün Karşılaştırma":
        # Product comparison visualization
        st.write("### Product Performance Comparison")
        
        # Calculate key metrics for each product
        product_metrics = []
        for product in selected_products:
            product_data = analysis_data[analysis_data['Product'] == product]
            
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
    elif sidebar_tab == "Müşteri Detayları":
        # Customer details
        st.write("### Customer Details")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            status_filter = st.selectbox("Filter by status", ["All", "Active", "Canceled"])
        
        with col2:
            product_filter = st.selectbox("Filter by product", ["All"] + list(selected_products))
        
        # Apply filters
        customer_details = analysis_data.copy()
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
    elif sidebar_tab == "Aylık Dağılım":
        # Monthly breakdown analysis
        st.write("### Monthly Breakdown Analysis")
        monthly_summary = []
        for month in all_months:
            month_str = str(month)
            month_data = analysis_data[
                (analysis_data['created_month'] <= month) &
                ((analysis_data['canceled_month'].isna()) | (analysis_data['canceled_month'] > month))
            ]
            for product in selected_products:
                product_month_data = month_data[month_data['Product'] == product]
                created_this_month = len(analysis_data[(analysis_data['created_month'] == month) & (analysis_data['Product'] == product)])
                canceled_this_month = len(analysis_data[(analysis_data['canceled_month'] == month) & (analysis_data['Product'] == product)])
                active_at_month_end = len(product_month_data)
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
        st.write("#### Monthly Summary by Product")
        st.dataframe(monthly_summary_df)
        st.write("#### Churn Rate Heatmap")
        churn_heatmap_data = monthly_summary_df.pivot(index='Month', columns='Product', values='Churn Rate (%)').fillna(0)
        fig = go.Figure(data=go.Heatmap(z=churn_heatmap_data.values, x=churn_heatmap_data.columns, y=churn_heatmap_data.index, colorscale='RdYlGn_r', colorbar=dict(title='Churn Rate (%)')))
        fig.update_layout(title='Monthly Churn Rate Heatmap by Product', xaxis_title='Product', yaxis_title='Month', height=600)
        st.plotly_chart(fig)
        st.write("#### Customer Growth Over Time")
        growth_data = monthly_summary_df.pivot(index='Month', columns='Product', values='Active').fillna(0)
        fig = go.Figure()
        for product in growth_data.columns:
            fig.add_trace(go.Scatter(x=growth_data.index, y=growth_data[product], name=product, stackgroup='one', mode='lines'))
        fig.update_layout(title='Customer Growth Over Time', xaxis_title='Month', yaxis_title='Number of Active Customers', height=500)
        st.plotly_chart(fig)
        if st.button("Export Monthly Summary"):
            monthly_summary_df.to_csv('monthly_summary.csv', index=False)
            st.success("Monthly summary exported to 'monthly_summary.csv'")
    elif sidebar_tab == "Aylık Ürün Analizi":
        # Monthly Product Analysis
        st.write("### Monthly Product Analysis")
        selected_product = st.selectbox("Select a product for detailed monthly analysis", selected_products, disabled=True) # Already filtered
        product_data = analysis_data # Already filtered for the selected product (GOLD or VIP)
        monthly_product_metrics = []
        for month in all_months:
            month_str = str(month)
            created_this_month = len(product_data[product_data['created_month'] == month])
            canceled_this_month = len(product_data[product_data['canceled_month'] == month])
            active_at_month_end = len(product_data[(product_data['created_month'] <= month) & ((product_data['canceled_month'].isna()) | (product_data['canceled_month'] > month))])
            churn_rate = (canceled_this_month / created_this_month * 100) if created_this_month > 0 else 0
            net_growth = created_this_month - canceled_this_month
            growth_rate = (net_growth / active_at_month_end * 100) if active_at_month_end > 0 else 0
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
        st.write(f"#### Monthly Metrics for {selected_product}")
        st.dataframe(monthly_product_df)
        st.write(f"#### Monthly Trends for {selected_product}")
        fig = make_subplots(rows=3, cols=1, subplot_titles=("Customer Counts", "Growth Metrics", "Rates"), vertical_spacing=0.1)
        fig.add_trace(go.Scatter(x=monthly_product_df['Month'], y=monthly_product_df['Created'], name="Created", line=dict(width=2, color='green')), row=1, col=1)
        fig.add_trace(go.Scatter(x=monthly_product_df['Month'], y=monthly_product_df['Canceled'], name="Canceled", line=dict(width=2, color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=monthly_product_df['Month'], y=monthly_product_df['Active'], name="Active", line=dict(width=2, color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=monthly_product_df['Month'], y=monthly_product_df['Net Growth'], name="Net Growth", line=dict(width=2, color='purple')), row=2, col=1)
        fig.add_trace(go.Scatter(x=monthly_product_df['Month'], y=monthly_product_df['Churn Rate (%)'], name="Churn Rate (%)", line=dict(width=2, color='red')), row=3, col=1)
        fig.add_trace(go.Scatter(x=monthly_product_df['Month'], y=monthly_product_df['Growth Rate (%)'], name="Growth Rate (%)", line=dict(width=2, color='green')), row=3, col=1)
        fig.add_trace(go.Scatter(x=monthly_product_df['Month'], y=monthly_product_df['Retention Rate (%)'], name="Retention Rate (%)", line=dict(width=2, color='blue')), row=3, col=1)
        fig.update_layout(height=900, title_text=f"Monthly Analysis for {selected_product}", showlegend=True)
        st.plotly_chart(fig)
        st.write(f"#### Key Metrics for {selected_product}")
        total_created = monthly_product_df['Created'].sum()
        total_canceled = monthly_product_df['Canceled'].sum()
        current_active = monthly_product_df['Active'].iloc[-1] if not monthly_product_df.empty else 0
        overall_churn_rate = (total_canceled / total_created * 100) if total_created > 0 else 0
        overall_retention_rate = 100 - overall_churn_rate
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
        if st.button(f"Export {selected_product} Monthly Analysis"):
            monthly_product_df.to_csv(f'{selected_product}_monthly_analysis.csv', index=False)
            st.success(f"Monthly analysis for {selected_product} exported to '{selected_product}_monthly_analysis.csv'")

else:
    st.info("Analizi başlatmak için tarih aralığı ve ürün seçip butona basın.")

# Save analysis results
st.success("Analysis Completed!") 