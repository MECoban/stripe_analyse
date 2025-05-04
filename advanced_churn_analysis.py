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

# Define Product IDs and Prices globally
VIP_PRODUCT_IDS_19 = ["prod_ReY9OvvKriHaxW", "prod_SBvjXlo061XZwu", "prod_ReWWPgA2gHZMUx"]  # $19/ay
VIP_PRODUCT_IDS_49 = ["prod_SBuNKVN5n7M3vy", "prod_SCw2DzI2S9QNMJ"]  # $49/ay
GOLD_PRODUCT_IDS = ["prod_QPlfKu5msYB4LD", "prod_QPl3bHPgziJbg7", "prod_R0TVW9SQX1HJXj"]  # $97/ay

PRODUCT_PRICES = {
    'GOLD': 97,
    'VIP_19': 19,
    'VIP_49': 49
}

# Define product start dates
PRODUCT_START_DATES = {
    "GOLD": datetime(2024, 7, 4),
    "VIP": datetime(2025, 1, 25)
}

# Cache the data fetching function - Reverted to using show_spinner, removed st.error
@st.cache_data(ttl=3600, show_spinner="Stripe verileri alınıyor...")
def fetch_product_specific_data(selected_product_label, min_creation_timestamp):
    # These IDs are now defined globally
    if selected_product_label == "GOLD":
        target_product_ids = set(GOLD_PRODUCT_IDS)
    elif selected_product_label == "VIP":
        target_product_ids = set(VIP_PRODUCT_IDS_19 + VIP_PRODUCT_IDS_49)
    else:
        return [], [], "Geçersiz ürün seçimi" # Return error message

    error_message = None # Variable to store potential errors

    # --- Step 1: Fetch Relevant Subscriptions --- 
    all_subscriptions = []
    has_more = True
    starting_after = None
    while has_more:
        params = {
            'limit': 100, 
            'status': 'all',
            'created': {'gte': min_creation_timestamp}
        }
        if starting_after:
            params['starting_after'] = starting_after
        try:
            response = stripe.Subscription.list(**params)
            if not response.data:
                has_more = False
                break
            all_subscriptions.extend(response.data)
            has_more = response.has_more
            if has_more:
                starting_after = response.data[-1].id
            else:
                has_more = False
        except Exception as e:
            error_message = f"Abonelikler alınırken hata: {e}"
            print(error_message) # Log error instead of st.error
            return [], [], error_message # Return error message
            
    # --- Step 2: Filter Subscriptions --- 
    filtered_subscriptions = []
    relevant_customer_ids = set()
    for sub in all_subscriptions:
        product_id = sub.plan.product if hasattr(sub, 'plan') and hasattr(sub.plan, 'product') else None
        if product_id in target_product_ids:
            filtered_subscriptions.append(sub)
            if sub.customer: 
                relevant_customer_ids.add(sub.customer) 
            
    if not relevant_customer_ids:
         return [], [], "İlgili abonelik bulunamadı." # Return message

    # --- Step 3: Fetch Relevant Customers --- 
    relevant_customers = []
    has_more = True
    starting_after = None
    while has_more:
        params = {'limit': 100}
        if starting_after:
            params['starting_after'] = starting_after
        try:
            response = stripe.Customer.list(**params)
            if not response.data:
                has_more = False
                break
            
            page_relevant_customers = [c for c in response.data if c.id in relevant_customer_ids]
            relevant_customers.extend(page_relevant_customers)
            
            if len(relevant_customers) >= len(relevant_customer_ids):
                has_more = False
                break
                
            has_more = response.has_more
            if has_more:
                starting_after = response.data[-1].id
            else:
                has_more = False
        except Exception as e:
            error_message = f"Müşteriler alınırken hata: {e}"
            print(error_message) # Log error instead of st.error
            # Return partially fetched data and error message
            return relevant_customers, filtered_subscriptions, error_message 

    return relevant_customers, filtered_subscriptions, error_message # Return data and None error if successful

st.title("Gelişmiş Müşteri Analiz Paneli")

stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
if not stripe.api_key:
    st.error("Stripe API anahtarı bulunamadı. Lütfen STRIPE_SECRET_KEY ortam değişkenini ayarlayın.")
    st.stop()

try:
    stripe.Customer.list(limit=1)
    st.success("Stripe API'sine başarıyla bağlanıldı!")
except Exception as e:
    st.error(f"Stripe'a bağlanırken hata: {str(e)}")
    st.stop()

# Product selector
product_selector = st.selectbox("Ürün Seçin", ["GOLD", "VIP"])

# Determine min creation timestamp based on selection
min_creation_date = PRODUCT_START_DATES.get(product_selector, datetime(1970, 1, 1))
min_creation_timestamp = int(min_creation_date.timestamp())

# Fetch data using the cached function
customers, subscriptions, fetch_error = fetch_product_specific_data(product_selector, min_creation_timestamp)

# Handle potential errors returned from the fetch function
if fetch_error:
    st.error(fetch_error)
    st.stop() # Stop execution if there was a fetch error

# Proceed only if data fetching was successful
if customers is not None and subscriptions is not None:
    # Process data into DataFrame (uses global IDs)
    customer_data = []
    for customer in customers: 
        customer_subs = [sub for sub in subscriptions if sub.customer == customer.id] 
        for sub in customer_subs: 
            product_id = sub.plan.product if hasattr(sub, 'plan') and hasattr(sub.plan, 'product') else 'Unknown Product'
            # Use globally defined IDs
            if product_id in GOLD_PRODUCT_IDS:
                product_label = 'GOLD'
                price = PRODUCT_PRICES['GOLD']
            elif product_id in VIP_PRODUCT_IDS_19:
                product_label = 'VIP_19'
                price = PRODUCT_PRICES['VIP_19']
            elif product_id in VIP_PRODUCT_IDS_49:
                product_label = 'VIP_49'
                price = PRODUCT_PRICES['VIP_49']
            else:
                product_label = product_id
                price = 0
            
            customer_data.append({
                'Customer ID': customer.id,
                'Customer Email': customer.email,
                'Product': product_label, # Should match product_selector now
                'Subscription ID': sub.id, # Added back to retrieve details later
                'Created (UTC)': datetime.fromtimestamp(sub.created).strftime('%Y-%m-%d %H:%M:%S'),
                'Canceled At (UTC)': datetime.fromtimestamp(sub.canceled_at).strftime('%Y-%m-%d %H:%M:%S') if sub.canceled_at else None,
                'Status': sub.status,
                'Fiyat': price
            })

    data = pd.DataFrame(customer_data)

    # --- Customer-Centric Deduplication ---
    if not data.empty:
        # Ensure 'Created (UTC)' is datetime for sorting
        data['Created_dt'] = pd.to_datetime(data['Created (UTC)'], errors='coerce')
        # Sort by email and latest creation date first
        data.sort_values(by=['Customer Email', 'Created_dt'], ascending=[True, False], inplace=True)
        # Keep only the first (latest) subscription record for each customer email
        data = data.drop_duplicates(subset=['Customer Email'], keep='first')
        # Drop the temporary datetime column if no longer needed
        # data.drop(columns=['Created_dt'], inplace=True)
    # --- End Deduplication ---

    # Filter for selected product AFTER deduplication
    if product_selector == "GOLD":
        analysis_data = data[data['Product'] == 'GOLD'].copy()
        selected_products = ['GOLD']
        product_header = "Ürün Analizi: GOLD"
    else:
        analysis_data = data[data['Product'] == 'VIP'].copy()
        selected_products = ['VIP']
        product_header = "Ürün Analizi: VIP"

    if analysis_data.empty:
        st.warning(f"Seçilen ürün için veri bulunamadı: {product_selector}")
        st.stop()

    # Define active condition based on NOW
    now = pd.Timestamp.utcnow().tz_localize(None) # Use UTC now but make it timezone-naive for comparison
    if not analysis_data.empty:
        # Ensure canceled_at_dt exists and is naive datetime
        if 'canceled_at_dt' not in analysis_data.columns or analysis_data['canceled_at_dt'].dt.tz is not None:
            analysis_data.loc[:, 'canceled_at_dt'] = pd.to_datetime(analysis_data['Canceled At (UTC)'], errors='coerce').dt.tz_localize(None)

        active_mask_now = (analysis_data['Status'] == 'active') | \
                      (analysis_data['Status'] == 'trialing') | \
                      (analysis_data['Status'] == 'overdue') | \
                      (analysis_data['Status'] == 'past_due') | \
                      ((analysis_data['Status'] == 'canceled') & (analysis_data['canceled_at_dt'].notna()) & (analysis_data['canceled_at_dt'] > now))
        active_subs_now_df = analysis_data[active_mask_now]
        
        # Calculate trialing count based on NOW
        trialing_mask_now = (analysis_data['Status'] == 'trialing')
        trialing_count_now = len(analysis_data[trialing_mask_now])

        # Calculate future-canceled count based on NOW
        future_canceled_mask_now = (analysis_data['Status'] == 'canceled') & (analysis_data['canceled_at_dt'].notna()) & (analysis_data['canceled_at_dt'] > now)
        future_canceled_count_now = len(analysis_data[future_canceled_mask_now])

        # Calculate overdue count based on NOW (now includes both overdue and past_due for display)
        overdue_mask_now = (analysis_data['Status'] == 'overdue') | (analysis_data['Status'] == 'past_due')
        overdue_count_now = len(analysis_data[overdue_mask_now])

        # Filter for past_due customers based on NOW
        past_due_mask_now = (analysis_data['Status'] == 'past_due')
        past_due_df_now = analysis_data[past_due_mask_now]

    else:
        active_subs_now_df = pd.DataFrame(columns=analysis_data.columns)
        trialing_count_now = 0
        future_canceled_count_now = 0
        overdue_count_now = 0

    # Convert dates for analysis
    analysis_data.loc[:, 'created_date'] = pd.to_datetime(analysis_data['Created (UTC)'], errors='coerce')
    analysis_data.loc[:, 'canceled_date'] = pd.to_datetime(analysis_data['Canceled At (UTC)'], errors='coerce')
    analysis_data.loc[:, 'created_month'] = analysis_data['created_date'].dt.to_period('M')
    analysis_data.loc[:, 'canceled_month'] = analysis_data['canceled_date'].dt.to_period('M')

    # Calculate average customer lifetime (moved here from product comparison)
    analysis_data['lifetime_days'] = (analysis_data['canceled_date'] - analysis_data['created_date']).dt.days
    # Calculate average only for customers who have actually canceled
    avg_lifetime = analysis_data.loc[analysis_data['canceled_date'].notna(), 'lifetime_days'].mean()
    # Format for display, handle NaN if no customers canceled
    avg_lifetime_display = f"{avg_lifetime:.0f}" if pd.notna(avg_lifetime) else "Yok"

    # Define all_months 
    if not analysis_data.empty and 'created_month' in analysis_data.columns:
        min_date = analysis_data['created_month'].min()
        max_date = pd.Timestamp.today().to_period('M')
        if pd.isna(min_date):
            all_months = pd.PeriodIndex([])
        else:
            # Ensure start is not after end
            if min_date > max_date:
                 all_months = pd.PeriodIndex([min_date]) # Or handle as error/empty
            else:
                 all_months = pd.period_range(start=min_date, end=max_date, freq='M')
    else:
        all_months = pd.PeriodIndex([])

    # Calculate active subscriptions based on the END OF THE LAST MONTH in the analysis range
    last_month_active_subs_df = pd.DataFrame(columns=analysis_data.columns)
    if not all_months.empty:
        last_month = all_months[-1]
        last_month_end_ts = last_month.to_timestamp(how='end').tz_localize(None)

        if not analysis_data.empty:
            # Ensure canceled_at_dt exists and is naive datetime
            if 'canceled_at_dt' not in analysis_data.columns or analysis_data['canceled_at_dt'].dt.tz is not None:
                analysis_data.loc[:, 'canceled_at_dt'] = pd.to_datetime(analysis_data['Canceled At (UTC)'], errors='coerce').dt.tz_localize(None)

            last_month_active_mask = \
                (analysis_data['created_month'] <= last_month) & \
                (
                    (analysis_data['Status'] == 'active') | \
                    (analysis_data['Status'] == 'trialing') | \
                    (analysis_data['Status'] == 'overdue') | \
                    (analysis_data['Status'] == 'past_due') | \
                    ((analysis_data['Status'] == 'canceled') & 
                     (analysis_data['canceled_at_dt'].notna()) & 
                     (analysis_data['canceled_at_dt'] > last_month_end_ts))
                )
            last_month_active_subs_df = analysis_data[last_month_active_mask]

    # Sidebar tab navigation in Turkish
    sidebar_tab = st.sidebar.radio(
        "Analiz Bölümü Seçin",
        [
            "Aylık Analiz",
            "Müşteri Detayları",
            "Aylık Dağılım",
            "Aylık Ürün Analizi"
        ]
    )

    # Analysis tabs
    if sidebar_tab == "Aylık Analiz":
        # Show header, metrics (using NOW active count), and main dataframe ONLY in this tab
        st.header(product_header)
        
        # Display metrics in two columns
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Toplam Abonelikler", len(analysis_data))
            st.metric("İptal Edilen Abonelikler (Toplam)", (analysis_data['Status'] == 'canceled').sum())
            st.metric("Gelecekte İptal Olacaklar (Şu An)", future_canceled_count_now)
            st.metric("Ort. Müşteri Ömrü (gün)", avg_lifetime_display)
        with col2:
            st.metric("Aktif Abonelikler (Şu An)", len(active_subs_now_df))
            st.metric("Deneme Sürümündeki Abonelikler (Şu An)", trialing_count_now)
            st.metric("Ödemesi Başarısız Olanlar (Şu An)", overdue_count_now)
            
        # Display Past Due customer list
        st.subheader("Ödemesi Başarısız Olan Abonelerin Listesi")
        if not past_due_df_now.empty:

            invoice_attempt_counts = [] # List for invoice attempt counts
            last_attempt_dates = [] # List for last attempt dates
            with st.spinner("Ödemesi başarısız abonelerin fatura detayları alınıyor..."):
                for sub_id in past_due_df_now['Subscription ID']:
                    inv_attempt_count = "Yok" # Default value
                    last_attempt_date = "Yok" # Default date
                    try:
                        sub = stripe.Subscription.retrieve(sub_id)
                        if sub.latest_invoice:
                            invoice = stripe.Invoice.retrieve(sub.latest_invoice)
                            inv_attempt_count = invoice.attempt_count

                            # Try to find the last failed payment event for this invoice
                            try:
                                events = stripe.Event.list(
                                    type='invoice.payment_failed',
                                    # Use the new recommended parameter 'resource' instead of 'related_object'
                                    # See Stripe API changelog 2022-08-01
                                    # It might vary depending on the stripe-python version
                                    # Falling back to related_object if resource is not standard yet in all versions
                                    # Let's try 'resource' first, but this might need adjustment based on library version
                                    # Using related_object for potentially broader compatibility for now:
                                    related_object=invoice.id, 
                                    limit=1 # Get only the most recent one
                                )
                                if events.data:
                                    latest_event = events.data[0]
                                    last_attempt_timestamp = latest_event.created
                                    last_attempt_date = datetime.fromtimestamp(last_attempt_timestamp).strftime('%Y-%m-%d %H:%M:%S')
                                else:
                                    # No failed event found, maybe the first attempt hasn't failed yet?
                                    # Or maybe the first attempt succeeded then subscription churned?
                                    # Check if attempt_count is > 0 but no failed event
                                    if inv_attempt_count > 0:
                                         last_attempt_date = "Başrsz Olay Yok" # Updated N/A message
                                    else: # attempt_count is 0 or None
                                         last_attempt_date = "Henüz Deneme Yok" # Updated N/A message

                            except stripe.error.PermissionError as event_perm_err:
                                print(f"Permission Error fetching events for invoice {invoice.id}: {event_perm_err}")
                                last_attempt_date = "İzin H. (Olay)" # Updated error message
                            except Exception as event_err:
                                print(f"Error fetching events for invoice {invoice.id}: {event_err}")
                                last_attempt_date = "Hata (Olay)" # Updated error message
                        else:
                            inv_attempt_count = "Fatura Yok" # Updated N/A message
                            last_attempt_date = "Fatura Yok" # Updated N/A message

                    except stripe.error.PermissionError as e:
                        print(f"Permission Error processing sub {sub_id}: {e}")
                        inv_attempt_count = "İzin Hatası" # Updated error message
                        last_attempt_date = "İzin Hatası" # Updated error message
                    except stripe.error.InvalidRequestError as e:
                         print(f"Invalid Request Error processing sub {sub_id}: {e}")
                         inv_attempt_count = "API Hatası" # Updated error message
                         last_attempt_date = "API Hatası" # Updated error message
                    except Exception as e:
                        print(f"General Error processing sub {sub_id}: {e}")
                        inv_attempt_count = "Hata" # Updated error message
                        last_attempt_date = "Hata" # Updated error message
                        
                    invoice_attempt_counts.append(inv_attempt_count)
                    last_attempt_dates.append(last_attempt_date) # Append the date

            # Create a display DataFrame with the new columns
            past_due_df_display = past_due_df_now[['Customer Email', 'Product', 'Status', 'Created (UTC)']].copy()
            past_due_df_display['Başarısız Ödeme Sayısı'] = invoice_attempt_counts
            past_due_df_display['Son Deneme (UTC)'] = last_attempt_dates # Add the date column
            
            # Display the table
            st.dataframe(past_due_df_display)
            
            # Optional: Add export for past due list including the new columns
            if st.button("Listeyi Dışa Aktar"):
                export_df = past_due_df_now.copy()
                export_df['Invoice Attempt Count'] = invoice_attempt_counts
                export_df['Last Attempt (UTC)'] = last_attempt_dates # Add date to export
                export_df.to_csv('odemesi_basarisiz_aboneler.csv', index=False)
                st.success("Ödemesi başarısız aboneler listesi 'odemesi_basarisiz_aboneler.csv' dosyasına aktarıldı")
        else:
            st.info("Şu anda 'past_due' (ödemesi başarısız) durumunda müşteri bulunmamaktadır.")
        
        # Calculate customers created per month
        created_per_month = analysis_data.groupby(['created_month', 'Product'])['Customer ID'].count().reset_index()
        created_per_month = created_per_month.pivot(index='created_month', columns='Product', values='Customer ID').fillna(0)

        # Calculate customers canceled per month
        canceled_per_month = analysis_data.groupby(['canceled_month', 'Product'])['Customer ID'].count().reset_index()
        canceled_per_month = canceled_per_month.pivot(index='canceled_month', columns='Product', values='Customer ID').fillna(0)

        # Calculate active customers per month correctly
        active_per_month = {}
        for month in all_months:
            # Apply the same active logic for the specific month end
            month_end = month.to_timestamp(how='end') # Get the timestamp for comparison
            
            # Ensure dates are comparable (convert month Period to timestamp for comparison with datetime)
            active_mask_month = \
                (analysis_data['created_month'] <= month) & \
                (
                    (analysis_data['Status'] == 'active') | \
                    (analysis_data['Status'] == 'trialing') | \
                    (analysis_data['Status'] == 'overdue') | \
                    (analysis_data['Status'] == 'past_due') | \
                    ((analysis_data['Status'] == 'canceled') & (analysis_data['canceled_at_dt'] > month_end))
                )
            active_customers = analysis_data[active_mask_month]
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
        st.write("### Ürünlere Göre Toplam Churn Oranı")
        total_churn_df = pd.DataFrame({
            'Ürün': all_products,
            'Toplam Oluşturulan': total_created.values,
            'Toplam İptal': total_canceled.values,
            'Toplam Churn Oranı (%)': total_churn_rate.values
        })
        st.dataframe(total_churn_df)
        
        # Display monthly analysis visualization
        st.write("### Ürünlere Göre Aylık Analiz")
        fig = make_subplots(rows=3, cols=1, subplot_titles=("Aktif Müşteriler", "Oluşturulan vs İptal Edilen", "Churn Oranı"), vertical_spacing=0.1)
        for product in selected_products:
            if product in active_per_month_df.columns:
                fig.add_trace(go.Scatter(x=active_per_month_df.index.astype(str), y=active_per_month_df[product], name=f"{product} - Aktif", line=dict(width=2)), row=1, col=1)
            if product in created_per_month.columns:
                fig.add_trace(go.Scatter(x=created_per_month.index.astype(str), y=created_per_month[product], name=f"{product} - Oluşturulan", line=dict(width=2)), row=2, col=1)
            if product in canceled_per_month.columns:
                fig.add_trace(go.Scatter(x=canceled_per_month.index.astype(str), y=canceled_per_month[product], name=f"{product} - İptal", line=dict(width=2)), row=2, col=1)
            if product in churn_rate_df.columns:
                fig.add_trace(go.Scatter(x=churn_rate_df.index.astype(str), y=churn_rate_df[product], name=f"{product} - Churn Oranı", line=dict(width=2)), row=3, col=1)
        fig.update_layout(height=900, title_text="Çoklu Ürün Müşteri Analizi", showlegend=True)
        st.plotly_chart(fig)
    elif sidebar_tab == "Müşteri Detayları":
        # Customer details
        st.header("Müşteri Detayları") # Keep header for the tab
        
        # Add the main dataframe here with the new title
        st.subheader("Tüm Abonelerin Detaylı Bilgileri")
        if 'analysis_data' in locals() and not analysis_data.empty:
            st.dataframe(analysis_data)
        else:
            st.warning("Gösterilecek abone verisi bulunamadı.")
            
        st.divider() # Add a divider before filter options
        st.write("### Filtreleme Seçenekleri") # Add a subheader for filters
        
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
        if st.button("Müşteri Detaylarını Dışa Aktar"):
            customer_details.to_csv('musteri_detaylari.csv', index=False)
            st.success("Müşteri detayları 'musteri_detaylari.csv' dosyasına aktarıldı")
    elif sidebar_tab == "Aylık Dağılım":
        # Monthly breakdown analysis
        st.write("### Aylık Dağılım Tablosu")
        monthly_summary = []
        now_naive = pd.Timestamp.utcnow().tz_localize(None) # Naive UTC now for comparison

        for month in all_months:
            month_str = str(month)
            month_end_ts = month.to_timestamp(how='end').tz_localize(None) # Naive timestamp for end of month

            # Use consistent active definition as of month_end_ts, now including 'overdue'
            active_at_month_end_mask = \
                (analysis_data['created_month'] <= month) & \
                (
                    (analysis_data['Status'] == 'active') | \
                    (analysis_data['Status'] == 'trialing') | \
                    (analysis_data['Status'] == 'overdue') | \
                    (analysis_data['Status'] == 'past_due') | \
                    ((analysis_data['Status'] == 'canceled') &
                     (analysis_data['canceled_at_dt'].notna()) & # Ensure canceled_at_dt is not NaT
                     (analysis_data['canceled_at_dt'] > month_end_ts))
                )
            
            # Calculate metrics for each product for this month
            for product in selected_products:
                 # Filter data specific to this product and active at month end
                product_month_data_active = analysis_data[active_at_month_end_mask & (analysis_data['Product'] == product)]
                active_at_month_end_count = len(product_month_data_active)

                # Created this month
                created_this_month = len(analysis_data[(analysis_data['created_month'] == month) & (analysis_data['Product'] == product)])
                
                # Canceled this specific month (Status changed to canceled or canceled_at is within this month)
                canceled_this_month = len(analysis_data[
                    (analysis_data['canceled_month'] == month) & 
                    (analysis_data['Product'] == product)
                ])

                # Original Churn Rate (vs Created)
                churn_rate_original = (canceled_this_month / created_this_month * 100) if created_this_month > 0 else 0

                # New Churn Rate Oguzhan (vs Active)
                churn_rate_oguzhan = (canceled_this_month / active_at_month_end_count * 100) if active_at_month_end_count > 0 else 0
                
                monthly_summary.append({
                    'Ay': month_str,
                    'Ürün': product,
                    'Oluşturulan': created_this_month,
                    'İptal': canceled_this_month,
                    'Aktif': active_at_month_end_count,
                    'Churn Oranı (%)': churn_rate_original, # Keep original name or rename?
                    'Churn Oranı (Aktife Göre)': churn_rate_oguzhan
                })
                
        monthly_summary_df = pd.DataFrame(monthly_summary)
        st.write("#### Ürün Bazında Aylık Özet")
        st.dataframe(monthly_summary_df)
        
        st.write("#### Zaman İçinde Müşteri Büyümesi")
        growth_data = monthly_summary_df.pivot(index='Ay', columns='Ürün', values='Aktif').fillna(0)
        fig = go.Figure()
        for product in growth_data.columns:
            fig.add_trace(go.Scatter(x=growth_data.index, y=growth_data[product], name=product, stackgroup='one', mode='lines'))
        fig.update_layout(title='Zaman İçinde Müşteri Büyümesi', xaxis_title='Ay', yaxis_title='Aktif Müşteri Sayısı', height=500)
        st.plotly_chart(fig)
        if st.button("Aylık Özeti Dışa Aktar"):
            monthly_summary_df.to_csv('aylik_ozet.csv', index=False)
            st.success("Aylık özet 'aylik_ozet.csv' dosyasına aktarıldı")
    elif sidebar_tab == "Aylık Ürün Analizi":
        # Monthly Product Analysis
        st.write("### Aylık Ürün Analizi")
        selected_product = st.selectbox("Detaylı analiz için ürün seçin", selected_products) 
        product_data = analysis_data # Already filtered
        monthly_product_metrics = []
        now_naive = pd.Timestamp.utcnow().tz_localize(None)

        for month in all_months:
            month_str = str(month)
            month_end_ts = month.to_timestamp(how='end').tz_localize(None)

            # Created this month for the selected product
            created_this_month = len(product_data[product_data['created_month'] == month])
            
            # Canceled this month for the selected product
            canceled_this_month = len(product_data[product_data['canceled_month'] == month])

            # Active at end of month using consistent definition, now including 'overdue'
            active_at_month_end_mask = \
                (product_data['created_month'] <= month) & \
                (
                    (product_data['Status'] == 'active') | \
                    (product_data['Status'] == 'trialing') | \
                    (product_data['Status'] == 'overdue') | \
                    (product_data['Status'] == 'past_due') | \
                    ((product_data['Status'] == 'canceled') &
                     (product_data['canceled_at_dt'].notna()) &
                     (product_data['canceled_at_dt'] > month_end_ts))
                )
            active_at_month_end_count = len(product_data[active_at_month_end_mask])

            churn_rate = (canceled_this_month / created_this_month * 100) if created_this_month > 0 else 0
            net_growth = created_this_month - canceled_this_month
            # Note: Growth rate definition might need refinement based on exact business logic
            growth_rate = (net_growth / active_at_month_end_count * 100) if active_at_month_end_count > 0 else 0 
            retention_rate = 100 - churn_rate

            monthly_product_metrics.append({
                'Ay': month_str,
                'Oluşturulan': created_this_month,
                'İptal': canceled_this_month,
                'Aktif': active_at_month_end_count,
                'Net Büyüme': net_growth,
                'Churn Oranı (%)': churn_rate,
                'Büyüme Oranı (%)': growth_rate,
                'Tutma Oranı (%)': retention_rate
            })
        monthly_product_df = pd.DataFrame(monthly_product_metrics)
        st.write(f"#### {selected_product} İçin Aylık Analiz")
        st.dataframe(monthly_product_df)
        st.write(f"#### {selected_product} İçin Aylık Trendler")
        fig = make_subplots(rows=3, cols=1, subplot_titles=("Müşteri Sayıları", "Büyüme Metrikleri", "Oranlar"), vertical_spacing=0.1)
        fig.add_trace(go.Scatter(x=monthly_product_df['Ay'], y=monthly_product_df['Oluşturulan'], name="Oluşturulan", line=dict(width=2, color='green')), row=1, col=1)
        fig.add_trace(go.Scatter(x=monthly_product_df['Ay'], y=monthly_product_df['İptal'], name="İptal", line=dict(width=2, color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=monthly_product_df['Ay'], y=monthly_product_df['Aktif'], name="Aktif", line=dict(width=2, color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=monthly_product_df['Ay'], y=monthly_product_df['Net Büyüme'], name="Net Büyüme", line=dict(width=2, color='purple')), row=2, col=1)
        fig.add_trace(go.Scatter(x=monthly_product_df['Ay'], y=monthly_product_df['Churn Oranı (%)'], name="Churn Oranı (%)", line=dict(width=2, color='red')), row=3, col=1)
        fig.add_trace(go.Scatter(x=monthly_product_df['Ay'], y=monthly_product_df['Büyüme Oranı (%)'], name="Büyüme Oranı (%)", line=dict(width=2, color='green')), row=3, col=1)
        fig.add_trace(go.Scatter(x=monthly_product_df['Ay'], y=monthly_product_df['Tutma Oranı (%)'], name="Tutma Oranı (%)", line=dict(width=2, color='blue')), row=3, col=1)
        fig.update_layout(height=900, title_text=f"{selected_product} İçin Aylık Analiz", showlegend=True)
        st.plotly_chart(fig)
        st.write(f"#### {selected_product} İçin Anahtar Metrikler")
        total_created = monthly_product_df['Oluşturulan'].sum()
        total_canceled = monthly_product_df['İptal'].sum()
        current_active = monthly_product_df['Aktif'].iloc[-1] if not monthly_product_df.empty else 0
        overall_churn_rate = (total_canceled / total_created * 100) if total_created > 0 else 0
        overall_retention_rate = 100 - overall_churn_rate
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Toplam Oluşturulan", total_created)
            st.metric("Toplam İptal", total_canceled)
        with col2:
            st.metric("Mevcut Aktif", current_active)
            st.metric("Net Büyüme", total_created - total_canceled)
        with col3:
            st.metric("Genel Churn Oranı (%)", f"{overall_churn_rate:.2f}%")
            st.metric("Genel Tutma Oranı (%)", f"{overall_retention_rate:.2f}%")
        if st.button(f"{selected_product} Aylık Analizini Dışa Aktar"):
            monthly_product_df.to_csv(f'{selected_product}_aylik_analiz.csv', index=False)
            st.success(f"{selected_product} için aylık analiz '{selected_product}_aylik_analiz.csv' dosyasına aktarıldı")

else:
    st.error("Veri alınamadı.")

# Save analysis results
st.success("Analysis Completed!") 