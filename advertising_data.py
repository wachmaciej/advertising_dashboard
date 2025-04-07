# Friday, April 4, 2025 at 5:12:30 PM BST

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import plotly.express as px
import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import re
from functools import reduce

# Filter warnings for a clean output
warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(
    page_title="YOY Dashboard - Advertising Data",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

# =============================================================================
# Common Functions for Advertising Data
# =============================================================================

@st.cache_data
def preprocess_ad_data(df):
    """Preprocess advertising data for analysis"""
    if df is None or df.empty:
         return pd.DataFrame()
    df_processed = df.copy() # Work on a copy
    # Attempt to convert 'WE Date', handle errors gracefully
    try:
        # Try common date formats first
        df_processed["WE Date"] = pd.to_datetime(df_processed["WE Date"], format="%d/%m/%Y", dayfirst=True, errors='coerce')
        if df_processed["WE Date"].isnull().any(): # If first format failed for some, try another
            # Try letting pandas infer if the first format didn't work for all rows
            df_processed["WE Date"] = pd.to_datetime(df_processed["WE Date"], errors='coerce')

        # Drop rows where date conversion failed completely
        original_rows = len(df_processed)
        df_processed.dropna(subset=["WE Date"], inplace=True)
        if len(df_processed) < original_rows:
             st.warning(f"Dropped {original_rows - len(df_processed)} rows due to invalid 'WE Date' format.")

        df_processed = df_processed.sort_values("WE Date")

    except KeyError:
        st.error("Input data is missing the required 'WE Date' column.")
        return pd.DataFrame() # Return empty DataFrame on critical error
    except Exception as e:
        st.error(f"Error processing 'WE Date': {e}")
        return pd.DataFrame()

    # Define numeric columns including the new 'Total Sales'
    numeric_cols = [
        "Impressions", "Clicks", "Spend", "Sales", "Orders", "Units", "Total Sales", # Added Total Sales
        "CTR", "CVR", "Orders %", "Spend %", "Sales %", "ACOS", "ROAS"
    ]
    # Ensure Year and Week columns are added *after* WE Date is confirmed valid datetime
    if 'WE Date' in df_processed.columns and pd.api.types.is_datetime64_any_dtype(df_processed['WE Date']):
        if 'Year' not in df_processed.columns: df_processed["Year"] = df_processed["WE Date"].dt.year
        if 'Week' not in df_processed.columns: df_processed["Week"] = df_processed["WE Date"].dt.isocalendar().week

    for col in numeric_cols:
        if col in df_processed.columns:
            # Convert to numeric, coercing errors to NaN
            df_processed[col] = pd.to_numeric(df_processed[col], errors="coerce")

    # Ensure Year/Week are integer types after potential creation/conversion, handle NaNs
    for col in ['Year', 'Week']:
         if col in df_processed.columns:
              df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
              # Drop rows if Year/Week conversion failed AFTER numeric conversion attempt
              pre_drop_len = len(df_processed)
              df_processed.dropna(subset=[col], inplace=True)
              if len(df_processed) < pre_drop_len:
                   st.warning(f"Dropped {pre_drop_len - len(df_processed)} rows due to invalid values in '{col}' column.")
              # Check if dataframe is empty after dropna before astype
              if not df_processed.empty:
                   try:
                        df_processed[col] = df_processed[col].astype(int)
                   except ValueError:
                        st.error(f"Could not convert '{col}' to integer after cleaning. Check data.")
                        # Decide handling: return empty or try proceeding without int conversion
                        return pd.DataFrame()
              elif col in ['Year', 'Week']: # If Year/Week were critical and now df is empty
                   st.warning(f"No valid data remaining after cleaning '{col}' column.")
                   return pd.DataFrame()

    return df_processed

@st.cache_data
def filter_data_by_timeframe(df, selected_years, selected_timeframe, selected_week):
    """
    Filters data for selected years based on timeframe.
    - "Specific Week": Filters all selected years for that specific week number.
    - "Last X Weeks": Determines the last X weeks based on the *latest* selected year's max week,
                      and filters *all* selected years for those *same* week numbers.
    Returns a concatenated dataframe across the selected years.
    """
    if not selected_years:
        return pd.DataFrame()
    if df is None or df.empty:
        st.warning("Input DataFrame to filter_data_by_timeframe is empty.")
        return pd.DataFrame()

    try:
        selected_years = [int(y) for y in selected_years]
    except ValueError:
        st.error("Selected years must be numeric.")
        return pd.DataFrame()

    df_copy = df.copy()

    # Check required columns created during preprocessing
    if not {"WE Date", "Year", "Week"}.issubset(df_copy.columns):
        st.error("Required 'WE Date', 'Year', or 'Week' columns missing for timeframe filtering. Check preprocessing.")
        return pd.DataFrame()
    # Ensure types are correct (should be handled by preprocess, but double check)
    if not pd.api.types.is_datetime64_any_dtype(df_copy['WE Date']): df_copy['WE Date'] = pd.to_datetime(df_copy['WE Date'], errors='coerce')
    if not pd.api.types.is_integer_dtype(df_copy.get('Year')): df_copy['Year'] = pd.to_numeric(df_copy['Year'], errors='coerce').astype('Int64')
    if not pd.api.types.is_integer_dtype(df_copy.get('Week')): df_copy['Week'] = pd.to_numeric(df_copy['Week'], errors='coerce').astype('Int64')
    df_copy.dropna(subset=["WE Date", "Year", "Week"], inplace=True)
    if df_copy.empty: return pd.DataFrame()
    df_copy['Year'] = df_copy['Year'].astype(int) # Convert back to standard int after dropna
    df_copy['Week'] = df_copy['Week'].astype(int)


    df_filtered_years = df_copy[df_copy["Year"].isin(selected_years)].copy()
    if df_filtered_years.empty:
        return pd.DataFrame()

    if selected_timeframe == "Specific Week":
        if selected_week is not None:
            try:
                selected_week_int = int(selected_week)
                return df_filtered_years[df_filtered_years["Week"] == selected_week_int]
            except ValueError:
                st.error(f"Invalid 'selected_week': {selected_week}. Must be a number.")
                return pd.DataFrame()
        else:
            return pd.DataFrame() # No specific week selected
    else: # Last X Weeks
        try:
            match = re.search(r'\d+', selected_timeframe)
            if match:
                 weeks_to_filter = int(match.group(0))
            else:
                 raise ValueError("Could not find number in timeframe string")
        except Exception as e:
            st.error(f"Could not parse weeks from timeframe: '{selected_timeframe}': {e}")
            return pd.DataFrame()

        if df_filtered_years.empty: return pd.DataFrame()
        latest_year_with_data = int(df_filtered_years["Year"].max())

        df_latest_year = df_filtered_years[df_filtered_years["Year"] == latest_year_with_data]
        if df_latest_year.empty:
            st.warning(f"No data found for the latest selected year ({latest_year_with_data}) to determine week range.")
            return pd.DataFrame()

        global_max_week = int(df_latest_year["Week"].max())
        start_week = max(1, global_max_week - weeks_to_filter + 1)
        target_weeks = list(range(start_week, global_max_week + 1))

        final_filtered_df = df_filtered_years[df_filtered_years["Week"].isin(target_weeks)]
        return final_filtered_df


# =============================================================================
# Basic Chart/Table/Insight/Styling Helpers
# =============================================================================

@st.cache_data
def create_metric_comparison_chart(df, metric, portfolio_name=None, campaign_type="Sponsored Products"):
    """Creates a bar chart comparing a metric by Portfolio Name. Now calculates CPC if possible."""
    required_cols_base = {"Product", "Portfolio Name"}
    required_cols_metric = {metric} # Base requirement is the metric itself or components

    # Define base components needed if metric needs calculation
    base_components = {}
    if metric == "CTR": base_components = {"Clicks", "Impressions"}
    elif metric == "CVR": base_components = {"Orders", "Clicks"}
    elif metric == "ACOS": base_components = {"Spend", "Sales"}
    elif metric == "ROAS": base_components = {"Sales", "Spend"}
    elif metric == "CPC": base_components = {"Spend", "Clicks"} # Added CPC components

    if df is None or df.empty:
        return go.Figure()

    if not required_cols_base.issubset(df.columns):
        missing = required_cols_base - set(df.columns)
        st.warning(f"Comparison chart missing base columns: {missing}")
        return go.Figure()

    filtered_df = df[df["Product"] == campaign_type].copy()
    if filtered_df.empty:
        return go.Figure()

    # Check if metric exists OR if base components for calculation exist
    metric_col_exists = metric in filtered_df.columns
    can_calculate_metric = bool(base_components) and base_components.issubset(filtered_df.columns)

    if not metric_col_exists and not can_calculate_metric:
        missing = {metric} if not base_components else base_components - set(filtered_df.columns)
        st.warning(f"Comparison chart cannot display '{metric}'. Missing required columns: {missing} in {campaign_type} data.")
        return go.Figure()

    # Handle portfolio filtering
    filtered_df["Portfolio Name"] = filtered_df["Portfolio Name"].fillna("Unknown Portfolio")
    if portfolio_name and portfolio_name != "All Portfolios":
        if portfolio_name in filtered_df["Portfolio Name"].unique():
             filtered_df = filtered_df[filtered_df["Portfolio Name"] == portfolio_name]
        else:
             st.warning(f"Portfolio '{portfolio_name}' not found for {campaign_type}. Showing all.")
             portfolio_name = "All Portfolios" # Reset variable to reflect change

    if filtered_df.empty: # Check again after potential portfolio filter
        return go.Figure()

    # Calculation logic
    grouped_df = pd.DataFrame() # Initialize
    group_col = "Portfolio Name"
    try:
        # Handle metrics calculated from aggregated base components
        if metric in ["CTR", "CVR", "ACOS", "ROAS", "CPC"]: # Added CPC here
            if metric == "CTR":
                agg_df = filtered_df.groupby(group_col).agg(Nominator=("Clicks", "sum"), Denominator=("Impressions", "sum")).reset_index()
                agg_df[metric] = agg_df.apply(lambda r: (r["Nominator"] / r["Denominator"] * 100) if r["Denominator"] else 0, axis=1).round(2)
            elif metric == "CVR":
                agg_df = filtered_df.groupby(group_col).agg(Nominator=("Orders", "sum"), Denominator=("Clicks", "sum")).reset_index()
                agg_df[metric] = agg_df.apply(lambda r: (r["Nominator"] / r["Denominator"] * 100) if r["Denominator"] else 0, axis=1).round(2)
            elif metric == "ACOS":
                agg_df = filtered_df.groupby(group_col).agg(Nominator=("Spend", "sum"), Denominator=("Sales", "sum")).reset_index()
                agg_df[metric] = agg_df.apply(lambda r: (r["Nominator"] / r["Denominator"] * 100) if r["Denominator"] else np.nan, axis=1).round(2)
            elif metric == "ROAS":
                agg_df = filtered_df.groupby(group_col).agg(Nominator=("Sales", "sum"), Denominator=("Spend", "sum")).reset_index()
                agg_df[metric] = agg_df.apply(lambda r: (r["Nominator"] / r["Denominator"]) if r["Denominator"] else np.nan, axis=1).round(2)
            elif metric == "CPC": # Added CPC calculation logic
                agg_df = filtered_df.groupby(group_col).agg(Nominator=("Spend", "sum"), Denominator=("Clicks", "sum")).reset_index()
                agg_df[metric] = agg_df.apply(lambda r: (r["Nominator"] / r["Denominator"]) if r["Denominator"] else np.nan, axis=1).round(2) # Use np.nan if clicks are 0

            agg_df[metric] = agg_df[metric].replace([np.inf, -np.inf], np.nan) # Handle division errors
            grouped_df = agg_df[[group_col, metric]].copy()

        # Handle metrics that are directly aggregatable (like sum)
        elif metric_col_exists:
             # Ensure the column is numeric before attempting sum aggregation
             if pd.api.types.is_numeric_dtype(filtered_df[metric]):
                 grouped_df = filtered_df.groupby(group_col).agg(**{metric: (metric, "sum")}).reset_index()
             else:
                  st.warning(f"Comparison chart cannot aggregate non-numeric column '{metric}'.")
                  return go.Figure()
        else:
            # This case should be caught earlier, but as a fallback:
            st.warning(f"Comparison chart cannot display '{metric}'. Column not found and no calculation rule defined.")
            return go.Figure()

    except Exception as e:
        st.warning(f"Error aggregating comparison chart for {metric}: {e}")
        return go.Figure()

    grouped_df = grouped_df.dropna(subset=[metric])
    if grouped_df.empty:
        # st.info(f"No valid data points for metric '{metric}' after aggregation.") # Can be noisy
        return go.Figure()

    grouped_df = grouped_df.sort_values(metric, ascending=False)

    title_suffix = f" - {portfolio_name}" if portfolio_name and portfolio_name != "All Portfolios" else ""
    chart_title = f"{metric} by Portfolio ({campaign_type}){title_suffix}"

    fig = px.bar(grouped_df, x=group_col, y=metric, title=chart_title, text_auto=True) # Use group_col variable

    # Apply formatting
    if metric in ["Spend", "Sales"]:
        fig.update_traces(texttemplate='%{y:$,.0f}')
        fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",.2f")
    elif metric in ["CTR", "CVR", "ACOS"]:
        fig.update_traces(texttemplate='%{y:.1f}%')
        fig.update_layout(yaxis_ticksuffix="%", yaxis_tickformat=".1f")
    elif metric == "ROAS":
        fig.update_traces(texttemplate='%{y:.2f}')
        fig.update_layout(yaxis_tickformat=".2f")
    elif metric == "CPC": # Added CPC formatting
        fig.update_traces(texttemplate='%{y:$,.2f}') # Currency format for text on bars
        fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",.2f") # Currency format for y-axis
    else: # Default formatting for Impressions, Clicks, Orders, Units (summed metrics)
        fig.update_traces(texttemplate='%{y:,.0f}')
        fig.update_layout(yaxis_tickformat=",.0f")

    fig.update_layout(margin=dict(t=50, b=50, l=20, r=20), height=400) # Adjust margins/height
    return fig
# <<< End of updated function >>>

@st.cache_data
def create_performance_metrics_table(df, portfolio_name=None, campaign_type="Sponsored Products"):
    """Creates portfolio breakdown and total summary tables"""
    # THIS FUNCTION IS NO LONGER DIRECTLY USED FOR SP/SB/SD TABS but kept for potential future use / other parts
    required_cols = {"Product", "Portfolio Name", "Impressions", "Clicks", "Spend", "Sales", "Orders"}
    if df is None or df.empty:
       return pd.DataFrame(), pd.DataFrame()

    if not required_cols.issubset(df.columns):
       missing = required_cols - set(df.columns)
       st.warning(f"Performance table missing required columns: {missing}")
       return pd.DataFrame(), pd.DataFrame()

    filtered_df = df[df["Product"] == campaign_type].copy()
    filtered_df["Portfolio Name"] = filtered_df["Portfolio Name"].fillna("Unknown Portfolio")

    if portfolio_name and portfolio_name != "All Portfolios":
       if portfolio_name in filtered_df["Portfolio Name"].unique():
             filtered_df = filtered_df[filtered_df["Portfolio Name"] == portfolio_name]
       else:
             st.warning(f"Portfolio '{portfolio_name}' not found for {campaign_type} in performance table.")
             return pd.DataFrame(), pd.DataFrame()

    if filtered_df.empty:
       return pd.DataFrame(), pd.DataFrame()

    try:
       metrics_by_portfolio = filtered_df.groupby("Portfolio Name").agg(
            Impressions=("Impressions", "sum"),
            Clicks=("Clicks", "sum"),
            Spend=("Spend", "sum"),
            Sales=("Sales", "sum"),
            Orders=("Orders", "sum")
       ).reset_index()
    except Exception as e:
       st.warning(f"Error aggregating performance table: {e}")
       return pd.DataFrame(), pd.DataFrame()

    metrics_by_portfolio["CTR"] = metrics_by_portfolio.apply(lambda r: (r["Clicks"] / r["Impressions"] * 100) if r.get("Impressions") else 0, axis=1)
    metrics_by_portfolio["CVR"] = metrics_by_portfolio.apply(lambda r: (r["Orders"] / r["Clicks"] * 100) if r.get("Clicks") else 0, axis=1)
    metrics_by_portfolio["ACOS"] = metrics_by_portfolio.apply(lambda r: (r["Spend"] / r["Sales"] * 100) if r.get("Sales") else np.nan, axis=1)
    metrics_by_portfolio["ROAS"] = metrics_by_portfolio.apply(lambda r: (r["Sales"] / r["Spend"]) if r.get("Spend") else np.nan, axis=1)
    metrics_by_portfolio = metrics_by_portfolio.replace([np.inf, -np.inf], np.nan)

    for col in ["CTR", "CVR", "ACOS"]:
        if col in metrics_by_portfolio.columns: metrics_by_portfolio[col] = metrics_by_portfolio[col].round(1)
    for col in ["Spend", "Sales", "ROAS"]:
        if col in metrics_by_portfolio.columns: metrics_by_portfolio[col] = metrics_by_portfolio[col].round(2)

    total_summary = pd.DataFrame()
    if not filtered_df.empty:
        sum_cols = ["Impressions", "Clicks", "Spend", "Sales", "Orders"]
        numeric_summary_data = filtered_df.copy()
        for col in sum_cols:
             if col in numeric_summary_data.columns:
                  numeric_summary_data[col] = pd.to_numeric(numeric_summary_data[col], errors='coerce').fillna(0)
             else: numeric_summary_data[col] = 0

        total_impressions = numeric_summary_data["Impressions"].sum()
        total_clicks = numeric_summary_data["Clicks"].sum()
        total_spend = numeric_summary_data["Spend"].sum()
        total_sales = numeric_summary_data["Sales"].sum()
        total_orders = numeric_summary_data["Orders"].sum()
        total_ctr = (total_clicks / total_impressions * 100) if total_impressions else 0
        total_cvr = (total_orders / total_clicks * 100) if total_clicks else 0
        total_acos = (total_spend / total_sales * 100) if total_sales else np.nan
        total_roas = (total_sales / total_spend) if total_spend else np.nan
        total_acos = np.nan if total_acos in [np.inf, -np.inf] else total_acos
        total_roas = np.nan if total_roas in [np.inf, -np.inf] else total_roas
        total_summary_data = {
            "Metric": ["Total"],
            "Impressions": [total_impressions], "Clicks": [total_clicks], "Orders": [total_orders],
            "Spend": [round(total_spend, 2)], "Sales": [round(total_sales, 2)],
            "CTR": [round(total_ctr, 1)], "CVR": [round(total_cvr, 1)],
            "ACOS": [total_acos], "ROAS": [total_roas]
        }
        total_summary = pd.DataFrame(total_summary_data)

    metrics_by_portfolio = metrics_by_portfolio.rename(columns={"Portfolio Name": "Portfolio"})
    return metrics_by_portfolio, total_summary

@st.cache_data
def create_metric_over_time_chart(data, metric, portfolio, product_type, show_yoy=True, weekly_total_sales_data=None): # Added weekly_total_sales_data
    """Create a chart showing metric over time with optional YoY comparison (Weekly YoY Overlay with Month Annotations)."""
    if data is None or data.empty:
        return go.Figure()

    base_required = {"Product", "Portfolio Name", "WE Date", "Year", "Week"}
    if not base_required.issubset(data.columns):
        missing = base_required - set(data.columns)
        st.warning(f"Metric over time chart missing required columns: {missing}")
        return go.Figure()
    if not pd.api.types.is_datetime64_any_dtype(data['WE Date']):
        st.warning("WE Date column is not datetime type for time chart.")
        return go.Figure()

    data_copy = data.copy() # Work on a copy

    filtered_data = data_copy[data_copy["Product"] == product_type].copy()
    filtered_data["Portfolio Name"] = filtered_data["Portfolio Name"].fillna("Unknown Portfolio")
    if portfolio != "All Portfolios":
        if portfolio in filtered_data["Portfolio Name"].unique():
            filtered_data = filtered_data[filtered_data["Portfolio Name"] == portfolio]
        else:
            st.warning(f"Portfolio '{portfolio}' not found for {product_type}. Showing all.")
            portfolio = "All Portfolios" # Update variable to reflect change

    if filtered_data.empty:
        return go.Figure()

    # --- Define required base components for derived metrics ---
    metric_required_cols = {metric}
    base_needed_for_metric = set()
    is_derived_metric = False
    if metric == "CTR": base_needed_for_metric.update({"Clicks", "Impressions"}); is_derived_metric = True
    elif metric == "CVR": base_needed_for_metric.update({"Orders", "Clicks"}); is_derived_metric = True
    elif metric == "ACOS": base_needed_for_metric.update({"Spend", "Sales"}); is_derived_metric = True
    elif metric == "ROAS": base_needed_for_metric.update({"Sales", "Spend"}); is_derived_metric = True
    elif metric == "CPC": base_needed_for_metric.update({"Spend", "Clicks"}); is_derived_metric = True
    elif metric == "Ad % Sale": base_needed_for_metric.update({"Sales"}); is_derived_metric = True # Also needs external denom

    # --- Check if necessary columns exist for the selected metric ---
    metric_exists_in_input = metric in filtered_data.columns
    base_components_exist = base_needed_for_metric.issubset(filtered_data.columns)

    ad_sale_check_passed = True
    if metric == "Ad % Sale":
        if not base_components_exist: # Check 'Sales' column exists
             st.warning(f"Metric chart requires 'Sales' column for 'Ad % Sale'.")
             ad_sale_check_passed = False
        if weekly_total_sales_data is None or weekly_total_sales_data.empty:
             st.info(f"Denominator data (weekly total sales) not available for 'Ad % Sale' calculation.") # Use info, maybe intended
             ad_sale_check_passed = False
        elif not {"Year", "Week", "Weekly_Total_Sales"}.issubset(weekly_total_sales_data.columns):
             st.warning(f"Passed 'weekly_total_sales_data' is missing required columns (Year, Week, Weekly_Total_Sales).")
             ad_sale_check_passed = False

    # If it's a derived metric, we MUST have its base components
    if is_derived_metric and not base_components_exist:
         # Specific check for Ad % Sale denominator availability
         if metric == "Ad % Sale" and not ad_sale_check_passed:
              st.warning(f"Cannot calculate 'Ad % Sale'. Check required 'Sales' column and denominator data source.")
              return go.Figure()
         elif metric != "Ad % Sale": # For other derived metrics if base components are missing
              missing_bases = base_needed_for_metric - set(filtered_data.columns)
              st.warning(f"Cannot calculate derived metric '{metric}'. Missing required base columns: {missing_bases}")
              return go.Figure()

    # If it's NOT a derived metric, it MUST exist in the input data
    if not is_derived_metric and not metric_exists_in_input:
         st.warning(f"Metric chart requires column '{metric}' in the data.")
         return go.Figure()

    # --- Start Plotting ---
    years = sorted(filtered_data["Year"].dropna().unique().astype(int))
    fig = go.Figure()

    if metric in ["CTR", "CVR", "ACOS", "Ad % Sale"]: hover_suffix = "%"; hover_format = ".1f"
    elif metric in ["Spend", "Sales"]: hover_suffix = ""; hover_format = "$,.2f"
    elif metric in ["ROAS", "CPC"]: hover_suffix = ""; hover_format = ".2f"
    else: hover_suffix = ""; hover_format = ",.0f" # Impressions, Clicks, Orders, Units
    base_hover_template = f"Date: %{{customdata[1]|%Y-%m-%d}}<br>Week: %{{customdata[0]}}<br>{metric}: %{{y:{hover_format}}}{hover_suffix}<extra></extra>"

    processed_years = []
    colors = px.colors.qualitative.Plotly

    # ========================
    # YoY Plotting Logic
    # ========================
    if show_yoy and len(years) > 1:
        # Define columns needed for aggregation: base components + WE Date
        cols_to_agg_yoy = list(base_needed_for_metric | {metric} | {"WE Date"})
        # Ensure only columns actually present in the data are included
        actual_cols_to_agg_yoy = list(set(cols_to_agg_yoy) & set(filtered_data.columns))

        if "WE Date" not in actual_cols_to_agg_yoy: # WE Date is critical for hover
             st.warning("Missing 'WE Date' for aggregation (YoY).")
             return go.Figure()

        try:
            agg_dict_yoy = {}
            numeric_aggregated = False
            for col in actual_cols_to_agg_yoy:
                if col == 'WE Date':
                    agg_dict_yoy[col] = 'min' # Get earliest date within the week for hover
                # Aggregate ONLY the base components or the original metric IF it's not derived
                elif pd.api.types.is_numeric_dtype(filtered_data[col]) and \
                     (col in base_needed_for_metric or (col == metric and not is_derived_metric)):
                    agg_dict_yoy[col] = "sum"
                    numeric_aggregated = True

            if not numeric_aggregated and not is_derived_metric:
                 st.warning(f"No numeric column found for metric '{metric}' to aggregate for the YoY chart.")
                 return go.Figure()
            elif not base_components_exist and is_derived_metric:
                 # This case should have been caught earlier, but double-check
                 st.warning(f"Cannot proceed with YoY chart for derived metric '{metric}' due to missing base columns.")
                 return go.Figure()

            # Aggregate: Sum up base components (and original metric if not derived) by Year/Week
            grouped = filtered_data.groupby(["Year", "Week"], as_index=False).agg(agg_dict_yoy)
            grouped["WE Date"] = pd.to_datetime(grouped["WE Date"])

        except Exception as e:
            st.warning(f"Could not group data by week for YoY chart: {e}")
            return go.Figure()

        # --- *** ALWAYS RECALCULATE DERIVED METRICS POST-AGGREGATION *** ---
        metric_calculated_successfully = False
        if is_derived_metric:
            if metric == "CTR":
                if {"Clicks", "Impressions"}.issubset(grouped.columns):
                    grouped[metric] = grouped.apply(lambda r: (r["Clicks"] / r["Impressions"] * 100) if r.get("Impressions") else 0, axis=1).round(1) # Added rounding
                    metric_calculated_successfully = True
            elif metric == "CVR":
                 if {"Orders", "Clicks"}.issubset(grouped.columns):
                    grouped[metric] = grouped.apply(lambda r: (r["Orders"] / r["Clicks"] * 100) if r.get("Clicks") else 0, axis=1).round(1) # Added rounding
                    metric_calculated_successfully = True
            elif metric == "ACOS":
                if {"Spend", "Sales"}.issubset(grouped.columns):
                    grouped[metric] = grouped.apply(lambda r: (r["Spend"] / r["Sales"] * 100) if r.get("Sales") else np.nan, axis=1).round(1) # Added rounding
                    metric_calculated_successfully = True
            elif metric == "ROAS":
                if {"Sales", "Spend"}.issubset(grouped.columns):
                    grouped[metric] = grouped.apply(lambda r: (r["Sales"] / r["Spend"]) if r.get("Spend") else np.nan, axis=1).round(2) # Added rounding
                    metric_calculated_successfully = True
            elif metric == "CPC":
                if {"Spend", "Clicks"}.issubset(grouped.columns):
                    grouped[metric] = grouped.apply(lambda r: (r["Spend"] / r["Clicks"]) if r.get("Clicks") else np.nan, axis=1).round(2) # Added rounding
                    metric_calculated_successfully = True
            elif metric == "Ad % Sale":
                if {"Sales"}.issubset(grouped.columns) and ad_sale_check_passed: # Use flag
                    try:
                        temp_denom = weekly_total_sales_data.copy()
                        # Ensure data types match for merge
                        if 'Year' in grouped.columns and 'Year' in temp_denom.columns: temp_denom['Year'] = temp_denom['Year'].astype(grouped['Year'].dtype)
                        if 'Week' in grouped.columns and 'Week' in temp_denom.columns: temp_denom['Week'] = temp_denom['Week'].astype(grouped['Week'].dtype)
                        # Perform merge safely
                        grouped_merged = pd.merge(grouped, temp_denom[['Year', 'Week', 'Weekly_Total_Sales']], on=['Year', 'Week'], how='left')
                        grouped_merged[metric] = grouped_merged.apply(lambda r: (r['Sales'] / r['Weekly_Total_Sales'] * 100) if pd.notna(r['Weekly_Total_Sales']) and r['Weekly_Total_Sales'] > 0 else np.nan, axis=1).round(1) # Added rounding
                        grouped = grouped_merged.drop(columns=['Weekly_Total_Sales'], errors='ignore') # Drop temp col
                        metric_calculated_successfully = True
                    except Exception as e:
                        st.warning(f"Failed to merge/calculate Ad % Sale for YoY chart: {e}")
                        grouped[metric] = np.nan # Ensure column exists even if calculation fails
                else:
                     grouped[metric] = np.nan # Ensure column exists if calculation wasn't possible

            if not metric_calculated_successfully:
                # This means components were missing post-aggregation somehow, should not happen if checks above passed
                st.error(f"Internal Error: Failed to recalculate derived metric '{metric}' (YoY). Check base columns post-aggregation.")
                return go.Figure()
        # --- End Recalculation Block ---

        # Ensure the final metric column exists (either summed directly or recalculated)
        if metric not in grouped.columns:
            st.warning(f"Metric column '{metric}' not found after aggregation/calculation (YoY).")
            return go.Figure()

        # Handle Inf/-Inf values after calculation/aggregation
        grouped[metric] = grouped[metric].replace([np.inf, -np.inf], np.nan)

        # --- Plotting YoY data ---
        min_week_data, max_week_data = 53, 0
        for i, year in enumerate(years):
            year_data = grouped[grouped["Year"] == year].sort_values("Week")
            if year_data.empty or year_data[metric].isnull().all(): continue

            processed_years.append(year)
            min_week_data = min(min_week_data, year_data["Week"].min())
            max_week_data = max(max_week_data, year_data["Week"].max())
            custom_data_hover = year_data[['Week', 'WE Date']] # WE Date from 'min' aggregation

            fig.add_trace(
                go.Scatter(x=year_data["Week"], y=year_data[metric], mode="lines+markers", name=f"{year}",
                           line=dict(color=colors[i % len(colors)], width=2), marker=dict(size=6),
                           customdata=custom_data_hover, hovertemplate=base_hover_template)
            )

        # Add month annotations if data was plotted
        if processed_years:
            month_approx_weeks = { 1: 2.5, 2: 6.5, 3: 10.5, 4: 15, 5: 19.5, 6: 24, 7: 28, 8: 32.5, 9: 37, 10: 41.5, 11: 46, 12: 50.5 }
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            for month_num, week_val in month_approx_weeks.items():
                if week_val >= min_week_data - 1 and week_val <= max_week_data + 1:
                    fig.add_annotation(x=week_val, y=-0.12, xref="x", yref="paper", text=month_names[month_num-1], showarrow=False, font=dict(size=10, color="grey"))
            fig.update_layout(xaxis_range=[max(0, min_week_data - 1), min(54, max_week_data + 1)])

        fig.update_layout(xaxis_title="Week of Year", xaxis_showticklabels=True, legend_title="Year", margin=dict(b=70))

    # ========================
    # Non-YoY Plotting Logic
    # ========================
    else:
        # Define columns needed for aggregation: base components + WE Date, Year, Week
        cols_to_agg_noyoy = list(base_needed_for_metric | {metric} | {"WE Date", "Year", "Week"})
        # Ensure only columns actually present in the data are included
        actual_cols_to_agg_noyoy = list(set(cols_to_agg_noyoy) & set(filtered_data.columns))

        if not {"WE Date", "Year", "Week"}.issubset(actual_cols_to_agg_noyoy): # Critical keys
             st.warning("Missing 'WE Date', 'Year', or 'Week' for aggregation (non-YoY).")
             return go.Figure()

        try:
            agg_dict_noyoy = {}
            numeric_aggregated = False
            grouping_keys_noyoy = ["WE Date", "Year", "Week"] # Group by specific date point
            for col in actual_cols_to_agg_noyoy:
                 if col not in grouping_keys_noyoy and pd.api.types.is_numeric_dtype(filtered_data[col]) and \
                    (col in base_needed_for_metric or (col == metric and not is_derived_metric)):
                    agg_dict_noyoy[col] = "sum"
                    numeric_aggregated = True

            if not numeric_aggregated and not is_derived_metric:
                 st.warning(f"No numeric column found for metric '{metric}' to aggregate for the time chart (non-YoY).")
                 return go.Figure()
            elif not base_components_exist and is_derived_metric:
                 st.warning(f"Cannot proceed with time chart for derived metric '{metric}' due to missing base columns (non-YoY).")
                 return go.Figure()

            # Aggregate: Sum up base components (and original metric if not derived) by Date/Year/Week
            grouped = filtered_data.groupby(grouping_keys_noyoy, as_index=False).agg(agg_dict_noyoy)
            grouped["WE Date"] = pd.to_datetime(grouped["WE Date"]) # Ensure datetime type

        except Exception as e:
            st.warning(f"Could not group data for time chart (non-YoY): {e}")
            return go.Figure()

        # --- *** ALWAYS RECALCULATE DERIVED METRICS POST-AGGREGATION *** ---
        metric_calculated_successfully = False
        if is_derived_metric:
            if metric == "CTR":
                if {"Clicks", "Impressions"}.issubset(grouped.columns):
                    grouped[metric] = grouped.apply(lambda r: (r["Clicks"] / r["Impressions"] * 100) if r.get("Impressions") else 0, axis=1).round(1) # Added rounding
                    metric_calculated_successfully = True
            elif metric == "CVR":
                 if {"Orders", "Clicks"}.issubset(grouped.columns):
                    grouped[metric] = grouped.apply(lambda r: (r["Orders"] / r["Clicks"] * 100) if r.get("Clicks") else 0, axis=1).round(1) # Added rounding
                    metric_calculated_successfully = True
            elif metric == "ACOS":
                if {"Spend", "Sales"}.issubset(grouped.columns):
                    grouped[metric] = grouped.apply(lambda r: (r["Spend"] / r["Sales"] * 100) if r.get("Sales") else np.nan, axis=1).round(1) # Added rounding
                    metric_calculated_successfully = True
            elif metric == "ROAS":
                if {"Sales", "Spend"}.issubset(grouped.columns):
                    grouped[metric] = grouped.apply(lambda r: (r["Sales"] / r["Spend"]) if r.get("Spend") else np.nan, axis=1).round(2) # Added rounding
                    metric_calculated_successfully = True
            elif metric == "CPC":
                if {"Spend", "Clicks"}.issubset(grouped.columns):
                    grouped[metric] = grouped.apply(lambda r: (r["Spend"] / r["Clicks"]) if r.get("Clicks") else np.nan, axis=1).round(2) # Added rounding
                    metric_calculated_successfully = True
            elif metric == "Ad % Sale":
                if {"Sales"}.issubset(grouped.columns) and ad_sale_check_passed: # Use flag
                    try:
                        temp_denom = weekly_total_sales_data.copy()
                        # Ensure data types match for merge
                        if 'Year' in grouped.columns and 'Year' in temp_denom.columns: temp_denom['Year'] = temp_denom['Year'].astype(grouped['Year'].dtype)
                        if 'Week' in grouped.columns and 'Week' in temp_denom.columns: temp_denom['Week'] = temp_denom['Week'].astype(grouped['Week'].dtype)
                        # Perform merge safely
                        grouped_merged = pd.merge(grouped, temp_denom[['Year', 'Week', 'Weekly_Total_Sales']], on=['Year', 'Week'], how='left')
                        grouped_merged[metric] = grouped_merged.apply(lambda r: (r['Sales'] / r['Weekly_Total_Sales'] * 100) if pd.notna(r['Weekly_Total_Sales']) and r['Weekly_Total_Sales'] > 0 else np.nan, axis=1).round(1) # Added rounding
                        grouped = grouped_merged.drop(columns=['Weekly_Total_Sales'], errors='ignore') # Drop temp col
                        metric_calculated_successfully = True
                    except Exception as e:
                        st.warning(f"Failed to merge/calculate Ad % Sale for non-YoY chart: {e}")
                        grouped[metric] = np.nan # Ensure column exists even if calculation fails
                else:
                     grouped[metric] = np.nan # Ensure column exists if calculation wasn't possible

            if not metric_calculated_successfully:
                st.error(f"Internal Error: Failed to recalculate derived metric '{metric}' (non-YoY). Check base columns post-aggregation.")
                return go.Figure()
        # --- End Recalculation Block ---

        # Ensure the final metric column exists
        if metric not in grouped.columns:
            st.warning(f"Metric column '{metric}' not found after aggregation/calculation (non-YoY).")
            return go.Figure()

        # Handle Inf/-Inf values
        grouped[metric] = grouped[metric].replace([np.inf, -np.inf], np.nan)

        # --- Plotting Non-YoY data ---
        if grouped[metric].isnull().all():
            st.info(f"No valid data points for metric '{metric}' over time (non-YoY).")
            return go.Figure() # Return empty figure if all values are NaN

        grouped = grouped.sort_values("WE Date")
        custom_data_hover_noyoy = grouped[['Week', 'WE Date']]
        fig.add_trace(
            go.Scatter(x=grouped["WE Date"], y=grouped[metric], mode="lines+markers", name=metric,
                       line=dict(color="#1f77b4", width=2), marker=dict(size=6),
                       customdata=custom_data_hover_noyoy, hovertemplate=base_hover_template)
        )
        fig.update_layout(xaxis_title="Date", showlegend=False)

    # --- Final Chart Layout ---
    portfolio_title = f" for {portfolio}" if portfolio != "All Portfolios" else " for All Portfolios"
    years_in_plot = processed_years if (show_yoy and len(years) > 1 and processed_years) else years # Get years actually plotted
    final_chart_title = f"{metric} "

    if show_yoy and len(years_in_plot) > 1:
        final_chart_title += f"Weekly Comparison {portfolio_title} ({product_type})"
        final_xaxis_title = "Week of Year"
    else:
        final_chart_title += f"Over Time (Weekly) {portfolio_title} ({product_type})"
        final_xaxis_title = "Week Ending Date"

    final_margin = dict(t=80, b=70, l=70, r=30)
    fig.update_layout(
        title=final_chart_title, xaxis_title=final_xaxis_title, yaxis_title=metric,
        hovermode="x unified", template="plotly_white", yaxis=dict(rangemode="tozero"), margin=final_margin
    )

    # Apply Y-axis formatting based on the metric
    if metric in ["Spend", "Sales", "CPC"]: fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",.2f")
    elif metric in ["CTR", "CVR", "ACOS", "Ad % Sale"]: fig.update_layout(yaxis_ticksuffix="%", yaxis_tickformat=".1f")
    elif metric == "ROAS": fig.update_layout(yaxis_tickformat=".2f")
    else: fig.update_layout(yaxis_tickformat=",.0f") # Impressions, Clicks, Orders, Units

    return fig


def style_dataframe(df, format_dict, highlight_cols=None, color_map_func=None, text_align='right', na_rep='N/A'):
    """Generic styling function for dataframes with alignment and NaN handling."""
    if df is None or df.empty: return None
    df_copy = df.copy().replace([np.inf, -np.inf], np.nan)
    valid_format_dict = {k: v for k, v in format_dict.items() if k in df_copy.columns}

    try:
        styled = df_copy.style.format(valid_format_dict, na_rep=na_rep)
    except Exception as e:
        st.error(f"Error applying format: {e}. Formatting dictionary: {valid_format_dict}")
        return df_copy.style # Basic styler on error

    if highlight_cols and color_map_func:
        if len(highlight_cols) == len(color_map_func):
            for col, func in zip(highlight_cols, color_map_func):
                if col in df_copy.columns:
                    try: styled = styled.applymap(func, subset=[col])
                    except Exception as e: st.warning(f"Styling failed for column '{col}': {e}")
        else:
             st.error("Mismatch between highlight_cols and color_map_func in style_dataframe.")

    cols_to_align = df_copy.columns
    if len(cols_to_align) > 0:
        try:
            first_col_idx = df_copy.columns.get_loc(cols_to_align[0])
            styles = [
                {'selector': 'th', 'props': [('text-align', text_align), ('white-space', 'nowrap')]},
                {'selector': 'td', 'props': [('text-align', text_align)]},
                {'selector': f'th.col_heading.level0.col{first_col_idx}', 'props': [('text-align', 'left')]},
                {'selector': f'td.col{first_col_idx}', 'props': [('text-align', 'left')]}
            ]
            styled = styled.set_table_styles(styles, overwrite=False)
        except Exception as e:
            st.warning(f"Could not apply specific alignment: {e}. Using general alignment.")
            styled = styled.set_properties(**{'text-align': text_align})
    else:
        styled = styled.set_properties(**{'text-align': text_align})

    return styled

def style_total_summary(df):
    """Styles the single-row total summary table"""
    # THIS FUNCTION IS NO LONGER DIRECTLY USED FOR SP/SB/SD TABS but kept
    format_dict = {
        "Impressions": "{:,.0f}", "Clicks": "{:,.0f}", "Orders": "{:,.0f}",
        "Spend": "${:,.2f}", "Sales": "${:,.2f}",
        "CTR": "{:.1f}%", "CVR": "{:.1f}%",
        "ACOS": lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A",
        "ROAS": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
    }
    def color_acos(val):
        if isinstance(val, str) or pd.isna(val): return "color: grey"
        try: val_f = float(val); return "color: green" if val_f <= 15 else ("color: orange" if val_f <= 20 else "color: red")
        except (ValueError, TypeError): return "color: grey"
    def color_roas(val):
        if isinstance(val, str) or pd.isna(val): return "color: grey"
        try: val_f = float(val); return "color: green" if val_f > 3 else "color: red"
        except (ValueError, TypeError): return "color: grey"

    styled = style_dataframe(df, format_dict, highlight_cols=["ACOS", "ROAS"], color_map_func=[color_acos, color_roas], na_rep="N/A")
    if styled: return styled.set_properties(**{"font-weight": "bold"})
    return None

def style_metrics_table(df):
    """Styles the multi-row performance metrics table (by Portfolio)"""
    # THIS FUNCTION IS NO LONGER DIRECTLY USED FOR SP/SB/SD TABS but kept
    format_dict = {
        "Impressions": "{:,.0f}", "Clicks": "{:,.0f}", "Orders": "{:,.0f}",
        "Spend": "${:,.2f}", "Sales": "${:,.2f}",
        "CTR": "{:.1f}%", "CVR": "{:.1f}%",
        "ACOS": lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A",
        "ROAS": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
    }
    if "Units" in df.columns: format_dict["Units"] = "{:,.0f}"
    if "CPC" in df.columns: format_dict["CPC"] = "${:,.2f}"
    id_cols = ["Portfolio", "Match Type", "RTW/Prospecting", "Campaign"]
    id_col_name = next((col for col in df.columns if col in id_cols), None)
    if id_col_name: format_dict[id_col_name] = "{}"

    def color_acos(val):
        if isinstance(val, str) or pd.isna(val): return "color: grey"
        try: val_f = float(val); return "color: green" if val_f <= 15 else ("color: orange" if val_f <= 20 else "color: red")
        except (ValueError, TypeError): return "color: grey"
    def color_roas(val):
        if isinstance(val, str) or pd.isna(val): return "color: grey"
        try: val_f = float(val); return "color: green" if val_f > 3 else "color: red"
        except (ValueError, TypeError): return "color: grey"

    styled = style_dataframe(df, format_dict, highlight_cols=["ACOS", "ROAS"], color_map_func=[color_acos, color_roas], na_rep="N/A")
    return styled

@st.cache_data
def generate_insights(total_metrics_series, campaign_type):
    """Generates text insights based on a summary row (Pandas Series)"""
    # --- Define Your Specific Thresholds Here ---
    acos_target = 15.0   # Acceptable ACOS is <= 15%
    roas_target = 5.0    # Good ROAS is >= 5
    ctr_target = 0.35  # Good CTR is >= 0.35%
    cvr_target = 10.0   # Good CVR is >= 10%
    # --- End of Threshold Definitions ---

    insights = []
    # Get the metric values from the input series
    acos = total_metrics_series.get("ACOS", np.nan)
    roas = total_metrics_series.get("ROAS", np.nan)
    ctr = total_metrics_series.get("CTR", np.nan)
    cvr = total_metrics_series.get("CVR", np.nan)
    sales = total_metrics_series.get("Sales", 0)
    spend = total_metrics_series.get("Spend", 0)

    if pd.isna(sales): sales = 0
    if pd.isna(spend): spend = 0

    # --- Insight Logic using the defined thresholds ---
    if spend > 0 and sales == 0:
        insights.append("⚠️ **Immediate Attention:** Spend occurred with zero attributed sales. Review targeting, keywords, and product pages urgently.")
        if pd.notna(ctr): insights.append(f"ℹ️ Click-through rate was {ctr:.2f}%.") # Use 2 decimal places for CTR display if needed
    else:
        # ACOS Insight
        if pd.isna(acos):
            if spend == 0 and sales == 0: insights.append("ℹ️ No spend or sales recorded for ACOS calculation.")
            elif sales == 0 and spend > 0: insights.append("ℹ️ ACOS is not applicable (No Sales from Spend).")
            elif spend == 0 and sales > 0: insights.append(f"✅ **ACOS:** ACOS is effectively 0% (Sales with no spend), which is below the target (≤{acos_target}%).") # Clarified message for 0% ACOS
            elif spend == 0: insights.append("ℹ️ ACOS is not applicable (No Spend).")
        elif acos > acos_target: # Compare with acos_target
            insights.append(f"📈 **High ACOS:** Overall ACOS ({acos:.1f}%) is above the target ({acos_target}%). Consider optimizing bids, keywords, or targeting.")
        else: # ACOS is <= acos_target
            insights.append(f"✅ **ACOS:** Overall ACOS ({acos:.1f}%) is within the acceptable range (≤{acos_target}%).")

        # ROAS Insight
        if pd.isna(roas):
            if spend == 0 and sales == 0: insights.append("ℹ️ No spend or sales recorded for ROAS calculation.")
            elif spend == 0 and sales > 0 : insights.append("✅ **ROAS:** ROAS is effectively infinite (Sales with No Spend).")
            elif spend > 0 and sales == 0: insights.append("ℹ️ ROAS is 0 (No Sales from Spend).")
        elif roas < roas_target: # Compare with roas_target
            insights.append(f"📉 **Low ROAS:** Overall ROAS ({roas:.2f}) is below the target of {roas_target}. Review performance and strategy.")
        else: # ROAS is >= roas_target
            insights.append(f"✅ **ROAS:** Overall ROAS ({roas:.2f}) is good (≥{roas_target}).")

        # CTR Insight
        if pd.isna(ctr):
            insights.append("ℹ️ Click-Through Rate (CTR) could not be calculated (likely no impressions).")
        elif ctr < ctr_target: # Compare with ctr_target
            insights.append(f"📉 **Low CTR:** Click-through rate ({ctr:.2f}%) is low (<{ctr_target}%). Review ad creative, relevance, or placement.") # Display CTR with 2 decimals for comparison
        else: # CTR is >= ctr_target
            insights.append(f"✅ **CTR:** Click-through rate ({ctr:.2f}%) is satisfactory (≥{ctr_target}%).")

        # CVR Insight
        if pd.isna(cvr):
            insights.append("ℹ️ Conversion Rate (CVR) could not be calculated (likely no clicks).")
        elif cvr < cvr_target: # Compare with cvr_target
            insights.append(f"📉 **Low CVR:** Conversion rate ({cvr:.1f}%) is below the target ({cvr_target}%). Review product listing pages and targeting.")
        else: # CVR is >= cvr_target
            insights.append(f"✅ **CVR:** Conversion rate ({cvr:.1f}%) is good (≥{cvr_target}%).")

    return insights

@st.cache_data
def create_yoy_grouped_table(df_filtered_period, group_by_col, selected_metrics, years_to_process, display_col_name=None):
    """Creates a merged YoY comparison table grouped by a specific column."""
    if df_filtered_period is None or df_filtered_period.empty: return pd.DataFrame()
    if group_by_col not in df_filtered_period.columns: st.warning(f"Grouping column '{group_by_col}' not found."); return pd.DataFrame()
    if not selected_metrics: st.warning("No metrics selected."); return pd.DataFrame()

    date_col = "WE Date"
    ad_sale_possible = ("Ad % Sale" in selected_metrics and {"Sales", "Total Sales", "WE Date"}.issubset(df_filtered_period.columns))

    if "Ad % Sale" in selected_metrics and not ad_sale_possible:
        st.warning("Cannot calculate 'Ad % Sale'. Requires 'Sales', 'Total Sales', and 'WE Date' columns.")
        selected_metrics = [m for m in selected_metrics if m != "Ad % Sale"]
        if not selected_metrics: return pd.DataFrame()

    df_filtered_period[group_by_col] = df_filtered_period[group_by_col].fillna(f"Unknown {group_by_col}")
    yearly_tables = []

    for yr in years_to_process:
        df_year = df_filtered_period[df_filtered_period["Year"] == yr].copy()
        if df_year.empty: continue

        base_metrics_to_sum_needed = set()
        for metric in selected_metrics:
            if metric in ["Impressions", "Clicks", "Spend", "Sales", "Orders", "Units"]: base_metrics_to_sum_needed.add(metric)
            elif metric == "CTR": base_metrics_to_sum_needed.update(["Clicks", "Impressions"])
            elif metric == "CVR": base_metrics_to_sum_needed.update(["Orders", "Clicks"])
            elif metric == "CPC": base_metrics_to_sum_needed.update(["Spend", "Clicks"])
            elif metric == "ACOS": base_metrics_to_sum_needed.update(["Spend", "Sales"])
            elif metric == "ROAS": base_metrics_to_sum_needed.update(["Sales", "Spend"])
            elif metric == "Ad % Sale": base_metrics_to_sum_needed.add("Sales")

        actual_base_present = {m for m in base_metrics_to_sum_needed if m in df_year.columns}
        if not actual_base_present and not any(m in df_year.columns for m in selected_metrics if m not in base_metrics_to_sum_needed): continue

        calculable_metrics_for_year = []
        for metric in selected_metrics:
             can_calc_yr = False
             if metric in df_year.columns: can_calc_yr = True
             elif metric == "CTR" and {"Clicks", "Impressions"}.issubset(actual_base_present): can_calc_yr = True
             elif metric == "CVR" and {"Orders", "Clicks"}.issubset(actual_base_present): can_calc_yr = True
             elif metric == "CPC" and {"Spend", "Clicks"}.issubset(actual_base_present): can_calc_yr = True
             elif metric == "ACOS" and {"Spend", "Sales"}.issubset(actual_base_present): can_calc_yr = True
             elif metric == "ROAS" and {"Sales", "Spend"}.issubset(actual_base_present): can_calc_yr = True
             elif metric == "Ad % Sale" and ad_sale_possible and "Sales" in actual_base_present: can_calc_yr = True
             if can_calc_yr: calculable_metrics_for_year.append(metric)

        if not calculable_metrics_for_year: continue

        total_sales_for_period = 0
        if "Ad % Sale" in calculable_metrics_for_year:
            try:
                df_year_valid_dates_total = df_year.dropna(subset=[date_col, 'Total Sales']) # Ensure Total Sales is not NaN for calc
                df_year_valid_dates_total['Total Sales'] = pd.to_numeric(df_year_valid_dates_total['Total Sales'], errors='coerce')
                df_year_valid_dates_total.dropna(subset=['Total Sales'], inplace=True)
                if not df_year_valid_dates_total.empty:
                    unique_subset = [date_col]
                    if "Marketplace" in df_year_valid_dates_total.columns: unique_subset.append("Marketplace")
                    unique_weekly_totals = df_year_valid_dates_total.drop_duplicates(subset=unique_subset)
                    total_sales_for_period = unique_weekly_totals['Total Sales'].sum()
            except Exception as e: st.warning(f"Could not calculate total sales denominator for year {yr}: {e}")

        agg_dict_final = {m: 'sum' for m in actual_base_present}
        if not agg_dict_final: df_pivot = pd.DataFrame({group_by_col: df_year[group_by_col].unique()})
        else:
            try: df_pivot = df_year.groupby(group_by_col).agg(agg_dict_final).reset_index()
            except Exception as e: st.warning(f"Error aggregating data for {group_by_col} in year {yr}: {e}"); continue

        if "CTR" in calculable_metrics_for_year: df_pivot["CTR"] = df_pivot.apply(lambda r: (r.get("Clicks",0) / r.get("Impressions",0) * 100) if r.get("Impressions") else 0, axis=1)
        if "CVR" in calculable_metrics_for_year: df_pivot["CVR"] = df_pivot.apply(lambda r: (r.get("Orders",0) / r.get("Clicks",0) * 100) if r.get("Clicks") else 0, axis=1)
        if "CPC" in calculable_metrics_for_year: df_pivot["CPC"] = df_pivot.apply(lambda r: (r.get("Spend",0) / r.get("Clicks",0)) if r.get("Clicks") else np.nan, axis=1)
        if "ACOS" in calculable_metrics_for_year: df_pivot["ACOS"] = df_pivot.apply(lambda r: (r.get("Spend",0) / r.get("Sales",0) * 100) if r.get("Sales") else np.nan, axis=1)
        if "ROAS" in calculable_metrics_for_year: df_pivot["ROAS"] = df_pivot.apply(lambda r: (r.get("Sales",0) / r.get("Spend",0)) if r.get("Spend") else np.nan, axis=1)
        if "Ad % Sale" in calculable_metrics_for_year: df_pivot["Ad % Sale"] = df_pivot.apply( lambda r: (r.get("Sales", 0) / total_sales_for_period * 100) if total_sales_for_period > 0 else np.nan, axis=1 )

        df_pivot = df_pivot.replace([np.inf, -np.inf], np.nan)
        final_cols_for_year = [group_by_col] + [m for m in calculable_metrics_for_year if m in df_pivot.columns]
        df_pivot_final = df_pivot[final_cols_for_year].rename(columns={m: f"{m} {yr}" for m in calculable_metrics_for_year})
        yearly_tables.append(df_pivot_final)

    if not yearly_tables: return pd.DataFrame()
    try: merged_table = reduce(lambda left, right: pd.merge(left, right, on=group_by_col, how="outer"), yearly_tables)
    except Exception as e: st.error(f"Error merging yearly {group_by_col} tables: {e}"); return pd.DataFrame()

    base_sum_metrics = ["Impressions", "Clicks", "Spend", "Sales", "Orders", "Units"]
    cols_to_fill_zero = [f"{m} {yr}" for yr in years_to_process for m in base_sum_metrics if f"{m} {yr}" in merged_table.columns]
    if cols_to_fill_zero: merged_table[cols_to_fill_zero] = merged_table[cols_to_fill_zero].fillna(0)

    ordered_cols = [group_by_col]
    actual_years_in_data = sorted(list({int(re.search(r'(\d{4})$', col).group(1)) for col in merged_table.columns if re.search(r'(\d{4})$', col)}))

    if len(actual_years_in_data) >= 2:
        current_year_sel, prev_year_sel = actual_years_in_data[-1], actual_years_in_data[-2]
        percentage_metrics = {"CTR", "CVR", "ACOS", "Ad % Sale"}
        for metric in selected_metrics:
            col_current, col_prev = f"{metric} {current_year_sel}", f"{metric} {prev_year_sel}"
            change_col_name = f"{metric} % Change"
            if col_prev in merged_table.columns: ordered_cols.append(col_prev)
            if col_current in merged_table.columns: ordered_cols.append(col_current)
            if col_current in merged_table.columns and col_prev in merged_table.columns:
                val_curr = merged_table[col_current]
                val_prev = merged_table[col_prev]
                if metric in percentage_metrics: merged_table[change_col_name] = val_curr - val_prev
                else:
                     merged_table[change_col_name] = np.where(
                         (val_prev != 0) & pd.notna(val_prev) & pd.notna(val_curr),
                         (val_curr - val_prev) / val_prev.abs() * 100, np.nan )
                merged_table[change_col_name] = merged_table[change_col_name].replace([np.inf, -np.inf], np.nan)
                ordered_cols.append(change_col_name)
    elif actual_years_in_data:
        yr_single = actual_years_in_data[0]
        ordered_cols.extend([f"{m} {yr_single}" for m in selected_metrics if f"{m} {yr_single}" in merged_table.columns])

    ordered_cols = [col for col in ordered_cols if col in merged_table.columns]
    merged_table_display = merged_table[ordered_cols].copy()
    final_display_col = display_col_name or group_by_col
    if group_by_col in merged_table_display.columns:
        merged_table_display = merged_table_display.rename(columns={group_by_col: final_display_col})

    if len(actual_years_in_data) >= 1:
        last_yr = actual_years_in_data[-1]
        sort_col = next((f"{m} {last_yr}" for m in selected_metrics if f"{m} {last_yr}" in merged_table_display.columns), None)
        if sort_col:
            try:
                merged_table_display[sort_col] = pd.to_numeric(merged_table_display[sort_col], errors='coerce')
                merged_table_display = merged_table_display.sort_values(sort_col, ascending=False, na_position='last')
            except Exception as e: st.warning(f"Could not sort table by column '{sort_col}': {e}")

    return merged_table_display



def style_yoy_comparison_table(df):
    """Styles the YoY comparison table with formats and % change coloring using applymap.
       Inverts colors for 'ACOS % Change'.
    """
    if df is None or df.empty: return None
    df_copy = df.copy().replace([np.inf, -np.inf], np.nan)

    format_dict = {}
    highlight_change_cols = []
    percentage_metrics = {"CTR", "CVR", "ACOS", "Ad % Sale"} # Metrics where change is absolute difference

    # --- Determine Formats and Identify Change Columns ---
    for col in df_copy.columns:
        base_metric_match = re.match(r"([a-zA-Z\s%]+)", col)
        base_metric = base_metric_match.group(1).strip() if base_metric_match else ""
        is_change_col = "% Change" in col
        is_metric_col = not is_change_col and any(char.isdigit() for char in col) # Basic check if year is in col name

        if is_change_col:
            base_metric_for_change = col.replace(" % Change", "").strip()
            # Format absolute change for percentage metrics, percentage change for others
            if base_metric_for_change in percentage_metrics:
                format_dict[col] = lambda x: f"{x:+.1f}%" if pd.notna(x) else 'N/A' # Use 'pt' for percentage points difference
            else:
                format_dict[col] = lambda x: f"{x:+.0f}%" if pd.notna(x) else 'N/A' # Standard percentage change
            highlight_change_cols.append(col)
        elif is_metric_col:
            # Apply standard metric formatting
            if base_metric in ["Impressions", "Clicks", "Orders", "Units"]: format_dict[col] = "{:,.0f}"
            elif base_metric in ["Spend", "Sales", "CPC"]: format_dict[col] = "${:,.2f}"
            elif base_metric in ["ACOS", "CTR", "CVR", "Ad % Sale"]: format_dict[col] = '{:.1f}%'
            elif base_metric == "ROAS": format_dict[col] = '{:.2f}'
            # Add other specific formats if needed
        elif df_copy[col].dtype == 'object': # Format the first (grouping) column as string
             format_dict[col] = "{}" # Ensure grouping column isn't formatted as number

    # --- Define Coloring Functions ---
    def color_pos_neg_standard(val):
        """Standard coloring: positive is green, negative is red."""
        if isinstance(val, str) and val == "N/A": return 'color: grey'
        numeric_val = pd.to_numeric(val, errors='coerce')
        if pd.isna(numeric_val): return 'color: grey'
        elif numeric_val > 0: return 'color: green'
        elif numeric_val < 0: return 'color: red'
        else: return 'color: inherit' # Or black/grey for zero change

    def color_pos_neg_inverted(val):
        """Inverted coloring (for ACOS): positive is red, negative is green."""
        if isinstance(val, str) and val == "N/A": return 'color: grey'
        numeric_val = pd.to_numeric(val, errors='coerce')
        if pd.isna(numeric_val): return 'color: grey'
        elif numeric_val > 0: return 'color: red'   # Positive change (ACOS increase) is red (bad)
        elif numeric_val < 0: return 'color: green' # Negative change (ACOS decrease) is green (good)
        else: return 'color: inherit' # Or black/grey for zero change

    # --- Apply Formatting ---
    try:
        styled_table = df_copy.style.format(format_dict, na_rep="N/A")
    except Exception as e:
        st.error(f"Error applying format to YOY table: {e}")
        return df_copy.style # Return basic styler on error

    # --- Apply Conditional Coloring ---
    for change_col in highlight_change_cols:
        if change_col in df_copy.columns:
            try:
                # Choose the appropriate coloring function based on the column name
                if change_col == "ACOS % Change":
                    color_func_to_apply = color_pos_neg_inverted
                else:
                    color_func_to_apply = color_pos_neg_standard

                # Apply the chosen function element-wise using applymap
                styled_table = styled_table.applymap(color_func_to_apply, subset=[change_col])

            except Exception as e:
                st.warning(f"Could not apply color style to YOY column '{change_col}': {e}")

    # --- Apply Alignment ---
    text_align='right'
    try:
        # Align first column left, others right
        first_col_name = df_copy.columns[0]
        # Using CSS selectors compatible with newer pandas/streamlit versions
        styles = [
             {'selector': 'th', 'props': [('text-align', text_align), ('white-space', 'nowrap')]},
             {'selector': 'td', 'props': [('text-align', text_align)]},
             # Target first column header and cells specifically by name/position if possible
             # This might need adjustment depending on exact HTML structure render
             {'selector': f'th.col_heading.level0.col0', 'props': [('text-align', 'left')]}, # Target first header
             {'selector': f'td:first-child', 'props': [('text-align', 'left')]}, # Target first data cell in each row
             {'selector': f'th[data-col-name="{first_col_name}"]', 'props': [('text-align', 'left !important')]}, # More specific targeting if available
             {'selector': f'td[data-col-name="{first_col_name}"]', 'props': [('text-align', 'left !important')]}
             ]
        # It's often simpler and more reliable to set general alignment and then override the first column
        styled_table = styled_table.set_properties(**{'text-align': text_align})
        styled_table = styled_table.set_properties(subset=[first_col_name], **{'text-align': 'left'})

    except Exception as e:
        st.warning(f"Could not apply specific alignment to YOY table: {e}. Using general alignment.")
        styled_table = styled_table.set_properties(**{'text-align': text_align}) # Fallback

    return styled_table

@st.cache_data
def calculate_yoy_summary_row(df, selected_metrics, years_to_process, id_col_name, id_col_value):
    """Calculates a single summary row with YoY comparison based on yearly totals."""
    if df is None or df.empty or not years_to_process: return pd.DataFrame()

    date_col = "WE Date"
    ad_sale_possible = ("Ad % Sale" in selected_metrics and {"Sales", "Total Sales", "WE Date"}.issubset(df.columns))
    if "Ad % Sale" in selected_metrics and not ad_sale_possible:
        selected_metrics = [m for m in selected_metrics if m != "Ad % Sale"]

    summary_row_data = {id_col_name: id_col_value}
    yearly_totals = {yr: {} for yr in years_to_process}
    yearly_total_sales_denom = {yr: 0 for yr in years_to_process}

    base_metrics_needed = set()
    for m in selected_metrics:
        if m in ["Impressions", "Clicks", "Spend", "Sales", "Orders", "Units"]: base_metrics_needed.add(m)
        elif m == "CTR": base_metrics_needed.update(["Clicks", "Impressions"])
        elif m == "CVR": base_metrics_needed.update(["Orders", "Clicks"])
        elif m == "CPC": base_metrics_needed.update(["Spend", "Clicks"])
        elif m == "ACOS": base_metrics_needed.update(["Spend", "Sales"])
        elif m == "ROAS": base_metrics_needed.update(["Sales", "Spend"])
        elif m == "Ad % Sale": base_metrics_needed.add("Sales")

    for yr in years_to_process:
        df_year = df[df["Year"] == yr]
        if df_year.empty: continue
        for base_m in base_metrics_needed:
            if base_m in df_year.columns: yearly_totals[yr][base_m] = pd.to_numeric(df_year[base_m], errors='coerce').fillna(0).sum()
            else: yearly_totals[yr][base_m] = 0

        if ad_sale_possible and "Ad % Sale" in selected_metrics:
            try:
                 df_year_valid_dates = df_year.dropna(subset=[date_col, 'Total Sales'])
                 df_year_valid_dates['Total Sales'] = pd.to_numeric(df_year_valid_dates['Total Sales'], errors='coerce')
                 df_year_valid_dates.dropna(subset=['Total Sales'], inplace=True)
                 if not df_year_valid_dates.empty:
                      unique_subset = [date_col]
                      if "Marketplace" in df_year_valid_dates.columns: unique_subset.append("Marketplace")
                      unique_totals = df_year_valid_dates.drop_duplicates(subset=unique_subset)
                      yearly_total_sales_denom[yr] = unique_totals['Total Sales'].sum()
            except Exception as e: yearly_total_sales_denom[yr] = 0

    for metric in selected_metrics:
        for yr in years_to_process:
            totals_yr = yearly_totals.get(yr, {})
            calc_val = np.nan
            try:
                 if metric == "CTR": calc_val = (totals_yr.get("Clicks", 0) / totals_yr.get("Impressions", 0) * 100) if totals_yr.get("Impressions", 0) > 0 else 0
                 elif metric == "CVR": calc_val = (totals_yr.get("Orders", 0) / totals_yr.get("Clicks", 0) * 100) if totals_yr.get("Clicks", 0) > 0 else 0
                 elif metric == "CPC": calc_val = (totals_yr.get("Spend", 0) / totals_yr.get("Clicks", 0)) if totals_yr.get("Clicks", 0) > 0 else np.nan
                 elif metric == "ACOS": calc_val = (totals_yr.get("Spend", 0) / totals_yr.get("Sales", 0) * 100) if totals_yr.get("Sales", 0) > 0 else np.nan
                 elif metric == "ROAS": calc_val = (totals_yr.get("Sales", 0) / totals_yr.get("Spend", 0)) if totals_yr.get("Spend", 0) > 0 else np.nan
                 elif metric == "Ad % Sale":
                      denom_yr = yearly_total_sales_denom.get(yr, 0)
                      calc_val = (totals_yr.get("Sales", 0) / denom_yr * 100) if denom_yr > 0 else np.nan
                 elif metric in totals_yr: calc_val = totals_yr.get(metric)
                 if isinstance(calc_val, (int, float)): calc_val = np.nan if calc_val in [np.inf, -np.inf] else calc_val
            except Exception as e: calc_val = np.nan
            if yr in yearly_totals: yearly_totals[yr][metric] = calc_val
            summary_row_data[f"{metric} {yr}"] = calc_val

    actual_years_in_row = sorted([yr for yr in years_to_process if yr in yearly_totals and yearly_totals[yr]])
    if len(actual_years_in_row) >= 2:
        curr_yr, prev_yr = actual_years_in_row[-1], actual_years_in_row[-2]
        percentage_metrics = {"CTR", "CVR", "ACOS", "Ad % Sale"}
        for metric in selected_metrics:
            val_curr = yearly_totals.get(curr_yr, {}).get(metric, np.nan)
            val_prev = yearly_totals.get(prev_yr, {}).get(metric, np.nan)
            pct_change = np.nan
            if pd.notna(val_curr) and pd.notna(val_prev):
                 if metric in percentage_metrics: pct_change = val_curr - val_prev
                 else:
                      if val_prev != 0: pct_change = ((val_curr - val_prev) / abs(val_prev)) * 100
                      elif val_curr == 0: pct_change = 0.0
            summary_row_data[f"{metric} % Change"] = np.nan if pct_change in [np.inf, -np.inf] else pct_change

    summary_df = pd.DataFrame([summary_row_data])
    ordered_summary_cols = [id_col_name]
    if len(actual_years_in_row) >= 2:
        curr_yr_o, prev_yr_o = actual_years_in_row[-1], actual_years_in_row[-2]
        for metric in selected_metrics:
            if f"{metric} {prev_yr_o}" in summary_df.columns: ordered_summary_cols.append(f"{metric} {prev_yr_o}")
            if f"{metric} {curr_yr_o}" in summary_df.columns: ordered_summary_cols.append(f"{metric} {curr_yr_o}")
            if f"{metric} % Change" in summary_df.columns: ordered_summary_cols.append(f"{metric} % Change")
    elif len(actual_years_in_row) == 1:
        yr_o = actual_years_in_row[0]
        ordered_summary_cols.extend([f"{metric} {yr_o}" for metric in selected_metrics if f"{metric} {yr_o}" in summary_df.columns])

    final_summary_cols = [col for col in ordered_summary_cols if col in summary_df.columns]
    return summary_df[final_summary_cols]


# =============================================================================
# --- Title and Logo ---
# =============================================================================
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Advertising Dashboard 📊")
with col2:
    try: st.image("logo.png", width=250) # Adjusted width slightly
    except Exception as e: st.warning(f"Could not load logo.png: {e}")


# =============================================================================
# --- Sidebar File Uploader & Initial Data Handling (REVISED for Visible Selector) ---
# =============================================================================
st.sidebar.header("Advertising Data")
advertising_file = st.sidebar.file_uploader("Upload Advertising Data (CSV)", type=["csv"], key="adv_file")

# --- Manage Session State for File Changes ---
if "current_file_name" not in st.session_state: st.session_state.current_file_name = None

# Detect new file upload or file removal
new_file_uploaded = (advertising_file is not None and advertising_file.name != st.session_state.current_file_name)
file_removed = (advertising_file is None and st.session_state.current_file_name is not None)

if new_file_uploaded or file_removed:
    # Store the new file name (or None if removed)
    st.session_state.current_file_name = advertising_file.name if advertising_file else None
    # Define keys to preserve during reset
    preserve_keys = ['current_file_name', 'adv_file']
    # Clear all other keys EXCEPT preserved ones
    keys_to_delete = [k for k in st.session_state.keys() if k not in preserve_keys]
    for key in keys_to_delete:
        if key in st.session_state: # Check if key exists before deleting
           del st.session_state[key]
    # Clear marketplace selector value on new file/removal to reset it
    if 'marketplace_selector_value' in st.session_state:
        del st.session_state['marketplace_selector_value']


# --- Process Uploaded File ---
if advertising_file is not None:
    # --- Marketplace Selector (Moved outside needs_processing) ---
    marketplace_options = ["All Marketplaces"]
    default_marketplace = "All Marketplaces"
    default_mp_index = 0

    # Load raw data if it's not already loaded for the *current* file
    # This ensures options are available even if not reprocessing
    if "ad_data_raw" not in st.session_state or st.session_state.get("processed_file_name") != advertising_file.name:
        try:
            st.session_state["ad_data_raw"] = pd.read_csv(advertising_file)
            st.session_state["processed_file_name"] = advertising_file.name # Track which file raw data belongs to
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            # Clean up potentially partially loaded state
            keys_to_del = [k for k in st.session_state.keys() if k not in ['current_file_name', 'adv_file']]
            for key in keys_to_del:
                if key in st.session_state: del st.session_state[key]
            st.stop() # Stop execution if file read fails

    # Populate marketplace options from raw data if available
    if "ad_data_raw" in st.session_state and "Marketplace" in st.session_state["ad_data_raw"].columns:
        raw_df_for_options = st.session_state["ad_data_raw"]
        if not raw_df_for_options.empty:
            available_marketplaces = sorted([str(mp) for mp in raw_df_for_options["Marketplace"].dropna().unique()])
            if available_marketplaces:
                marketplace_options.extend(available_marketplaces)
                # Try to set 'US' as default if available, otherwise stick to 'All Marketplaces'
                target_default = "US"
                if target_default in marketplace_options:
                    default_marketplace = target_default
                    default_mp_index = marketplace_options.index(target_default)

    # Ensure the session state value for the widget exists before rendering
    if 'marketplace_selector_value' not in st.session_state:
        st.session_state.marketplace_selector_value = default_marketplace

    # Get the current index based on the session state value, default to 0 if not found
    try:
        # Ensure the value in session state is actually in the current options
        current_value = st.session_state.marketplace_selector_value
        if current_value not in marketplace_options:
            current_value = default_marketplace # Reset to default if invalid
            st.session_state.marketplace_selector_value = current_value # Update state
        current_mp_index = marketplace_options.index(current_value)
    except ValueError:
        current_mp_index = default_mp_index # Fallback to default if state value is somehow invalid
        st.session_state.marketplace_selector_value = marketplace_options[current_mp_index] # Correct the state value

    # Display the widget - its value is stored/retrieved using the key
    selected_mp_widget = st.sidebar.selectbox(
        "Select Marketplace",
        options=marketplace_options,
        #index=current_mp_index,
        key="marketplace_selector_value" # Key links widget to session state
    )
    # selected_mp_widget now holds the value selected in this run

    # --- Check if Reprocessing is Needed ---
    # Reprocessing is needed if:
    # 1. Processed data doesn't exist yet OR
    # 2. The marketplace selected now is different from the one used for the existing processed data OR
    # 3. The file itself has changed (handled by session state reset earlier, but double-check processed_file_name)
    needs_processing = False
    if "ad_data_processed" not in st.session_state:
        needs_processing = True
    elif st.session_state.get("processed_marketplace") != selected_mp_widget:
         needs_processing = True
    elif st.session_state.get("processed_file_name") != advertising_file.name: # If somehow raw data loaded but not processed
         needs_processing = True


    # --- Perform Filtering and Preprocessing ONLY if needed ---
    if needs_processing:
        with st.spinner("Processing data..."): # Use spinner for feedback
            # Use the value directly from the widget for this run's processing
            current_selection_for_processing = selected_mp_widget

            try:
                # Ensure raw data is available (should be loaded above)
                if "ad_data_raw" not in st.session_state:
                    st.error("Raw data not found for processing. Please re-upload the file.")
                    st.stop()

                ad_data_to_filter = st.session_state["ad_data_raw"]

                # --- Filter by Marketplace ---
                if current_selection_for_processing != "All Marketplaces":
                     # Check if Marketplace column exists before filtering
                     if "Marketplace" in ad_data_to_filter.columns:
                         st.session_state["ad_data_filtered"] = ad_data_to_filter[
                             ad_data_to_filter["Marketplace"].astype(str) == current_selection_for_processing
                         ].copy()
                     else:
                         st.warning("'Marketplace' column not found in data. Cannot filter by Marketplace.")
                         st.session_state["ad_data_filtered"] = ad_data_to_filter.copy() # Use unfiltered
                else:
                    st.session_state["ad_data_filtered"] = ad_data_to_filter.copy()

                # --- PREPROCESSING ---
                if not st.session_state["ad_data_filtered"].empty:
                    st.session_state["ad_data_processed"] = preprocess_ad_data(st.session_state["ad_data_filtered"])

                    if st.session_state["ad_data_processed"].empty:
                        st.error("Preprocessing resulted in empty data. Please check data quality and formats for the selected marketplace.")
                        # Keep potentially existing 'processed_marketplace' state as is, but data is empty
                    else:
                        # Store which marketplace was successfully used for this processed data
                        st.session_state.processed_marketplace = current_selection_for_processing
                        st.session_state.processed_file_name = advertising_file.name # Also track file name associated with processed data
                        # st.sidebar.success("Processing complete.") # Can be noisy, spinner is often enough

                else:
                    st.warning(f"Data is empty after filtering for Marketplace: '{current_selection_for_processing}'.")
                    # Clear processed data if filtering results in empty
                    if "ad_data_processed" in st.session_state: del st.session_state["ad_data_processed"]
                    st.session_state.processed_marketplace = current_selection_for_processing # Store the MP that resulted in empty data
                    st.session_state.processed_file_name = advertising_file.name

            except Exception as e:
                st.error(f"Error processing data: {e}")
                # Clean up session state on processing error
                keys_to_del = [k for k in st.session_state.keys() if k not in ['current_file_name', 'adv_file', 'marketplace_selector_value']]
                for key in keys_to_del:
                    if key in st.session_state: del st.session_state[key]
                # No st.stop() here, allow app to continue perhaps showing error

    # If needs_processing was False, the existing st.session_state["ad_data_processed"] is used by the rest of the app.

# =============================================================================
# Display Dashboard Tabs Only When Data is Uploaded and Processed
# =============================================================================
if "ad_data_processed" in st.session_state and not st.session_state["ad_data_processed"].empty:

    tabs_adv = st.tabs([
        "YOY Comparison",
        "Sponsored Products",
        "Sponsored Brands",
        "Sponsored Display"
    ])

    # -------------------------------
    # Tab 0: YOY Comparison (Unchanged from previous state)
    # -------------------------------
    with tabs_adv[0]:
        st.markdown("### YOY Comparison")
        ad_data_overview = st.session_state["ad_data_processed"].copy()

        st.markdown("#### Select Comparison Criteria")
        col1_yoy, col2_yoy, col3_yoy, col4_yoy = st.columns(4)

        with col1_yoy:
            available_years_yoy = sorted(ad_data_overview["Year"].unique())
            default_years_yoy = available_years_yoy[-2:] if len(available_years_yoy) >= 2 else available_years_yoy
            selected_years_yoy = st.multiselect("Select Year(s):", available_years_yoy, default=default_years_yoy, key="yoy_years")
        with col2_yoy:
            timeframe_options_yoy = ["Specific Week", "Last 4 Weeks", "Last 8 Weeks", "Last 12 Weeks"]
            default_tf_index_yoy = timeframe_options_yoy.index("Last 4 Weeks") if "Last 4 Weeks" in timeframe_options_yoy else 0
            selected_timeframe_yoy = st.selectbox("Select Timeframe:", timeframe_options_yoy, index=default_tf_index_yoy, key="yoy_timeframe")
        with col3_yoy:
            available_weeks_str_yoy = ["Select..."]
            if selected_years_yoy:
                try:
                    weeks_in_selected_years = ad_data_overview[ad_data_overview["Year"].isin(selected_years_yoy)]["Week"].unique()
                    available_weeks_yoy = sorted([int(w) for w in weeks_in_selected_years if pd.notna(w)])
                    available_weeks_str_yoy.extend([str(w) for w in available_weeks_yoy])
                except Exception as e: st.warning(f"Could not retrieve weeks: {e}")
            is_specific_week_yoy = (selected_timeframe_yoy == "Specific Week")
            selected_week_option_yoy = st.selectbox("Select Week:", available_weeks_str_yoy, index=0, key="yoy_week", disabled=(not is_specific_week_yoy))
            selected_week_yoy = int(selected_week_option_yoy) if is_specific_week_yoy and selected_week_option_yoy != "Select..." else None
        with col4_yoy:
            all_metrics_yoy = ["Impressions", "Clicks", "Spend", "Sales", "Orders", "Units", "CTR", "CVR", "CPC", "ACOS", "ROAS", "Ad % Sale"]
            calculable_metrics_yoy = {"CTR", "CVR", "CPC", "ACOS", "ROAS", "Ad % Sale"}
            original_cols_yoy = set(st.session_state["ad_data_processed"].columns)
            available_display_metrics_yoy = []
            for m in all_metrics_yoy:
                if m in original_cols_yoy: available_display_metrics_yoy.append(m)
                elif m in calculable_metrics_yoy:
                    can_calc_m = False
                    if m == "CTR" and {"Clicks", "Impressions"}.issubset(original_cols_yoy): can_calc_m = True
                    elif m == "CVR" and {"Orders", "Clicks"}.issubset(original_cols_yoy): can_calc_m = True
                    elif m == "CPC" and {"Spend", "Clicks"}.issubset(original_cols_yoy): can_calc_m = True
                    elif m == "ACOS" and {"Spend", "Sales"}.issubset(original_cols_yoy): can_calc_m = True
                    elif m == "ROAS" and {"Sales", "Spend"}.issubset(original_cols_yoy): can_calc_m = True
                    elif m == "Ad % Sale" and {"Sales", "Total Sales", "WE Date"}.issubset(original_cols_yoy): can_calc_m = True
                    if can_calc_m: available_display_metrics_yoy.append(m)
            default_metrics_list_yoy = ["Spend", "Sales", "Ad % Sale", "ACOS"]
            default_metrics_yoy = [m for m in default_metrics_list_yoy if m in available_display_metrics_yoy]
            selected_metrics_yoy = st.multiselect("Select Metrics:", available_display_metrics_yoy, default=default_metrics_yoy, key="yoy_metrics")
            if not selected_metrics_yoy:
                selected_metrics_yoy = default_metrics_yoy[:1] if default_metrics_yoy else available_display_metrics_yoy[:1]
                if not selected_metrics_yoy: st.warning("No metrics available for selection.")

        if not selected_years_yoy: st.warning("Please select at least one year.")
        elif not selected_metrics_yoy: st.warning("Please select at least one metric.")
        else:
            filtered_data_yoy = filter_data_by_timeframe(ad_data_overview, selected_years_yoy, selected_timeframe_yoy, selected_week_yoy)
            if filtered_data_yoy.empty: st.info("No data available for the selected YOY criteria (Years/Timeframe/Week).")
            else:
                years_to_process_yoy = sorted(filtered_data_yoy['Year'].unique())
                st.markdown("---"); st.markdown("#### Overview by Product Type")
                st.caption("*Aggregated data for selected years/timeframe, showing only selected metrics.*")
                product_overview_yoy_table = create_yoy_grouped_table(df_filtered_period=filtered_data_yoy, group_by_col="Product", selected_metrics=selected_metrics_yoy, years_to_process=years_to_process_yoy, display_col_name="Product")
                if not product_overview_yoy_table.empty:
                    styled_product_overview_yoy = style_yoy_comparison_table(product_overview_yoy_table)
                    if styled_product_overview_yoy: st.dataframe(styled_product_overview_yoy, use_container_width=True)
                else: st.info("No product overview data available.")

                portfolio_col_yoy = next((col for col in ["Portfolio Name", "Portfolio"] if col in filtered_data_yoy.columns), None)
                if portfolio_col_yoy:
                    st.markdown("---"); st.markdown("#### Portfolio Performance")
                    st.caption("*Aggregated data for selected years/timeframe, showing only selected metrics. Optionally filter by Product Type below.*")
                    portfolio_table_data_yoy = filtered_data_yoy.copy()
                    selected_product_portfolio_yoy = "All"
                    if "Product" in filtered_data_yoy.columns:
                        product_types_portfolio_yoy = ["All"] + sorted(filtered_data_yoy["Product"].unique().tolist())
                        selected_product_portfolio_yoy = st.selectbox("Filter Portfolio Table by Product Type:", product_types_portfolio_yoy, index=0, key="portfolio_product_filter_yoy")
                        if selected_product_portfolio_yoy != "All": portfolio_table_data_yoy = portfolio_table_data_yoy[portfolio_table_data_yoy["Product"] == selected_product_portfolio_yoy]
                    if portfolio_table_data_yoy.empty: st.info(f"No Portfolio data available for Product Type '{selected_product_portfolio_yoy}'.")
                    else:
                        portfolio_yoy_table = create_yoy_grouped_table(df_filtered_period=portfolio_table_data_yoy, group_by_col=portfolio_col_yoy, selected_metrics=selected_metrics_yoy, years_to_process=years_to_process_yoy, display_col_name="Portfolio")
                        if not portfolio_yoy_table.empty:
                            styled_portfolio_yoy = style_yoy_comparison_table(portfolio_yoy_table)
                            if styled_portfolio_yoy: st.dataframe(styled_portfolio_yoy, use_container_width=True)
                            portfolio_summary_row_yoy = calculate_yoy_summary_row(df=portfolio_table_data_yoy, selected_metrics=selected_metrics_yoy, years_to_process=years_to_process_yoy, id_col_name="Portfolio", id_col_value=f"TOTAL - {selected_product_portfolio_yoy}")
                            if not portfolio_summary_row_yoy.empty:
                                st.markdown("###### YoY Total (Selected Period & Product Filter)")
                                styled_portfolio_summary_yoy = style_yoy_comparison_table(portfolio_summary_row_yoy)
                                if styled_portfolio_summary_yoy: st.dataframe(styled_portfolio_summary_yoy.set_properties(**{'font-weight': 'bold'}), use_container_width=True)
                        else: st.info(f"No displayable portfolio data for Product Type '{selected_product_portfolio_yoy}'.")

                if {"Product", "Match Type"}.issubset(filtered_data_yoy.columns):
                    st.markdown("---"); st.markdown("#### Match Type Performance")
                    st.caption("*Aggregated data for selected years/timeframe, showing only selected metrics, broken down by Product Type.*")
                    product_types_match_yoy = ["Sponsored Products", "Sponsored Brands", "Sponsored Display"]
                    for product_type_m in product_types_match_yoy:
                        product_data_match_yoy = filtered_data_yoy[filtered_data_yoy["Product"] == product_type_m].copy()
                        if product_data_match_yoy.empty: continue
                        st.subheader(product_type_m)
                        match_type_yoy_table = create_yoy_grouped_table(df_filtered_period=product_data_match_yoy, group_by_col="Match Type", selected_metrics=selected_metrics_yoy, years_to_process=years_to_process_yoy, display_col_name="Match Type")
                        if not match_type_yoy_table.empty:
                            styled_match_type_yoy = style_yoy_comparison_table(match_type_yoy_table)
                            if styled_match_type_yoy: st.dataframe(styled_match_type_yoy, use_container_width=True)
                            match_type_summary_row_yoy = calculate_yoy_summary_row(df=product_data_match_yoy, selected_metrics=selected_metrics_yoy, years_to_process=years_to_process_yoy, id_col_name="Match Type", id_col_value=f"TOTAL - {product_type_m}")
                            if not match_type_summary_row_yoy.empty:
                                st.markdown("###### YoY Total (Selected Period)")
                                styled_match_type_summary_yoy = style_yoy_comparison_table(match_type_summary_row_yoy)
                                if styled_match_type_summary_yoy: st.dataframe(styled_match_type_summary_yoy.set_properties(**{'font-weight': 'bold'}), use_container_width=True)
                        else: st.info(f"No Match Type data available for {product_type_m}.")

                if {"Product", "RTW/Prospecting"}.issubset(filtered_data_yoy.columns):
                    st.markdown("---"); st.markdown("#### RTW/Prospecting Performance")
                    st.caption("*Aggregated data for selected years/timeframe, showing only selected metrics. Choose a Product Type below.*")
                    rtw_product_types_yoy = ["Sponsored Products", "Sponsored Brands", "Sponsored Display"]
                    available_rtw_products_yoy = sorted([pt for pt in filtered_data_yoy["Product"].unique() if pt in rtw_product_types_yoy])
                    if available_rtw_products_yoy:
                        selected_rtw_product_yoy = st.selectbox("Select Product Type for RTW/Prospecting Analysis:", available_rtw_products_yoy, key="rtw_product_selector_yoy")
                        rtw_filtered_product_data_yoy = filtered_data_yoy[filtered_data_yoy["Product"] == selected_rtw_product_yoy].copy()
                        if not rtw_filtered_product_data_yoy.empty:
                            rtw_yoy_table = create_yoy_grouped_table(df_filtered_period=rtw_filtered_product_data_yoy, group_by_col="RTW/Prospecting", selected_metrics=selected_metrics_yoy, years_to_process=years_to_process_yoy, display_col_name="RTW/Prospecting")
                            if not rtw_yoy_table.empty:
                                styled_rtw_yoy = style_yoy_comparison_table(rtw_yoy_table)
                                if styled_rtw_yoy: st.dataframe(styled_rtw_yoy, use_container_width=True)
                                rtw_summary_row_yoy = calculate_yoy_summary_row(df=rtw_filtered_product_data_yoy, selected_metrics=selected_metrics_yoy, years_to_process=years_to_process_yoy, id_col_name="RTW/Prospecting", id_col_value=f"TOTAL - {selected_rtw_product_yoy}")
                                if not rtw_summary_row_yoy.empty:
                                    st.markdown("###### YoY Total (Selected Period)")
                                    styled_rtw_summary_yoy = style_yoy_comparison_table(rtw_summary_row_yoy)
                                    if styled_rtw_summary_yoy: st.dataframe(styled_rtw_summary_yoy.set_properties(**{'font-weight': 'bold'}), use_container_width=True)
                            else: st.info(f"No RTW/Prospecting data available for {selected_rtw_product_yoy}.")
                        else: st.info(f"No {selected_rtw_product_yoy} data in selected period.")
                    else: st.info("No relevant Product Types found for RTW/Prospecting analysis.")

                campaign_col_yoy = "Campaign Name"
                if campaign_col_yoy in filtered_data_yoy.columns:
                    st.markdown("---"); st.markdown(f"#### {campaign_col_yoy} Performance")
                    st.caption("*Aggregated data for selected years/timeframe, showing only selected metrics.*")
                    campaign_yoy_table = create_yoy_grouped_table(df_filtered_period=filtered_data_yoy, group_by_col=campaign_col_yoy, selected_metrics=selected_metrics_yoy, years_to_process=years_to_process_yoy, display_col_name="Campaign")
                    if not campaign_yoy_table.empty:
                        styled_campaign_yoy = style_yoy_comparison_table(campaign_yoy_table)
                        if styled_campaign_yoy: st.dataframe(styled_campaign_yoy, use_container_width=True, height=600)
                        campaign_summary_row_yoy = calculate_yoy_summary_row(df=filtered_data_yoy, selected_metrics=selected_metrics_yoy, years_to_process=years_to_process_yoy, id_col_name="Campaign", id_col_value="TOTAL - All Campaigns")
                        if not campaign_summary_row_yoy.empty:
                            st.markdown("###### YoY Total (Selected Period)")
                            styled_campaign_summary_yoy = style_yoy_comparison_table(campaign_summary_row_yoy)
                            if styled_campaign_summary_yoy: st.dataframe(styled_campaign_summary_yoy.set_properties(**{'font-weight': 'bold'}), use_container_width=True)
                    else: st.info(f"No displayable {campaign_col_yoy} data.")

    # =========================================================================
    # == SP / SB / SD Tabs (REVISED STRUCTURE with YOY Tables & Visible Selector Fix) ==
    # =========================================================================
    for i, product_type_tab in enumerate(["Sponsored Products", "Sponsored Brands", "Sponsored Display"]):
        with tabs_adv[i+1]: # Start from tab index 1
            st.markdown(f"### {product_type_tab} Performance")
            st.caption("Charts use filters below. Tables show YoY comparison for the selected date range & metrics.") # Updated caption

            # Use unique keys for widgets in each tab
            tab_key_prefix = product_type_tab.lower().replace(" ", "_")

            ad_data_tab = st.session_state["ad_data_processed"].copy()

            # Check if product type exists in the data
            if "Product" not in ad_data_tab.columns or product_type_tab not in ad_data_tab["Product"].unique():
                st.warning(f"No '{product_type_tab}' data found in the uploaded file for the selected Marketplace.")
                continue # Skip to the next tab

            ad_data_tab_filtered_initial = ad_data_tab[ad_data_tab["Product"] == product_type_tab].copy()
            if ad_data_tab_filtered_initial.empty:
                 st.warning(f"No {product_type_tab} data available after initial filtering (check selected marketplace/dates).")
                 continue

            # --- Filters ---
            with st.expander("Filters", expanded=True):
                col1_tab, col2_tab, col3_tab = st.columns(3) # Adjusted columns for new selector
                selected_metric_tab = None # Initialize for charts
                selected_yoy_metrics_tab = [] # Initialize for tables
                can_calc_ad_sale_tab = {"Sales", "Total Sales", "WE Date"}.issubset(st.session_state["ad_data_processed"].columns)

                # --- Determine Available Metrics (for both charts and tables) ---
                all_possible_metrics = ["Impressions", "Clicks", "Spend", "Sales", "Orders", "Units", "CTR", "CVR", "ACOS", "ROAS", "CPC", "Ad % Sale"]
                original_cols_tab = set(st.session_state["ad_data_processed"].columns) # Use processed data cols
                available_metrics_tab = []
                # Check against initially filtered data for this product type to be more specific
                product_specific_cols = set(ad_data_tab_filtered_initial.columns)
                for m in all_possible_metrics:
                    can_display_m = False
                    # Check if base columns exist in the original processed data (needed for calculations)
                    # Check if the metric itself exists in the product-specific data (for direct display)
                    if m in product_specific_cols: can_display_m = True
                    elif m == "CTR" and {"Clicks", "Impressions"}.issubset(original_cols_tab): can_display_m = True
                    elif m == "CVR" and {"Orders", "Clicks"}.issubset(original_cols_tab): can_display_m = True
                    elif m == "ACOS" and {"Spend", "Sales"}.issubset(original_cols_tab): can_display_m = True
                    elif m == "ROAS" and {"Sales", "Spend"}.issubset(original_cols_tab): can_display_m = True
                    elif m == "CPC" and {"Spend", "Clicks"}.issubset(original_cols_tab): can_display_m = True
                    elif m == "Ad % Sale" and can_calc_ad_sale_tab: can_display_m = True # Requires cols in original data
                    if can_display_m: available_metrics_tab.append(m)
                available_metrics_tab = sorted(list(set(available_metrics_tab)))

                with col1_tab:
                    # Chart Metric Selector (Single Select)
                    default_metric_chart_tab = "Spend" if "Spend" in available_metrics_tab else available_metrics_tab[0] if available_metrics_tab else None
                    sel_metric_index_tab = available_metrics_tab.index(default_metric_chart_tab) if default_metric_chart_tab in available_metrics_tab else 0
                    if available_metrics_tab:
                        selected_metric_tab = st.selectbox("Select Metric for Charts", options=available_metrics_tab, index=sel_metric_index_tab, key=f"{tab_key_prefix}_metric")
                    else: st.warning(f"No metrics available for chart selection in {product_type_tab} tab.")

                with col2_tab:
                    # Portfolio Selector (Single Select for Charts)
                    portfolio_col_tab = next((col for col in ["Portfolio Name", "Portfolio"] if col in ad_data_tab_filtered_initial.columns), None)
                    if not portfolio_col_tab:
                        selected_portfolio_tab = "All Portfolios"
                        st.info("Portfolio filtering (for charts) not available ('Portfolio Name' column missing).")
                    else:
                        ad_data_tab_filtered_initial[portfolio_col_tab] = ad_data_tab_filtered_initial[portfolio_col_tab].fillna("Unknown Portfolio")
                        portfolio_options_tab = ["All Portfolios"] + sorted(ad_data_tab_filtered_initial[portfolio_col_tab].unique().tolist())
                        selected_portfolio_tab = st.selectbox("Select Portfolio (for Charts)", options=portfolio_options_tab, index=0, key=f"{tab_key_prefix}_portfolio")

                with col3_tab:
                    # Table Metrics Selector (Multi Select) - NEW
                    default_metrics_table_list = ["Spend", "Sales", "Ad % Sale", "ACOS"]
                    default_metrics_table_tab = [m for m in default_metrics_table_list if m in available_metrics_tab]
                    if not default_metrics_table_tab and available_metrics_tab:
                         default_metrics_table_tab = available_metrics_tab[:1] # Fallback

                    if available_metrics_tab:
                        selected_yoy_metrics_tab = st.multiselect(
                            "Select Metrics for YOY Tables",
                            options=available_metrics_tab,
                            default=default_metrics_table_tab,
                            key=f"{tab_key_prefix}_yoy_metrics"
                        )
                    else: st.warning(f"No metrics available for YOY table selection in {product_type_tab} tab.")

                show_yoy_tab = st.checkbox("Show Year-over-Year Comparison (Chart - Weekly Points)", value=True, key=f"{tab_key_prefix}_show_yoy")

                # Date Range Selector
                date_range_tab = None
                min_date_tab, max_date_tab = None, None
                if "WE Date" in ad_data_tab_filtered_initial.columns and not ad_data_tab_filtered_initial["WE Date"].dropna().empty:
                     try:
                         min_date_tab = ad_data_tab_filtered_initial["WE Date"].min().date()
                         max_date_tab = ad_data_tab_filtered_initial["WE Date"].max().date()
                         if min_date_tab <= max_date_tab:
                             date_range_tab = st.date_input("Select Date Range", value=(min_date_tab, max_date_tab), min_value=min_date_tab, max_value=max_date_tab, key=f"{tab_key_prefix}_date_range")
                         else: st.warning(f"Invalid date range found in {product_type_tab} data.")
                     except Exception as e: st.warning(f"Error setting date range for {product_type_tab}: {e}")
                else: st.warning(f"Cannot determine date range for {product_type_tab} tab ('WE Date' missing or empty).")

            # Apply Date Range Filter (to data specifically for this product type tab)
            ad_data_tab_date_filtered = ad_data_tab_filtered_initial.copy()
            # Also filter the *original* processed data to calculate the *overall* total sales denominator correctly across all product types
            original_data_date_filtered_tab = st.session_state["ad_data_processed"].copy()
            if date_range_tab and len(date_range_tab) == 2 and min_date_tab and max_date_tab:
                start_date_tab, end_date_tab = date_range_tab
                if start_date_tab >= min_date_tab and end_date_tab <= max_date_tab and start_date_tab <= end_date_tab:
                    ad_data_tab_date_filtered = ad_data_tab_date_filtered[ (ad_data_tab_date_filtered["WE Date"].dt.date >= start_date_tab) & (ad_data_tab_date_filtered["WE Date"].dt.date <= end_date_tab) ]
                    original_data_date_filtered_tab = original_data_date_filtered_tab[ (original_data_date_filtered_tab["WE Date"].dt.date >= start_date_tab) & (original_data_date_filtered_tab["WE Date"].dt.date <= end_date_tab) ]
                else:
                    st.warning("Selected date range is invalid or outside data bounds. Using full data range.")
                    # Keep initial data if range is bad
                    ad_data_tab_date_filtered = ad_data_tab_filtered_initial.copy()
                    original_data_date_filtered_tab = st.session_state["ad_data_processed"].copy()

            # --- Prepare Data for Ad % Sale Chart (Denominator) ---
            weekly_denominator_df_tab = pd.DataFrame()
            if selected_metric_tab == "Ad % Sale" and can_calc_ad_sale_tab:
                if not original_data_date_filtered_tab.empty:
                     try:
                          temp_denom_df = original_data_date_filtered_tab.copy()
                          temp_denom_df["WE Date"] = pd.to_datetime(temp_denom_df["WE Date"], errors='coerce')
                          temp_denom_df['Total Sales'] = pd.to_numeric(temp_denom_df['Total Sales'], errors='coerce')
                          if 'Year' not in temp_denom_df.columns: temp_denom_df["Year"] = temp_denom_df["WE Date"].dt.year
                          if 'Week' not in temp_denom_df.columns: temp_denom_df["Week"] = temp_denom_df["WE Date"].dt.isocalendar().week
                          temp_denom_df['Year'] = pd.to_numeric(temp_denom_df['Year'], errors='coerce')
                          temp_denom_df['Week'] = pd.to_numeric(temp_denom_df['Week'], errors='coerce')
                          temp_denom_df.dropna(subset=["WE Date", "Total Sales", "Year", "Week"], inplace=True)
                          if not temp_denom_df.empty:
                              temp_denom_df['Year'] = temp_denom_df['Year'].astype(int)
                              temp_denom_df['Week'] = temp_denom_df['Week'].astype(int)
                              unique_subset_denom = ['Year', 'Week']
                              if "Marketplace" in temp_denom_df.columns: unique_subset_denom.append("Marketplace")
                              unique_totals = temp_denom_df.drop_duplicates(subset=unique_subset_denom)
                              weekly_denominator_df_tab = unique_totals.groupby(['Year', 'Week'], as_index=False)['Total Sales'].sum()
                              weekly_denominator_df_tab = weekly_denominator_df_tab.rename(columns={'Total Sales': 'Weekly_Total_Sales'})
                     except Exception as e: st.warning(f"Could not calculate weekly total sales denominator for Ad % Sale chart ({product_type_tab}): {e}")
                else: st.warning(f"Cannot calculate Ad % Sale denominator: No original data in selected date range for {product_type_tab}.")

            # --- Display Charts (Unchanged from previous structure) ---
            if ad_data_tab_date_filtered.empty:
                # This case is handled below before attempting table generation
                pass # st.warning(f"No {product_type_tab} data available for the selected filters.")
            elif selected_metric_tab is None:
                st.warning("Please select a metric to visualize the charts.")
            else:
                # Time Chart
                st.subheader(f"{selected_metric_tab} Over Time")
                fig1_tab = create_metric_over_time_chart(data=ad_data_tab_date_filtered, metric=selected_metric_tab, portfolio=selected_portfolio_tab, product_type=product_type_tab, show_yoy=show_yoy_tab, weekly_total_sales_data=weekly_denominator_df_tab)
                st.plotly_chart(fig1_tab, use_container_width=True, key=f"{tab_key_prefix}_time_chart")
                # Comparison Chart
                if selected_portfolio_tab == "All Portfolios" and portfolio_col_tab:
                    st.subheader(f"{selected_metric_tab} by Portfolio")
                    # Display Ad % Sale info message BEFORE calling the chart function
                    if selected_metric_tab == "Ad % Sale":
                         st.info("'Ad % Sale' cannot be displayed in the Portfolio Comparison bar chart.")
                    else:
                         # Call the comparison chart function (now handles CPC)
                         fig2_tab = create_metric_comparison_chart(ad_data_tab_date_filtered, selected_metric_tab, None, product_type_tab)
                         st.plotly_chart(fig2_tab, use_container_width=True, key=f"{tab_key_prefix}_portfolio_chart")


            # --- NEW: Display YOY Tables ---
            st.markdown("---")
            st.subheader("Year-over-Year Portfolio Performance (Selected Period & Metrics)")

            if not portfolio_col_tab:
                st.warning("Cannot generate YOY Portfolio table: 'Portfolio Name' column not found.")
            elif not selected_yoy_metrics_tab:
                st.warning("Please select at least one metric in the 'Select Metrics for YOY Tables' filter to display the table.")
            elif ad_data_tab_date_filtered.empty:
                st.info(f"No {product_type_tab} data available for the selected date range to build the YOY table.")
            else:
                years_in_tab_data = sorted(ad_data_tab_date_filtered['Year'].unique())

                # Prepare data for table, potentially adding 'Total Sales' if needed
                data_for_yoy_table = ad_data_tab_date_filtered.copy()
                if "Ad % Sale" in selected_yoy_metrics_tab and "Total Sales" in original_data_date_filtered_tab.columns:
                    merge_cols = ['WE Date', 'Year', 'Week']
                    if "Marketplace" in data_for_yoy_table.columns: merge_cols.append("Marketplace")
                    total_sales_data = original_data_date_filtered_tab[merge_cols + ['Total Sales']].drop_duplicates(subset=merge_cols)
                    if 'Total Sales' in data_for_yoy_table.columns: data_for_yoy_table = data_for_yoy_table.drop(columns=['Total Sales'])
                    data_for_yoy_table = pd.merge(data_for_yoy_table, total_sales_data, on=merge_cols, how='left')

                # Create Portfolio Breakdown Table
                portfolio_yoy_table_tab = create_yoy_grouped_table(df_filtered_period=data_for_yoy_table, group_by_col=portfolio_col_tab, selected_metrics=selected_yoy_metrics_tab, years_to_process=years_in_tab_data, display_col_name="Portfolio")
                # Create Summary Row
                portfolio_yoy_summary_tab = calculate_yoy_summary_row(df=data_for_yoy_table, selected_metrics=selected_yoy_metrics_tab, years_to_process=years_in_tab_data, id_col_name="Portfolio", id_col_value="TOTAL")

                # Display Tables
                if not portfolio_yoy_table_tab.empty:
                    st.markdown("###### YOY Portfolio Breakdown")
                    styled_portfolio_yoy_tab = style_yoy_comparison_table(portfolio_yoy_table_tab)
                    if styled_portfolio_yoy_tab: st.dataframe(styled_portfolio_yoy_tab, use_container_width=True)
                else: st.info("No portfolio breakdown data available for the selected YOY metrics and period.")

                if not portfolio_yoy_summary_tab.empty:
                    st.markdown("###### YOY Total")
                    styled_portfolio_summary_yoy_tab = style_yoy_comparison_table(portfolio_yoy_summary_tab)
                    if styled_portfolio_summary_yoy_tab: st.dataframe(styled_portfolio_summary_yoy_tab.set_properties(**{'font-weight': 'bold'}), use_container_width=True)
                else: st.info("No summary data available for the selected YOY metrics and period.")

            # --- REVISED Insights Section ---
            st.markdown("---")
            st.subheader("Key Insights (Latest Year in Selected Period)")

            # Ensure date-filtered data for the tab exists and the list of years is available
            if 'ad_data_tab_date_filtered' in locals() and not ad_data_tab_date_filtered.empty and 'years_in_tab_data' in locals() and years_in_tab_data:
                latest_year_tab = years_in_tab_data[-1]
                # Filter the date-filtered data to get only the latest year's data
                data_latest_year = ad_data_tab_date_filtered[ad_data_tab_date_filtered['Year'] == latest_year_tab].copy()

                if not data_latest_year.empty:
                    # Calculate totals for core metrics needed for insights directly from latest year data
                    # Use .get(col, 0) or similar if columns might be missing, or ensure they exist
                    # Convert to numeric, coercing errors and filling NaN with 0 before summing
                    total_spend = pd.to_numeric(data_latest_year.get("Spend"), errors='coerce').fillna(0).sum()
                    total_sales = pd.to_numeric(data_latest_year.get("Sales"), errors='coerce').fillna(0).sum()
                    total_clicks = pd.to_numeric(data_latest_year.get("Clicks"), errors='coerce').fillna(0).sum()
                    total_impressions = pd.to_numeric(data_latest_year.get("Impressions"), errors='coerce').fillna(0).sum()
                    total_orders = pd.to_numeric(data_latest_year.get("Orders"), errors='coerce').fillna(0).sum()

                    # Calculate derived metrics specifically for insights
                    insight_acos = (total_spend / total_sales * 100) if total_sales > 0 else np.nan
                    insight_roas = (total_sales / total_spend) if total_spend > 0 else np.nan
                    # Use 0 for CTR/CVR if denominator is 0, or np.nan if you prefer insights function to handle NaN
                    insight_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
                    insight_cvr = (total_orders / total_clicks * 100) if total_clicks > 0 else 0

                    # Handle potential inf values from division by zero if ROAS/ACOS were calculated differently
                    insight_acos = np.nan if insight_acos in [np.inf, -np.inf] else insight_acos
                    insight_roas = np.nan if insight_roas in [np.inf, -np.inf] else insight_roas


                    # Prepare data Series for the generate_insights function
                    # Ensure it contains the keys the function expects ('ACOS', 'ROAS', 'CTR', 'CVR', 'Sales', 'Spend')
                    summary_series_for_insights = pd.Series({
                        "ACOS": insight_acos,
                        "ROAS": insight_roas,
                        "CTR": insight_ctr,
                        "CVR": insight_cvr,
                        "Sales": total_sales, # Pass base values too, generate_insights uses them
                        "Spend": total_spend
                    })

                    # Generate and display insights using the independently calculated data
                    insights_tab = generate_insights(summary_series_for_insights, product_type_tab)
                    for insight in insights_tab:
                        st.markdown(f"- {insight}")

                else:
                    # Handle case where there's no data for the very latest year within the filter
                    st.info(f"No data found for the latest year ({latest_year_tab}) in the selected period to generate insights.")
            else:
                # Handle case where date filtering resulted in empty data overall for the tab
                st.info("No summary data available to generate insights (check date range and filters).")
            # --- End of revised Insights Section ---


# =============================================================================
# Final Fallback Messages (If no data was processed or loaded)
# =============================================================================
# This block executes only if the initial 'if "ad_data_processed"...' check fails.
elif advertising_file is None:
    st.info("Please upload an Advertising Data CSV file using the sidebar to begin.")
else:
    # This implies file was uploaded, but processing failed or resulted in no data.
    # Check if raw data exists but processed doesn't
    if "ad_data_raw" in st.session_state and ("ad_data_processed" not in st.session_state or st.session_state["ad_data_processed"].empty):
         st.warning("Data loaded but processing resulted in no usable data for the selected filters (e.g., Marketplace, Date Range). Please check the file content and filter selections.")
    else: # Fallback for other unexpected states
         st.warning("Could not display dashboard. Please check the uploaded file format, content, and sidebar selections.")
