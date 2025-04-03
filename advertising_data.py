import streamlit as st
import pandas as pd
import numpy as np  # Added numpy import
import warnings
import plotly.express as px
import datetime
# import calendar # Removed as not explicitly used
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import re
import io # Removed as not explicitly used
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
# MOVED UP: Common Functions for Advertising Data (Needed Early)
# =============================================================================
def preprocess_ad_data(df):
    """Preprocess advertising data for analysis"""
    if df is None or df.empty:
          return pd.DataFrame()
    df_processed = df.copy() # Work on a copy
    # Attempt to convert 'WE Date', handle errors gracefully
    try:
        # Try common date formats
        df_processed["WE Date"] = pd.to_datetime(df_processed["WE Date"], format="%d/%m/%Y", dayfirst=True, errors='coerce')
        if df_processed["WE Date"].isnull().any(): # If first format failed for some, try another
             df_processed["WE Date"] = pd.to_datetime(df["WE Date"], errors='coerce') # Let pandas infer

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
    for col in numeric_cols:
        if col in df_processed.columns:
            # Convert to numeric, coercing errors to NaN
            df_processed[col] = pd.to_numeric(df_processed[col], errors="coerce")
            # Optionally, fill NaNs if appropriate (e.g., fill 0 for sums, leave NaN for rates)
            # Example: Fill sum-based columns with 0
            # if col in ["Impressions", "Clicks", "Spend", "Sales", "Orders", "Units", "Total Sales"]:
            #     df_processed[col] = df_processed[col].fillna(0)

    return df_processed


# --- CORRECTED filter_data_by_timeframe function ---
def filter_data_by_timeframe(df, selected_years, selected_timeframe, selected_week):
    """
    Filters data for selected years based on timeframe.
    - "Specific Week": Filters all selected years for that specific week number.
    - "Last X Weeks": Determines the last X weeks based on the *latest* selected year's max week,
                      and filters *all* selected years for those *same* week numbers.
    Returns a concatenated dataframe across the selected years.
    """
    # --- Input Validation ---
    if not selected_years:
        return pd.DataFrame() # Return empty if no years are selected

    if df is None or df.empty:
        st.warning("Input DataFrame to filter_data_by_timeframe is empty.")
        return pd.DataFrame()


    # Ensure selected_years are numeric if possible, handle potential errors
    try:
        selected_years = [int(y) for y in selected_years]
    except ValueError:
        st.error("Selected years must be numeric.") # Use st.error if in Streamlit context
        return pd.DataFrame()

    df_copy = df.copy() # Work on a copy to avoid modifying original

    # --- Ensure Base Columns Exist and Prepare Data ---
    # Check for Year/Week only *after* confirming WE Date exists implicitly (as it's needed to create them)
    if 'WE Date' not in df_copy.columns or df_copy['WE Date'].isnull().all():
        st.error("Required 'WE Date' column is missing or empty for timeframe filtering.")
        return pd.DataFrame()

    # Ensure WE Date is datetime first
    df_copy['WE Date'] = pd.to_datetime(df_copy['WE Date'], errors='coerce')
    df_copy.dropna(subset=['WE Date'], inplace=True)
    if df_copy.empty: return pd.DataFrame() # Return empty if no valid dates

    # Now add Year/Week if they don't exist
    if 'Year' not in df_copy.columns: df_copy["Year"] = df_copy["WE Date"].dt.year
    if 'Week' not in df_copy.columns: df_copy["Week"] = df_copy["WE Date"].dt.isocalendar().week

    # Ensure 'Year' and 'Week' columns are numeric, drop rows where conversion fails
    df_copy['Year'] = pd.to_numeric(df_copy['Year'], errors='coerce')
    df_copy['Week'] = pd.to_numeric(df_copy['Week'], errors='coerce')
    df_copy.dropna(subset=['Year', 'Week'], inplace=True)
    if df_copy.empty: return pd.DataFrame() # Return empty if no valid Year/Week data
    df_copy['Year'] = df_copy['Year'].astype(int)
    df_copy['Week'] = df_copy['Week'].astype(int)

    # --- Initial Filter for Selected Years ---
    df_filtered_years = df_copy[df_copy["Year"].isin(selected_years)].copy()

    if df_filtered_years.empty:
        # st.warning(f"No data found for the selected year(s): {selected_years}") # Optional warning
        return pd.DataFrame() # Return empty if no data for selected years

    # --- Handle Timeframe Logic ---
    if selected_timeframe == "Specific Week":
        if selected_week is not None:
            try:
                # Ensure selected_week is an integer
                selected_week_int = int(selected_week)
                # Filter the already year-filtered data for the specific week
                return df_filtered_years[df_filtered_years["Week"] == selected_week_int]
            except ValueError:
                st.error(f"Invalid 'selected_week': {selected_week}. Must be a number.")
                return pd.DataFrame()
        else:
            # If "Specific Week" is selected but no week number is provided (e.g., "Select..." was chosen)
            # Return an empty DataFrame, as no specific week was actually chosen to filter by.
            return pd.DataFrame()

    # --- Handle "Last X Weeks" Logic ---
    else:
        try:
            # Robustly extract number of weeks
            match = re.search(r'\d+', selected_timeframe)
            if match:
                 weeks_to_filter = int(match.group(0))
            else:
                 raise ValueError("Could not find number in timeframe string")
        except (IndexError, ValueError, TypeError):
            st.error(f"Could not parse weeks from timeframe: '{selected_timeframe}'")
            return pd.DataFrame()

        # Find the latest year *among the selected years* that actually has data in df_filtered_years
        # Check if df_filtered_years is not empty before calling .max()
        if df_filtered_years.empty:
            return pd.DataFrame()
        latest_year_with_data = df_filtered_years["Year"].max()


        # Get the data just for that latest year to find its max week
        df_latest_year = df_filtered_years[df_filtered_years["Year"] == latest_year_with_data]

        if df_latest_year.empty:
            # This can happen if the latest selected year has no data at all
            st.warning(f"No data found for the latest selected year ({latest_year_with_data}) to determine week range.")
            return pd.DataFrame()

        # Find the max week within that latest year's data
        global_max_week = df_latest_year["Week"].max()

        # Calculate the target week range based on the latest year's max week
        start_week = global_max_week - weeks_to_filter + 1
        # Ensure start_week isn't less than 1
        start_week = max(1, start_week)

        # Create a list of target week numbers
        target_weeks = list(range(start_week, global_max_week + 1))

        # Filter the *entire* selected years' data (df_filtered_years) for these target weeks
        final_filtered_df = df_filtered_years[df_filtered_years["Week"].isin(target_weeks)]

        return final_filtered_df


# =============================================================================
# REST of Helper Functions Start Here
# =============================================================================

# --- Basic Chart/Table/Insight/Styling Helpers ---

def create_metric_comparison_chart(df, metric, portfolio_name=None, campaign_type="Sponsored Products"):
    # Add checks for required columns
    required_cols_base = {"Product", "Portfolio Name"}
    required_cols_metric = {metric}
    if metric == "CTR": required_cols_metric.update({"Clicks", "Impressions"})
    elif metric == "CVR": required_cols_metric.update({"Orders", "Clicks"})
    elif metric == "ACOS": required_cols_metric.update({"Spend", "Sales"})
    elif metric == "ROAS": required_cols_metric.update({"Sales", "Spend"}) # Added ROAS requirement

    if df is None or df.empty: # Check if df is valid
        st.warning(f"Comparison chart received empty data for {campaign_type}.")
        return go.Figure()

    if not required_cols_base.issubset(df.columns):
        missing = required_cols_base - set(df.columns)
        st.warning(f"Comparison chart missing base columns: {missing}")
        return go.Figure() # Return empty figure

    # Filter by campaign type first
    filtered_df = df[df["Product"] == campaign_type].copy()
    if filtered_df.empty: # Check if data exists for this campaign type
        # st.info(f"No data found for {campaign_type} to generate comparison chart.") # Can be verbose
        return go.Figure()


    # Now check specific metric columns on the filtered data
    if not required_cols_metric.issubset(filtered_df.columns):
        missing = required_cols_metric - set(filtered_df.columns)
        st.warning(f"Comparison chart missing columns for metric '{metric}' in {campaign_type} data: {missing}")
        return go.Figure()


    if portfolio_name and portfolio_name != "All Portfolios":
        # Check if the specific portfolio exists in the filtered data
        if portfolio_name in filtered_df["Portfolio Name"].unique():
            filtered_df = filtered_df[filtered_df["Portfolio Name"] == portfolio_name]
        else:
             st.warning(f"Portfolio '{portfolio_name}' not found for {campaign_type}. Showing all.")
             portfolio_name = "All Portfolios" # Update variable to reflect change


    # Ensure Portfolio Name column is suitable for grouping (e.g., handle NaN)
    filtered_df["Portfolio Name"] = filtered_df["Portfolio Name"].fillna("Unknown Portfolio")

    if filtered_df.empty:
        # This check might be redundant if already checked after campaign type filter
        # st.warning(f"No data for {campaign_type}" + (f" and portfolio '{portfolio_name}'" if portfolio_name and portfolio_name != "All Portfolios" else "") + " to create comparison chart.")
        return go.Figure()

    # Calculation logic (with added safety)
    if metric in ["CTR", "CVR", "ACOS", "ROAS"]:
        if metric == "CTR":
            portfolio_agg = filtered_df.groupby("Portfolio Name").agg(Clicks=("Clicks", "sum"), Impressions=("Impressions", "sum")).reset_index()
            portfolio_agg[metric] = portfolio_agg.apply(lambda row: (row["Clicks"] / row["Impressions"] * 100) if row["Impressions"] else 0, axis=1).round(2)
        elif metric == "CVR":
            portfolio_agg = filtered_df.groupby("Portfolio Name").agg(Orders=("Orders", "sum"), Clicks=("Clicks", "sum")).reset_index()
            portfolio_agg[metric] = portfolio_agg.apply(lambda row: (row["Orders"] / row["Clicks"] * 100) if row["Clicks"] else 0, axis=1).round(2)
        elif metric == "ACOS":
            portfolio_agg = filtered_df.groupby("Portfolio Name").agg(Spend=("Spend", "sum"), Sales=("Sales", "sum")).reset_index()
            portfolio_agg[metric] = portfolio_agg.apply(lambda row: (row["Spend"] / row["Sales"] * 100) if row["Sales"] else np.nan, axis=1).round(2) # Use NaN
        elif metric == "ROAS":
            portfolio_agg = filtered_df.groupby("Portfolio Name").agg(Sales=("Sales", "sum"), Spend=("Spend", "sum")).reset_index()
            portfolio_agg[metric] = portfolio_agg.apply(lambda row: (row["Sales"] / row["Spend"]) if row["Spend"] else np.nan, axis=1).round(2) # Use NaN

        # Replace inf generated during calculation before grouping
        if metric in ["ACOS", "ROAS"]:
             portfolio_agg[metric] = portfolio_agg[metric].replace([np.inf, -np.inf], np.nan)

        grouped_df = portfolio_agg
    else: # Direct aggregation for base metrics
        # Ensure the metric column exists before grouping
        if metric not in filtered_df.columns:
             st.warning(f"Metric column '{metric}' not found for direct aggregation.")
             return go.Figure()
        try:
             grouped_df = filtered_df.groupby("Portfolio Name").agg(**{metric: (metric, "sum")}).reset_index()
        except Exception as e:
             st.warning(f"Error aggregating comparison chart for {metric}: {e}")
             return go.Figure()


    # Drop rows where the metric value is NaN before sorting and plotting
    grouped_df = grouped_df.dropna(subset=[metric])

    if grouped_df.empty:
        st.warning(f"No valid data points for metric '{metric}' after aggregation.")
        return go.Figure()

    grouped_df = grouped_df.sort_values(metric, ascending=False)

    # Generate Plot Title dynamically
    title_suffix = ""
    if portfolio_name and portfolio_name != "All Portfolios":
        title_suffix = f" - {portfolio_name}"
    chart_title = f"{metric} by Portfolio ({campaign_type}){title_suffix}"


    fig = px.bar(
        grouped_df,
        x="Portfolio Name",
        y=metric,
        title=chart_title,
        text_auto=True
    )
    # Apply formatting based on metric type
    if metric in ["Spend", "Sales"]:
        fig.update_traces(texttemplate='%{y:$,.0f}') # Format text inside bars
        fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",.2f")
    elif metric in ["CTR", "CVR", "ACOS"]:
        fig.update_traces(texttemplate='%{y:.1f}%') # Format text inside bars
        fig.update_layout(yaxis_ticksuffix="%", yaxis_tickformat=".1f") # Use .1f for consistency
    elif metric == "ROAS":
        fig.update_traces(texttemplate='%{y:.2f}')
        fig.update_layout(yaxis_tickformat=".2f")
    else: # Impressions, Clicks, Orders, Units
        fig.update_traces(texttemplate='%{y:,.0f}')
        fig.update_layout(yaxis_tickformat=",.0f")

    fig.update_layout(margin=dict(t=50, b=50)) # Adjust margins as needed
    return fig


def create_performance_metrics_table(df, portfolio_name=None, campaign_type="Sponsored Products"):
     # Check required columns
     required_cols = {"Product", "Portfolio Name", "Impressions", "Clicks", "Spend", "Sales", "Orders"}
     if df is None or df.empty:
        st.warning(f"Performance table received empty data for {campaign_type}.")
        return pd.DataFrame(), pd.DataFrame()

     if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        st.warning(f"Performance table missing required columns: {missing}")
        return pd.DataFrame(), pd.DataFrame()

     filtered_df = df[df["Product"] == campaign_type].copy()
     # Handle potential NaN in Portfolio Name before filtering/grouping
     filtered_df["Portfolio Name"] = filtered_df["Portfolio Name"].fillna("Unknown Portfolio")

     if portfolio_name and portfolio_name != "All Portfolios":
        # Check if portfolio exists
        if portfolio_name in filtered_df["Portfolio Name"].unique():
             filtered_df = filtered_df[filtered_df["Portfolio Name"] == portfolio_name]
        else:
             st.warning(f"Portfolio '{portfolio_name}' not found for {campaign_type} in performance table.")
             return pd.DataFrame(), pd.DataFrame()


     if filtered_df.empty:
        # st.warning(f"No data for {campaign_type}" + (f" and portfolio '{portfolio_name}'" if portfolio_name and portfolio_name != "All Portfolios" else "") + " for performance table.") # Can be verbose
        return pd.DataFrame(), pd.DataFrame()

     # Aggregate base metrics
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


     # Calculate derived metrics safely
     metrics_by_portfolio["CTR"] = metrics_by_portfolio.apply(lambda r: (r["Clicks"] / r["Impressions"] * 100) if r["Impressions"] else 0, axis=1).round(1)
     metrics_by_portfolio["CVR"] = metrics_by_portfolio.apply(lambda r: (r["Orders"] / r["Clicks"] * 100) if r["Clicks"] else 0, axis=1).round(1)
     metrics_by_portfolio["ACOS"] = metrics_by_portfolio.apply(lambda r: (r["Spend"] / r["Sales"] * 100) if r["Sales"] else np.nan, axis=1) # Use NaN for no sales
     metrics_by_portfolio["ROAS"] = metrics_by_portfolio.apply(lambda r: (r["Sales"] / r["Spend"]) if r["Spend"] else np.nan, axis=1) # Use NaN for no spend

     # Handle inf results specifically for ACOS/ROAS if division occurred
     metrics_by_portfolio['ACOS'] = metrics_by_portfolio['ACOS'].replace([np.inf, -np.inf], np.nan)
     metrics_by_portfolio['ROAS'] = metrics_by_portfolio['ROAS'].replace([np.inf, -np.inf], np.nan)


     # Round currency values
     metrics_by_portfolio["Spend"] = metrics_by_portfolio["Spend"].round(2)
     metrics_by_portfolio["Sales"] = metrics_by_portfolio["Sales"].round(2)


     # Calculate Total Summary Row
     # Check if metrics_by_portfolio is empty before summing
     if metrics_by_portfolio.empty:
        total_summary = pd.DataFrame() # Return empty if no portfolio data
     else:
        total_impressions = metrics_by_portfolio["Impressions"].sum()
        total_clicks = metrics_by_portfolio["Clicks"].sum()
        total_spend = metrics_by_portfolio["Spend"].sum() # Use already rounded sum
        total_sales = metrics_by_portfolio["Sales"].sum() # Use already rounded sum
        total_orders = metrics_by_portfolio["Orders"].sum()

        total_ctr = (total_clicks / total_impressions * 100) if total_impressions else 0
        total_cvr = (total_orders / total_clicks * 100) if total_clicks else 0
        total_acos = (total_spend / total_sales * 100) if total_sales else np.nan
        total_roas = (total_sales / total_spend) if total_spend else np.nan

        # Ensure total ACOS/ROAS are not infinite
        total_acos = np.nan if total_acos in [np.inf, -np.inf] else total_acos
        total_roas = np.nan if total_roas in [np.inf, -np.inf] else total_roas

        total_summary_data = {
             "Metric": ["Total"], # Keep simple label for the total row identifier
             "Impressions": [total_impressions],
             "Clicks": [total_clicks],
             "Spend": [total_spend], # Already rounded
             "Sales": [total_sales], # Already rounded
             "Orders": [total_orders],
             "CTR": [round(total_ctr, 1)],
             "CVR": [round(total_cvr, 1)],
             "ACOS": [total_acos], # Keep as NaN if applicable, format later
             "ROAS": [total_roas]  # Keep as NaN if applicable, format later
        }
        total_summary = pd.DataFrame(total_summary_data)

     # Rename column for clarity in the main table display
     metrics_by_portfolio = metrics_by_portfolio.rename(columns={"Portfolio Name": "Portfolio"})


     return metrics_by_portfolio, total_summary

def create_metric_over_time_chart(data, metric, portfolio, product_type, show_yoy=True):
    """Create a chart showing metric over time with optional YoY comparison (Weekly YoY Overlay with Month Annotations)."""
    # --- Input Data Check ---
    if data is None or data.empty:
        st.warning(f"Time chart received empty data for {product_type}.")
        return go.Figure()

    # --- Base Column & WE Date Checks ---
    base_required = {"Product", "Portfolio Name", "WE Date"}
    if not base_required.issubset(data.columns):
        missing = base_required - set(data.columns)
        st.warning(f"Metric over time chart missing required base columns: {missing}")
        return go.Figure()

    # Ensure WE Date is datetime first, and drop rows where it fails
    data_copy = data.copy() # Work on a copy
    data_copy["WE Date"] = pd.to_datetime(data_copy["WE Date"], errors='coerce')
    data_copy.dropna(subset=["WE Date"], inplace=True)
    if data_copy.empty:
        st.warning("No valid 'WE Date' data remains after conversion.")
        return go.Figure()

    # --- Filter Data ---
    filtered_data = data_copy[data_copy["Product"] == product_type].copy()
    if "Portfolio Name" not in filtered_data.columns: # Should be caught by initial check, but safe
        if portfolio != "All Portfolios": st.warning("'Portfolio Name' column missing...")
        portfolio = "All Portfolios"
    else:
        filtered_data["Portfolio Name"] = filtered_data["Portfolio Name"].fillna("Unknown Portfolio")
        if portfolio != "All Portfolios":
            if portfolio in filtered_data["Portfolio Name"].unique():
                filtered_data = filtered_data[filtered_data["Portfolio Name"] == portfolio]
            else:
                st.warning(f"Portfolio '{portfolio}' not found... Showing all.")
                portfolio = "All Portfolios"

    if filtered_data.empty:
        return go.Figure()

    # --- Ensure Time Components Exist ---
    try:
        if 'Year' not in filtered_data.columns: filtered_data["Year"] = filtered_data["WE Date"].dt.year
        if 'Week' not in filtered_data.columns: filtered_data["Week"] = filtered_data["WE Date"].dt.isocalendar().week # Use ISO week
        # Ensure types AFTER creation
        filtered_data['Year'] = pd.to_numeric(filtered_data['Year'], errors='coerce').astype('Int64')
        filtered_data['Week'] = pd.to_numeric(filtered_data['Week'], errors='coerce').astype('Int64')
        filtered_data.dropna(subset=['Year', 'Week'], inplace=True) # Drop if Year/Week couldn't be coerced
    except Exception as e:
        st.error(f"Error creating time components: {e}")
        return go.Figure()

    if filtered_data.empty:
        st.warning("No data remains after creating time components (Year/Week).")
        return go.Figure()

    # --- Check if metric exists or can be calculated ---
    metric_required_cols = {metric}
    base_needed_for_metric = set()
    if metric == "CTR": base_needed_for_metric.update({"Clicks", "Impressions"})
    elif metric == "CVR": base_needed_for_metric.update({"Orders", "Clicks"})
    elif metric == "ACOS": base_needed_for_metric.update({"Spend", "Sales"})
    elif metric == "ROAS": base_needed_for_metric.update({"Sales", "Spend"})
    elif metric == "CPC": base_needed_for_metric.update({"Spend", "Clicks"})

    metric_exists = metric in filtered_data.columns
    can_calculate = base_needed_for_metric.issubset(filtered_data.columns)

    if not metric_exists and not can_calculate:
        missing = (base_needed_for_metric | metric_required_cols) - set(filtered_data.columns)
        st.warning(f"Metric chart requires column(s): {missing}")
        return go.Figure()
    # --- End Column Checks ---


    years = sorted(filtered_data["Year"].dropna().unique())
    fig = go.Figure()

    # --- Define Hover Templates ---
    if metric in ["CTR", "CVR", "ACOS"]: hover_suffix = "%"; hover_format = ".1f"
    elif metric in ["Spend", "Sales"]: hover_suffix = ""; hover_format = "$,.2f" # Add $ prefix
    elif metric in ["ROAS", "CPC"]: hover_suffix = ""; hover_format = ".2f" # Added CPC
    else: hover_suffix = ""; hover_format = ",.0f" # Impressions, Clicks etc.
    # Updated hover template to show Date (from customdata) and Week
    base_hover_template = f"Date: %{{customdata[1]}}<br>Week: %{{customdata[0]}}<br>{metric}: %{{y:{hover_format}}}{hover_suffix}<extra></extra>"

    # --- Plotting Logic ---
    processed_years = []
    colors = px.colors.qualitative.Plotly

    if show_yoy and len(years) > 1:
        # --- Aggregate by Week for YoY Overlay ---
        base_needed = base_needed_for_metric if not metric_exists else {metric} | base_needed_for_metric
        agg_cols = list(base_needed & set(filtered_data.columns))
        if not agg_cols:
             st.warning(f"Missing base columns required for metric '{metric}'")
             return go.Figure()

        try:
            agg_dict = {col: "sum" for col in agg_cols}
            agg_dict["WE Date"] = "mean" # Keep a representative date for the week
            grouped = filtered_data.groupby(["Year", "Week"], as_index=False).agg(agg_dict)
            grouped["WE Date"] = pd.to_datetime(grouped["WE Date"])
        except Exception as e:
            st.warning(f"Could not group data by week for YoY chart: {e}")
            return go.Figure()

        # Calculate derived metric if it wasn't directly aggregated
        metric_calculated = metric in grouped.columns
        if not metric_calculated:
            # ... (metric calculation logic) ...
            if metric == "CTR":
                if {"Clicks", "Impressions"}.issubset(grouped.columns):
                    grouped[metric] = grouped.apply(lambda r: (r["Clicks"] / r["Impressions"] * 100) if r.get("Impressions") else 0, axis=1)
                    metric_calculated = True
            elif metric == "CVR":
                if {"Orders", "Clicks"}.issubset(grouped.columns):
                    grouped[metric] = grouped.apply(lambda r: (r["Orders"] / r["Clicks"] * 100) if r.get("Clicks") else 0, axis=1)
                    metric_calculated = True
            elif metric == "ACOS":
                if {"Spend", "Sales"}.issubset(grouped.columns):
                    grouped[metric] = grouped.apply(lambda r: (r["Spend"] / r["Sales"] * 100) if r.get("Sales") else np.nan, axis=1)
                    metric_calculated = True
            elif metric == "ROAS":
                if {"Sales", "Spend"}.issubset(grouped.columns):
                    grouped[metric] = grouped.apply(lambda r: (r["Sales"] / r["Spend"]) if r.get("Spend") else np.nan, axis=1)
                    metric_calculated = True
            elif metric == "CPC":
                if {"Spend", "Clicks"}.issubset(grouped.columns):
                    grouped[metric] = grouped.apply(lambda r: (r["Spend"] / r["Clicks"]) if r.get("Clicks") else np.nan, axis=1)
                    metric_calculated = True

        if not metric_calculated:
             st.warning(f"Could not calculate metric '{metric}' for weekly YoY chart.")
             return go.Figure()

        if metric in grouped.columns:
            grouped[metric] = grouped[metric].replace([np.inf, -np.inf], np.nan)
        else:
            st.warning(f"Metric column '{metric}' missing after calculation.")
            return go.Figure()

        # Plot each year's weekly data against Week Number
        for i, year in enumerate(years):
            year_data = grouped[grouped["Year"] == year].sort_values("Week")
            if year_data.empty or year_data[metric].isnull().all():
                continue

            processed_years.append(year)

            # Prepare customdata with Week and Formatted WE Date string
            custom_data_for_hover = year_data[['Week']].copy()
            custom_data_for_hover['DateStr'] = year_data['WE Date'].dt.strftime('%Y-%m-%d') # Use the representative date

            fig.add_trace(
                go.Scatter(
                    x=year_data["Week"], # **** Plot against Week number ****
                    y=year_data[metric],
                    mode="lines+markers",
                    name=f"{year}",
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=6),
                    customdata=custom_data_for_hover[['Week', 'DateStr']], # Pass Week and Formatted Date
                    hovertemplate=base_hover_template
                )
            )

        # --- Add Month Annotations Below Axis --- ## MODIFIED SECTION ##
        month_approx_weeks = {
            1: 2.5, 2: 6.5, 3: 10.5, 4: 15, 5: 19.5, 6: 24,
            7: 28, 8: 32.5, 9: 37, 10: 41.5, 11: 46, 12: 50.5
        } # Week numbers approximating middle of month
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        min_week_data = grouped['Week'].min()
        max_week_data = grouped['Week'].max()

        # Remove the fig.add_vrect loop (background shading)

        # Add month annotations below the axis
        for month_num, week_val in month_approx_weeks.items():
            # Only add annotation if the week is within the plotted data range
            if week_val >= min_week_data - 1 and week_val <= max_week_data + 1:
                 fig.add_annotation(
                     x=week_val,      # Position based on week number
                     y=-0.12,         # Position below the plot area (adjust as needed)
                     xref="x",        # X position refers to data (week number)
                     yref="paper",    # Y position refers to paper (0=bottom, 1=top)
                     text=month_names[month_num-1],
                     showarrow=False,
                     font=dict(size=10, color="grey"),
                 )

        fig.update_layout(
             xaxis_title="Week of Year", # Keep title accurate
             xaxis_showticklabels=True, # Show the actual week numbers
             xaxis_range=[max(0, min_week_data - 1), min(54, max_week_data + 1)], # Pad axis slightly
             legend_title="Year",
             margin=dict(b=70) # Ensure enough bottom margin for annotations
        )
        ## --- END MODIFIED SECTION --- ##

    else: # --- Plotting Logic for Single Trace (No YoY or only one year) ---
        # This part still plots against WE Date
        base_needed_no_yoy = base_needed_for_metric if not metric_exists else {metric} | base_needed_for_metric
        agg_cols_no_yoy = base_needed_no_yoy & set(filtered_data.columns)

        if not agg_cols_no_yoy:
            st.warning(f"Missing base columns to calculate '{metric}' over time.")
            return go.Figure()

        try:
            grouped = filtered_data.groupby(["WE Date", "Week"], as_index=False)[list(agg_cols_no_yoy)].sum()
        except Exception as e:
             st.warning(f"Could not group data for time chart: {e}")
             return go.Figure()

        # Calculate derived metric
        metric_calculated_no_yoy = metric in grouped.columns
        if not metric_calculated_no_yoy:
            # ... (metric calculation logic) ...
            if metric == "CTR" and {"Clicks", "Impressions"}.issubset(agg_cols_no_yoy):
                grouped[metric] = grouped.apply(lambda r: (r["Clicks"] / r["Impressions"] * 100) if r.get("Impressions", 0) else 0, axis=1)
                metric_calculated_no_yoy = True
            elif metric == "CVR" and {"Orders", "Clicks"}.issubset(agg_cols_no_yoy):
                grouped[metric] = grouped.apply(lambda r: (r["Orders"] / r["Clicks"] * 100) if r.get("Clicks", 0) else 0, axis=1)
                metric_calculated_no_yoy = True
            elif metric == "ACOS" and {"Spend", "Sales"}.issubset(agg_cols_no_yoy):
                grouped[metric] = grouped.apply(lambda r: (r["Spend"] / r["Sales"] * 100) if r.get("Sales", 0) else np.nan, axis=1)
                metric_calculated_no_yoy = True
            elif metric == "ROAS" and {"Sales", "Spend"}.issubset(agg_cols_no_yoy):
                grouped[metric] = grouped.apply(lambda r: (r["Sales"] / r["Spend"]) if r.get("Spend", 0) else np.nan, axis=1)
                metric_calculated_no_yoy = True
            elif metric == "CPC" and {"Spend", "Clicks"}.issubset(agg_cols_no_yoy):
                grouped[metric] = grouped.apply(lambda r: (r["Spend"] / r["Clicks"]) if r.get("Clicks", 0) else np.nan, axis=1)
                metric_calculated_no_yoy = True

        if not metric_calculated_no_yoy:
             st.warning(f"Could not calculate '{metric}' over time.")
             return go.Figure()

        if metric in grouped.columns:
             grouped[metric] = grouped[metric].replace([np.inf, -np.inf], np.nan)
        else:
             st.warning(f"Metric column '{metric}' missing after calculation.")
             return go.Figure()

        if grouped[metric].isnull().all():
             st.warning(f"No valid '{metric}' data found for plotting over time.")
             return go.Figure()

        grouped = grouped.sort_values("WE Date")

        # Prepare customdata with Week and Formatted WE Date string for single trace
        custom_data_for_hover_single = grouped[['Week']].copy()
        custom_data_for_hover_single['DateStr'] = grouped['WE Date'].dt.strftime('%Y-%m-%d')

        fig.add_trace(
            go.Scatter(
                x=grouped["WE Date"], # Plot against actual date here
                y=grouped[metric],
                mode="lines+markers",
                name=metric,
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=6),
                customdata=custom_data_for_hover_single[['Week', 'DateStr']], # Pass Week and Date
                hovertemplate=base_hover_template
            )
        )
        fig.update_layout(xaxis_title="Date", showlegend=False)


    # --- General Layout Updates ---
    portfolio_title = f" for {portfolio}" if portfolio != "All Portfolios" else " for All Portfolios"
    # Adjust title and axis labels based on view
    years_in_plot = processed_years if processed_years else years # Use years actually plotted or initial list
    if show_yoy and len(years_in_plot) > 1:
        final_chart_title = f"{metric} Weekly Comparison {portfolio_title} ({product_type})"
        final_xaxis_title = "Week of Year" # Updated axis label
    else:
        final_chart_title = f"{metric} Over Time (Weekly) {portfolio_title} ({product_type})"
        final_xaxis_title = "Week Ending Date"


    # Consolidate margin update
    final_margin = dict(t=80, b=70, l=70, r=30) # Ensure bottom margin is sufficient
    fig.update_layout(
        title=final_chart_title,
        xaxis_title=final_xaxis_title,
        yaxis_title=metric,
        hovermode="x unified",
        template="plotly_white",
        yaxis=dict(rangemode="tozero"), # Ensure y axis includes 0
        margin=final_margin # Apply updated margin
    )
    # Apply specific y-axis formatting (no change needed here)
    if metric in ["Spend", "Sales"]: fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",.2f")
    elif metric in ["CTR", "CVR", "ACOS"]: fig.update_layout(yaxis_ticksuffix="%", yaxis_tickformat=".1f")
    elif metric in ["ROAS", "CPC"]: fig.update_layout(yaxis_tickformat=".2f")

    return fig

def style_dataframe(df, format_dict, highlight_cols=None, color_map_func=None, text_align='right', na_rep='NaN'):
    """Generic styling function for dataframes with alignment and NaN handling."""
    if df is None or df.empty:
        return None # Return None if input df is empty
    df_copy = df.copy()
    # Replace inf values before styling
    df_copy = df_copy.replace([np.inf, -np.inf], np.nan)

    # Ensure format_dict keys exist in df columns
    valid_format_dict = {k: v for k, v in format_dict.items() if k in df_copy.columns}

    try:
        styled = df_copy.style.format(valid_format_dict, na_rep=na_rep)
    except Exception as e:
        st.error(f"Error applying format: {e}. Formatting dictionary: {valid_format_dict}")
        return df_copy.style # Return basic styler on error


    if highlight_cols and color_map_func:
        # Ensure funcs list matches cols list length
        if len(highlight_cols) != len(color_map_func):
             st.error("Mismatch between highlight_cols and color_map_func in style_dataframe.")
        else:
            for col, func in zip(highlight_cols, color_map_func):
                if col in df_copy.columns:
                    try: # Add try-except for safety during applymap
                         styled = styled.applymap(func, subset=[col])
                    except Exception as e:
                         st.warning(f"Styling failed for column '{col}': {e}")

    # Apply text alignment to all columns except potentially the first one if it's labels
    cols_to_align = df_copy.columns
    if len(cols_to_align) > 0 and df_copy[cols_to_align[0]].dtype == 'object':
        # Optionally align first column left, others right
        # Apply styles using a dictionary for clarity
        try:
            styles = [
                 {'selector': 'th', 'props': [('text-align', text_align)]}, # Align headers
                 {'selector': 'td', 'props': [('text-align', text_align)]}, # Align all cells
                 {'selector': 'th:first-child', 'props': [('text-align', 'left')]}, # Align first header left
                 {'selector': 'td:first-child', 'props': [('text-align', 'left')]}  # Align first column cells left
                 ]
            styled = styled.set_table_styles(styles, overwrite=False)
        except Exception as e:
            st.warning(f"Failed to apply specific alignment: {e}")
            styled = styled.set_properties(**{'text-align': text_align})


    else: # Align all right if first column isn't object type
        styled = styled.set_properties(**{'text-align': text_align})


    return styled

def style_total_summary(df):
    format_dict = {
        "Impressions": "{:,.0f}", "Clicks": "{:,.0f}", "Orders": "{:,.0f}",
        "Spend": "${:,.2f}", "Sales": "${:,.2f}",
        "CTR": "{:.1f}%", "CVR": "{:.1f}%",
        "ACOS": lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A", # Handle NaN
        "ROAS": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"  # Handle NaN
    }
    # Define coloring functions
    def color_acos(val):
        # Check type before comparison, handle string 'N/A'
        if isinstance(val, str) or pd.isna(val): return "color: grey"
        if val <= 15: return "color: green"
        elif val <= 20: return "color: orange"
        else: return "color: red"

    def color_roas(val):
        if isinstance(val, str) or pd.isna(val): return "color: grey"
        return "color: green" if val > 3 else "color: red"

    styled = style_dataframe(df, format_dict,
                             highlight_cols=["ACOS", "ROAS"],
                             color_map_func=[color_acos, color_roas],
                             na_rep="N/A") # Pass na_rep for consistency
    if styled: # Check if styling was successful
        # Ensure bold property is applied
        return styled.set_properties(**{"font-weight": "bold"})
    return None # Return None if styling failed


def style_metrics_table(df):
    # Similar formatting, adjust if Units column is present/needed
    format_dict = {
        "Impressions": "{:,.0f}", "Clicks": "{:,.0f}", "Orders": "{:,.0f}",
        "Spend": "${:,.2f}", "Sales": "${:,.2f}",
        "CTR": "{:.1f}%", "CVR": "{:.1f}%",
        "ACOS": lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A", # Handle NaN
        "ROAS": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"  # Handle NaN
        # Add formats for other potential columns if needed
    }
    # Dynamically add format for existing columns only
    if "Units" in df.columns: format_dict["Units"] = "{:,.0f}"
    # Add CPC format if exists
    if "CPC" in df.columns: format_dict["CPC"] = "${:,.2f}" # Added CPC

    # Check for portfolio identifier columns
    id_cols = ["Portfolio", "Match Type", "RTW/Prospecting", "Campaign"] # Added Campaign
    id_col_name = next((col for col in df.columns if col in id_cols), None)
    if id_col_name: format_dict[id_col_name] = "{}" # Ensure identifier isn't formatted numerically


    # Define coloring functions (same as summary)
    def color_acos(val):
        if isinstance(val, str) or pd.isna(val): return "color: grey"
        if val <= 15: return "color: green"
        elif val <= 20: return "color: orange"
        else: return "color: red"

    def color_roas(val):
        if isinstance(val, str) or pd.isna(val): return "color: grey"
        return "color: green" if val > 3 else "color: red"

    styled = style_dataframe(df, format_dict,
                             highlight_cols=["ACOS", "ROAS"],
                             color_map_func=[color_acos, color_roas],
                             na_rep="N/A")
    return styled # Returns the styled object or None


def generate_insights(total_metrics_series, campaign_type):
    # Expects a pandas Series (like df.iloc[0])
    # Determine ACOS threshold based on campaign type
    acos_threshold = 15 if campaign_type == "Sponsored Brands" else 30

    insights = []
    # Access metrics safely using .get() with a default of NaN
    acos = total_metrics_series.get("ACOS", np.nan)
    roas = total_metrics_series.get("ROAS", np.nan)
    ctr = total_metrics_series.get("CTR", np.nan)
    cvr = total_metrics_series.get("CVR", np.nan)
    sales = total_metrics_series.get("Sales", 0)
    spend = total_metrics_series.get("Spend", 0)

    # Check for Spend without Sales first
    if spend > 0 and sales == 0:
        insights.append("⚠️ **Immediate Attention:** There are ad expenses but absolutely no sales attributed. Review campaigns, targeting, and product pages urgently.")
        # Add specific metric insights only if relevant
        if pd.notna(ctr):
             if ctr < 0.3: insights.append("📉 Click-through rate is also low (<0.3%). Ad visibility or relevance might be poor.")
             else: insights.append(f"ℹ️ Click-through rate is {ctr:.2f}%.")
        # Skip ACOS/ROAS/CVR insights as they are meaningless without sales/orders

    else: # Proceed with normal insights if there are sales or no spend
        # ACOS Insight
        if pd.isna(acos):
             if spend == 0 and sales == 0: insights.append("ℹ️ No spend or sales recorded for ACOS calculation.")
             elif sales == 0 and spend > 0: insights.append("ℹ️ ACOS is not applicable (No Sales from Spend).")
             elif spend == 0: insights.append("ℹ️ ACOS is not applicable (No Spend).")
        elif acos > acos_threshold:
             insights.append(f"📈 **High ACOS:** Overall ACOS ({acos:.1f}%) is above the target ({acos_threshold}%). Consider optimizing bids, keywords, or targeting to improve efficiency.")
        else:
             insights.append(f"✅ **ACOS:** Overall ACOS ({acos:.1f}%) is within the acceptable range (≤{acos_threshold}%).")

        # ROAS Insight
        if pd.isna(roas):
             if spend == 0 and sales == 0: insights.append("ℹ️ No spend or sales recorded for ROAS calculation.")
             elif spend == 0 and sales > 0 : insights.append("✅ **ROAS:** ROAS is effectively infinite (Sales with No Spend).")
             elif spend > 0 and sales == 0: insights.append("ℹ️ ROAS is not applicable (No Sales from Spend).")
        elif roas < 3:
             insights.append(f"📉 **Low ROAS:** Overall ROAS ({roas:.2f}) is below the common target of 3. Review keyword performance, bid strategy, and product conversion rates.")
        else:
             insights.append(f"✅ **ROAS:** Overall ROAS ({roas:.2f}) is good (≥3), indicating efficient ad spend relative to sales generated.")

        # CTR Insight
        if pd.isna(ctr):
             insights.append("ℹ️ Click-Through Rate (CTR) could not be calculated (likely no impressions).")
        elif ctr < 0.3:
             insights.append("📉 **Low CTR:** Click-through rate ({ctr:.2f}%) is low (<0.3%). Consider improving ad creative (images, headlines), relevance, or placement.")
        else:
             insights.append(f"✅ **CTR:** Click-through rate ({ctr:.2f}%) is satisfactory (≥0.3%).")

        # CVR Insight
        if pd.isna(cvr):
             insights.append("ℹ️ Conversion Rate (CVR) could not be calculated (likely no clicks).")
        elif cvr < 10:
             insights.append("📉 **Low CVR:** Conversion rate ({cvr:.1f}%) is below 10%. Review product listing pages for clarity, pricing, reviews, and ensure targeting attracts relevant shoppers.")
        else:
             insights.append(f"✅ **CVR:** Conversion rate ({cvr:.1f}%) is good (≥10%).")

    return insights

# =============================================================================
# <<< MODIFIED >>> Helper Function for YOY Grouped Tables (Correct Denominator & % Change Logic)
# =============================================================================
def create_yoy_grouped_table(df_filtered_period, group_by_col, selected_metrics, years_to_process, display_col_name=None):
    """
    Creates a merged YoY comparison table grouped by a specific column.
    Calculates 'Ad % Sale' using the SUM of unique weekly Total Sales
    for the period as the denominator for all groups.
    Calculates absolute percentage point change for % metrics, relative otherwise.

    Args:
        df_filtered_period (pd.DataFrame): Data already filtered by year/timeframe.
        group_by_col (str): The column name to group the data by (e.g., "Portfolio Name").
        selected_metrics (list): List of metrics selected by the user.
        years_to_process (list): Sorted list of unique years present in the data.
        display_col_name (str, optional): Name for the grouping column in the final output table. Defaults to group_by_col.

    Returns:
        pd.DataFrame: A dataframe ready for display and styling, or empty if no data.
    """
    if df_filtered_period is None or df_filtered_period.empty: return pd.DataFrame()
    if group_by_col not in df_filtered_period.columns: st.warning(f"Grouping column '{group_by_col}' not found."); return pd.DataFrame()
    if not selected_metrics: st.warning("No metrics selected."); return pd.DataFrame()

    date_col = "WE Date" if "WE Date" in df_filtered_period.columns else None
    # Critical check for Ad % Sale calculation
    ad_sale_possible = ("Ad % Sale" in selected_metrics and
                       "Sales" in df_filtered_period.columns and
                       "Total Sales" in df_filtered_period.columns and
                       date_col)
    if "Ad % Sale" in selected_metrics and not ad_sale_possible:
       st.warning("Cannot calculate 'Ad % Sale'. Requires 'Sales', 'Total Sales', and 'WE Date' columns.")
       selected_metrics = [m for m in selected_metrics if m != "Ad % Sale"] # Remove it if not possible
       if not selected_metrics: return pd.DataFrame()

    df_filtered_period[group_by_col] = df_filtered_period[group_by_col].fillna(f"Unknown {group_by_col}")
    yearly_tables = []

    for yr in years_to_process:
        df_year = df_filtered_period[df_filtered_period["Year"] == yr].copy()
        if df_year.empty: continue

        # --- Determine base metrics needed for SUM aggregation ---
        base_metrics_to_sum_needed = set()
        for metric in selected_metrics:
            # Add metrics needed for SUM aggregation
            if metric in ["Impressions", "Clicks", "Spend", "Sales", "Orders", "Units"]: base_metrics_to_sum_needed.add(metric)
            elif metric == "CTR": base_metrics_to_sum_needed.update(["Clicks", "Impressions"])
            elif metric == "CVR": base_metrics_to_sum_needed.update(["Orders", "Clicks"])
            elif metric == "CPC": base_metrics_to_sum_needed.update(["Spend", "Clicks"])
            elif metric == "ACOS": base_metrics_to_sum_needed.update(["Spend", "Sales"])
            elif metric == "ROAS": base_metrics_to_sum_needed.update(["Sales", "Spend"])
            elif metric == "Ad % Sale": base_metrics_to_sum_needed.add("Sales") # Need Ad Sales summed

        # Check data availability only for SUM metrics
        actual_base_present = {m for m in base_metrics_to_sum_needed if m in df_year.columns}

        # Filter calculable metrics based *only* on columns needed for SUM or direct existence
        # Ad % Sale check happens separately below
        missing_for_calc = set()
        for metric in selected_metrics:
            if metric == "Ad % Sale": continue # Skip Ad % Sale check here
            # ... (checks for CTR, CVR, CPC, ACOS, ROAS as before using actual_base_present) ...
            if metric == "CTR" and not {"Clicks", "Impressions"}.issubset(actual_base_present): missing_for_calc.add(metric)
            elif metric == "CVR" and not {"Orders", "Clicks"}.issubset(actual_base_present): missing_for_calc.add(metric)
            elif metric == "CPC" and not {"Spend", "Clicks"}.issubset(actual_base_present): missing_for_calc.add(metric)
            elif metric == "ACOS" and not {"Spend", "Sales"}.issubset(actual_base_present): missing_for_calc.add(metric)
            elif metric == "ROAS" and not {"Sales", "Spend"}.issubset(actual_base_present): missing_for_calc.add(metric)
            elif metric in ["Impressions", "Clicks", "Spend", "Sales", "Orders", "Units"] and metric not in actual_base_present: missing_for_calc.add(metric)

        calculable_selected_metrics = [m for m in selected_metrics if m not in missing_for_calc]
        # Re-add Ad % Sale if it was possible initially
        if ad_sale_possible and "Ad % Sale" not in calculable_selected_metrics:
            calculable_selected_metrics.append("Ad % Sale")
        if not calculable_selected_metrics: continue

        # --- Calculate CORRECT Total Sales for the entire period/year ---
        total_sales_for_period = 0 # Default value
        if ad_sale_possible: # Check again if still possible
            try:
                df_year[date_col] = pd.to_datetime(df_year[date_col], errors='coerce')
                df_year_valid_dates_total = df_year.dropna(subset=[date_col])
                if not df_year_valid_dates_total.empty:
                    # --- MODIFIED SECTION ---
                    # Check if 'Marketplace' column exists and use it for uniqueness
                    unique_subset = [date_col]
                    if "Marketplace" in df_year_valid_dates_total.columns:
                         unique_subset.append("Marketplace")

                    unique_weekly_totals_for_sum = df_year_valid_dates_total.drop_duplicates(subset=unique_subset)
                    # --- END MODIFIED SECTION ---

                    total_sales_for_period = pd.to_numeric(unique_weekly_totals_for_sum['Total Sales'], errors='coerce').fillna(0).sum()
            except Exception as e:
                st.warning(f"Could not calculate total sales for period for year {yr}: {e}")
                total_sales_for_period = 0 # Default to 0 on error


        # --- Build Aggregation Dictionary (Only SUMS needed) ---
        agg_dict_final = {}
        sum_metrics_to_agg = list(actual_base_present) # Use only metrics present
        for m_sum in sum_metrics_to_agg:
            agg_dict_final[m_sum] = 'sum'

        # Perform Aggregation (No Total Sales needed here)
        if not agg_dict_final: df_pivot = pd.DataFrame({group_by_col: df_year[group_by_col].unique()})
        else:
            try: df_pivot = df_year.groupby(group_by_col).agg(agg_dict_final).reset_index()
            except Exception as e: st.warning(f"Error aggregating SUM data for {group_by_col} in year {yr}: {e}"); continue

        # --- Calculate Derived Metrics ---
        # Use .get() for safety on df_pivot columns
        if "CTR" in calculable_selected_metrics: df_pivot["CTR"] = df_pivot.apply(lambda r: (r.get("Clicks",0) / r.get("Impressions",0) * 100) if r.get("Impressions") else 0, axis=1)
        if "CVR" in calculable_selected_metrics: df_pivot["CVR"] = df_pivot.apply(lambda r: (r.get("Orders",0) / r.get("Clicks",0) * 100) if r.get("Clicks") else 0, axis=1)
        if "CPC" in calculable_selected_metrics: df_pivot["CPC"] = df_pivot.apply(lambda r: (r.get("Spend",0) / r.get("Clicks",0)) if r.get("Clicks") else np.nan, axis=1)
        if "ACOS" in calculable_selected_metrics: df_pivot["ACOS"] = df_pivot.apply(lambda r: (r.get("Spend",0) / r.get("Sales",0) * 100) if r.get("Sales") else np.nan, axis=1)
        if "ROAS" in calculable_selected_metrics: df_pivot["ROAS"] = df_pivot.apply(lambda r: (r.get("Sales",0) / r.get("Spend",0)) if r.get("Spend") else np.nan, axis=1)

        # --- FINAL Ad % Sale Calculation ---
        if "Ad % Sale" in calculable_selected_metrics:
            # Use the pre-calculated total_sales_for_period as the denominator for ALL rows
            if total_sales_for_period > 0:
                df_pivot["Ad % Sale"] = df_pivot.apply(
                    lambda r: (r.get("Sales", 0) / total_sales_for_period * 100), axis=1 )
            else:
                 df_pivot["Ad % Sale"] = np.nan # Assign NaN if total sales for period is 0 or couldn't be calculated
        # --- End Final Ad % Sale Calculation ---

        # Handle NaN/inf results
        for m in ["CPC", "ACOS", "ROAS", "CTR", "CVR", "Ad % Sale"]:
            if m in df_pivot.columns: df_pivot[m] = df_pivot[m].replace([np.inf, -np.inf], np.nan)


        # --- Select and Rename Columns for Output ---
        rename_cols = {m: f"{m} {yr}" for m in calculable_selected_metrics if m in df_pivot.columns}
        cols_to_keep = [group_by_col] + list(rename_cols.keys())
        cols_to_keep = [col for col in cols_to_keep if col in df_pivot.columns]
        if len(cols_to_keep) > 1:
           df_pivot_final = df_pivot[cols_to_keep].rename(columns=rename_cols)
           yearly_tables.append(df_pivot_final)

    # --- Merging and Final Processing ---
    if not yearly_tables: return pd.DataFrame()
    valid_tables = [tbl for tbl in yearly_tables if group_by_col in tbl.columns and not tbl.empty]
    if not valid_tables: return pd.DataFrame()
    try: merged_table = reduce(lambda left, right: pd.merge(left, right, on=group_by_col, how="outer"), valid_tables)
    except Exception as e: st.error(f"Error merging yearly {group_by_col} tables: {e}"); return pd.DataFrame()
    if merged_table.empty: return pd.DataFrame()

    # Fill NaNs ONLY for base SUM metrics after outer merge
    base_sum_metrics_all_years = set()
    for yr_proc in years_to_process: base_sum_metrics_all_years.update({f"{m} {yr_proc}" for m in ["Impressions", "Clicks", "Spend", "Sales", "Orders", "Units"]})
    cols_to_fill_zero = list(base_sum_metrics_all_years & set(merged_table.columns))
    if cols_to_fill_zero: merged_table[cols_to_fill_zero] = merged_table[cols_to_fill_zero].fillna(0)

    # Calculate % Change
    change_cols = []; ordered_cols = [group_by_col]
    actual_years_in_data = []
    if len(years_to_process) >= 2:
        actual_years_in_data = sorted(list(set([int(re.search(r'(\d{4})$', col).group(1)) for col in merged_table.columns if re.search(r'(\d{4})$', col)])))
        if len(actual_years_in_data) >= 2:
            current_year_sel, prev_year_sel = actual_years_in_data[-1], actual_years_in_data[-2]
            # Define the set of metrics that are already percentages << NEW
            percentage_metrics = {"CTR", "CVR", "ACOS", "Ad % Sale"}

            for metric in selected_metrics: # Use originally selected metrics for iteration
                col_current, col_prev = f"{metric} {current_year_sel}", f"{metric} {prev_year_sel}"
                change_col_name = f"{metric} % Change"

                # Determine column order first
                if col_prev in merged_table.columns: ordered_cols.append(col_prev)
                if col_current in merged_table.columns: ordered_cols.append(col_current)

                # Calculate change only if both columns exist
                if col_current in merged_table.columns and col_prev in merged_table.columns:
                    # <<< MODIFIED: Calculate % Change or Absolute Point Change >>>
                    if metric in percentage_metrics:
                        # --- Calculate Absolute Percentage Point Difference ---
                        merged_table[change_col_name] = merged_table.apply(
                            lambda r: (r[col_current] - r[col_prev]) if pd.notna(r[col_current]) and pd.notna(r[col_prev]) else np.nan,
                            axis=1
                        )
                    else:
                        # --- Calculate Relative Percentage Change (Original Logic) ---
                        merged_table[change_col_name] = merged_table.apply(
                            lambda r: ((r[col_current] - r[col_prev]) / abs(r[col_prev]) * 100) if pd.notna(r[col_prev]) and r[col_prev] != 0 else np.nan,
                            axis=1
                        )

                    # Ensure inf/-inf generated by relative change are NaN
                    merged_table[change_col_name] = merged_table[change_col_name].replace([np.inf, -np.inf], np.nan)

                    ordered_cols.append(change_col_name); change_cols.append(change_col_name)

        elif actual_years_in_data: # Only one actual year in data after merge
             yr_single = actual_years_in_data[0]; ordered_cols.extend([f"{m} {yr_single}" for m in selected_metrics if f"{m} {yr_single}" in merged_table.columns])
    elif len(years_to_process) == 1: # Only one year selected initially
        yr_single = years_to_process[0]; ordered_cols.extend([f"{m} {yr_single}" for m in selected_metrics if f"{m} {yr_single}" in merged_table.columns])

    # Final column selection and ordering
    ordered_cols = [col for col in ordered_cols if col in merged_table.columns] # Ensure all exist
    merged_table_display = merged_table[ordered_cols].copy() if len(ordered_cols) > 1 else pd.DataFrame({group_by_col: merged_table[group_by_col]})

    # Rename group_by column for display
    final_display_col = display_col_name or group_by_col
    if group_by_col in merged_table_display.columns: merged_table_display = merged_table_display.rename(columns={group_by_col: final_display_col})

    # Sorting logic (Optional) - Sort by last year's first available metric
    sort_col = None
    if actual_years_in_data: sort_col = next((f"{m} {actual_years_in_data[-1]}" for m in selected_metrics if f"{m} {actual_years_in_data[-1]}" in merged_table_display.columns), None)
    if sort_col and sort_col in merged_table_display:
       merged_table_display[sort_col] = pd.to_numeric(merged_table_display[sort_col], errors='coerce')
       merged_table_display = merged_table_display.dropna(subset=[sort_col]).sort_values(sort_col, ascending=False)

    return merged_table_display


# =============================================================================
# <<< MODIFIED >>> Styling Function for YOY Grouped Tables (Using applymap for Color)
# =============================================================================
def style_yoy_comparison_table(df):
    """Styles the YoY comparison table with formats and % change coloring using applymap."""
    if df is None or df.empty:
        return None

    df_copy = df.copy() # Avoid modifying original df passed to styler
    # Replace inf/-inf with NaN *before* formatting attempt
    df_copy = df_copy.replace([np.inf, -np.inf], np.nan)

    format_dict = {}
    highlight_change_cols = []
    # Define the set of metrics that use percentage point difference
    percentage_metrics_for_styling = {"CTR", "CVR", "ACOS", "Ad % Sale"}

    for col in df_copy.columns:
        # Use regex that allows spaces for multi-word metrics like "Ad % Sale"
        base_metric_match = re.match(r"([a-zA-Z\s%]+)", col) # Match metric name (letters, space, %) at the start
        base_metric = base_metric_match.group(1).strip() if base_metric_match else "" # Strip trailing space

        is_change_col = "% Change" in col
        is_metric_col = not is_change_col and any(char.isdigit() for char in col) # Basic check if year suffix exists

        if is_change_col:
            # Get the base metric name from the change column more robustly
            base_metric_for_change = col.replace(" % Change", "").strip()
            # You can probably remove the debug prints now if the logic was confirmed correct
            # print(f"DEBUG: Column='{col}', BaseMetric='{base_metric_for_change}'")

            if base_metric_for_change in percentage_metrics_for_styling:
                # print(f"DEBUG: Applying 'pp' format to {col}")
                # Format absolute difference with "pp"
                format_dict[col] = lambda x: f"{x:+.1f} %" if pd.notna(x) else 'N/A' # Using 'pp'
            else:
                # print(f"DEBUG: Applying '%' format to {col}")
                # Format relative difference with "%"
                format_dict[col] = lambda x: f"{x:+.1f}%" if pd.notna(x) else 'N/A'

            highlight_change_cols.append(col)

        elif is_metric_col: # Apply formatting only to metric columns (with year suffix)
            if base_metric in ["Impressions", "Clicks", "Orders", "Units"]:
                format_dict[col] = "{:,.0f}"
            elif base_metric in ["Spend", "Sales"]:
                format_dict[col] = "${:,.2f}"
            elif base_metric == "CPC":
                format_dict[col] = "${:,.2f}"
            elif base_metric in ["ACOS", "CTR", "CVR", "Ad % Sale"]:
                 if pd.api.types.is_numeric_dtype(df_copy[col].dropna()):
                      format_dict[col] = '{:.1f}%'
                 else:
                      format_dict[col] = '{}'
            elif base_metric == "ROAS":
                 if pd.api.types.is_numeric_dtype(df_copy[col].dropna()):
                      format_dict[col] = '{:.2f}'
                 else:
                      format_dict[col] = '{}'
        elif df_copy[col].dtype == 'object':
            format_dict[col] = "{}"

    # Apply base styling and formatting
    try:
        styled_table = df_copy.style.format(format_dict, na_rep="N/A")
    except Exception as e:
        st.error(f"Error applying format to YOY table: {e}")
        st.dataframe(df_copy) # Display raw data on format error
        return None

    # --- Define the coloring function ---
    def color_pos_neg(val):
        """Applies green to positive, red to negative, grey to NaN/non-numeric."""
        # Attempt to convert to numeric, handling potential errors from formatted strings etc.
        numeric_val = pd.to_numeric(val, errors='coerce')

        if pd.isna(numeric_val):
            return 'color: grey' # Color for NaN or non-numeric strings like 'N/A'
        elif numeric_val > 0:
            return 'color: green'
        elif numeric_val < 0:
            return 'color: red'
        else: # Value is zero
            return 'color: inherit' # Use default text color for zero

    # --- Apply coloring using applymap ---
    # Loop through the identified "% Change" columns
    for change_col in highlight_change_cols:
        if change_col in df_copy.columns:
            try:
                # Apply the color_pos_neg function element-wise to the specific column
                styled_table = styled_table.applymap(
                    color_pos_neg,
                    subset=[change_col]
                )
            except Exception as e:
                # Add more specific error handling if needed
                st.warning(f"Could not apply color styling via applymap to column '{change_col}': {e}")
    # --- End applying coloring ---

    # Apply text alignment (first column left, others right) - reusing existing logic
    cols_to_align = df_copy.columns
    text_align='right'
    if len(cols_to_align) > 0: # Check if there are columns
        try:
            first_col_idx = df_copy.columns.get_loc(cols_to_align[0])
            styles = [
                 {'selector': 'th', 'props': [('text-align', text_align)]}, # Align headers
                 {'selector': 'td', 'props': [('text-align', text_align)]}, # Align all cells
                 {'selector': f'th.col_heading.level0.col{first_col_idx}', 'props': [('text-align', 'left')]}, # Align first header left using index
                 {'selector': f'td.col{first_col_idx}', 'props': [('text-align', 'left')]}  # Align first column cells left using index
                 ]
            styled_table = styled_table.set_table_styles(styles, overwrite=False)
        except Exception as e:
            st.warning(f"Could not apply alignment styles: {e}")
            # Fallback alignment
            styled_table = styled_table.set_properties(**{'text-align': text_align})
    else:
        styled_table = styled_table.set_properties(**{'text-align': text_align})

    return styled_table

# =============================================================================
# <<< MODIFIED >>> Helper Function for YoY Aggregated Summary Rows (% Change Logic)
# =============================================================================
def calculate_yoy_summary_row(df, selected_metrics, years_to_process, id_col_name, id_col_value):
    """
    Calculates a single summary row with YoY comparison based on yearly totals.
    Includes calculation for 'Ad % Sale' based on unique weekly Total Sales.
    Calculates absolute percentage point change for % metrics, relative otherwise.

    Args:
        df (pd.DataFrame): The data to summarize (already filtered by time and potentially product type).
        selected_metrics (list): List of metrics selected by the user to display.
        years_to_process (list): Sorted list of unique years present in the data.
        id_col_name (str): The name for the identifier column (e.g., "Portfolio").
        id_col_value (str): The text to put in the identifier column for the total row.

    Returns:
        pd.DataFrame: A single-row DataFrame with the YoY summary, or empty if error/no data.
    """
    if df is None or df.empty or not years_to_process:
        return pd.DataFrame()

    date_col = "WE Date" if "WE Date" in df.columns else None
    ad_sale_possible = ("Ad % Sale" in selected_metrics and
                       "Sales" in df.columns and
                       "Total Sales" in df.columns and
                       date_col)
    if "Ad % Sale" in selected_metrics and not ad_sale_possible:
        selected_metrics = [m for m in selected_metrics if m != "Ad % Sale"] # Remove if not possible

    summary_row_data = {id_col_name: id_col_value}
    yearly_totals = {yr: {} for yr in years_to_process} # To store sum/calculated metrics per year
    yearly_total_sales_denom = {yr: 0 for yr in years_to_process} # Store the unique total sales denominator per year

    # --- Determine base metrics needed ---
    base_metrics_needed = set()
    for m in selected_metrics:
        if m in ["Impressions", "Clicks", "Spend", "Sales", "Orders", "Units"]: base_metrics_needed.add(m)
        elif m == "CTR": base_metrics_needed.update(["Clicks", "Impressions"])
        elif m == "CVR": base_metrics_needed.update(["Orders", "Clicks"])
        elif m == "CPC": base_metrics_needed.update(["Spend", "Clicks"])
        elif m == "ACOS": base_metrics_needed.update(["Spend", "Sales"])
        elif m == "ROAS": base_metrics_needed.update(["Sales", "Spend"])
        # Ad % Sale only needs 'Sales' for the numerator sum, denominator calculated separately

    # --- Calculate Yearly Totals for Base Metrics AND the Denominator for Ad % Sale ---
    for yr in years_to_process:
        if 'Year' in df.columns and pd.api.types.is_numeric_dtype(df['Year']):
            df_year = df[df["Year"] == yr]
        else:
            df_year = df # Fallback, though likely incorrect for multi-year data
            # st.warning(f"Could not properly filter by year {yr} for summary row.")

        if df_year.empty: continue # Skip if no data for this year in the subset

        # Calculate sums for base metrics
        for base_m in base_metrics_needed:
            if base_m in df_year.columns:
                try:
                    yearly_totals[yr][base_m] = pd.to_numeric(df_year[base_m], errors='coerce').sum()
                except Exception as e:
                    yearly_totals[yr][base_m] = 0
            else:
                yearly_totals[yr][base_m] = 0 # Default if base metric missing

        # Calculate the unique total sales denominator for Ad % Sale for this year
        if ad_sale_possible:
             try:
                 df_year[date_col] = pd.to_datetime(df_year[date_col], errors='coerce')
                 df_year_valid_dates_total = df_year.dropna(subset=[date_col])
                 if not df_year_valid_dates_total.empty:
                     # --- MODIFIED SECTION ---
                     unique_subset_summary = [date_col]
                     if "Marketplace" in df_year_valid_dates_total.columns:
                          unique_subset_summary.append("Marketplace")
                     unique_weekly_totals_for_sum = df_year_valid_dates_total.drop_duplicates(subset=unique_subset_summary)
                     # --- END MODIFIED SECTION ---
                     yearly_total_sales_denom[yr] = pd.to_numeric(unique_weekly_totals_for_sum['Total Sales'], errors='coerce').fillna(0).sum()
             except Exception as e:
                 # st.warning(f"Could not calculate total sales denominator for summary year {yr}: {e}")
                 yearly_total_sales_denom[yr] = 0


    # --- Calculate Yearly Totals for Derived Metrics & Populate summary_row_data ---
    for metric in selected_metrics:
        for yr in years_to_process:
            totals_yr = yearly_totals.get(yr, {}) # Get the dict for the current year

            calculated_value = np.nan # Default
            if metric == "CTR":
                 calculated_value = (totals_yr.get("Clicks", 0) / totals_yr.get("Impressions", 0) * 100) if totals_yr.get("Impressions", 0) > 0 else 0
            elif metric == "CVR":
                 calculated_value = (totals_yr.get("Orders", 0) / totals_yr.get("Clicks", 0) * 100) if totals_yr.get("Clicks", 0) > 0 else 0
            elif metric == "CPC":
                 calculated_value = (totals_yr.get("Spend", 0) / totals_yr.get("Clicks", 0)) if totals_yr.get("Clicks", 0) > 0 else np.nan
            elif metric == "ACOS":
                 calculated_value = (totals_yr.get("Spend", 0) / totals_yr.get("Sales", 0) * 100) if totals_yr.get("Sales", 0) > 0 else np.nan
            elif metric == "ROAS":
                 calculated_value = (totals_yr.get("Sales", 0) / totals_yr.get("Spend", 0)) if totals_yr.get("Spend", 0) > 0 else np.nan
            elif metric == "Ad % Sale": # Use the pre-calculated denominator for this year
                 total_sales_denom_yr = yearly_total_sales_denom.get(yr, 0)
                 if total_sales_denom_yr > 0:
                     calculated_value = (totals_yr.get("Sales", 0) / total_sales_denom_yr * 100)
                 else:
                     calculated_value = np.nan # Denominator is 0 or missing
            elif metric in totals_yr: # It was a base metric already summed
                 calculated_value = totals_yr.get(metric)

            # Handle NaN/inf
            if isinstance(calculated_value, (int, float)):
                if calculated_value in [np.inf, -np.inf]:
                    calculated_value = np.nan

            # Store the final value for the metric and year
            if yr in yearly_totals: # Ensure year dict exists
                yearly_totals[yr][metric] = calculated_value # Update the dictionary
            summary_row_data[f"{metric} {yr}"] = calculated_value # Add to the final row data

    # --- Calculate % Change (if applicable) <<< MODIFIED >>> ---
    if len(years_to_process) >= 2:
        curr_yr, prev_yr = years_to_process[-1], years_to_process[-2]
        # Define the set of metrics that are already percentages
        percentage_metrics = {"CTR", "CVR", "ACOS", "Ad % Sale"}

        for metric in selected_metrics:
            val_curr = yearly_totals.get(curr_yr, {}).get(metric, np.nan)
            val_prev = yearly_totals.get(prev_yr, {}).get(metric, np.nan)

            pct_change = np.nan # Default

            if pd.notna(val_curr) and pd.notna(val_prev):
                if metric in percentage_metrics:
                    # --- Calculate Absolute Percentage Point Difference ---
                    pct_change = val_curr - val_prev
                else:
                    # --- Calculate Relative Percentage Change (Original Logic) ---
                    if val_prev != 0:
                        pct_change = ((val_curr - val_prev) / abs(val_prev)) * 100
                    elif val_curr == 0: # Previous was 0, current is 0 -> 0% change
                        pct_change = 0.0
                    # else: remains NaN (e.g., growth from 0, handle as NaN in summary)

            # Handle cases where only one value exists (remains NaN from default init)

            summary_row_data[f"{metric} % Change"] = pct_change
    # --- End % Change Calculation ---


    # --- Create DataFrame and Order Columns ---
    summary_df = pd.DataFrame([summary_row_data])

    ordered_summary_cols = [id_col_name]
    if len(years_to_process) >= 2:
        curr_yr, prev_yr = years_to_process[-1], years_to_process[-2]
        for metric in selected_metrics: # Iterate based on originally selected (after initial filter)
            ordered_summary_cols.append(f"{metric} {prev_yr}")
            ordered_summary_cols.append(f"{metric} {curr_yr}")
            ordered_summary_cols.append(f"{metric} % Change")
    elif len(years_to_process) == 1:
        yr = years_to_process[0]
        for metric in selected_metrics:
            ordered_summary_cols.append(f"{metric} {yr}")

    final_summary_cols = [col for col in ordered_summary_cols if col in summary_df.columns]
    summary_df = summary_df[final_summary_cols]

    return summary_df

# =============================================================================
# --- Title and Logo ---
# =============================================================================
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Advertising Dashboard 📊")
with col2:
    # Wrap image loading in a try-except block
    try:
        st.image("logo.png", width=300)
    except Exception as e:
        st.warning(f"Could not load logo.png: {e}")


# =============================================================================
# --- Sidebar File Uploader for Advertising Data <<< MODIFIED >>> ---
# =============================================================================
st.sidebar.header("Advertising Data")
advertising_file = st.sidebar.file_uploader("Upload Advertising Data (CSV)", type=["csv"], key="adv_file")
if advertising_file is not None:
    # --- Data Loading and Initial Filtering ---
    try:
        st.session_state["ad_data"] = pd.read_csv(advertising_file)

        # --- Marketplace Selector ---
        if "Marketplace" in st.session_state["ad_data"].columns:
            available_marketplaces = sorted(st.session_state["ad_data"]["Marketplace"].unique())
            # Ensure options are strings for selectbox, handle potential NaN
            available_marketplaces = [str(mp) for mp in available_marketplaces if pd.notna(mp)]

            if available_marketplaces: # Only show selector if marketplaces exist
                # --- MODIFIED SECTION START ---
                # Define the desired default marketplace
                target_default_marketplace = "US"

                # Construct the full list of options for the selectbox
                full_options = ["All Marketplaces"] + available_marketplaces

                # Determine the default index
                default_index = 0 # Start with the default index for "All Marketplaces"
                if target_default_marketplace in full_options:
                    try:
                        # Find the index of the target default marketplace
                        default_index = full_options.index(target_default_marketplace)
                    except ValueError:
                        # This case should technically not happen because of the 'in' check,
                        # but it's safe to leave the default_index at 0.
                        st.sidebar.warning(f"'{target_default_marketplace}' was found but index lookup failed. Defaulting to 'All Marketplaces'.")
                else:
                    # Optional: Warn if the desired default isn't even in the list
                    # st.sidebar.info(f"Default marketplace '{target_default_marketplace}' not found in data. Defaulting to 'All Marketplaces'.")
                    pass # Keep default_index = 0

                # Create the selectbox using the determined default index
                selected_marketplace = st.sidebar.selectbox(
                    "Select Marketplace",
                    options=full_options, # Use the full list here
                    index=default_index,  # Set the calculated default index
                    key="marketplace_selector"
                )
                # --- MODIFIED SECTION END ---

                # Filter data based on selected marketplace
                if selected_marketplace != "All Marketplaces":
                    st.session_state["filtered_ad_data"] = st.session_state["ad_data"][
                        st.session_state["ad_data"]["Marketplace"].astype(str) == selected_marketplace
                    ].copy() # Use .copy() to avoid SettingWithCopyWarning
                else:
                    st.session_state["filtered_ad_data"] = st.session_state["ad_data"].copy()
            else:
                st.sidebar.warning("No valid marketplaces found in 'Marketplace' column.")
                st.session_state["filtered_ad_data"] = st.session_state["ad_data"].copy() # Use all data if no valid marketplaces
        else:
            # If no Marketplace column exists, use all data
            st.session_state["filtered_ad_data"] = st.session_state["ad_data"].copy()

        # --- PREPROCESSING STEP (Now defined above) ---
        # Perform preprocessing ONCE and store it
        if "filtered_ad_data" in st.session_state and not st.session_state["filtered_ad_data"].empty:
            # Call the function which is now defined above this point
            st.session_state["ad_data_processed"] = preprocess_ad_data(st.session_state["filtered_ad_data"])
        else:
            # Handle case where filtered data might be empty even if file loaded
            if "ad_data_processed" in st.session_state: del st.session_state["ad_data_processed"]
        # --- End Preprocessing Step ---

    except Exception as e:
        st.error(f"Error reading or processing CSV file: {e}")
        # Reset session state if error occurs
        if "ad_data" in st.session_state: del st.session_state["ad_data"]
        if "filtered_ad_data" in st.session_state: del st.session_state["filtered_ad_data"]
        if "ad_data_processed" in st.session_state: del st.session_state["ad_data_processed"] # Also clear processed data

# =============================================================================
# Display Dashboard Tabs Only When Data is Uploaded and Processed
# =============================================================================
# Check if processed data exists and is not empty
if "ad_data_processed" in st.session_state and not st.session_state["ad_data_processed"].empty:

    tabs_adv = st.tabs([
        "YOY Comparison",
        "Sponsored Products",
        "Sponsored Brands",
        "Sponsored Display"
    ])

# -------------------------------
    # Tab 0: YOY Comparison <<< MODIFIED >>>
    # -------------------------------
    with tabs_adv[0]:
        st.markdown("### YOY Comparison")
        # Use the processed data stored in session state
        ad_data_overview = st.session_state["ad_data_processed"].copy()

        # ----------------------------------------------------------------
        # Prepare data specifically for YOY tab selectors and initial overview
        # Requires Year and Week columns
        # ----------------------------------------------------------------
        # Ensure required columns 'WE Date', 'Year', 'Week' exist or are created
        if "WE Date" not in ad_data_overview.columns:
             st.error("'WE Date' column missing after preprocessing. Cannot proceed.")
             st.stop()
        else:
            # Add Year/Week if not present (might be added in preprocess already)
            if 'Year' not in ad_data_overview.columns:
                ad_data_overview["Year"] = ad_data_overview["WE Date"].dt.year
            if 'Week' not in ad_data_overview.columns:
                # Use .dt.isocalendar().week which returns a Series
                ad_data_overview["Week"] = ad_data_overview["WE Date"].dt.isocalendar().week

            # Ensure Year/Week are integer types after creation/check, handle potential NaNs
            error_in_prep = False
            for col in ['Year', 'Week']:
                if col in ad_data_overview.columns:
                    ad_data_overview[col] = pd.to_numeric(ad_data_overview[col], errors='coerce')
                    ad_data_overview.dropna(subset=[col], inplace=True)
                    # Check if dataframe is empty after dropna before astype
                    if not ad_data_overview.empty:
                        ad_data_overview[col] = ad_data_overview[col].astype(int)
                    else:
                         st.warning(f"No valid data remaining after cleaning '{col}' column.")
                         error_in_prep = True; break
                else:
                    st.error(f"Column '{col}' could not be prepared for YOY analysis.")
                    error_in_prep = True; break

            if error_in_prep or ad_data_overview.empty:
                st.error("Could not prepare Year/Week columns. Stopping YOY tab.")
                st.stop() # Use st.stop() to halt execution if critical prep fails
            else:
                # --- Selectors ---
                st.markdown("#### Select Comparison Criteria")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    available_years = sorted(ad_data_overview["Year"].unique())
                    default_years = available_years[-2:] if len(available_years) >= 2 else available_years
                    selected_years = st.multiselect("Select Year(s):", available_years, default=default_years, key="yoy_years")
                with col2:
                    timeframe_options = ["Specific Week", "Last 4 Weeks", "Last 8 Weeks", "Last 12 Weeks"]
                    default_tf_index = timeframe_options.index("Last 4 Weeks") if "Last 4 Weeks" in timeframe_options else 0
                    selected_timeframe = st.selectbox("Select Timeframe:", timeframe_options, index=default_tf_index, key="yoy_timeframe")
                with col3:
                    if selected_years:
                         weeks_in_selected_years = ad_data_overview[ad_data_overview["Year"].isin(selected_years)]["Week"].unique()
                         available_weeks = sorted([w for w in weeks_in_selected_years if pd.notna(w)])
                         available_weeks_str = ["Select..."] + [str(int(w)) for w in available_weeks]
                    else: available_weeks_str = ["Select..."]
                    is_specific_week = (selected_timeframe == "Specific Week")
                    selected_week_option = st.selectbox("Select Week:", available_weeks_str, index=0, key="yoy_week", disabled=(not is_specific_week))

                with col4:
                    # <<< MODIFIED Metrics Selection >>>
                    all_metrics = [
                        "Impressions", "Clicks", "Spend", "Sales", "Orders", "Units",
                        "CTR", "CVR", "CPC", "ACOS", "ROAS", "Ad % Sale"
                    ]
                    calculable_metrics = {"CTR", "CVR", "CPC", "ACOS", "ROAS", "Ad % Sale"}
                    required_grouping_cols = {"Product", "Portfolio Name", "Match Type", "RTW/Prospecting", "Campaign Name", "WE Date", "Total Sales"} & set(ad_data_overview.columns)

                    # Available metrics are those directly in columns OR those that can be calculated
                    available_display_metrics = []
                    for m in all_metrics:
                        if m in ad_data_overview.columns:
                            available_display_metrics.append(m)
                        elif m in calculable_metrics:
                            # Check if required base cols for this specific calculable metric exist
                            can_calc_m = False
                            if m == "CTR" and {"Clicks", "Impressions"}.issubset(ad_data_overview.columns): can_calc_m = True
                            elif m == "CVR" and {"Orders", "Clicks"}.issubset(ad_data_overview.columns): can_calc_m = True
                            elif m == "CPC" and {"Spend", "Clicks"}.issubset(ad_data_overview.columns): can_calc_m = True
                            elif m == "ACOS" and {"Spend", "Sales"}.issubset(ad_data_overview.columns): can_calc_m = True
                            elif m == "ROAS" and {"Sales", "Spend"}.issubset(ad_data_overview.columns): can_calc_m = True
                            elif m == "Ad % Sale" and {"Sales", "Total Sales", "WE Date"}.issubset(ad_data_overview.columns): can_calc_m = True # Check specific cols for Ad % Sale
                            if can_calc_m: available_display_metrics.append(m)

                    # --- MODIFIED Default Logic ---
                    # Define the NEW desired default list of metrics
                    default_metrics_list = ["Spend", "Sales", "Ad % Sale", "ACOS"]
                    # Filter the desired default list based on actually available metrics
                    default_metrics = [m for m in default_metrics_list if m in available_display_metrics]
                    # --- End MODIFIED Default Logic ---

                    # Create the multiselect widget
                    selected_metrics = st.multiselect("Select Metrics:",
                                                      available_display_metrics,
                                                      default=default_metrics, # Uses the filtered default list
                                                      key="yoy_metrics")

                    # Fallback logic if user deselects all
                    if not selected_metrics:
                        st.warning("Please select at least one metric.")
                        # Fallback: try to use the desired defaults that ARE available,
                        # otherwise use the first available metric overall.
                        fallback_defaults = default_metrics[:2] if len(default_metrics) >= 2 else default_metrics[:1] if default_metrics else available_display_metrics[:1]
                        selected_metrics = fallback_defaults
                    # <<< End MODIFIED Metrics Selection >>>


                selected_week = int(selected_week_option) if is_specific_week and selected_week_option != "Select..." else None

                # --- Display Section ---
                if not selected_years:
                     st.warning("Please select at least one year.")
                else:
                    # ----------------------------------------------------------------
                    # Filter data using the CORRECTED function based on top selections
                    # ----------------------------------------------------------------
                    filtered_data_for_tables = filter_data_by_timeframe(ad_data_overview, selected_years, selected_timeframe, selected_week)

                    if filtered_data_for_tables.empty:
                        st.info("No data available for the selected criteria (Years/Timeframe).")
                    else:
                        # Determine unique years present *after* filtering
                        years_to_process = sorted(filtered_data_for_tables['Year'].unique())

                        # ----------------------------------------------------------------
                        # Detailed Overview by Product Type (Multi-Year)
                        # ----------------------------------------------------------------
                        st.markdown("---")
                        st.markdown("#### Overview by Product Type")
                        st.caption("*Aggregated data for selected years/timeframe, showing only selected metrics.*")

                        # Call the UPDATED helper function
                        product_overview_yoy_table = create_yoy_grouped_table(
                            df_filtered_period=filtered_data_for_tables,
                            group_by_col="Product",
                            selected_metrics=selected_metrics,
                            years_to_process=years_to_process,
                            display_col_name="Product"
                        )

                        # Style and display using the UPDATED styling function
                        if not product_overview_yoy_table.empty:
                            styled_product_overview_yoy = style_yoy_comparison_table(product_overview_yoy_table)
                            if styled_product_overview_yoy:
                                st.dataframe(styled_product_overview_yoy, use_container_width=True)
                        else:
                            st.info("No product overview data available for the selected criteria.")


                        # ----------------------------------------------------------------
                        # Portfolio Performance Table
                        # ----------------------------------------------------------------
                        portfolio_col_name = next((col for col in ["Portfolio Name", "Portfolio", "PortfolioName", "Portfolio_Name"] if col in filtered_data_for_tables.columns), None)

                        if portfolio_col_name and not filtered_data_for_tables.empty:
                            st.markdown("---")
                            st.markdown("#### Portfolio Performance")
                            st.caption("*Aggregated data for selected years/timeframe, showing only selected metrics. Optionally filter by Product Type below.*")

                            if "Product" in filtered_data_for_tables.columns:
                                product_types_portfolio = ["All"] + sorted(filtered_data_for_tables["Product"].unique().tolist())
                                selected_product_type_portfolio = st.selectbox(
                                    "Filter Portfolio Table by Product Type:",
                                    product_types_portfolio, index=0, key="portfolio_product_filter_yoy"
                                )
                                portfolio_table_data = filtered_data_for_tables.copy()
                                if selected_product_type_portfolio != "All":
                                    portfolio_table_data = portfolio_table_data[portfolio_table_data["Product"] == selected_product_type_portfolio]
                            else:
                                st.warning("Cannot filter Portfolio Table by Product Type ('Product' column missing).")
                                portfolio_table_data = filtered_data_for_tables.copy()

                            if portfolio_table_data.empty:
                                if selected_product_type_portfolio != "All" and "Product" in filtered_data_for_tables.columns:
                                    st.info(f"No Portfolio data available for Product Type '{selected_product_type_portfolio}' in the selected period.")
                            else:
                                portfolio_yoy_table = create_yoy_grouped_table(
                                    df_filtered_period=portfolio_table_data,
                                    group_by_col=portfolio_col_name,
                                    selected_metrics=selected_metrics,
                                    years_to_process=years_to_process,
                                    display_col_name="Portfolio"
                                )
                                if not portfolio_yoy_table.empty:
                                    styled_portfolio_yoy = style_yoy_comparison_table(portfolio_yoy_table)
                                    if styled_portfolio_yoy: st.dataframe(styled_portfolio_yoy, use_container_width=True)

                                    # --- Portfolio Summary Row ---
                                    portfolio_summary_row = calculate_yoy_summary_row(
                                        df=portfolio_table_data,
                                        selected_metrics=selected_metrics,
                                        years_to_process=years_to_process,
                                        id_col_name="Portfolio",
                                        id_col_value=f"TOTAL - {selected_product_type_portfolio}"
                                    )
                                    if not portfolio_summary_row.empty:
                                        st.markdown("###### YoY Total (Selected Period & Product Filter)")
                                        styled_portfolio_summary = style_yoy_comparison_table(portfolio_summary_row)
                                        if styled_portfolio_summary:
                                            st.dataframe(styled_portfolio_summary.set_properties(**{'font-weight': 'bold'}), use_container_width=True)
                                    # --- End Summary Row ---
                                else:
                                    st.info(f"No displayable portfolio data for Product Type '{selected_product_type_portfolio}' after processing.")

                        elif not filtered_data_for_tables.empty:
                            st.info("Portfolio analysis requires a 'Portfolio Name' column.")


                        # ----------------------------------------------------------------
                        # Match Type Performance Analysis
                        # ----------------------------------------------------------------
                        if {"Product", "Match Type"}.issubset(filtered_data_for_tables.columns) and not filtered_data_for_tables.empty:
                            st.markdown("---")
                            st.markdown("#### Match Type Performance")
                            st.caption("*Aggregated data for selected years/timeframe, showing only selected metrics, broken down by Product Type.*")
                            product_types_match = ["Sponsored Products", "Sponsored Brands", "Sponsored Display"]
                            for product_type in product_types_match:
                                product_data_match = filtered_data_for_tables[filtered_data_for_tables["Product"] == product_type].copy()
                                if product_data_match.empty: continue

                                st.subheader(product_type)
                                match_type_yoy_table = create_yoy_grouped_table(
                                    df_filtered_period=product_data_match,
                                    group_by_col="Match Type",
                                    selected_metrics=selected_metrics,
                                    years_to_process=years_to_process,
                                    display_col_name="Match Type"
                                )
                                if not match_type_yoy_table.empty:
                                    styled_match_type_yoy = style_yoy_comparison_table(match_type_yoy_table)
                                    if styled_match_type_yoy: st.dataframe(styled_match_type_yoy, use_container_width=True)

                                    # --- Match Type Summary Row ---
                                    match_type_summary_row = calculate_yoy_summary_row(
                                        df=product_data_match,
                                        selected_metrics=selected_metrics,
                                        years_to_process=years_to_process,
                                        id_col_name="Match Type",
                                        id_col_value=f"TOTAL - {product_type}"
                                    )
                                    if not match_type_summary_row.empty:
                                        st.markdown("###### YoY Total (Selected Period)")
                                        styled_match_type_summary = style_yoy_comparison_table(match_type_summary_row)
                                        if styled_match_type_summary:
                                            st.dataframe(styled_match_type_summary.set_properties(**{'font-weight': 'bold'}), use_container_width=True)
                                    # --- End Summary Row ---
                                else:
                                    st.info(f"No Match Type data available to display for {product_type} based on selected metrics.")

                        elif not filtered_data_for_tables.empty:
                            st.info("Match Type analysis requires 'Product' and 'Match Type' columns.")


                        # ----------------------------------------------------------------
                        # RTW/Prospecting Performance Analysis
                        # ----------------------------------------------------------------
                        if {"Product", "RTW/Prospecting"}.issubset(filtered_data_for_tables.columns) and not filtered_data_for_tables.empty:
                            st.markdown("---")
                            st.markdown("#### RTW/Prospecting Performance")
                            st.caption("*Aggregated data for selected years/timeframe, showing only selected metrics. Choose a Product Type below.*")
                            rtw_product_types = ["Sponsored Products", "Sponsored Brands", "Sponsored Display"]
                            available_rtw_products = sorted([pt for pt in filtered_data_for_tables["Product"].unique() if pt in rtw_product_types])

                            if available_rtw_products:
                                selected_rtw_product = st.selectbox(
                                    "Select Product Type for RTW/Prospecting Analysis:",
                                    options=available_rtw_products, key="rtw_product_selector_yoy"
                                )
                                rtw_filtered_product_data = filtered_data_for_tables[filtered_data_for_tables["Product"] == selected_rtw_product].copy()

                                if not rtw_filtered_product_data.empty:
                                    rtw_yoy_table = create_yoy_grouped_table(
                                        df_filtered_period=rtw_filtered_product_data,
                                        group_by_col="RTW/Prospecting",
                                        selected_metrics=selected_metrics,
                                        years_to_process=years_to_process,
                                        display_col_name="RTW/Prospecting"
                                    )
                                    if not rtw_yoy_table.empty:
                                        styled_rtw_yoy = style_yoy_comparison_table(rtw_yoy_table)
                                        if styled_rtw_yoy: st.dataframe(styled_rtw_yoy, use_container_width=True)

                                        # --- RTW Summary Row ---
                                        rtw_summary_row = calculate_yoy_summary_row(
                                            df=rtw_filtered_product_data,
                                            selected_metrics=selected_metrics,
                                            years_to_process=years_to_process,
                                            id_col_name="RTW/Prospecting",
                                            id_col_value=f"TOTAL - {selected_rtw_product}"
                                        )
                                        if not rtw_summary_row.empty:
                                            st.markdown("###### YoY Total (Selected Period)")
                                            styled_rtw_summary = style_yoy_comparison_table(rtw_summary_row)
                                            if styled_rtw_summary:
                                                st.dataframe(styled_rtw_summary.set_properties(**{'font-weight': 'bold'}), use_container_width=True)
                                        # --- End Summary Row ---
                                    else:
                                        st.info(f"No RTW/Prospecting data available to display for {selected_rtw_product} based on selected metrics.")
                                else:
                                    st.info(f"No {selected_rtw_product} data found in the selected period for RTW analysis.")
                            else:
                                st.info("No Product Types with RTW/Prospecting data available in the selected period.")

                        elif not filtered_data_for_tables.empty:
                            st.info("RTW/Prospecting analysis requires 'Product' and 'RTW/Prospecting' columns.")


                        # ----------------------------------------------------------------
                        # Campaign Performance Table
                        # ----------------------------------------------------------------
                        campaign_col_name = "Campaign Name" # Adjust if your column is named differently
                        if campaign_col_name in filtered_data_for_tables.columns and not filtered_data_for_tables.empty:
                            st.markdown("---")
                            st.markdown(f"#### {campaign_col_name} Performance")
                            st.caption("*Aggregated data for selected years/timeframe, showing only selected metrics.*")

                            campaign_yoy_table = create_yoy_grouped_table(
                                df_filtered_period=filtered_data_for_tables,
                                group_by_col=campaign_col_name,
                                selected_metrics=selected_metrics,
                                years_to_process=years_to_process,
                                display_col_name="Campaign"
                            )
                            if not campaign_yoy_table.empty:
                                styled_campaign_yoy = style_yoy_comparison_table(campaign_yoy_table)
                                if styled_campaign_yoy:
                                    st.dataframe(styled_campaign_yoy, use_container_width=True, height=600)

                                # --- Campaign Summary Row ---
                                campaign_summary_row = calculate_yoy_summary_row(
                                    df=filtered_data_for_tables,
                                    selected_metrics=selected_metrics,
                                    years_to_process=years_to_process,
                                    id_col_name="Campaign",
                                    id_col_value="TOTAL - All Campaigns"
                                )
                                if not campaign_summary_row.empty:
                                    st.markdown("###### YoY Total (Selected Period)")
                                    styled_campaign_summary = style_yoy_comparison_table(campaign_summary_row)
                                    if styled_campaign_summary:
                                        st.dataframe(styled_campaign_summary.set_properties(**{'font-weight': 'bold'}), use_container_width=True)
                                # --- End Summary Row ---
                            else:
                                st.info(f"No displayable {campaign_col_name} data after processing (check metric availability).")

                        elif not filtered_data_for_tables.empty:
                            st.info(f"Campaign-level analysis requires a '{campaign_col_name}' column.")

    # -------------------------------
    # Tab 1: Sponsored Products
    # -------------------------------
    with tabs_adv[1]:
        st.markdown("### Sponsored Products Performance")
        st.caption("Charts below use the filters specific to this tab. Tables show a standard summary for the selected date range.")
        # Use processed data from session state
        if "ad_data_processed" in st.session_state and not st.session_state["ad_data_processed"].empty:
            ad_data_sp = st.session_state["ad_data_processed"].copy()
            product_type_sp = "Sponsored Products"

            if "Product" in ad_data_sp.columns and product_type_sp in ad_data_sp["Product"].unique():
                ad_data_sp_filtered_initial = ad_data_sp[ad_data_sp["Product"] == product_type_sp].copy()

                if ad_data_sp_filtered_initial.empty:
                    st.warning(f"No {product_type_sp} data found after initial filtering.")
                else:
                    # --- Filters for SP Tab ---
                    with st.expander("Filters", expanded=True):
                        col1_sp, col2_sp = st.columns(2)
                        with col1_sp:
                            available_metrics_sp = ["Impressions", "Clicks", "Spend", "Sales", "Orders", "Units", "CTR", "CVR", "ACOS", "ROAS", "CPC"]
                            metrics_exist_sp = [m for m in available_metrics_sp if m in ad_data_sp_filtered_initial.columns or m in ["CTR", "CVR", "ACOS", "ROAS", "CPC"]]
                            sel_metric_index_sp = 0 if metrics_exist_sp else -1
                            if sel_metric_index_sp != -1:
                                selected_metric_sp = st.selectbox("Select Metric for Charts", options=metrics_exist_sp, index=sel_metric_index_sp, key="sp_metric")
                            else:
                                st.warning("No metrics available for selection in SP tab."); selected_metric_sp = None
                        with col2_sp:
                            if "Portfolio Name" not in ad_data_sp_filtered_initial.columns:
                                st.warning("Missing 'Portfolio Name' column for SP tab filtering.")
                                portfolio_options_sp = ["All Portfolios"]; selected_portfolio_sp = "All Portfolios"
                            else:
                                ad_data_sp_filtered_initial["Portfolio Name"] = ad_data_sp_filtered_initial["Portfolio Name"].fillna("Unknown Portfolio")
                                portfolio_options_sp = ["All Portfolios"] + sorted(ad_data_sp_filtered_initial["Portfolio Name"].unique().tolist())
                                selected_portfolio_sp = st.selectbox("Select Portfolio", options=portfolio_options_sp, index=0, key="sp_portfolio")

                        show_yoy_sp = st.checkbox("Show Year-over-Year Comparison (Weekly Points)", value=True, key="sp_show_yoy")

                        date_range_sp = None
                        if "WE Date" in ad_data_sp_filtered_initial.columns and not ad_data_sp_filtered_initial["WE Date"].dropna().empty:
                            min_date_sp = ad_data_sp_filtered_initial["WE Date"].min().date()
                            max_date_sp = ad_data_sp_filtered_initial["WE Date"].max().date()
                            if min_date_sp <= max_date_sp:
                                date_range_sp = st.date_input("Select Date Range", value=(min_date_sp, max_date_sp), min_value=min_date_sp, max_value=max_date_sp, key="sp_date_range")
                            else: st.warning("Invalid date range found in SP data.")
                        else: st.warning("Cannot determine date range for SP tab.")

                    # Apply Date Range Filter
                    ad_data_sp_date_filtered = ad_data_sp_filtered_initial.copy()
                    if date_range_sp and len(date_range_sp) == 2:
                        start_date_sp, end_date_sp = date_range_sp
                        ad_data_sp_date_filtered = ad_data_sp_date_filtered[(ad_data_sp_date_filtered["WE Date"].dt.date >= start_date_sp) & (ad_data_sp_date_filtered["WE Date"].dt.date <= end_date_sp)]

                    # Display Charts and Tables
                    if ad_data_sp_date_filtered.empty:
                        st.warning("No Sponsored Products data available for the selected filters.")
                    elif selected_metric_sp is None:
                        st.warning("Please select a metric to visualize the charts.")
                    else:
                        st.subheader(f"{selected_metric_sp} Over Time")
                        fig1_sp = create_metric_over_time_chart(ad_data_sp_date_filtered, selected_metric_sp, selected_portfolio_sp, product_type_sp, show_yoy=show_yoy_sp)
                        st.plotly_chart(fig1_sp, use_container_width=True, key="sp_time_chart")

                        if selected_portfolio_sp == "All Portfolios":
                            st.subheader(f"{selected_metric_sp} by Portfolio")
                            fig2_sp = create_metric_comparison_chart(ad_data_sp_date_filtered, selected_metric_sp, None, product_type_sp)
                            st.plotly_chart(fig2_sp, use_container_width=True, key="sp_portfolio_chart")

                        st.subheader("Performance Summary")
                        metrics_table_sp, total_summary_sp = create_performance_metrics_table(ad_data_sp_date_filtered, selected_portfolio_sp, product_type_sp)

                        if not total_summary_sp.empty:
                            st.markdown("###### Overall Totals (Selected Period)")
                            styled_total_sp = style_total_summary(total_summary_sp)
                            if styled_total_sp: st.dataframe(styled_total_sp, use_container_width=True)

                        if not metrics_table_sp.empty:
                            st.markdown("###### Performance by Portfolio (Selected Period)")
                            styled_metrics_sp = style_metrics_table(metrics_table_sp)
                            if styled_metrics_sp: st.dataframe(styled_metrics_sp, use_container_width=True)
                        else: st.info("No portfolio breakdown available for the current selection.")

                        st.subheader("Key Insights (Selected Period)")
                        if not total_summary_sp.empty:
                            insights_sp = generate_insights(total_summary_sp.iloc[0], product_type_sp)
                            for insight in insights_sp: st.markdown(f"- {insight}")
                        else: st.info("No summary data to generate insights.")
            else:
                st.warning(f"No {product_type_sp} data found in the uploaded file or 'Product' column missing.")

    # -------------------------------
    # Tab 2: Sponsored Brands
    # -------------------------------
    with tabs_adv[2]:
        st.markdown("### Sponsored Brands Performance")
        st.caption("Charts below use the filters specific to this tab. Tables show a standard summary for the selected date range.")
        # Use processed data from session state
        if "ad_data_processed" in st.session_state and not st.session_state["ad_data_processed"].empty:
            ad_data_sb = st.session_state["ad_data_processed"].copy()
            product_type_sb = "Sponsored Brands"

            if "Product" in ad_data_sb.columns and product_type_sb in ad_data_sb["Product"].unique():
                ad_data_sb_filtered_initial = ad_data_sb[ad_data_sb["Product"] == product_type_sb].copy()

                if ad_data_sb_filtered_initial.empty:
                    st.warning(f"No {product_type_sb} data found after initial filtering.")
                else:
                    # --- Filters for SB Tab ---
                    with st.expander("Filters", expanded=True):
                        col1_sb, col2_sb = st.columns(2)
                        with col1_sb:
                            available_metrics_sb = ["Impressions", "Clicks", "Spend", "Sales", "Orders", "Units", "CTR", "CVR", "ACOS", "ROAS", "CPC"]
                            metrics_exist_sb = [m for m in available_metrics_sb if m in ad_data_sb_filtered_initial.columns or m in ["CTR", "CVR", "ACOS", "ROAS", "CPC"]]
                            sel_metric_index_sb = 0 if metrics_exist_sb else -1
                            if sel_metric_index_sb != -1:
                                selected_metric_sb = st.selectbox("Select Metric for Charts", options=metrics_exist_sb, index=sel_metric_index_sb, key="sb_metric")
                            else:
                                st.warning("No metrics available for selection in SB tab."); selected_metric_sb = None
                        with col2_sb:
                            if "Portfolio Name" not in ad_data_sb_filtered_initial.columns:
                                st.warning("Missing 'Portfolio Name' column for SB tab filtering.")
                                portfolio_options_sb = ["All Portfolios"]; selected_portfolio_sb = "All Portfolios"
                            else:
                                ad_data_sb_filtered_initial["Portfolio Name"] = ad_data_sb_filtered_initial["Portfolio Name"].fillna("Unknown")
                                portfolio_options_sb = ["All Portfolios"] + sorted(ad_data_sb_filtered_initial["Portfolio Name"].unique().tolist())
                                selected_portfolio_sb = st.selectbox("Select Portfolio", options=portfolio_options_sb, index=0, key="sb_portfolio")

                        show_yoy_sb = st.checkbox("Show Year-over-Year Comparison (Weekly Points)", value=True, key="sb_show_yoy")

                        date_range_sb = None
                        if "WE Date" in ad_data_sb_filtered_initial.columns and not ad_data_sb_filtered_initial["WE Date"].dropna().empty:
                            min_date_sb = ad_data_sb_filtered_initial["WE Date"].min().date()
                            max_date_sb = ad_data_sb_filtered_initial["WE Date"].max().date()
                            if min_date_sb <= max_date_sb:
                                date_range_sb = st.date_input("Select Date Range", value=(min_date_sb, max_date_sb), min_value=min_date_sb, max_value=max_date_sb, key="sb_date_range")
                            else: st.warning("Invalid date range found in SB data.")
                        else: st.warning("Cannot determine date range for SB tab.")

                    # Apply Date Range Filter
                    ad_data_sb_date_filtered = ad_data_sb_filtered_initial.copy()
                    if date_range_sb and len(date_range_sb) == 2:
                        start_date_sb, end_date_sb = date_range_sb
                        ad_data_sb_date_filtered = ad_data_sb_date_filtered[(ad_data_sb_date_filtered["WE Date"].dt.date >= start_date_sb) & (ad_data_sb_date_filtered["WE Date"].dt.date <= end_date_sb)]

                    # Display Charts and Tables
                    if ad_data_sb_date_filtered.empty:
                        st.warning("No Sponsored Brands data available for the selected filters.")
                    elif selected_metric_sb is None:
                        st.warning("Please select a metric to visualize the charts.")
                    else:
                        st.subheader(f"{selected_metric_sb} Over Time")
                        fig1_sb = create_metric_over_time_chart(ad_data_sb_date_filtered, selected_metric_sb, selected_portfolio_sb, product_type_sb, show_yoy=show_yoy_sb)
                        st.plotly_chart(fig1_sb, use_container_width=True, key="sb_time_chart")

                        if selected_portfolio_sb == "All Portfolios":
                            st.subheader(f"{selected_metric_sb} by Portfolio")
                            fig2_sb = create_metric_comparison_chart(ad_data_sb_date_filtered, selected_metric_sb, None, product_type_sb)
                            st.plotly_chart(fig2_sb, use_container_width=True, key="sb_portfolio_chart")

                        st.subheader("Performance Summary")
                        metrics_table_sb, total_summary_sb = create_performance_metrics_table(ad_data_sb_date_filtered, selected_portfolio_sb, product_type_sb)

                        if not total_summary_sb.empty:
                            st.markdown("###### Overall Totals (Selected Period)")
                            styled_total_sb = style_total_summary(total_summary_sb)
                            if styled_total_sb: st.dataframe(styled_total_sb, use_container_width=True)

                        if not metrics_table_sb.empty:
                            st.markdown("###### Performance by Portfolio (Selected Period)")
                            styled_metrics_sb = style_metrics_table(metrics_table_sb)
                            if styled_metrics_sb: st.dataframe(styled_metrics_sb, use_container_width=True)
                        else: st.info("No portfolio breakdown available.")

                        st.subheader("Key Insights (Selected Period)")
                        if not total_summary_sb.empty:
                            insights_sb = generate_insights(total_summary_sb.iloc[0], product_type_sb)
                            for insight in insights_sb: st.markdown(f"- {insight}")
                        else: st.info("No summary data for insights.")
            else:
                st.warning(f"No {product_type_sb} data found in the uploaded file or 'Product' column missing.")
        else:
            st.warning("Could not load processed data for SB tab.")


    # -------------------------------
    # Tab 3: Sponsored Display
    # -------------------------------
    with tabs_adv[3]:
        st.markdown("### Sponsored Display Performance")
        st.caption("Charts below use the filters specific to this tab. Tables show a standard summary for the selected date range.")
        # Use processed data from session state
        if "ad_data_processed" in st.session_state and not st.session_state["ad_data_processed"].empty:
            ad_data_sd = st.session_state["ad_data_processed"].copy()
            product_type_sd = "Sponsored Display"

            if "Product" in ad_data_sd.columns and product_type_sd in ad_data_sd["Product"].unique():
                ad_data_sd_filtered_initial = ad_data_sd[ad_data_sd["Product"] == product_type_sd].copy()

                if ad_data_sd_filtered_initial.empty:
                     st.warning(f"No {product_type_sd} data found after initial filtering.")
                else:
                    # --- Filters for SD Tab ---
                    with st.expander("Filters", expanded=True):
                        col1_sd, col2_sd = st.columns(2)
                        with col1_sd:
                            available_metrics_sd = ["Impressions", "Clicks", "Spend", "Sales", "Orders", "Units", "CTR", "CVR", "ACOS", "ROAS", "CPC"]
                            metrics_exist_sd = [m for m in available_metrics_sd if m in ad_data_sd_filtered_initial.columns or m in ["CTR", "CVR", "ACOS", "ROAS", "CPC"]]
                            sel_metric_index_sd = 0 if metrics_exist_sd else -1
                            if sel_metric_index_sd != -1:
                                selected_metric_sd = st.selectbox("Select Metric for Charts", options=metrics_exist_sd, index=sel_metric_index_sd, key="sd_metric")
                            else:
                                st.warning("No metrics available for selection in SD tab."); selected_metric_sd = None
                        with col2_sd:
                            if "Portfolio Name" not in ad_data_sd_filtered_initial.columns:
                                st.warning("Missing 'Portfolio Name' column for SD tab filtering.")
                                portfolio_options_sd = ["All Portfolios"]; selected_portfolio_sd = "All Portfolios"
                            else:
                                ad_data_sd_filtered_initial["Portfolio Name"] = ad_data_sd_filtered_initial["Portfolio Name"].fillna("Unknown")
                                portfolio_options_sd = ["All Portfolios"] + sorted(ad_data_sd_filtered_initial["Portfolio Name"].unique().tolist())
                                selected_portfolio_sd = st.selectbox("Select Portfolio", options=portfolio_options_sd, index=0, key="sd_portfolio")

                        show_yoy_sd = st.checkbox("Show Year-over-Year Comparison (Weekly Points)", value=True, key="sd_show_yoy")

                        date_range_sd = None
                        if "WE Date" in ad_data_sd_filtered_initial.columns and not ad_data_sd_filtered_initial["WE Date"].dropna().empty:
                            min_date_sd = ad_data_sd_filtered_initial["WE Date"].min().date()
                            max_date_sd = ad_data_sd_filtered_initial["WE Date"].max().date()
                            if min_date_sd <= max_date_sd:
                                date_range_sd = st.date_input("Select Date Range", value=(min_date_sd, max_date_sd), min_value=min_date_sd, max_value=max_date_sd, key="sd_date_range")
                            else: st.warning("Invalid date range found in SD data.")
                        else: st.warning("Cannot determine date range for SD tab.")

                    # Apply Date Range Filter
                    ad_data_sd_date_filtered = ad_data_sd_filtered_initial.copy()
                    if date_range_sd and len(date_range_sd) == 2:
                        start_date_sd, end_date_sd = date_range_sd
                        ad_data_sd_date_filtered = ad_data_sd_date_filtered[(ad_data_sd_date_filtered["WE Date"].dt.date >= start_date_sd) & (ad_data_sd_date_filtered["WE Date"].dt.date <= end_date_sd)]

                    # Display Charts and Tables
                    if ad_data_sd_date_filtered.empty:
                        st.warning("No Sponsored Display data available for the selected filters.")
                    elif selected_metric_sd is None:
                        st.warning("Please select a metric to visualize the charts.")
                    else:
                        st.subheader(f"{selected_metric_sd} Over Time")
                        fig1_sd = create_metric_over_time_chart(ad_data_sd_date_filtered, selected_metric_sd, selected_portfolio_sd, product_type_sd, show_yoy=show_yoy_sd)
                        st.plotly_chart(fig1_sd, use_container_width=True, key="sd_time_chart")

                        if selected_portfolio_sd == "All Portfolios":
                            st.subheader(f"{selected_metric_sd} by Portfolio")
                            fig2_sd = create_metric_comparison_chart(ad_data_sd_date_filtered, selected_metric_sd, None, product_type_sd)
                            st.plotly_chart(fig2_sd, use_container_width=True, key="sd_portfolio_chart")

                        st.subheader("Performance Summary")
                        metrics_table_sd, total_summary_sd = create_performance_metrics_table(ad_data_sd_date_filtered, selected_portfolio_sd, product_type_sd)

                        if not total_summary_sd.empty:
                            st.markdown("###### Overall Totals (Selected Period)")
                            styled_total_sd = style_total_summary(total_summary_sd)
                            if styled_total_sd: st.dataframe(styled_total_sd, use_container_width=True)

                        if not metrics_table_sd.empty:
                            st.markdown("###### Performance by Portfolio (Selected Period)")
                            styled_metrics_sd = style_metrics_table(metrics_table_sd)
                            if styled_metrics_sd: st.dataframe(styled_metrics_sd, use_container_width=True)
                        else: st.info("No portfolio breakdown available.")

                        st.subheader("Key Insights (Selected Period)")
                        if not total_summary_sd.empty:
                            insights_sd = generate_insights(total_summary_sd.iloc[0], product_type_sd)
                            for insight in insights_sd: st.markdown(f"- {insight}")
                        else: st.info("No summary data for insights.")
            else:
                st.warning(f"No {product_type_sd} data found in the uploaded file or 'Product' column missing.")
        else:
             st.warning("Could not load processed data for SD tab.")

# Display initial message if no data is loaded/processed yet
elif advertising_file is None:
    st.info("Please upload an Advertising Data CSV file using the sidebar to begin.")
else: # File uploaded but something went wrong during initial load/filter/preprocess
    st.warning("Please check the uploaded file or sidebar selections. No data available to display.")
