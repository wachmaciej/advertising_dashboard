import streamlit as st
import pandas as pd
import warnings
import plotly.express as px
import datetime
import calendar
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import re
import io
from functools import reduce

# Filter warnings for a clean output
warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(
    page_title="YOY Dashboard - Advertising Data",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

# --- Title and Logo ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Advertising Dashboard 📊")
with col2:
    st.image("logo.png", width=300)

# --- Sidebar File Uploader for Advertising Data ---
st.sidebar.header("Advertising Data")
advertising_file = st.sidebar.file_uploader("Upload Advertising Data (CSV)", type=["csv"], key="adv_file")
if advertising_file is not None:
    st.session_state["ad_data"] = pd.read_csv(advertising_file)

# =============================================================================
# Common Functions for Advertising Data
# =============================================================================
def preprocess_ad_data(df):
    """Preprocess advertising data for analysis"""
    df["WE Date"] = pd.to_datetime(df["WE Date"], format="%d/%m/%Y", dayfirst=True)
    df = df.sort_values("WE Date")
    numeric_cols = [
        "Impressions", "Clicks", "Spend", "Sales", "Orders", "Units",
        "CTR", "CVR", "Orders %", "Spend %", "Sales %", "ACOS", "ROAS"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def filter_data_by_timeframe(df, selected_years, selected_timeframe, selected_week):
    """
    For each selected year, filter data based on:
      - "Specific Week" => only that week.
      - Otherwise, filter for the last X weeks (e.g. 4, 8, 12) using the maximum week in that year.
    Returns a concatenated dataframe across the selected years.
    """
    filtered = pd.DataFrame()
    for yr in selected_years:
        df_year = df[df["Year"] == yr].copy()
        if selected_timeframe == "Specific Week":
            df_year = df_year[df_year["Week"] == selected_week]
            if selected_week is not None:
                df_year = df_year[df_year["Week"] == selected_week]
        else:
            weeks_to_filter = int(selected_timeframe.split()[1])  # gets 4, 8, or 12
            if not df_year.empty:
                max_week = df_year["Week"].max()
                df_year = df_year[df_year["Week"] >= (max_week - weeks_to_filter + 1)]
        filtered = pd.concat([filtered, df_year], ignore_index=True)
    return filtered

def create_metric_comparison_chart(df, metric, portfolio_name=None, campaign_type="Sponsored Products"):
    filtered_df = df[df["Product"] == campaign_type].copy()
    if portfolio_name and portfolio_name != "All Portfolios":
        filtered_df = filtered_df[filtered_df["Portfolio Name"] == portfolio_name]
    if metric in ["CTR", "CVR", "ACOS"]:
        if metric == "CTR":
            portfolio_agg = filtered_df.groupby("Portfolio Name").agg({
                "Clicks": "sum",
                "Impressions": "sum"
            }).reset_index()
            portfolio_agg[metric] = (portfolio_agg["Clicks"] / portfolio_agg["Impressions"] * 100).round(2)
        elif metric == "CVR":
            portfolio_agg = filtered_df.groupby("Portfolio Name").agg({
                "Orders": "sum",
                "Clicks": "sum"
            }).reset_index()
            portfolio_agg[metric] = (portfolio_agg["Orders"] / portfolio_agg["Clicks"] * 100).round(2)
        elif metric == "ACOS":
            portfolio_agg = filtered_df.groupby("Portfolio Name").agg({
                "Spend": "sum",
                "Sales": "sum"
            }).reset_index()
            portfolio_agg[metric] = (portfolio_agg["Spend"] / portfolio_agg["Sales"] * 100).round(2)
        grouped_df = portfolio_agg
    else:
        grouped_df = filtered_df.groupby("Portfolio Name").agg({
            metric: "sum"
        }).reset_index()
    grouped_df = grouped_df.sort_values(metric, ascending=False)
    fig = px.bar(
        grouped_df,
        x="Portfolio Name",
        y=metric,
        title=f"{metric} by Portfolio" + (f" - {portfolio_name}" if portfolio_name and portfolio_name != "All Portfolios" else ""),
        text_auto=True
    )
    if metric in ["Spend", "Sales"]:
        fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",.2f")
    elif metric in ["CTR", "CVR", "ACOS"]:
        fig.update_layout(yaxis_ticksuffix="%", yaxis_tickformat=".2f")
    fig.update_layout(margin=dict(t=50, b=50))
    return fig

def create_performance_metrics_table(df, portfolio_name=None, campaign_type="Sponsored Products"):
    filtered_df = df[df["Product"] == campaign_type].copy()
    if portfolio_name and portfolio_name != "All Portfolios":
        filtered_df = filtered_df[filtered_df["Portfolio Name"] == portfolio_name]
    metrics_by_portfolio = filtered_df.groupby("Portfolio Name").agg({
        "Impressions": "sum",
        "Clicks": "sum",
        "Spend": "sum",
        "Sales": "sum",
        "Orders": "sum"
    }).reset_index()
    metrics_by_portfolio["CTR"] = (metrics_by_portfolio["Clicks"] / metrics_by_portfolio["Impressions"] * 100).round(1)
    metrics_by_portfolio["CVR"] = (metrics_by_portfolio["Orders"] / metrics_by_portfolio["Clicks"] * 100).round(1)
    metrics_by_portfolio["ACOS"] = (metrics_by_portfolio["Spend"] / metrics_by_portfolio["Sales"] * 100).round(1)
    metrics_by_portfolio["ROAS"] = (metrics_by_portfolio["Sales"] / metrics_by_portfolio["Spend"]).round(2)
    metrics_by_portfolio["Spend"] = metrics_by_portfolio["Spend"].round(2)
    metrics_by_portfolio["Sales"] = metrics_by_portfolio["Sales"].round(2)
    total_summary = pd.DataFrame({
        "Metric": ["Total"],
        "Impressions": [metrics_by_portfolio["Impressions"].sum()],
        "Clicks": [metrics_by_portfolio["Clicks"].sum()],
        "Spend": [metrics_by_portfolio["Spend"].sum()],
        "Sales": [metrics_by_portfolio["Sales"].sum()],
        "Orders": [metrics_by_portfolio["Orders"].sum()],
        "CTR": [(metrics_by_portfolio["Clicks"].sum() / metrics_by_portfolio["Impressions"].sum() * 100).round(1)],
        "CVR": [(metrics_by_portfolio["Orders"].sum() / metrics_by_portfolio["Clicks"].sum() * 100).round(1)],
        "ACOS": [(metrics_by_portfolio["Spend"].sum() / metrics_by_portfolio["Sales"].sum() * 100).round(1)],
        "ROAS": [(metrics_by_portfolio["Sales"].sum() / metrics_by_portfolio["Spend"].sum()).round(2)]
    })
    return metrics_by_portfolio, total_summary

def create_metric_over_time_chart(data, metric, portfolio, product_type, show_yoy=True):
    filtered_data = data[data["Product"] == product_type].copy()
    if portfolio != "All Portfolios":
        filtered_data = filtered_data[filtered_data["Portfolio Name"] == portfolio]
    filtered_data["Year"] = filtered_data["WE Date"].dt.year
    filtered_data["Month"] = filtered_data["WE Date"].dt.month
    filtered_data["MonthName"] = filtered_data["WE Date"].dt.strftime("%b")
    filtered_data["Week"] = filtered_data["WE Date"].dt.isocalendar().week
    years = sorted(filtered_data["Year"].unique())
    fig = go.Figure()
    if metric in ["CTR", "CVR", "ACOS"]:
        hover_template = "%{y:.1f}%<extra></extra>"
    elif metric in ["ROAS"]:
        hover_template = "%{y:.1f}<extra></extra>"
    elif metric in ["Spend", "Sales"]:
        hover_template = "%{y:.1f}<extra></extra>"
    else:
        hover_template = "%{y:,.1f}<extra></extra>"
    if show_yoy and len(years) > 1:
        colors = {
            years[0]: "#1f77b4",
            years[1]: "#ff7f0e"
        }
        month_order = []
        month_values = {}
        for year in years:
            year_data = filtered_data[filtered_data["Year"] == year]
            if year_data.empty:
                continue
            grouped = year_data.groupby(["Month", "MonthName"]).agg({
                "Impressions": "sum",
                "Clicks": "sum",
                "Spend": "sum",
                "Sales": "sum",
                "Orders": "sum",
                "Units": "sum"
            }).reset_index()
            grouped["CTR"] = (grouped["Clicks"] / grouped["Impressions"]) * 100
            grouped["CVR"] = (grouped["Orders"] / grouped["Clicks"]) * 100 if "Orders" in grouped.columns else 0
            grouped["ACOS"] = (grouped["Spend"] / grouped["Sales"]) * 100
            grouped["ROAS"] = grouped["Sales"] / grouped["Spend"]
            grouped = grouped.replace([float("inf"), -float("inf")], float("nan"))
            grouped = grouped.sort_values("Month")
            for _, row in grouped.iterrows():
                month_name = row["MonthName"]
                if month_name not in month_order:
                    month_order.append(month_name)
                    month_values[month_name] = row["Month"]
            fig.add_trace(
                go.Scatter(
                    x=grouped["MonthName"],
                    y=grouped[metric],
                    mode="lines+markers",
                    name=f"{year}",
                    line=dict(color=colors.get(year, "#000000"), width=2),
                    marker=dict(size=8),
                    hovertemplate=f"{year}: " + hover_template
                )
            )
        if month_order:
            month_order = sorted(month_order, key=lambda m: month_values[m])
            fig.update_layout(xaxis=dict(categoryorder="array", categoryarray=month_order))

    else:
        grouped = filtered_data.groupby("WE Date").agg({
            "Impressions": "sum",
            "Clicks": "sum",
            "Spend": "sum",
            "Sales": "sum",
            "Orders": "sum",
            "Units": "sum"
        }).reset_index()
        grouped["CTR"] = (grouped["Clicks"] / grouped["Impressions"]) * 100
        grouped["CVR"] = (grouped["Orders"] / grouped["Clicks"]) * 100 if "Orders" in grouped.columns else 0
        grouped["ACOS"] = (grouped["Spend"] / grouped["Sales"]) * 100
        grouped["ROAS"] = grouped["Sales"] / grouped["Spend"]
        grouped = grouped.replace([float("inf"), -float("inf")], float("nan"))
        grouped = grouped.sort_values("WE Date")
        fig.add_trace(
            go.Scatter(
                x=grouped["WE Date"],
                y=grouped[metric],
                mode="lines+markers",
                name=metric,
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=8),
                hovertemplate=hover_template
            )
        )
    if show_yoy and len(years) > 1:
        monthly_agg = filtered_data.groupby(["Year", "Month", "MonthName"]).agg({
            "Impressions": "sum",
            "Clicks": "sum",
            "Spend": "sum",
            "Sales": "sum",
            "Orders": "sum",
            "Units": "sum"
        }).reset_index()
        monthly_agg["CTR"] = (monthly_agg["Clicks"] / monthly_agg["Impressions"]) * 100
        monthly_agg["CVR"] = (monthly_agg["Orders"] / monthly_agg["Clicks"]) * 100 if "Orders" in monthly_agg.columns else 0
        monthly_agg["ACOS"] = (monthly_agg["Spend"] / monthly_agg["Sales"]) * 100
        monthly_agg["ROAS"] = monthly_agg["Sales"] / monthly_agg["Spend"]
        monthly_agg = monthly_agg.replace([float("inf"), -float("inf")], float("nan"))
        pivot_df = monthly_agg.pivot(index=["Month", "MonthName"], columns="Year", values=metric).reset_index()
        changes = []
        for _, row in pivot_df.iterrows():
            if pd.notna(row[years[0]]) and pd.notna(row[years[1]]) and row[years[0]] != 0:
                month_name = row["MonthName"]
                old_value = row[years[0]]
                new_value = row[years[1]]
                pct_change = (new_value - old_value) / old_value * 100
                changes.append({
                    "Month": row["Month"],
                    "MonthName": month_name,
                    "PctChange": pct_change
                })

        if changes:
            for change in changes:
                direction = "▲" if change["PctChange"] > 0 else "▼"
                color = "green" if change["PctChange"] > 0 else "red"
                fig.add_annotation(
                    x=change["MonthName"],
                    y=max(pivot_df.loc[pivot_df["MonthName"] == change["MonthName"], years].max().max() * 1.1, 1),
                    text=f"{direction} {abs(change['PctChange']):.1f}%",
                    showarrow=False,
                    font=dict(size=10, color=color)
                )
            avg_change = sum(change["PctChange"] for change in changes) / len(changes)
            direction = "increase" if avg_change > 0 else "decrease"
            fig.add_annotation(
                x=0.5,
                y=1.12,
                xref="paper",
                yref="paper",
                text=f"Average YoY {direction}: {abs(avg_change):.1f}%",
                showarrow=False,
                font=dict(size=12, color="white", family="Arial Black"),
                bgcolor="rgba(50, 50, 50, 0.9)",
                bordercolor="rgba(255, 255, 255, 0.8)",
                borderwidth=2,
                borderpad=6
            )
    portfolio_title = f"for {portfolio}" if portfolio != "All Portfolios" else "for All Portfolios"
    fig.update_layout(
        title=f"{metric} By Month {portfolio_title}",
        xaxis_title="Month",
        yaxis_title=metric,
        legend_title="Year",
        hovermode="x unified",
        template="plotly_white",
        yaxis=dict(rangemode="tozero")
    )
    if metric in ["CTR", "CVR", "ACOS"]:
        fig.update_layout(yaxis=dict(ticksuffix="%", rangemode="tozero"))
    elif metric in ["Spend", "Sales"]:
        fig.update_layout(yaxis=dict(tickprefix="$", rangemode="tozero"))
    return fig

def style_total_summary(df):
    df_copy = df.copy()
    df_copy = df_copy.replace([float("inf"), -float("inf")], float("nan"))
    def format_acos(val):
        if pd.isna(val):
            return "No Sales"
        return f"{val:.1f}%"
    def format_roas(val):
        if pd.isna(val):
            return "No Sales"
        return f"{val:.2f}"
    formatted = df_copy.style.format({
        "Impressions": "{:,}",
        "Clicks": "{:,}",
        "Spend": "${:,.2f}",
        "Sales": "${:,.2f}",
        "Orders": "{:,}",
        "CTR": "{:.1f}%",
        "CVR": "{:.1f}%",
        "ACOS": format_acos,
        "ROAS": format_roas
    })
    def color_acos(val):
        if pd.isna(val):
            return "color: red"
        if val <= 15:
            return "color: green"
        elif val <= 20:
            return "color: orange"
        else:
            return "color: red"
    def color_roas(val):
        if pd.isna(val):
            return "color: red"
        color = "green" if val > 3 else "red"
        return f"color: {color}"
    formatted = formatted.applymap(color_acos, subset=["ACOS"])
    formatted = formatted.applymap(color_roas, subset=["ROAS"])
    return formatted.set_properties(**{"font-weight": "bold"})


def style_metrics_table(df):
    df_copy = df.copy()
    df_copy = df_copy.replace([float("inf"), -float("inf")], float("nan"))
    def format_acos(val):
        if pd.isna(val):
            return "No Sales"
        return f"{val:.1f}%"
    def format_roas(val):
        if pd.isna(val):
            return "No Sales"
        return f"{val:.2f}"
    formatted = df_copy.style.format({
        "Impressions": "{:,}",
        "Clicks": "{:,}",
        "Spend": "${:,.2f}",
        "Sales": "${:,.2f}",
        "Orders": "{:,}",
        "Units": "{:,}",
        "CTR": "{:.1f}%",
        "CVR": "{:.1f}%",
        "ACOS": format_acos,
        "ROAS": format_roas
    })
    def color_acos(val):
        if pd.isna(val):
            return "color: red"
        if val <= 15:
            return "color: green"
        elif val <= 20:
            return "color: orange"
        else:
            return "color: red"
    def color_roas(val):
        if pd.isna(val):
            return "color: red"
        color = "green" if val > 3 else "red"
        return f"color: {color}"
    formatted = formatted.applymap(color_acos, subset=["ACOS"])
    formatted = formatted.applymap(color_roas, subset=["ROAS"])
    return formatted

def generate_insights(total_metrics, campaign_type):
    if campaign_type == "Sponsored Brands":
        acos_threshold = 15
    else:
        acos_threshold = 30

    insights = []

    if pd.isna(total_metrics["ACOS"]):
        insights.append("⚠️ There are ad expenses but no sales. Review your campaigns immediately.")
    elif total_metrics["ACOS"] > acos_threshold:
        insights.append(f"⚠️ Overall ACOS is high (above {acos_threshold}%). Consider optimizing campaigns to improve efficiency.")
    else:
        insights.append("✅ Overall ACOS is within acceptable range.")

    if pd.isna(total_metrics["ROAS"]):
        insights.append("⚠️ No return on ad spend. Your ads are not generating sales.")
    elif total_metrics["ROAS"] < 3:
        insights.append("⚠️ Overall ROAS is below target (less than 3). Review keyword and bid strategies.")
    else:
        insights.append("✅ Overall ROAS is good, indicating efficient ad spend.")

    if total_metrics["CTR"] < 0.3:
        insights.append("⚠️ Click-through rate is low. Consider improving ad creative or targeting.")
    else:
        insights.append("✅ Click-through rate is satisfactory.")

    if total_metrics["CVR"] < 10:
        insights.append("⚠️ Conversion rate is below 10%. Review product listings and targeting.")
    else:
        insights.append("✅ Conversion rate is good.")

    return insights

# =============================================================================
# Display Dashboard Tabs Only When Data is Uploaded
# =============================================================================
if "ad_data" in st.session_state and st.session_state["ad_data"] is not None:

    tabs_adv = st.tabs([
        "YOY Comparison",
        "Sponsored Products",
        "Sponsored Brands",
        "Sponsored Display"
    ])

    # -------------------------------
    # Tab 0: YOY Comparison 
    # -------------------------------
    with tabs_adv[0]:
        st.markdown("### YOY Comparison")
        ad_data = preprocess_ad_data(st.session_state["ad_data"])

        # ----------------------------------------------------------------
        # Updated General Overview Section with selectors including metrics
        # ----------------------------------------------------------------
        st.markdown("#### General Overview by Product Type")
        ad_data_overview = st.session_state["ad_data"].copy()
        ad_data_overview["WE Date"] = pd.to_datetime(ad_data_overview["WE Date"], dayfirst=True, errors="coerce")
        ad_data_overview["Year"] = ad_data_overview["WE Date"].dt.year
        ad_data_overview["Week"] = ad_data_overview["WE Date"].dt.isocalendar().week

        # --- New selectors in one row: Year, Overview Timeframe, Week, and Metrics ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            available_years = sorted(ad_data_overview["Year"].dropna().unique())
            # Select the last two years by default if there are at least 2 years available
            default_years = available_years[-2:] if len(available_years) >= 2 else [available_years[-1]]
            selected_years = st.multiselect("Select Year(s) for Overview:", available_years, default=default_years)
        with col2:
            timeframe_options = ["Specific Week", "Last 4 Weeks", "Last 8 Weeks", "Last 12 Weeks"]
            selected_timeframe = st.selectbox("Select Overview Timeframe:", timeframe_options, index=1)
        with col3:
            available_weeks = sorted(ad_data_overview[ad_data_overview["Year"].isin(selected_years)]["Week"].dropna().unique())
            available_weeks_with_select = ["Select..."] + [str(int(w)) for w in available_weeks]
            selected_week_option = st.selectbox("Select Week for Overview:", available_weeks_with_select, index=0)
        
        # New metrics selector
        with col4:
            all_metrics = ["Impressions", "Clicks", "Spend", "Sales", "Orders", "Units", "CTR", "CVR", "CPC", "ACOS"]
            default_metrics = ["Spend", "Sales", "CTR", "CVR", "ACOS"]  # Default selection
            selected_metrics = st.multiselect("Select Metrics to Display:", all_metrics, default=default_metrics)
            
            # Ensure there's at least one metric selected
            if not selected_metrics:
                st.warning("Please select at least one metric to display")
                selected_metrics = ["Spend", "Sales"]  # Fallback if nothing selected

        # Convert selected option back to numeric if it's not "Select..."
        if selected_week_option != "Select...":
            selected_week = int(selected_week_option)
        else:
            selected_week = None

        # ----------------------------------------------------------------
        # Detailed Overview by Product (Multi-Year) with selected metrics
        # ----------------------------------------------------------------
        st.markdown("##### ")
        yearly_tables = []
        for yr in selected_years:
            df_year = ad_data_overview[ad_data_overview["Year"] == yr].copy()
            if selected_timeframe == "Specific Week":
                df_year = df_year[df_year["Week"] == selected_week]
            else:
                weeks_to_filter = int(selected_timeframe.split()[1])
                if not df_year.empty:
                    max_week = df_year["Week"].max()
                    df_year = df_year[df_year["Week"] >= (max_week - weeks_to_filter + 1)]
            if df_year.empty:
                continue
            
            # Always aggregate all base metrics initially
            df_pivot = df_year.groupby("Product").agg({
                "Impressions": "sum",
                "Clicks": "sum",
                "Spend": "sum",
                "Sales": "sum",
                "Orders": "sum",
                "Units": "sum"
            }).reset_index()
            
            # Calculate derived metrics
            df_pivot["CTR"] = df_pivot.apply(
                lambda row: (row["Clicks"] / row["Impressions"] * 100) if row["Impressions"] else 0,
                axis=1
            )
            df_pivot["CVR"] = df_pivot.apply(
                lambda row: (row["Orders"] / row["Clicks"] * 100) if row["Clicks"] else 0,
                axis=1
            )
            df_pivot["CPC"] = df_pivot.apply(
                lambda row: (row["Spend"] / row["Clicks"]) if row["Clicks"] else 0,
                axis=1
            )
            df_pivot["ACOS"] = df_pivot.apply(
                lambda row: (row["Spend"] / row["Sales"] * 100) if row["Sales"] else 0,
                axis=1
            )
            
            # Rename columns with year suffix but only for selected metrics
            rename_cols = {}
            for metric in ["Impressions", "Clicks", "Spend", "Sales", "Orders", "Units", "CTR", "CVR", "CPC", "ACOS"]:
                if metric in selected_metrics:
                    rename_cols[metric] = f"{metric} {yr}"
            
            # Filter to keep only Product and selected metrics
            df_pivot = df_pivot[["Product"] + list(rename_cols.keys())]
            df_pivot = df_pivot.rename(columns=rename_cols)
            yearly_tables.append(df_pivot)

        if yearly_tables:
            merged_table = reduce(lambda left, right: pd.merge(left, right, on="Product", how="outer"), yearly_tables)
            merged_table.fillna(0, inplace=True)
            if len(selected_years) >= 2:
                current_year_sel = max(selected_years)
                if (current_year_sel - 1) in selected_years:
                    prev_year_sel = current_year_sel - 1
                    for metric in selected_metrics:  # Only process selected metrics
                        col_current = f"{metric} {current_year_sel}"
                        col_prev = f"{metric} {prev_year_sel}"
                        if col_current in merged_table.columns and col_prev in merged_table.columns:
                            merged_table[f"{metric} {current_year_sel} % Change"] = merged_table.apply(
                                lambda row: ((row[col_current] - row[col_prev]) / row[col_prev] * 100)
                                if row[col_prev] != 0 else None,
                                axis=1
                            ).round(0)
            ordered_cols = ["Product"]
            for metric in selected_metrics:  # Only include selected metrics in column order
                if (len(selected_years) >= 2 and
                    (max(selected_years) - 1) in selected_years and
                    f"{metric} {max(selected_years) - 1}" in merged_table.columns and
                    f"{metric} {max(selected_years)}" in merged_table.columns and
                    f"{metric} {max(selected_years)} % Change" in merged_table.columns):
                    ordered_cols.append(f"{metric} {max(selected_years) - 1}")
                    ordered_cols.append(f"{metric} {max(selected_years)}")
                    ordered_cols.append(f"{metric} {max(selected_years)} % Change")
                else:
                    for yr in sorted(selected_years):
                        col_name = f"{metric} {yr}"
                        if col_name in merged_table.columns:
                            ordered_cols.append(col_name)
            # Filter to ensure all columns exist
            ordered_cols = [col for col in ordered_cols if col in merged_table.columns]
            merged_table = merged_table[ordered_cols]

            format_dict = {}
            for col in merged_table.columns:
                if "% Change" in col:
                    format_dict[col] = "{:.0f}%"
                elif any(metric in col for metric in ["Impressions", "Clicks", "Orders", "Units"]):
                    format_dict[col] = "{:,.0f}"
                elif any(metric in col for metric in ["Spend", "Sales"]):
                    format_dict[col] = "{:,.2f}"
                elif any(metric in col for metric in ["CTR", "CVR", "ACOS"]):
                    format_dict[col] = "{:.1f}%"
                elif "CPC" in col:
                    format_dict[col] = "{:,.2f}"
            styled_merged_table = merged_table.style.format(format_dict)
            pct_change_cols = [col for col in merged_table.columns if "% Change" in col]
            styled_merged_table = styled_merged_table.applymap(
                lambda v: "color: green" if (isinstance(v, (int, float)) and v > 0)
                else ("color: red" if (isinstance(v, (int, float)) and v < 0) else ""),
                subset=pct_change_cols
            )
            st.dataframe(styled_merged_table, use_container_width=True)
        else:
            st.warning("No data available for the selected criteria.")

        # ----------------------------------------------------------------
        # Portfolio Performance Table (using filtered_overview_data)
        # ----------------------------------------------------------------
        filtered_overview_data = filter_data_by_timeframe(ad_data_overview, selected_years, selected_timeframe, selected_week)
        if filtered_overview_data.empty:
            st.warning("No data available for the selected criteria.")
        else:
            required_cols = ["Product", "Impressions", "Clicks", "Spend", "Sales", "Orders", "Units"]
            missing_cols = [col for col in required_cols if col not in filtered_overview_data.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                st.stop()

            st.markdown("---")
            st.markdown("#### Portfolio Performance Table")
            st.markdown("*Click on any column header to sort*")

            if "Product" in filtered_overview_data.columns:
                product_types = ["All"] + sorted(filtered_overview_data["Product"].unique().tolist())
                selected_product_type = st.selectbox("Filter by Product Type:", product_types)
                if selected_product_type != "All":
                    filtered_portfolio_data = filtered_overview_data[filtered_overview_data["Product"] == selected_product_type].copy()
                else:
                    filtered_portfolio_data = filtered_overview_data.copy()
            else:
                st.warning("No 'Product' column found in the data. Showing all products.")
                filtered_portfolio_data = filtered_overview_data.copy()

            portfolio_col = None
            for possible_col in ["Portfolio Name", "Portfolio", "PortfolioName", "Portfolio_Name"]:
                if possible_col in filtered_portfolio_data.columns:
                    portfolio_col = possible_col
                    break
            if portfolio_col is None:
                st.error("No portfolio identifier column found in the data. Please ensure your data includes a column for portfolio names.")
                st.stop()

            # Add Year column if not already present
            if "Year" not in filtered_portfolio_data.columns:
                filtered_portfolio_data["Year"] = filtered_portfolio_data["WE Date"].dt.year

            # Check if we have at least 2 years of data to compare
            years = sorted(filtered_portfolio_data["Year"].unique())
            has_multiple_years = len(years) >= 2

            if has_multiple_years:
                current_year = max(years)
                previous_year = current_year - 1

                # Create a pivot table for each year
                current_year_pivot = filtered_portfolio_data[filtered_portfolio_data["Year"] == current_year].groupby(portfolio_col).agg({
                    "Impressions": "sum",
                    "Clicks": "sum",
                    "Spend": "sum",
                    "Sales": "sum",
                    "Orders": "sum",
                    "Units": "sum"
                }).reset_index()

                if not current_year_pivot.empty:
                    current_year_pivot["CTR"] = (current_year_pivot["Clicks"] / current_year_pivot["Impressions"] * 100).replace([float("inf"), -float("inf")], 0)
                    current_year_pivot["CVR"] = (current_year_pivot["Orders"] / current_year_pivot["Clicks"] * 100).replace([float("inf"), -float("inf")], 0)
                    current_year_pivot["CPC"] = (current_year_pivot["Spend"] / current_year_pivot["Clicks"]).replace([float("inf"), -float("inf")], 0)
                    current_year_pivot["ACOS"] = (current_year_pivot["Spend"] / current_year_pivot["Sales"] * 100).replace([float("inf"), -float("inf")], 0)

                previous_year_pivot = filtered_portfolio_data[filtered_portfolio_data["Year"] == previous_year].groupby(portfolio_col).agg({
                    "Impressions": "sum",
                    "Clicks": "sum",
                    "Spend": "sum",
                    "Sales": "sum",
                    "Orders": "sum",
                    "Units": "sum"
                }).reset_index()

                if not previous_year_pivot.empty:
                    previous_year_pivot["CTR"] = (previous_year_pivot["Clicks"] / previous_year_pivot["Impressions"] * 100).replace([float("inf"), -float("inf")], 0)
                    previous_year_pivot["CVR"] = (previous_year_pivot["Orders"] / previous_year_pivot["Clicks"] * 100).replace([float("inf"), -float("inf")], 0)
                    previous_year_pivot["CPC"] = (previous_year_pivot["Spend"] / previous_year_pivot["Clicks"]).replace([float("inf"), -float("inf")], 0)
                    previous_year_pivot["ACOS"] = (previous_year_pivot["Spend"] / previous_year_pivot["Sales"] * 100).replace([float("inf"), -float("inf")], 0)

                # Merge both pivots
                all_portfolios = pd.concat([
                    current_year_pivot[portfolio_col],
                    previous_year_pivot[portfolio_col]
                ]).drop_duplicates()

                # Create a new table structure with year columns for each metric
                yoy_table = pd.DataFrame({portfolio_col: all_portfolios})

                # Add metrics for each year and calculate YoY change
                metrics = ["Impressions", "Clicks", "Spend", "Sales", "Orders", "Units", "CTR", "CVR", "CPC", "ACOS"]
                
                # Filter metrics based on user selection
                filtered_metrics = [m for m in metrics if m in selected_metrics]

                for metric in filtered_metrics:  # Only use selected metrics
                    # Add previous year data
                    if not previous_year_pivot.empty:
                        yoy_table = yoy_table.merge(
                            previous_year_pivot[[portfolio_col, metric]].rename(columns={metric: f"{metric} {previous_year}"}),
                            on=portfolio_col, how="left"
                        )
                    else:
                        yoy_table[f"{metric} {previous_year}"] = 0

                    # Add current year data
                    if not current_year_pivot.empty:
                        yoy_table = yoy_table.merge(
                            current_year_pivot[[portfolio_col, metric]].rename(columns={metric: f"{metric} {current_year}"}),
                            on=portfolio_col, how="left"
                        )
                    else:
                        yoy_table[f"{metric} {current_year}"] = 0

                    # Calculate percent change
                    yoy_table[f"{metric} % Change"] = yoy_table.apply(
                        lambda row: ((row[f"{metric} {current_year}"] - row[f"{metric} {previous_year}"]) / row[f"{metric} {previous_year}"] * 100)
                        if row[f"{metric} {previous_year}"] != 0 else None,
                        axis=1
                    )
                yoy_table = yoy_table.fillna(0)
                yoy_table = yoy_table.rename(columns={portfolio_col: "Row Labels"})

                # Format the table for display
                format_dict = {}
                for col in yoy_table.columns:
                    if "% Change" in col:
                        format_dict[col] = "{:.0f}%"
                    elif any(metric in col for metric in ["Impressions", "Clicks", "Orders", "Units"]):
                        format_dict[col] = "{:,.0f}"
                    elif any(metric in col for metric in ["Spend", "Sales"]):
                        format_dict[col] = "${:,.2f}"
                    elif any(metric in col for metric in ["CTR", "CVR", "ACOS"]):
                        format_dict[col] = "{:.1f}%"
                    elif "CPC" in col:
                        format_dict[col] = "${:.2f}"
                styled_yoy_table = yoy_table.style.format(format_dict)

                # Color the percent change columns
                pct_change_cols = [col for col in yoy_table.columns if "% Change" in col]
                styled_yoy_table = styled_yoy_table.applymap(
                    lambda v: "color: green" if (isinstance(v, (int, float)) and v > 0)
                    else ("color: red" if (isinstance(v, (int, float)) and v < 0) else ""),
                    subset=pct_change_cols
                )

                st.dataframe(styled_yoy_table, use_container_width=True, height=400)

                # Create a grand total row with YoY comparison
                grand_total = {
                    "Row Labels": [f'Grand Total - {selected_product_type if selected_product_type != "All" else "All Products"}']
                }

                # Calculate totals and YoY changes for each selected metric
                for metric in filtered_metrics:  # Only use selected metrics
                    # Get values for each year
                    prev_value = yoy_table[f"{metric} {previous_year}"].sum()
                    curr_value = yoy_table[f"{metric} {current_year}"].sum()

                    # For rate metrics (CTR, CVR, ACOS), recalculate from base metrics
                    if metric == "CTR":
                        prev_clicks = yoy_table[f"Clicks {previous_year}"].sum() if f"Clicks {previous_year}" in yoy_table.columns else 0
                        prev_impressions = yoy_table[f"Impressions {previous_year}"].sum() if f"Impressions {previous_year}" in yoy_table.columns else 0
                        curr_clicks = yoy_table[f"Clicks {current_year}"].sum() if f"Clicks {current_year}" in yoy_table.columns else 0
                        curr_impressions = yoy_table[f"Impressions {current_year}"].sum() if f"Impressions {current_year}" in yoy_table.columns else 0
                        prev_value = (prev_clicks / prev_impressions * 100) if prev_impressions > 0 else 0
                        curr_value = (curr_clicks / curr_impressions * 100) if curr_impressions > 0 else 0
                    elif metric == "CVR":
                        prev_orders = yoy_table[f"Orders {previous_year}"].sum() if f"Orders {previous_year}" in yoy_table.columns else 0
                        prev_clicks = yoy_table[f"Clicks {previous_year}"].sum() if f"Clicks {previous_year}" in yoy_table.columns else 0
                        curr_orders = yoy_table[f"Orders {current_year}"].sum() if f"Orders {current_year}" in yoy_table.columns else 0
                        curr_clicks = yoy_table[f"Clicks {current_year}"].sum() if f"Clicks {current_year}" in yoy_table.columns else 0
                        prev_value = (prev_orders / prev_clicks * 100) if prev_clicks > 0 else 0
                        curr_value = (curr_orders / curr_clicks * 100) if curr_clicks > 0 else 0
                    elif metric == "CPC":
                        prev_spend = yoy_table[f"Spend {previous_year}"].sum() if f"Spend {previous_year}" in yoy_table.columns else 0
                        prev_clicks = yoy_table[f"Clicks {previous_year}"].sum() if f"Clicks {previous_year}" in yoy_table.columns else 0
                        curr_spend = yoy_table[f"Spend {current_year}"].sum() if f"Spend {current_year}" in yoy_table.columns else 0
                        curr_clicks = yoy_table[f"Clicks {current_year}"].sum() if f"Clicks {current_year}" in yoy_table.columns else 0
                        prev_value = (prev_spend / prev_clicks) if prev_clicks > 0 else 0
                        curr_value = (curr_spend / curr_clicks) if curr_clicks > 0 else 0
                    elif metric == "ACOS":
                        prev_spend = yoy_table[f"Spend {previous_year}"].sum() if f"Spend {previous_year}" in yoy_table.columns else 0
                        prev_sales = yoy_table[f"Sales {previous_year}"].sum() if f"Sales {previous_year}" in yoy_table.columns else 0
                        curr_spend = yoy_table[f"Spend {current_year}"].sum() if f"Spend {current_year}" in yoy_table.columns else 0
                        curr_sales = yoy_table[f"Sales {current_year}"].sum() if f"Sales {current_year}" in yoy_table.columns else 0
                        prev_value = (prev_spend / prev_sales * 100) if prev_sales > 0 else 0
                        curr_value = (curr_spend / curr_sales * 100) if curr_sales > 0 else 0

                    grand_total[f"{metric} {previous_year}"] = [prev_value]
                    grand_total[f"{metric} {current_year}"] = [curr_value]

                    if prev_value != 0:
                        pct_change = ((curr_value - prev_value) / prev_value) * 100
                    else:
                        pct_change = float("inf") if curr_value > 0 else 0
                    grand_total[f"{metric} % Change"] = [pct_change]
                grand_total_df = pd.DataFrame(grand_total)

                def style_grand_total_yoy(df):
                    format_dict = {}
                    for col in df.columns:
                        if "% Change" in col:
                            format_dict[col] = "{:.0f}%"
                        elif any(metric in col for metric in ["Impressions", "Clicks", "Orders", "Units"]):
                            format_dict[col] = "{:,.0f}"
                        elif any(metric in col for metric in ["Spend", "Sales"]):
                            format_dict[col] = "${:,.2f}"
                        elif any(metric in col for metric in ["CTR", "CVR", "ACOS"]):
                            format_dict[col] = "{:.1f}%"
                        elif "CPC" in col:
                            format_dict[col] = "${:.2f}"
                    styled = df.style.format(format_dict)
                    pct_change_cols = [col for col in df.columns if "% Change" in col]
                    styled = styled.applymap(
                        lambda v: "color: green" if (isinstance(v, (int, float)) and v > 0)
                        else ("color: red" if (isinstance(v, (int, float)) and v < 0) else ""),
                        subset=pct_change_cols
                    )
                    styled = styled.set_properties(**{"font-weight": "bold"})
                    return styled

                st.markdown("###### Grand Total")
                st.dataframe(style_grand_total_yoy(grand_total_df), use_container_width=True)

            else:
                # Single year display (original implementation)
                portfolio_pivot = filtered_portfolio_data.groupby(portfolio_col).agg({
                    "Impressions": "sum",
                    "Clicks": "sum",
                    "Spend": "sum",
                    "Sales": "sum",
                    "Orders": "sum",
                    "Units": "sum"
                }).reset_index()
                
                # Calculate derived metrics
                portfolio_pivot["CTR"] = (portfolio_pivot["Clicks"] / portfolio_pivot["Impressions"] * 100).replace([float("inf"), -float("inf")], 0)
                portfolio_pivot["CVR"] = (portfolio_pivot["Orders"] / portfolio_pivot["Clicks"] * 100).replace([float("inf"), -float("inf")], 0)
                portfolio_pivot["CPC"] = (portfolio_pivot["Spend"] / portfolio_pivot["Clicks"]).replace([float("inf"), -float("inf")], 0)
                portfolio_pivot["ACOS"] = (portfolio_pivot["Spend"] / portfolio_pivot["Sales"] * 100).replace([float("inf"), -float("inf")], 0)
                
                # Filter to show only selected metrics plus the identifier column
                display_columns = [portfolio_col] + selected_metrics
                portfolio_pivot = portfolio_pivot[display_columns]
                
                portfolio_pivot = portfolio_pivot.sort_values("Spend", ascending=False)
                portfolio_pivot = portfolio_pivot.rename(columns={portfolio_col: "Row Labels"})

                def format_portfolio_data(df):
                    formatted_df = df.copy()
                    for col in formatted_df.columns:
                        if col == "Row Labels":
                            continue
                        elif col in ["Impressions", "Clicks", "Orders", "Units"]:
                            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:,}")
                        elif col in ["Spend", "Sales"]:
                            formatted_df[col] = formatted_df[col].apply(lambda x: f"${x:,.0f}")
                        elif col in ["CTR", "CVR", "ACOS"]:
                            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.1f}%")
                        elif col == "CPC":
                            formatted_df[col] = formatted_df[col].apply(lambda x: f"${x:.2f}")
                    return formatted_df
                
                # Separate the summary row from the main table.
                summary_row = {
                    "Row Labels": f"TOTAL - {selected_product_type if selected_product_type != 'All' else 'All Products'}",
                }
                
                # Only include selected metrics in the summary
                for metric in selected_metrics:
                    if metric in portfolio_pivot.columns:
                        summary_row[metric] = portfolio_pivot[metric].sum()
                
                # Recalculate derived metrics
                if "CTR" in selected_metrics and "Clicks" in selected_metrics and "Impressions" in selected_metrics:
                    summary_row["CTR"] = (summary_row["Clicks"] / summary_row["Impressions"] * 100) if summary_row.get("Impressions", 0) > 0 else 0
                if "CVR" in selected_metrics and "Orders" in selected_metrics and "Clicks" in selected_metrics:
                    summary_row["CVR"] = (summary_row["Orders"] / summary_row["Clicks"] * 100) if summary_row.get("Clicks", 0) > 0 else 0
                if "CPC" in selected_metrics and "Spend" in selected_metrics and "Clicks" in selected_metrics:
                    summary_row["CPC"] = (summary_row["Spend"] / summary_row["Clicks"]) if summary_row.get("Clicks", 0) > 0 else 0
                if "ACOS" in selected_metrics and "Spend" in selected_metrics and "Sales" in selected_metrics:
                    summary_row["ACOS"] = (summary_row["Spend"] / summary_row["Sales"] * 100) if summary_row.get("Sales", 0) > 0 else 0

                main_table = portfolio_pivot.copy()  # Main table without summary
                summary_df = pd.DataFrame([summary_row])

                st.dataframe(format_portfolio_data(main_table), use_container_width=True, height=400)

                st.markdown("###### Grand Total")
                st.dataframe(format_portfolio_data(summary_df), use_container_width=True)

            # ----------------------------------------------------------------
            # Match Type Performance Analysis (using filtered_overview_data)
            # ----------------------------------------------------------------
            st.markdown("---")
            st.markdown("#### Match Type Performance Analysis")
            if "Product" in filtered_overview_data.columns and "Match Type" in filtered_overview_data.columns:
                match_type_data = filtered_overview_data.copy()
                match_type_data["Match Type"] = match_type_data["Match Type"].fillna("Other")

                # Add Year column if not already present
                if "Year" not in match_type_data.columns:
                    match_type_data["Year"] = match_type_data["WE Date"].dt.year

                # Get years for comparison
                years = sorted(match_type_data["Year"].unique())
                has_multiple_years = len(years) >= 2

                product_types = ["Sponsored Products", "Sponsored Brands", "Sponsored Display"]
                
                # Only use selected metrics
                match_type_metrics = [m for m in ["Impressions", "Clicks", "Spend", "Sales", "Orders", "Units", "CTR", "CVR", "CPC", "ACOS"] if m in selected_metrics]

            for product_type in product_types:
                product_data = match_type_data[match_type_data["Product"] == product_type].copy()
                if product_data.empty:
                    continue

                st.subheader(product_type)

                if has_multiple_years:
                    current_year = max(years)
                    previous_year = current_year - 1

                    # Create pivots for each year
                    current_year_data = product_data[product_data["Year"] == current_year]
                    current_pivot = current_year_data.groupby("Match Type", as_index=False).agg({
                        "Impressions": "sum",
                        "Clicks": "sum",
                        "Spend": "sum",
                        "Sales": "sum",
                        "Orders": "sum",
                        "Units": "sum"
                    })

                    if not current_pivot.empty:
                        current_pivot["CTR"] = (current_pivot["Clicks"] / current_pivot["Impressions"] * 100).replace([float("inf"), -float("inf")], 0)
                        current_pivot["CVR"] = (current_pivot["Orders"] / current_pivot["Clicks"] * 100).replace([float("inf"), -float("inf")], 0)
                        current_pivot["CPC"] = (current_pivot["Spend"] / current_pivot["Clicks"]).replace([float("inf"), -float("inf")], 0)
                        current_pivot["ACOS"] = (current_pivot["Spend"] / current_pivot["Sales"] * 100).replace([float("inf"), -float("inf")], 0)

                    previous_year_data = product_data[product_data["Year"] == previous_year]
                    previous_pivot = previous_year_data.groupby("Match Type", as_index=False).agg({
                        "Impressions": "sum",
                        "Clicks": "sum",
                        "Spend": "sum",
                        "Sales": "sum",
                        "Orders": "sum",
                        "Units": "sum"
                    })

                    if not previous_pivot.empty:
                        previous_pivot["CTR"] = (previous_pivot["Clicks"] / previous_pivot["Impressions"] * 100).replace([float("inf"), -float("inf")], 0)
                        previous_pivot["CVR"] = (previous_pivot["Orders"] / previous_pivot["Clicks"] * 100).replace([float("inf"), -float("inf")], 0)
                        previous_pivot["CPC"] = (previous_pivot["Spend"] / previous_pivot["Clicks"]).replace([float("inf"), -float("inf")], 0)
                        previous_pivot["ACOS"] = (previous_pivot["Spend"] / previous_pivot["Sales"] * 100).replace([float("inf"), -float("inf")], 0)

                    # Create a unified table with all match types from both years
                    all_match_types = pd.concat([
                        current_pivot["Match Type"] if not current_pivot.empty else pd.Series(),
                        previous_pivot["Match Type"] if not previous_pivot.empty else pd.Series()
                    ]).drop_duplicates()

                    # Build YoY comparison table
                    yoy_match_type = pd.DataFrame({"Match Type": all_match_types})

                    # Add metrics for each year and calculate YoY change - only for selected metrics
                    for metric in match_type_metrics:
                        # Add previous year data
                        if not previous_pivot.empty:
                            yoy_match_type = yoy_match_type.merge(
                                previous_pivot[["Match Type", metric]].rename(columns={metric: f"{metric} {previous_year}"}),
                                on="Match Type", how="left"
                            )
                        else:
                            yoy_match_type[f"{metric} {previous_year}"] = 0

                        # Add current year data
                        if not current_pivot.empty:
                            yoy_match_type = yoy_match_type.merge(
                                current_pivot[["Match Type", metric]].rename(columns={metric: f"{metric} {current_year}"}),
                                on="Match Type", how="left"
                            )
                        else:
                            yoy_match_type[f"{metric} {current_year}"] = 0

                        # Fill NaN values with 0
                        yoy_match_type[f"{metric} {previous_year}"] = yoy_match_type[f"{metric} {previous_year}"].fillna(0)
                        yoy_match_type[f"{metric} {current_year}"] = yoy_match_type[f"{metric} {current_year}"].fillna(0)

                        # Calculate percent change
                        yoy_match_type[f"{metric} % Change"] = yoy_match_type.apply(
                            lambda row: ((row[f"{metric} {current_year}"] - row[f"{metric} {previous_year}"]) / row[f"{metric} {previous_year}"] * 100)
                            if row[f"{metric} {previous_year}"] != 0 else float("inf") if row[f"{metric} {current_year}"] > 0 else 0,
                            axis=1
                        )

                        # Sort by current year spend (if available, otherwise use first selected metric)
                    sort_metric = "Spend" if "Spend" in match_type_metrics else match_type_metrics[0]
                    yoy_match_type = yoy_match_type.sort_values(f"{sort_metric} {current_year}", ascending=False)

                    # Build the summary row separately instead of appending it
                    summary_row = {"Match Type": f"TOTAL - {product_type}"}
                    for metric in match_type_metrics:
                        summary_row[f"{metric} {previous_year}"] = yoy_match_type[f"{metric} {previous_year}"].sum()
                        summary_row[f"{metric} {current_year}"] = yoy_match_type[f"{metric} {current_year}"].sum()
                        
                        # Recalculate derived metrics for total
                        if metric == "CTR":
                            prev_impressions = yoy_match_type[f"Impressions {previous_year}"].sum() if f"Impressions {previous_year}" in yoy_match_type.columns else 0
                            prev_clicks = yoy_match_type[f"Clicks {previous_year}"].sum() if f"Clicks {previous_year}" in yoy_match_type.columns else 0
                            curr_impressions = yoy_match_type[f"Impressions {current_year}"].sum() if f"Impressions {current_year}" in yoy_match_type.columns else 0
                            curr_clicks = yoy_match_type[f"Clicks {current_year}"].sum() if f"Clicks {current_year}" in yoy_match_type.columns else 0
                            summary_row[f"CTR {previous_year}"] = (prev_clicks / prev_impressions * 100) if prev_impressions > 0 else 0
                            summary_row[f"CTR {current_year}"] = (curr_clicks / curr_impressions * 100) if curr_impressions > 0 else 0
                        elif metric == "CVR":
                            prev_clicks = yoy_match_type[f"Clicks {previous_year}"].sum() if f"Clicks {previous_year}" in yoy_match_type.columns else 0
                            prev_orders = yoy_match_type[f"Orders {previous_year}"].sum() if f"Orders {previous_year}" in yoy_match_type.columns else 0
                            curr_clicks = yoy_match_type[f"Clicks {current_year}"].sum() if f"Clicks {current_year}" in yoy_match_type.columns else 0
                            curr_orders = yoy_match_type[f"Orders {current_year}"].sum() if f"Orders {current_year}" in yoy_match_type.columns else 0
                            summary_row[f"CVR {previous_year}"] = (prev_orders / prev_clicks * 100) if prev_clicks > 0 else 0
                            summary_row[f"CVR {current_year}"] = (curr_orders / curr_clicks * 100) if curr_clicks > 0 else 0
                        elif metric == "CPC":
                            prev_clicks = yoy_match_type[f"Clicks {previous_year}"].sum() if f"Clicks {previous_year}" in yoy_match_type.columns else 0
                            prev_spend = yoy_match_type[f"Spend {previous_year}"].sum() if f"Spend {previous_year}" in yoy_match_type.columns else 0
                            curr_clicks = yoy_match_type[f"Clicks {current_year}"].sum() if f"Clicks {current_year}" in yoy_match_type.columns else 0
                            curr_spend = yoy_match_type[f"Spend {current_year}"].sum() if f"Spend {current_year}" in yoy_match_type.columns else 0
                            summary_row[f"CPC {previous_year}"] = (prev_spend / prev_clicks) if prev_clicks > 0 else 0
                            summary_row[f"CPC {current_year}"] = (curr_spend / curr_clicks) if curr_clicks > 0 else 0
                        elif metric == "ACOS":
                            prev_spend = yoy_match_type[f"Spend {previous_year}"].sum() if f"Spend {previous_year}" in yoy_match_type.columns else 0
                            prev_sales = yoy_match_type[f"Sales {previous_year}"].sum() if f"Sales {previous_year}" in yoy_match_type.columns else 0
                            curr_spend = yoy_match_type[f"Spend {current_year}"].sum() if f"Spend {current_year}" in yoy_match_type.columns else 0
                            curr_sales = yoy_match_type[f"Sales {current_year}"].sum() if f"Sales {current_year}" in yoy_match_type.columns else 0
                            summary_row[f"ACOS {previous_year}"] = (prev_spend / prev_sales * 100) if prev_sales > 0 else 0
                            summary_row[f"ACOS {current_year}"] = (curr_spend / curr_sales * 100) if curr_sales > 0 else 0
                        
                        # Calculate percent change for the summary row
                        prev_val = summary_row[f"{metric} {previous_year}"]
                        curr_val = summary_row[f"{metric} {current_year}"]
                        if prev_val != 0:
                            pct_change = ((curr_val - prev_val) / prev_val) * 100
                        else:
                            pct_change = float("inf") if curr_val > 0 else 0
                        summary_row[f"{metric} % Change"] = pct_change

                    # Separate main table and summary table
                    main_match_table = yoy_match_type.copy()
                    summary_df = pd.DataFrame([summary_row])
    
                    format_dict = {}
                    for col in yoy_match_type.columns:
                        if "% Change" in col:
                            format_dict[col] = "{:.0f}%"
                        elif any(metric in col for metric in ["Impressions", "Clicks", "Orders", "Units"]):
                            format_dict[col] = "{:,.0f}"
                        elif any(metric in col for metric in ["Spend", "Sales"]):
                            format_dict[col] = "${:,.2f}"
                        elif any(metric in col for metric in ["CTR", "CVR", "ACOS"]):
                            format_dict[col] = "{:.1f}%"
                        elif "CPC" in col:
                            format_dict[col] = "${:.2f}"

    
                    styled_main_table = main_match_table.style.format(format_dict)
                    pct_change_cols = [col for col in main_match_table.columns if "% Change" in col]
                    styled_main_table = styled_main_table.applymap(
                        lambda v: "color: green" if (isinstance(v, (int, float)) and v > 0)
                        else ("color: red" if (isinstance(v, (int, float)) and v < 0) else ""),
                        subset=pct_change_cols
                    )
                    
                    st.dataframe(styled_main_table, use_container_width=True)
    
                    st.markdown("###### Total")
                    styled_summary_table = summary_df.style.format(format_dict)
                    styled_summary_table = styled_summary_table.set_properties(**{"font-weight": "bold"})
                    st.dataframe(styled_summary_table, use_container_width=True)
                else:
                    # Single year display (filter for selected metrics)
                    match_type_pivot = product_data.groupby("Match Type", as_index=False).agg({
                        "Impressions": "sum",
                        "Clicks": "sum",
                        "Spend": "sum",
                        "Sales": "sum",
                        "Orders": "sum",
                        "Units": "sum"
                    })
                    match_type_pivot["CTR"] = (match_type_pivot["Clicks"] / match_type_pivot["Impressions"] * 100).replace([float("inf"), -float("inf")], 0)
                    match_type_pivot["CVR"] = (match_type_pivot["Orders"] / match_type_pivot["Clicks"] * 100).replace([float("inf"), -float("inf")], 0)
                    match_type_pivot["CPC"] = (match_type_pivot["Spend"] / match_type_pivot["Clicks"]).replace([float("inf"), -float("inf")], 0)
                    match_type_pivot["ACOS"] = (match_type_pivot["Spend"] / match_type_pivot["Sales"] * 100).replace([float("inf"), -float("inf")], 0)
                    
                    # Filter to only include selected metrics
                    filtered_cols = ["Match Type"] + match_type_metrics
                    match_type_pivot = match_type_pivot[filtered_cols]
                    
                    # Sort by Spend if available
                    if "Spend" in match_type_metrics:
                        match_type_pivot = match_type_pivot.sort_values("Spend", ascending=False)
                    
                    summary_row = {
                        "Match Type": f"TOTAL - {product_type}"
                    }
                    
                    # Calculate totals only for selected metrics
                    for metric in match_type_metrics:
                        if metric in ["Impressions", "Clicks", "Spend", "Sales", "Orders", "Units"]:
                            summary_row[metric] = match_type_pivot[metric].sum()
                    
                    # Calculate derived metrics for totals
                    if "CTR" in match_type_metrics and "Clicks" in match_type_metrics and "Impressions" in match_type_metrics:
                        summary_row["CTR"] = (summary_row["Clicks"] / summary_row["Impressions"] * 100) if summary_row.get("Impressions", 0) > 0 else 0
                    if "CVR" in match_type_metrics and "Orders" in match_type_metrics and "Clicks" in match_type_metrics:
                        summary_row["CVR"] = (summary_row["Orders"] / summary_row["Clicks"] * 100) if summary_row.get("Clicks", 0) > 0 else 0
                    if "CPC" in match_type_metrics and "Spend" in match_type_metrics and "Clicks" in match_type_metrics:
                        summary_row["CPC"] = (summary_row["Spend"] / summary_row["Clicks"]) if summary_row.get("Clicks", 0) > 0 else 0
                    if "ACOS" in match_type_metrics and "Spend" in match_type_metrics and "Sales" in match_type_metrics:
                        summary_row["ACOS"] = (summary_row["Spend"] / summary_row["Sales"] * 100) if summary_row.get("Sales", 0) > 0 else 0
                    
                    main_table = match_type_pivot.copy()  # main table without summary row
                    summary_df = pd.DataFrame([summary_row])

                    def style_match_type_table(df):
                        format_dict = {}
                        for col in df.columns:
                            if col == "Match Type":
                                continue
                            elif col in ["Impressions", "Clicks", "Orders", "Units"]:
                                format_dict[col] = "{:,}"
                            elif col in ["Spend", "Sales"]:
                                format_dict[col] = "${:,.0f}"
                            elif col in ["CTR", "CVR", "ACOS"]:
                                format_dict[col] = "{:.1f}%"
                            elif col == "CPC":
                                format_dict[col] = "${:.2f}"
                        return df.style.format(format_dict)
    
                    st.dataframe(style_match_type_table(main_table), use_container_width=True)
                    st.markdown("###### Total")
                    st.dataframe(style_match_type_table(summary_df), use_container_width=True)

            # ----------------------------------------------------------------
            # RTW/Prospecting Performance Analysis (using filtered_overview_data)
            # ----------------------------------------------------------------
            st.markdown("---")
            st.markdown("#### RTW/Prospecting Performance Analysis")
            if "Product" in filtered_overview_data.columns and "RTW/Prospecting" in filtered_overview_data.columns:
                rtw_data = filtered_overview_data.copy()
                if "Year" not in rtw_data.columns:
                    rtw_data["Year"] = rtw_data["WE Date"].dt.year
                rtw_product_types = ["Sponsored Products", "Sponsored Brands", "Sponsored Display"]
                rtw_product_types = [pt for pt in rtw_product_types if pt in rtw_data["Product"].unique()]
                selected_rtw_product = st.selectbox(
                    "Select Product Type for RTW/Prospecting Analysis:",
                    options=rtw_product_types,
                    key="rtw_product_selector"
                )
                rtw_filtered_data = rtw_data[rtw_data["Product"] == selected_rtw_product].copy()
                rtw_filtered_data["RTW/Prospecting"] = rtw_filtered_data["RTW/Prospecting"].fillna("Unknown")
                years = sorted(rtw_filtered_data["Year"].unique())
                has_multiple_years = len(years) >= 2
                
                # Filter metrics based on selection for RTW analysis
                rtw_metrics = [m for m in ["Impressions", "Clicks", "Spend", "Sales", "Orders", "Units", "CTR", "CVR", "CPC", "ACOS"] if m in selected_metrics]

            if has_multiple_years:
                current_year = max(years)
                previous_year = current_year - 1

                current_year_data = rtw_filtered_data[rtw_filtered_data["Year"] == current_year]
                current_pivot = current_year_data.groupby("RTW/Prospecting", as_index=False).agg({
                    "Impressions": "sum",
                    "Clicks": "sum",
                    "Spend": "sum",
                    "Sales": "sum",
                    "Orders": "sum",
                    "Units": "sum"
                })
                if not current_pivot.empty:
                    current_pivot["CTR"] = (current_pivot["Clicks"] / current_pivot["Impressions"] * 100).replace([float("inf"), -float("inf")], 0)
                    current_pivot["CVR"] = (current_pivot["Orders"] / current_pivot["Clicks"] * 100).replace([float("inf"), -float("inf")], 0)
                    current_pivot["CPC"] = (current_pivot["Spend"] / current_pivot["Clicks"]).replace([float("inf"), -float("inf")], 0)
                    current_pivot["ACOS"] = (current_pivot["Spend"] / current_pivot["Sales"] * 100).replace([float("inf"), -float("inf")], 0)

                previous_year_data = rtw_filtered_data[rtw_filtered_data["Year"] == previous_year]
                previous_pivot = previous_year_data.groupby("RTW/Prospecting", as_index=False).agg({
                    "Impressions": "sum",
                    "Clicks": "sum",
                    "Spend": "sum",
                    "Sales": "sum",
                    "Orders": "sum",
                    "Units": "sum"
                })
                if not previous_pivot.empty:
                    previous_pivot["CTR"] = (previous_pivot["Clicks"] / previous_pivot["Impressions"] * 100).replace([float("inf"), -float("inf")], 0)
                    previous_pivot["CVR"] = (previous_pivot["Orders"] / previous_pivot["Clicks"] * 100).replace([float("inf"), -float("inf")], 0)
                    previous_pivot["CPC"] = (previous_pivot["Spend"] / previous_pivot["Clicks"]).replace([float("inf"), -float("inf")], 0)
                    previous_pivot["ACOS"] = (previous_pivot["Spend"] / previous_pivot["Sales"] * 100).replace([float("inf"), -float("inf")], 0)

                all_categories = pd.concat([
                    current_pivot["RTW/Prospecting"] if not current_pivot.empty else pd.Series(),
                    previous_pivot["RTW/Prospecting"] if not previous_pivot.empty else pd.Series()
                ]).drop_duplicates()

                yoy_rtw = pd.DataFrame({"RTW/Prospecting": all_categories})

                for metric in rtw_metrics:  # Only use selected metrics
                    if not previous_pivot.empty:
                        yoy_rtw = yoy_rtw.merge(
                            previous_pivot[["RTW/Prospecting", metric]].rename(columns={metric: f"{metric} {previous_year}"}),
                            on="RTW/Prospecting", how="left"
                        )
                    else:
                        yoy_rtw[f"{metric} {previous_year}"] = 0

                    if not current_pivot.empty:
                        yoy_rtw = yoy_rtw.merge(
                            current_pivot[["RTW/Prospecting", metric]].rename(columns={metric: f"{metric} {current_year}"}),
                            on="RTW/Prospecting", how="left"
                        )
                    else:
                        yoy_rtw[f"{metric} {current_year}"] = 0

                    yoy_rtw[f"{metric} {previous_year}"] = yoy_rtw[f"{metric} {previous_year}"].fillna(0)
                    yoy_rtw[f"{metric} {current_year}"] = yoy_rtw[f"{metric} {current_year}"].fillna(0)

                    yoy_rtw[f"{metric} % Change"] = yoy_rtw.apply(
                        lambda row: ((row[f"{metric} {current_year}"] - row[f"{metric} {previous_year}"]) / row[f"{metric} {previous_year}"] * 100)
                        if row[f"{metric} {previous_year}"] != 0 else float("inf") if row[f"{metric} {current_year}"] > 0 else 0,
                        axis=1
                    )
                    
                # Sort by current year spend if available
                if "Spend" in rtw_metrics:
                    yoy_rtw = yoy_rtw.sort_values(f"Spend {current_year}", ascending=False)
                elif len(rtw_metrics) > 0:
                    yoy_rtw = yoy_rtw.sort_values(f"{rtw_metrics[0]} {current_year}", ascending=False)

                summary_row = {"RTW/Prospecting": f"TOTAL - {selected_rtw_product}"}
                for metric in rtw_metrics:
                    summary_row[f"{metric} {previous_year}"] = yoy_rtw[f"{metric} {previous_year}"].sum()
                    summary_row[f"{metric} {current_year}"] = yoy_rtw[f"{metric} {current_year}"].sum()
                    
                    # Recalculate derived metrics for summary
                    if metric == "CTR":
                        prev_impressions = yoy_rtw[f"Impressions {previous_year}"].sum() if f"Impressions {previous_year}" in yoy_rtw.columns else 0
                        prev_clicks = yoy_rtw[f"Clicks {previous_year}"].sum() if f"Clicks {previous_year}" in yoy_rtw.columns else 0
                        curr_impressions = yoy_rtw[f"Impressions {current_year}"].sum() if f"Impressions {current_year}" in yoy_rtw.columns else 0
                        curr_clicks = yoy_rtw[f"Clicks {current_year}"].sum() if f"Clicks {current_year}" in yoy_rtw.columns else 0
                        summary_row[f"CTR {previous_year}"] = (prev_clicks / prev_impressions * 100) if prev_impressions > 0 else 0
                        summary_row[f"CTR {current_year}"] = (curr_clicks / curr_impressions * 100) if curr_impressions > 0 else 0
                    elif metric == "CVR":
                        prev_clicks = yoy_rtw[f"Clicks {previous_year}"].sum() if f"Clicks {previous_year}" in yoy_rtw.columns else 0
                        prev_orders = yoy_rtw[f"Orders {previous_year}"].sum() if f"Orders {previous_year}" in yoy_rtw.columns else 0
                        curr_clicks = yoy_rtw[f"Clicks {current_year}"].sum() if f"Clicks {current_year}" in yoy_rtw.columns else 0
                        curr_orders = yoy_rtw[f"Orders {current_year}"].sum() if f"Orders {current_year}" in yoy_rtw.columns else 0
                        summary_row[f"CVR {previous_year}"] = (prev_orders / prev_clicks * 100) if prev_clicks > 0 else 0
                        summary_row[f"CVR {current_year}"] = (curr_orders / curr_clicks * 100) if curr_clicks > 0 else 0
                    elif metric == "CPC":
                        prev_clicks = yoy_rtw[f"Clicks {previous_year}"].sum() if f"Clicks {previous_year}" in yoy_rtw.columns else 0
                        prev_spend = yoy_rtw[f"Spend {previous_year}"].sum() if f"Spend {previous_year}" in yoy_rtw.columns else 0
                        curr_clicks = yoy_rtw[f"Clicks {current_year}"].sum() if f"Clicks {current_year}" in yoy_rtw.columns else 0
                        curr_spend = yoy_rtw[f"Spend {current_year}"].sum() if f"Spend {current_year}" in yoy_rtw.columns else 0
                        summary_row[f"CPC {previous_year}"] = (prev_spend / prev_clicks) if prev_clicks > 0 else 0
                        summary_row[f"CPC {current_year}"] = (curr_spend / curr_clicks) if curr_clicks > 0 else 0
                    elif metric == "ACOS":
                        prev_spend = yoy_rtw[f"Spend {previous_year}"].sum() if f"Spend {previous_year}" in yoy_rtw.columns else 0
                        prev_sales = yoy_rtw[f"Sales {previous_year}"].sum() if f"Sales {previous_year}" in yoy_rtw.columns else 0
                        curr_spend = yoy_rtw[f"Spend {current_year}"].sum() if f"Spend {current_year}" in yoy_rtw.columns else 0
                        curr_sales = yoy_rtw[f"Sales {current_year}"].sum() if f"Sales {current_year}" in yoy_rtw.columns else 0
                        summary_row[f"ACOS {previous_year}"] = (prev_spend / prev_sales * 100) if prev_sales > 0 else 0
                        summary_row[f"ACOS {current_year}"] = (curr_spend / curr_sales * 100) if curr_sales > 0 else 0

                    # Calculate percent change for summary
                    prev_val = summary_row[f"{metric} {previous_year}"]
                    curr_val = summary_row[f"{metric} {current_year}"]
                    if prev_val != 0:
                        pct_change = ((curr_val - prev_val) / prev_val) * 100
                    else:
                        pct_change = float("inf") if curr_val > 0 else 0
                    summary_row[f"{metric} % Change"] = pct_change
                    
                # Separate main table and summary row for RTW
                main_rtw_table = yoy_rtw.copy()
                summary_df = pd.DataFrame([summary_row])

                format_dict = {}
                for col in yoy_rtw.columns:
                    if "% Change" in col:
                        format_dict[col] = "{:.0f}%"
                    elif any(metric in col for metric in ["Impressions", "Clicks", "Orders", "Units"]):
                        format_dict[col] = "{:,.0f}"
                    elif any(metric in col for metric in ["Spend", "Sales"]):
                        format_dict[col] = "${:,.2f}"
                    elif any(metric in col for metric in ["CTR", "CVR", "ACOS"]):
                        format_dict[col] = "{:.1f}%"
                    elif "CPC" in col:
                        format_dict[col] = "${:.2f}"
                        
                styled_table = yoy_rtw.style.format(format_dict)
                pct_change_cols = [col for col in yoy_rtw.columns if "% Change" in col]
                styled_table = styled_table.applymap(
                    lambda v: "color: green" if (isinstance(v, (int, float)) and v > 0)
                    else ("color: red" if (isinstance(v, (int, float)) and v < 0) else ""),
                    subset=pct_change_cols
                )
                st.dataframe(styled_table, use_container_width=True)

                st.markdown("###### Total")
                styled_summary_table = summary_df.style.format(format_dict)
                styled_summary_table = styled_summary_table.set_properties(**{"font-weight": "bold"})
                st.dataframe(styled_summary_table, use_container_width=True)

            else:
                # Single year RTW analysis with selected metrics
                rtw_pivot = rtw_filtered_data.groupby("RTW/Prospecting", as_index=False).agg({
                    "Impressions": "sum",
                    "Clicks": "sum",
                    "Spend": "sum",
                    "Sales": "sum",
                    "Orders": "sum",
                    "Units": "sum"
                })
                rtw_pivot["CTR"] = (rtw_pivot["Clicks"] / rtw_pivot["Impressions"] * 100).replace([float("inf"), -float("inf")], 0)
                rtw_pivot["CVR"] = (rtw_pivot["Orders"] / rtw_pivot["Clicks"] * 100).replace([float("inf"), -float("inf")], 0)
                rtw_pivot["CPC"] = (rtw_pivot["Spend"] / rtw_pivot["Clicks"]).replace([float("inf"), -float("inf")], 0)
                rtw_pivot["ACOS"] = (rtw_pivot["Spend"] / rtw_pivot["Sales"] * 100).replace([float("inf"), -float("inf")], 0)
                
                # Filter to include only selected metrics
                filtered_cols = ["RTW/Prospecting"] + rtw_metrics
                rtw_pivot = rtw_pivot[filtered_cols]
                
                # Sort by Spend if available
                if "Spend" in rtw_metrics:
                    rtw_pivot = rtw_pivot.sort_values("Spend", ascending=False)
                    
                # Create summary row for selected metrics
                summary_row = {
                    "RTW/Prospecting": f"TOTAL - {selected_rtw_product}"
                }
                
                # Calculate totals for selected metrics
                for metric in rtw_metrics:
                    if metric in ["Impressions", "Clicks", "Spend", "Sales", "Orders", "Units"]:
                        summary_row[metric] = rtw_pivot[metric].sum()
                        
                # Calculate derived metrics for summary
                if "CTR" in rtw_metrics and "Clicks" in rtw_metrics and "Impressions" in rtw_metrics:
                    summary_row["CTR"] = (summary_row["Clicks"] / summary_row["Impressions"] * 100) if summary_row.get("Impressions", 0) > 0 else 0
                if "CVR" in rtw_metrics and "Orders" in rtw_metrics and "Clicks" in rtw_metrics:
                    summary_row["CVR"] = (summary_row["Orders"] / summary_row["Clicks"] * 100) if summary_row.get("Clicks", 0) > 0 else 0
                if "CPC" in rtw_metrics and "Spend" in rtw_metrics and "Clicks" in rtw_metrics:
                    summary_row["CPC"] = (summary_row["Spend"] / summary_row["Clicks"]) if summary_row.get("Clicks", 0) > 0 else 0
                if "ACOS" in rtw_metrics and "Spend" in rtw_metrics and "Sales" in rtw_metrics:
                    summary_row["ACOS"] = (summary_row["Spend"] / summary_row["Sales"] * 100) if summary_row.get("Sales", 0) > 0 else 0
                
                main_table = rtw_pivot.copy()
                summary_df = pd.DataFrame([summary_row])

                def style_rtw_table(df):
                    format_dict = {}
                    for col in df.columns:
                        if col == "RTW/Prospecting":
                            continue
                        elif col in ["Impressions", "Clicks", "Orders", "Units"]:
                            format_dict[col] = "{:,}"
                        elif col in ["Spend", "Sales"]:
                            format_dict[col] = "${:,.0f}"
                        elif col in ["CTR", "CVR", "ACOS"]:
                            format_dict[col] = "{:.1f}%"
                        elif col == "CPC":
                            format_dict[col] = "${:.2f}"
                    return df.style.format(format_dict)

                st.dataframe(style_rtw_table(main_table), use_container_width=True)
                st.markdown("###### Total")
                st.dataframe(style_rtw_table(summary_df), use_container_width=True)
            st.markdown("")
        
        # -------------------------------
        # Tab 1: Sponsored Products
        # -------------------------------
        with tabs_adv[1]:
            st.markdown("### Sponsored Products Performance")
            ad_data = preprocess_ad_data(st.session_state["ad_data"])
            if "Product" in ad_data.columns and "Sponsored Products" in ad_data["Product"].unique():
                with st.expander("Filters", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        available_metrics = [
                            "Impressions", "Clicks", "Spend", "Sales", "Orders", "Units",
                            "CTR", "CVR", "ACOS", "ROAS"
                        ]
                        available_metrics = [m for m in available_metrics if m in ad_data.columns]
                        selected_metric = st.selectbox(
                            "Select Metric to Visualize",
                            options=available_metrics,
                            index=0,
                            key="sp_metric"
                        )
                    with col2:
                        portfolio_options = ["All Portfolios"] + sorted(ad_data["Portfolio Name"].unique().tolist())
                        selected_portfolio = st.selectbox(
                            "Select Portfolio",
                            options=portfolio_options,
                            index=0,
                            key="sp_portfolio"
                        )
                    show_yoy = st.checkbox("Show Year-over-Year Comparison", value=True, key="sp_show_yoy")
                    date_range = st.date_input(
                        "Select Date Range",
                        value=(ad_data["WE Date"].min().date(), ad_data["WE Date"].max().date()),
                        min_value=ad_data["WE Date"].min().date(),
                        max_value=ad_data["WE Date"].max().date(),
                        key="sp_date_range"
                    )
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    date_filtered_data = ad_data[
                        (ad_data["WE Date"].dt.date >= start_date) &
                        (ad_data["WE Date"].dt.date <= end_date)
                    ]
                else:
                    date_filtered_data = ad_data
                st.subheader(f"{selected_metric} Over Time")
                fig1 = create_metric_over_time_chart(
                    date_filtered_data,
                    selected_metric,
                    selected_portfolio,
                    "Sponsored Products",
                    show_yoy=show_yoy
                )
                st.plotly_chart(fig1, use_container_width=True)
                if selected_portfolio == "All Portfolios":
                    st.subheader(f"{selected_metric} by Portfolio")
                    fig2 = create_metric_comparison_chart(
                        date_filtered_data,
                        selected_metric,
                        None,
                        "Sponsored Products"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                st.subheader("Total Summary")
                metrics_table, total_summary = create_performance_metrics_table(
                    date_filtered_data,
                    selected_portfolio if selected_portfolio != "All Portfolios" else None,
                    "Sponsored Products"
                )
                st.dataframe(style_total_summary(total_summary), use_container_width=True)
                st.subheader("Performance Metrics by Portfolio")
                st.dataframe(style_metrics_table(metrics_table), use_container_width=True)
                st.subheader("Key Insights")
                total_metrics = total_summary.iloc[0]
                insights = generate_insights(total_metrics, "Sponsored Products")
                for insight in insights:
                    st.markdown(insight)
            else:
                st.warning("No Sponsored Products data found in the uploaded file. Please check your data.")
        # -------------------------------
        # Tab 2: Sponsored Brands
        # -------------------------------
        with tabs_adv[2]:
            st.markdown("### Sponsored Brands Performance")
            ad_data = preprocess_ad_data(st.session_state["ad_data"])
            if "Product" in ad_data.columns and "Sponsored Brands" in ad_data["Product"].unique():
                with st.expander("Filters", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        available_metrics = [
                            "Impressions", "Clicks", "Spend", "Sales", "Orders", "Units",
                            "CTR", "CVR", "ACOS", "ROAS"
                        ]
                        available_metrics = [m for m in available_metrics if m in ad_data.columns]
                        selected_metric = st.selectbox(
                            "Select Metric to Visualize",
                            options=available_metrics,
                            index=0,
                            key="sb_metric"
                        )
                    with col2:
                        portfolio_options = ["All Portfolios"] + sorted(ad_data["Portfolio Name"].unique().tolist())
                        selected_portfolio = st.selectbox(
                            "Select Portfolio",
                            options=portfolio_options,
                            index=0,
                            key="sb_portfolio"
                        )
                    show_yoy = st.checkbox("Show Year-over-Year Comparison", value=True, key="sb_show_yoy")
                    date_range = st.date_input(
                        "Select Date Range",
                        value=(ad_data["WE Date"].min().date(), ad_data["WE Date"].max().date()),
                        min_value=ad_data["WE Date"].min().date(),
                        max_value=ad_data["WE Date"].max().date(),
                        key="sb_date_range"
                    )
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    date_filtered_data = ad_data[
                        (ad_data["WE Date"].dt.date >= start_date) &
                        (ad_data["WE Date"].dt.date <= end_date)
                    ]
                else:
                    date_filtered_data = ad_data
                st.subheader(f"{selected_metric} Over Time")
                fig1 = create_metric_over_time_chart(
                    date_filtered_data,
                    selected_metric,
                    selected_portfolio,
                    "Sponsored Brands",
                    show_yoy=show_yoy
                )
                st.plotly_chart(fig1, use_container_width=True)
                if selected_portfolio == "All Portfolios":
                    st.subheader(f"{selected_metric} by Portfolio")
                    fig2 = create_metric_comparison_chart(
                        date_filtered_data,
                        selected_metric,
                        None,
                        "Sponsored Brands"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                st.subheader("Total Summary")
                metrics_table, total_summary = create_performance_metrics_table(
                    date_filtered_data,
                    selected_portfolio if selected_portfolio != "All Portfolios" else None,
                    "Sponsored Brands"
                )
                st.dataframe(style_total_summary(total_summary), use_container_width=True)
                st.subheader("Performance Metrics by Portfolio")
                st.dataframe(style_metrics_table(metrics_table), use_container_width=True)
                st.subheader("Key Insights")
                total_metrics = total_summary.iloc[0]
                insights = []
                if pd.isna(total_metrics["ACOS"]):
                    insights.append("⚠️ There are ad expenses but no sales. Review your campaigns immediately.")
                elif total_metrics["ACOS"] > 15:
                    insights.append("⚠️ Overall ACOS is high (above 15%). Consider optimizing campaigns to improve efficiency.")
                else:
                    insights.append("✅ Overall ACOS is within acceptable range.")
                if pd.isna(total_metrics["ROAS"]):
                    insights.append("⚠️ No return on ad spend. Your ads are not generating sales.")
                elif total_metrics["ROAS"] < 3:
                    insights.append("⚠️ Overall ROAS is below target (less than 3). Review keyword and bid strategies.")
                else:
                    insights.append("✅ Overall ROAS is good, indicating efficient ad spend.")
                if total_metrics["CTR"] < 0.3:
                    insights.append("⚠️ Click-through rate is low. Consider improving ad creative or targeting.")
                else:
                    insights.append("✅ Click-through rate is satisfactory.")
                if total_metrics["CVR"] < 10:
                    insights.append("⚠️ Conversion rate is below 10%. Review product listings and targeting.")
                else:
                    insights.append("✅ Conversion rate is good.")
                for insight in insights:
                    st.markdown(insight)
            else:
                st.warning("No Sponsored Brands data found in the uploaded file. Please check your data.")
        # -------------------------------
        # Tab 3: Sponsored Display
        # -------------------------------
        with tabs_adv[3]:
            st.markdown("### Sponsored Display Performance")
            ad_data = preprocess_ad_data(st.session_state["ad_data"])
            if "Product" in ad_data.columns and "Sponsored Display" in ad_data["Product"].unique():
                with st.expander("Filters", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        available_metrics = [
                            "Impressions", "Clicks", "Spend", "Sales", "Orders", "Units",
                            "CTR", "CVR", "ACOS", "ROAS"
                        ]
                        available_metrics = [m for m in available_metrics if m in ad_data.columns]
                        selected_metric = st.selectbox(
                            "Select Metric to Visualize",
                            options=available_metrics,
                            index=0,
                            key="sd_metric"
                        )
                    with col2:
                        portfolio_options = ["All Portfolios"] + sorted(ad_data["Portfolio Name"].unique().tolist())
                        selected_portfolio = st.selectbox(
                            "Select Portfolio",
                            options=portfolio_options,
                            index=0,
                            key="sd_portfolio"
                        )
                    show_yoy = st.checkbox("Show Year-over-Year Comparison", value=True, key="sd_show_yoy")
                    date_range = st.date_input(
                        "Select Date Range",
                        value=(ad_data["WE Date"].min().date(), ad_data["WE Date"].max().date()),
                        min_value=ad_data["WE Date"].min().date(),
                        max_value=ad_data["WE Date"].max().date(),
                        key="sd_date_range"
                    )
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    date_filtered_data = ad_data[
                        (ad_data["WE Date"].dt.date >= start_date) &
                        (ad_data["WE Date"].dt.date <= end_date)
                    ]
                else:
                    date_filtered_data = ad_data
                st.subheader(f"{selected_metric} Over Time")
                fig1 = create_metric_over_time_chart(
                    date_filtered_data,
                    selected_metric,
                    selected_portfolio,
                    "Sponsored Display",
                    show_yoy=show_yoy
                )
                st.plotly_chart(fig1, use_container_width=True)
                if selected_portfolio == "All Portfolios":
                    st.subheader(f"{selected_metric} by Portfolio")
                    fig2 = create_metric_comparison_chart(
                        date_filtered_data,
                        selected_metric,
                        None,
                        "Sponsored Display"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                st.subheader("Total Summary")
                metrics_table, total_summary = create_performance_metrics_table(
                    date_filtered_data,
                    selected_portfolio if selected_portfolio != "All Portfolios" else None,
                    "Sponsored Display"
                )
                st.dataframe(style_total_summary(total_summary), use_container_width=True)
                st.subheader("Performance Metrics by Portfolio")
                st.dataframe(style_metrics_table(metrics_table), use_container_width=True)
                st.subheader("Key Insights")
                total_metrics = total_summary.iloc[0]
                insights = generate_insights(total_metrics, "Sponsored Display")
                for insight in insights:
                    st.markdown(insight)
            else:
                st.warning("No Sponsored Display data found in the uploaded file. Please check your data.")