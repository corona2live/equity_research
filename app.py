import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
import json
from io import StringIO
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="IT Companies Financial Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# App title and description
st.title("IT Companies Financial Model")
st.write("This tool analyzes IT companies based on financial metrics and ranks them according to investment criteria including ROE, ROCE, free cash flow, PE ratio, and principles from Warren Buffett and Peter Lynch. Upload your company data to get started!")

# Custom CSS for styling
st.markdown("""
<style>
    .stSlider > div > div > div > div {
        background-color: #FF4B4B;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1rem;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f0ff;
        border-bottom: 2px solid #4a76c7;
    }
    .upload-container {
        border: 2px dashed #ccc;
        border-radius: 5px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f9f9f9;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #4a76c7;
    }
    footer {display: none !important;}
    .viewerBadge_container__1QSob {display: none !important;}
</style>
""", unsafe_allow_html=True)

# Function to normalize column values
def normalize_column(df, column_name):
    """Normalize column values to 0-100 scale"""
    if column_name not in df.columns:
        return pd.Series([0] * len(df))
        
    min_val = df[column_name].min()
    max_val = df[column_name].max()
    
    if max_val == min_val:
        return pd.Series([50] * len(df))  # If all values are the same
    
    return 100 * (df[column_name] - min_val) / (max_val - min_val)

# Function to rank companies based on given weights
def rank_companies(df, weights):
    """Rank companies based on given weights and financial metrics"""
    # Create copy of dataframe to avoid modifying original
    ranked_df = df.copy()
    
    # Dictionary to map metrics to score columns (also handles inverted metrics where lower is better)
    metrics_mapping = {
        'Return_on_equity': {'score_col': 'ROE_score', 'invert': False},
        'Debt_to_equity': {'score_col': 'DE_score', 'invert': True},
        'Return_on_capital_employed': {'score_col': 'ROCE_score', 'invert': False},
        'Free_cash_flow_last_year': {'score_col': 'FCF_score', 'invert': False},
        'Price_to_Earning': {'score_col': 'PE_score', 'invert': True},
        'OPM': {'score_col': 'OPM_score', 'invert': False}
    }
    
    # Normalize all metrics
    for metric, info in metrics_mapping.items():
        if metric in ranked_df.columns:
            score = normalize_column(ranked_df, metric)
            if info['invert']:
                score = 100 - score
            ranked_df[info['score_col']] = score
    
    # Calculate overall score based on weights
    ranked_df['Total_Score'] = 0
    for metric, info in metrics_mapping.items():
        if metric in weights and info['score_col'] in ranked_df.columns:
            ranked_df['Total_Score'] += ranked_df[info['score_col']] * weights[metric]
    
    # Sort by total score (descending)
    ranked_df = ranked_df.sort_values('Total_Score', ascending=False).reset_index(drop=True)
    
    return ranked_df

# Function to create radar chart
def create_radar_chart(df, metrics):
    """Create a radar chart comparing top companies across metrics"""
    # Limit to top 5 companies
    df = df.head(5)
    
    # Prepare data for radar chart
    companies = df['Name'].tolist() if 'Name' in df.columns else [f"Company {i+1}" for i in range(len(df))]
    
    # Normalize data for radar chart
    radar_df = pd.DataFrame()
    for metric in metrics:
        if metric in df.columns:
            values = normalize_column(df, metric)
            # For metrics where lower is better, invert the score
            if metric in ['Price_to_Earning', 'Debt_to_equity']:
                values = 100 - values
            radar_df[metric] = values
    
    # Number of variables
    categories = radar_df.columns.tolist()
    N = len(categories)
    
    # Create angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Draw one company at a time
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, company in enumerate(companies):
        values = radar_df.iloc[i].values.tolist()
        values += values[:1]  # Close the loop
        
        # Plot company data
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=company, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace('_', ' ').title() for c in categories], fontsize=12)
    
    # Draw y labels
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(['0', '25', '50', '75', '100'])
    ax.set_ylim(0, 100)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    return fig

# Function to create bar charts
def create_bar_charts(df, metrics):
    """Create bar charts for each metric"""
    # Limit to top 5 companies
    top5 = df.head(5).copy()
    
    # Available metrics
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        return None
    
    # Create subplots
    fig, axes = plt.subplots(len(available_metrics), 1, figsize=(10, len(available_metrics)*3))
    
    if len(available_metrics) == 1:
        axes = [axes]  # Make iterable if only one metric is available
    
    for i, metric in enumerate(available_metrics):
        # Create horizontal bar chart
        company_names = top5['Name'] if 'Name' in top5.columns else top5.index
        ax = axes[i]
        bars = ax.barh(company_names, top5[metric], color='#4a76c7')
        ax.set_title(f"{metric.replace('_', ' ').title()}", fontsize=14)
        ax.invert_yaxis()  # To have highest value at the top
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 0.01 * max(top5[metric]) if width > 0 else 0.01 * max(top5[metric])
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                   va='center', fontsize=10)
    
    plt.tight_layout()
    return fig

# Main app function
def main():
    # Create two columns: left for upload and parameters, right for display
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.header("Upload Data & Set Parameters")
        
        # File uploader
        st.markdown("<div class='upload-container'>", unsafe_allow_html=True)
        st.write("Upload CSV file with company data")
        uploaded_file = st.file_uploader("Drag and drop file here", type=["csv"], 
                                        help="Limit 200MB per file • CSV")
        
        if not uploaded_file:
            st.button("Browse files", disabled=False)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Weights sliders
        st.header("Adjust Ranking Weights")
        st.write("Set the importance of each factor in the ranking (total: 100%)")
        
        # Create sliders and store weights
        weights = {}
        metrics = [
            ('Return_on_equity', 'Return on Equity'),
            ('Debt_to_equity', 'Debt to Equity'),
            ('Return_on_capital_employed', 'Return on Capital Employed'),
            ('Free_cash_flow_last_year', 'Free Cash Flow'),
            ('Price_to_Earning', 'Price to Earning'),
            ('OPM', 'Operating Profit Margin')
        ]
        
        default_weights = {
            'Return_on_equity': 0.20,
            'Debt_to_equity': 0.10,
            'Return_on_capital_employed': 0.20,
            'Free_cash_flow_last_year': 0.20,
            'Price_to_Earning': 0.15,
            'OPM': 0.15
        }
        
        # Create sliders for each metric
        for metric_key, metric_name in metrics:
            weights[metric_key] = st.slider(
                metric_name, 
                0.0, 1.0, 
                default_weights[metric_key], 
                0.01, 
                format="%.2f"
            )
        
        # Calculate and display total weight
        total_weight = sum(weights.values())
        st.write(f"Total weight: {total_weight:.2f}")
        
        if abs(total_weight - 1.0) > 0.01:
            st.warning(f"Total weight should be 1.0. Current total: {total_weight:.2f}")
    
    # Right column for results
    with col2:
        if uploaded_file is not None:
            # Process the CSV file
            try:
                df = pd.read_csv(uploaded_file)
                
                # Display success message
                st.success("File uploaded successfully!")
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["📊 Overview", "�� Detailed Analysis", "📈 Visualizations"])
                
                with tab1:
                    # Display the raw data
                    st.subheader("Uploaded Company Data")
                    st.dataframe(df)
                    
                    # Process and rank companies
                    ranked_df = rank_companies(df, weights)
                    
                    # Display top companies
                    st.subheader("Top Recommended Companies")
                    st.dataframe(
                        ranked_df[['Name', 'NSE_Code', 'Total_Score'] + 
                                 [col for col in ranked_df.columns if col in weights.keys()]].head(5),
                        use_container_width=True
                    )
                
                with tab2:
                    # Process and rank companies
                    if 'ranked_df' not in locals():
                        ranked_df = rank_companies(df, weights)
                    
                    # Display individual metrics
                    st.subheader("Company Performance by Metric")
                    
                    # Show top 5 companies for each metric
                    metric_cols = [col for col in df.columns if col in weights.keys()]
                    
                    for i, metric in enumerate(metric_cols):
                        display_metric = metric.replace('_', ' ').title()
                        invert = metric in ['Debt_to_equity', 'Price_to_Earning'] 
                        
                        # Sort based on whether higher or lower is better
                        sorted_df = df.sort_values(by=metric, ascending=invert)
                        
                        # Create a metric card
                        with st.expander(f"{display_metric}", expanded=i==0):
                            st.dataframe(
                                sorted_df[['Name', 'NSE_Code', metric]].head(5),
                                use_container_width=True
                            )
                            
                            # Show industry average if there are enough companies
                            if len(df) > 1:
                                industry_avg = df[metric].mean()
                                st.metric(
                                    "Industry Average", 
                                    f"{industry_avg:.2f}", 
                                    delta=None
                                )
                
                with tab3:
                    # Process and rank companies
                    if 'ranked_df' not in locals():
                        ranked_df = rank_companies(df, weights)
                    
                    # Create visualizations
                    st.subheader("Financial Metrics Comparison")
                    
                    # Bar charts
                    metric_cols = [col for col in df.columns if col in weights.keys()]
                    fig_bars = create_bar_charts(ranked_df, metric_cols)
                    if fig_bars:
                        st.pyplot(fig_bars)
                    
                    # Radar chart if there are enough metrics
                    if len(metric_cols) >= 3:
                        st.subheader("Comparative Analysis (Radar Chart)")
                        fig_radar = create_radar_chart(ranked_df, metric_cols)
                        st.pyplot(fig_radar)
                    
                    # Total score chart
                    st.subheader("Overall Ranking")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    top5 = ranked_df.head(5)
                    bars = ax.barh(top5['Name'], top5['Total_Score'], color='#4a76c7')
                    ax.set_title("Total Score", fontsize=14)
                    ax.invert_yaxis()
                    
                    # Add value labels
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                              va='center', fontsize=10)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("Please make sure your CSV file has the required format.")
        else:
            # Information about required format
            st.info("Please upload a CSV file with company financial data to begin analysis.")
            
            # Display required CSV format
            st.subheader("Required CSV Format")
            st.write("Your CSV file should include the following columns:")
            
            requirements = [
                "**Name**: Company name",
                "**NSE_Code**: Stock symbol on NSE",
                "**Return_on_equity**: ROE percentage",
                "**Return_on_capital_employed**: ROCE percentage",
                "**Free_cash_flow_last_year**: FCF in crores",
                "**Price_to_Earning**: P/E ratio",
                "**Debt_to_equity**: D/E ratio",
                "**OPM**: Operating Profit Margin percentage"
            ]
            
            for req in requirements:
                st.markdown(f"* {req}")
                
            # Sample data to show expected format
            sample_data = {
                "Name": ["Infosys", "TCS", "Wipro", "HCL Tech", "Tech Mahindra"],
                "NSE_Code": ["INFY", "TCS", "WIPRO", "HCLTECH", "TECHM"],
                "Return_on_equity": [24.8, 25.6, 17.2, 19.5, 15.8],
                "Return_on_capital_employed": [29.7, 38.2, 21.5, 24.3, 18.7],
                "Free_cash_flow_last_year": [12500, 32600, 7800, 9700, 4200],
                "Price_to_Earning": [23.5, 27.8, 19.2, 18.7, 15.6],
                "Debt_to_equity": [0.12, 0.08, 0.21, 0.15, 0.23],
                "OPM": [24.5, 26.8, 18.9, 21.3, 16.7]
            }
            
            st.subheader("Sample Data Format")
            st.dataframe(pd.DataFrame(sample_data))
            
            # Download sample template
            sample_df = pd.DataFrame(sample_data)
            csv = sample_df.to_csv(index=False)
            st.download_button(
                label="Download Sample Template",
                data=csv,
                file_name="sample_it_companies_template.csv",
                mime="text/csv"
            )

# Add a footer with deployment information
st.markdown("""
---
Made with ❤️ by Financial Analytics Team
""")

# Run the app
if __name__ == "__main__":
    main()import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
import json
from io import StringIO
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="IT Companies Financial Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# App title and description
st.title("IT Companies Financial Model")
st.write("This tool analyzes IT companies based on financial metrics and ranks them according to investment criteria including ROE, ROCE, free cash flow, PE ratio, and principles from Warren Buffett and Peter Lynch. Upload your company data to get started!")

# Custom CSS for styling
st.markdown("""
<style>
    .stSlider > div > div > div > div {
        background-color: #FF4B4B;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1rem;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f0ff;
        border-bottom: 2px solid #4a76c7;
    }
    .upload-container {
        border: 2px dashed #ccc;
        border-radius: 5px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f9f9f9;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #4a76c7;
    }
    footer {display: none !important;}
    .viewerBadge_container__1QSob {display: none !important;}
</style>
""", unsafe_allow_html=True)

# Function to normalize column values
def normalize_column(df, column_name):
    """Normalize column values to 0-100 scale"""
    if column_name not in df.columns:
        return pd.Series([0] * len(df))
        
    min_val = df[column_name].min()
    max_val = df[column_name].max()
    
    if max_val == min_val:
        return pd.Series([50] * len(df))  # If all values are the same
    
    return 100 * (df[column_name] - min_val) / (max_val - min_val)

# Function to rank companies based on given weights
def rank_companies(df, weights):
    """Rank companies based on given weights and financial metrics"""
    # Create copy of dataframe to avoid modifying original
    ranked_df = df.copy()
    
    # Dictionary to map metrics to score columns (also handles inverted metrics where lower is better)
    metrics_mapping = {
        'Return_on_equity': {'score_col': 'ROE_score', 'invert': False},
        'Debt_to_equity': {'score_col': 'DE_score', 'invert': True},
        'Return_on_capital_employed': {'score_col': 'ROCE_score', 'invert': False},
        'Free_cash_flow_last_year': {'score_col': 'FCF_score', 'invert': False},
        'Price_to_Earning': {'score_col': 'PE_score', 'invert': True},
        'OPM': {'score_col': 'OPM_score', 'invert': False}
    }
    
    # Normalize all metrics
    for metric, info in metrics_mapping.items():
        if metric in ranked_df.columns:
            score = normalize_column(ranked_df, metric)
            if info['invert']:
                score = 100 - score
            ranked_df[info['score_col']] = score
    
    # Calculate overall score based on weights
    ranked_df['Total_Score'] = 0
    for metric, info in metrics_mapping.items():
        if metric in weights and info['score_col'] in ranked_df.columns:
            ranked_df['Total_Score'] += ranked_df[info['score_col']] * weights[metric]
    
    # Sort by total score (descending)
    ranked_df = ranked_df.sort_values('Total_Score', ascending=False).reset_index(drop=True)
    
    return ranked_df

# Function to create radar chart
def create_radar_chart(df, metrics):
    """Create a radar chart comparing top companies across metrics"""
    # Limit to top 5 companies
    df = df.head(5)
    
    # Prepare data for radar chart
    companies = df['Name'].tolist() if 'Name' in df.columns else [f"Company {i+1}" for i in range(len(df))]
    
    # Normalize data for radar chart
    radar_df = pd.DataFrame()
    for metric in metrics:
        if metric in df.columns:
            values = normalize_column(df, metric)
            # For metrics where lower is better, invert the score
            if metric in ['Price_to_Earning', 'Debt_to_equity']:
                values = 100 - values
            radar_df[metric] = values
    
    # Number of variables
    categories = radar_df.columns.tolist()
    N = len(categories)
    
    # Create angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Draw one company at a time
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, company in enumerate(companies):
        values = radar_df.iloc[i].values.tolist()
        values += values[:1]  # Close the loop
        
        # Plot company data
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=company, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace('_', ' ').title() for c in categories], fontsize=12)
    
    # Draw y labels
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(['0', '25', '50', '75', '100'])
    ax.set_ylim(0, 100)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    return fig

# Function to create bar charts
def create_bar_charts(df, metrics):
    """Create bar charts for each metric"""
    # Limit to top 5 companies
    top5 = df.head(5).copy()
    
    # Available metrics
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        return None
    
    # Create subplots
    fig, axes = plt.subplots(len(available_metrics), 1, figsize=(10, len(available_metrics)*3))
    
    if len(available_metrics) == 1:
        axes = [axes]  # Make iterable if only one metric is available
    
    for i, metric in enumerate(available_metrics):
        # Create horizontal bar chart
        company_names = top5['Name'] if 'Name' in top5.columns else top5.index
        ax = axes[i]
        bars = ax.barh(company_names, top5[metric], color='#4a76c7')
        ax.set_title(f"{metric.replace('_', ' ').title()}", fontsize=14)
        ax.invert_yaxis()  # To have highest value at the top
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 0.01 * max(top5[metric]) if width > 0 else 0.01 * max(top5[metric])
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                   va='center', fontsize=10)
    
    plt.tight_layout()
    return fig

# Main app function
def main():
    # Create two columns: left for upload and parameters, right for display
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.header("Upload Data & Set Parameters")
        
        # File uploader
        st.markdown("<div class='upload-container'>", unsafe_allow_html=True)
        st.write("Upload CSV file with company data")
        uploaded_file = st.file_uploader("Drag and drop file here", type=["csv"], 
                                        help="Limit 200MB per file • CSV")
        
        if not uploaded_file:
            st.button("Browse files", disabled=False)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Weights sliders
        st.header("Adjust Ranking Weights")
        st.write("Set the importance of each factor in the ranking (total: 100%)")
        
        # Create sliders and store weights
        weights = {}
        metrics = [
            ('Return_on_equity', 'Return on Equity'),
            ('Debt_to_equity', 'Debt to Equity'),
            ('Return_on_capital_employed', 'Return on Capital Employed'),
            ('Free_cash_flow_last_year', 'Free Cash Flow'),
            ('Price_to_Earning', 'Price to Earning'),
            ('OPM', 'Operating Profit Margin')
        ]
        
        default_weights = {
            'Return_on_equity': 0.20,
            'Debt_to_equity': 0.10,
            'Return_on_capital_employed': 0.20,
            'Free_cash_flow_last_year': 0.20,
            'Price_to_Earning': 0.15,
            'OPM': 0.15
        }
        
        # Create sliders for each metric
        for metric_key, metric_name in metrics:
            weights[metric_key] = st.slider(
                metric_name, 
                0.0, 1.0, 
                default_weights[metric_key], 
                0.01, 
                format="%.2f"
            )
        
        # Calculate and display total weight
        total_weight = sum(weights.values())
        st.write(f"Total weight: {total_weight:.2f}")
        
        if abs(total_weight - 1.0) > 0.01:
            st.warning(f"Total weight should be 1.0. Current total: {total_weight:.2f}")
    
    # Right column for results
    with col2:
        if uploaded_file is not None:
            # Process the CSV file
            try:
                df = pd.read_csv(uploaded_file)
                
                # Display success message
                st.success("File uploaded successfully!")
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["📊 Overview", "�� Detailed Analysis", "📈 Visualizations"])
                
                with tab1:
                    # Display the raw data
                    st.subheader("Uploaded Company Data")
                    st.dataframe(df)
                    
                    # Process and rank companies
                    ranked_df = rank_companies(df, weights)
                    
                    # Display top companies
                    st.subheader("Top Recommended Companies")
                    st.dataframe(
                        ranked_df[['Name', 'NSE_Code', 'Total_Score'] + 
                                 [col for col in ranked_df.columns if col in weights.keys()]].head(5),
                        use_container_width=True
                    )
                
                with tab2:
                    # Process and rank companies
                    if 'ranked_df' not in locals():
                        ranked_df = rank_companies(df, weights)
                    
                    # Display individual metrics
                    st.subheader("Company Performance by Metric")
                    
                    # Show top 5 companies for each metric
                    metric_cols = [col for col in df.columns if col in weights.keys()]
                    
                    for i, metric in enumerate(metric_cols):
                        display_metric = metric.replace('_', ' ').title()
                        invert = metric in ['Debt_to_equity', 'Price_to_Earning'] 
                        
                        # Sort based on whether higher or lower is better
                        sorted_df = df.sort_values(by=metric, ascending=invert)
                        
                        # Create a metric card
                        with st.expander(f"{display_metric}", expanded=i==0):
                            st.dataframe(
                                sorted_df[['Name', 'NSE_Code', metric]].head(5),
                                use_container_width=True
                            )
                            
                            # Show industry average if there are enough companies
                            if len(df) > 1:
                                industry_avg = df[metric].mean()
                                st.metric(
                                    "Industry Average", 
                                    f"{industry_avg:.2f}", 
                                    delta=None
                                )
                
                with tab3:
                    # Process and rank companies
                    if 'ranked_df' not in locals():
                        ranked_df = rank_companies(df, weights)
                    
                    # Create visualizations
                    st.subheader("Financial Metrics Comparison")
                    
                    # Bar charts
                    metric_cols = [col for col in df.columns if col in weights.keys()]
                    fig_bars = create_bar_charts(ranked_df, metric_cols)
                    if fig_bars:
                        st.pyplot(fig_bars)
                    
                    # Radar chart if there are enough metrics
                    if len(metric_cols) >= 3:
                        st.subheader("Comparative Analysis (Radar Chart)")
                        fig_radar = create_radar_chart(ranked_df, metric_cols)
                        st.pyplot(fig_radar)
                    
                    # Total score chart
                    st.subheader("Overall Ranking")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    top5 = ranked_df.head(5)
                    bars = ax.barh(top5['Name'], top5['Total_Score'], color='#4a76c7')
                    ax.set_title("Total Score", fontsize=14)
                    ax.invert_yaxis()
                    
                    # Add value labels
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                              va='center', fontsize=10)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("Please make sure your CSV file has the required format.")
        else:
            # Information about required format
            st.info("Please upload a CSV file with company financial data to begin analysis.")
            
            # Display required CSV format
            st.subheader("Required CSV Format")
            st.write("Your CSV file should include the following columns:")
            
            requirements = [
                "**Name**: Company name",
                "**NSE_Code**: Stock symbol on NSE",
                "**Return_on_equity**: ROE percentage",
                "**Return_on_capital_employed**: ROCE percentage",
                "**Free_cash_flow_last_year**: FCF in crores",
                "**Price_to_Earning**: P/E ratio",
                "**Debt_to_equity**: D/E ratio",
                "**OPM**: Operating Profit Margin percentage"
            ]
            
            for req in requirements:
                st.markdown(f"* {req}")
                
            # Sample data to show expected format
            sample_data = {
                "Name": ["Infosys", "TCS", "Wipro", "HCL Tech", "Tech Mahindra"],
                "NSE_Code": ["INFY", "TCS", "WIPRO", "HCLTECH", "TECHM"],
                "Return_on_equity": [24.8, 25.6, 17.2, 19.5, 15.8],
                "Return_on_capital_employed": [29.7, 38.2, 21.5, 24.3, 18.7],
                "Free_cash_flow_last_year": [12500, 32600, 7800, 9700, 4200],
                "Price_to_Earning": [23.5, 27.8, 19.2, 18.7, 15.6],
                "Debt_to_equity": [0.12, 0.08, 0.21, 0.15, 0.23],
                "OPM": [24.5, 26.8, 18.9, 21.3, 16.7]
            }
            
            st.subheader("Sample Data Format")
            st.dataframe(pd.DataFrame(sample_data))
            
            # Download sample template
            sample_df = pd.DataFrame(sample_data)
            csv = sample_df.to_csv(index=False)
            st.download_button(
                label="Download Sample Template",
                data=csv,
                file_name="sample_it_companies_template.csv",
                mime="text/csv"
            )

# Add a footer with deployment information
st.markdown("""
---
Made with ❤️ by Financial Analytics Team
""")

# Run the app
if __name__ == "__main__":
    main()
