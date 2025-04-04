import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="IT Companies Financial Model", layout="wide")

# Define the main function
def main():
    st.title("IT Companies Financial Model")
    
    st.markdown("""
    This tool analyzes IT companies based on financial metrics and ranks them according to investment criteria 
    including ROE, ROCE, free cash flow, PE ratio, and principles from Warren Buffett and Peter Lynch.
    Upload your company data to get started!
    """)
    
    # File upload section
    st.header("Upload Data & Set Parameters")
    
    uploaded_file = st.file_uploader("Upload CSV file with company data", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Display the uploaded data
            st.subheader("Uploaded Data")
            st.dataframe(df)
            
            # Check if required columns exist
            required_columns = [
                'Name', 'NSE_Code', 'Return_on_equity', 'Return_on_capital_employed',
                'Free_cash_flow', 'Price_to_Earning', 'Debt_to_equity', 'OPM'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.info("Please ensure your CSV has the following columns: Name, NSE_Code, Return_on_equity, Return_on_capital_employed, Free_cash_flow, Price_to_Earning, Debt_to_equity, OPM")
            else:
                # Continue with analysis
                analyze_data(df)
        except Exception as e:
            st.error(f"Error reading the file: {e}")
    else:
        # Show required CSV format when no file is uploaded
        display_csv_format()

# Function to display required CSV format
def display_csv_format():
    st.subheader("Required CSV Format")
    st.write("Your CSV file should include the following columns:")
    
    format_info = [
        {"Column": "Name", "Description": "Company name"},
        {"Column": "NSE_Code", "Description": "Stock symbol on NSE"},
        {"Column": "Return_on_equity", "Description": "ROE percentage"},
        {"Column": "Return_on_capital_employed", "Description": "ROCE percentage"},
        {"Column": "Free_cash_flow", "Description": "FCF in crores"},
        {"Column": "Price_to_Earning", "Description": "P/E ratio"},
        {"Column": "Debt_to_equity", "Description": "D/E ratio"},
        {"Column": "OPM", "Description": "Operating Profit Margin percentage"}
    ]
    
    st.table(pd.DataFrame(format_info))
    
    # Sample data
    st.subheader("Sample Data")
    sample_data = {
        "Name": ["TCS", "Infosys", "Wipro"],
        "NSE_Code": ["TCS", "INFY", "WIPRO"],
        "Return_on_equity": [42.5, 27.8, 17.3],
        "Return_on_capital_employed": [38.2, 33.5, 19.8],
        "Free_cash_flow": [32500, 18700, 8900],
        "Price_to_Earning": [28.5, 24.7, 19.2],
        "Debt_to_equity": [0.12, 0.08, 0.31],
        "OPM": [25.3, 24.1, 18.7]
    }
    st.dataframe(pd.DataFrame(sample_data))
    
# Function to analyze uploaded data
def analyze_data(df):
    st.header("Analysis & Ranking")
    
    # Sidebar for weights
    st.sidebar.header("Adjust Ranking Weights")
    st.sidebar.write("Set the importance of each factor in the ranking (total: 100%)")
    
    # Get user weights
    roe_weight = st.sidebar.slider("Return on Equity", 0.0, 0.5, 0.20, 0.01)
    debt_equity_weight = st.sidebar.slider("Debt to Equity", 0.0, 0.5, 0.10, 0.01)
    roce_weight = st.sidebar.slider("Return on Capital Employed", 0.0, 0.5, 0.20, 0.01)
    profit_growth_weight = st.sidebar.slider("Profit Growth", 0.0, 0.5, 0.10, 0.01)
    fcf_weight = st.sidebar.slider("Free Cash Flow", 0.0, 0.5, 0.20, 0.01)
    pe_weight = st.sidebar.slider("Price to Earning", 0.0, 0.5, 0.10, 0.01)
    opm_weight = st.sidebar.slider("Operating Profit Margin", 0.0, 0.5, 0.10, 0.01)
    
    # Calculate total weight
    total_weight = round(roe_weight + debt_equity_weight + roce_weight + profit_growth_weight + 
                          fcf_weight + pe_weight + opm_weight, 2)
    
    st.sidebar.write(f"Total weight: {total_weight * 100}%")
    
    if total_weight != 1.0:
        st.sidebar.warning("Total weight should be 100%")
    
    # Normalize the data for ranking
    df_normalized = df.copy()
    
    # Higher is better
    for col in ['Return_on_equity', 'Return_on_capital_employed', 'Free_cash_flow', 'OPM']:
        if col in df.columns:
            max_val = df[col].max()
            min_val = df[col].min()
            df_normalized[f'{col}_normalized'] = (df[col] - min_val) / (max_val - min_val) if max_val != min_val else 0.5
    
    # Lower is better
    for col in ['Price_to_Earning', 'Debt_to_equity']:
        if col in df.columns:
            max_val = df[col].max()
            min_val = df[col].min()
            df_normalized[f'{col}_normalized'] = 1 - ((df[col] - min_val) / (max_val - min_val)) if max_val != min_val else 0.5
    
    # Calculate weighted score
    df_normalized['weighted_score'] = (
        roe_weight * df_normalized['Return_on_equity_normalized'] +
        debt_equity_weight * df_normalized['Debt_to_equity_normalized'] +
        roce_weight * df_normalized['Return_on_capital_employed_normalized'] +
        fcf_weight * df_normalized['Free_cash_flow_normalized'] +
        pe_weight * df_normalized['Price_to_Earning_normalized'] +
        opm_weight * df_normalized['OPM_normalized']
    )
    
    # Rank companies
    df_normalized['rank'] = df_normalized['weighted_score'].rank(ascending=False)
    df_ranked = df_normalized.sort_values('rank')
    
    # Display ranked companies
    st.subheader("Ranked Companies")
    ranked_display = df_ranked[['Name', 'NSE_Code', 'rank', 'weighted_score']].copy()
    ranked_display['weighted_score'] = ranked_display['weighted_score'].round(3)
    st.dataframe(ranked_display)
    
    # Visualizations
    st.header("Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 5 Companies by Score")
        top_5 = df_ranked.head(5)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='weighted_score', y='Name', data=top_5, ax=ax)
        ax.set_title('Top 5 Companies by Weighted Score')
        ax.set_xlabel('Weighted Score')
        ax.set_ylabel('Company')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Return on Equity vs ROCE")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='Return_on_equity', y='Return_on_capital_employed', 
                        size='Free_cash_flow', hue='Debt_to_equity',
                        sizes=(50, 400), data=df, ax=ax)
        for i, row in df.iterrows():
            ax.text(row['Return_on_equity'], row['Return_on_capital_employed'], row['Name'], 
                   fontsize=9)
        ax.set_title('ROE vs ROCE (Size: FCF, Color: Debt/Equity)')
        ax.set_xlabel('Return on Equity (%)')
        ax.set_ylabel('Return on Capital Employed (%)')
        st.pyplot(fig)
    
    # Financial Metrics Comparison
    st.subheader("Financial Metrics Comparison")
    
    # Select companies to compare
    companies_to_compare = st.multiselect(
        "Select companies to compare",
        options=df['Name'].tolist(),
        default=df_ranked['Name'].head(3).tolist()
    )
    
    if companies_to_compare:
        comparison_df = df[df['Name'].isin(companies_to_compare)]
        
        # Radar chart
        st.subheader("Financial Metrics Radar Chart")
        
        # Prepare data for radar chart
        metrics = ['Return_on_equity', 'Return_on_capital_employed', 'OPM', 
                  'Free_cash_flow', 'Price_to_Earning', 'Debt_to_equity']
        
        # Normalize data for radar chart
        radar_df = comparison_df.copy()
        for metric in metrics:
            max_val = df[metric].max()
            min_val = df[metric].min()
            
            # For metrics where higher is better
            if metric in ['Return_on_equity', 'Return_on_capital_employed', 'OPM', 'Free_cash_flow']:
                radar_df[f'{metric}_norm'] = (radar_df[metric] - min_val) / (max_val - min_val) if max_val != min_val else 0.5
            # For metrics where lower is better
            else:
                radar_df[f'{metric}_norm'] = 1 - ((radar_df[metric] - min_val) / (max_val - min_val)) if max_val != min_val else 0.5
        
        # Create radar chart
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # Number of variables
        categories = ['ROE', 'ROCE', 'OPM', 'FCF', 'P/E', 'D/E']
        N = len(categories)
        
        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], categories)
        
        # Draw the chart for each company
        for i, company in enumerate(radar_df['Name']):
            values = [
                radar_df.loc[radar_df['Name'] == company, 'Return_on_equity_norm'].values[0],
                radar_df.loc[radar_df['Name'] == company, 'Return_on_capital_employed_norm'].values[0],
                radar_df.loc[radar_df['Name'] == company, 'OPM_norm'].values[0],
                radar_df.loc[radar_df['Name'] == company, 'Free_cash_flow_norm'].values[0],
                radar_df.loc[radar_df['Name'] == company, 'Price_to_Earning_norm'].values[0],
                radar_df.loc[radar_df['Name'] == company, 'Debt_to_equity_norm'].values[0]
            ]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=company)
            ax.fill(angles, values, alpha=0.1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        st.pyplot(fig)
        
        # Bar chart comparison
        st.subheader("Key Metrics Comparison")
        
        metric_to_compare = st.selectbox(
            "Select metric to compare",
            options=['Return_on_equity', 'Return_on_capital_employed', 'Free_cash_flow', 
                    'Price_to_Earning', 'Debt_to_equity', 'OPM']
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Name', y=metric_to_compare, data=comparison_df, ax=ax)
        ax.set_title(f'Comparison of {metric_to_compare}')
        ax.set_xlabel('Company')
        ax.set_ylabel(metric_to_compare)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Investment Recommendations
    st.header("Investment Recommendations")
    
    # Get top companies
    top_companies = df_ranked.head(3)
    
    for idx, row in top_companies.iterrows():
        st.subheader(f"{int(row['rank'])}. {row['Name']} ({row['NSE_Code']})")
        
        # Calculate strengths and weaknesses
        strengths = []
        weaknesses = []
        
        if row['Return_on_equity'] > df['Return_on_equity'].median():
            strengths.append(f"Strong ROE of {row['Return_on_equity']}%")
        else:
            weaknesses.append(f"Below average ROE of {row['Return_on_equity']}%")
            
        if row['Return_on_capital_employed'] > df['Return_on_capital_employed'].median():
            strengths.append(f"High ROCE of {row['Return_on_capital_employed']}%")
        else:
            weaknesses.append(f"Below average ROCE of {row['Return_on_capital_employed']}%")
            
        if row['Debt_to_equity'] < df['Debt_to_equity'].median():
            strengths.append(f"Low debt-to-equity ratio of {row['Debt_to_equity']}")
        else:
            weaknesses.append(f"High debt-to-equity ratio of {row['Debt_to_equity']}")
            
        if row['Price_to_Earning'] < df['Price_to_Earning'].median():
            strengths.append(f"Attractive P/E ratio of {row['Price_to_Earning']}")
        else:
            weaknesses.append(f"High P/E ratio of {row['Price_to_Earning']}")
            
        if row['Free_cash_flow'] > df['Free_cash_flow'].median():
            strengths.append(f"Strong free cash flow of {row['Free_cash_flow']} crores")
        else:
            weaknesses.append(f"Below average free cash flow of {row['Free_cash_flow']} crores")
            
        if row['OPM'] > df['OPM'].median():
            strengths.append(f"Good operating profit margin of {row['OPM']}%")
        else:
            weaknesses.append(f"Below average operating profit margin of {row['OPM']}%")
        
        # Display strengths and weaknesses
        st.write("**Strengths:**")
        for strength in strengths:
            st.write(f"- {strength}")
            
        st.write("**Weaknesses:**")
        for weakness in weaknesses:
            st.write(f"- {weakness}")
        
        # Investment thesis
        st.write("**Investment Thesis:**")
        if len(strengths) > len(weaknesses):
            st.write(f"{row['Name']} shows strong financial performance across key metrics, particularly in {', '.join(strengths[:2])}. With a weighted score of {row['weighted_score']:.3f}, it ranks {int(row['rank'])} among the analyzed companies. Based on Warren Buffett's principles of investing in companies with strong fundamentals, {row['Name']} appears to be a solid investment candidate.")
        else:
            st.write(f"Despite ranking {int(row['rank'])} with a weighted score of {row['weighted_score']:.3f}, {row['Name']} has some concerning metrics, particularly {', '.join(weaknesses[:2])}. While it shows strength in {', '.join(strengths[:1]) if strengths else 'few areas'}, potential investors should carefully consider these weaknesses before investing.")
        
        st.write("---")

    # Disclaimer
    st.header("Disclaimer")
    st.write("""
    This analysis is for informational purposes only and does not constitute investment advice. 
    The rankings are based on the provided data and the weightings assigned to different metrics.
    Always conduct thorough research and consult with a financial advisor before making investment decisions.
    """)

# Run the app
if __name__ == "__main__":
    main()