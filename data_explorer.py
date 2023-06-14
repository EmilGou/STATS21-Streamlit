import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# Add custom styles for dataframes and markdown text
st.markdown(
    """
    <style>
    table {
        white-space: nowrap;
        text-align: center;
        font-family: Arial, sans-serif;
        color: #6c757d;
        font-size: 0.9em;
    }
    thead tr th {
        color: #495057;
        background-color: #e9ecef;
    }
    tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def get_dtype_count(df, dtype):
    return df.select_dtypes(include=[dtype]).shape[1]

def main():
    st.title("Data Explorer")
    uploaded_file = st.sidebar.file_uploader("Choose a file")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        bool_cols = [col for col in df if df[col].isin([0, 1]).all()]

        for col in bool_cols:
            df[col] = df[col].astype(bool)
        
        st.header("Dataset Preview")
        st.write(df)

        st.header("Dataset Information")
        st.markdown(f"*Number of rows: {df.shape[0]}*")
        st.markdown(f"*Number of columns: {df.shape[1]}*")
        
        types_to_count = {'Numeric': np.number, 'Boolean': 'bool', 'Categorical': 'object'}
        
        for dtype_name, dtype in types_to_count.items():
            count = get_dtype_count(df, dtype)
            st.markdown(f'*Number of {dtype_name} Columns: {count}*')

        df_cols = df.columns.tolist()
        st.header("Data Analysis")
        selected_columns = st.sidebar.multiselect('Select columns', df_cols)

        if selected_columns:
            num_cols = df[selected_columns].select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                st.write(df[num_cols].describe().drop(['count', 'mean','std']).T)

                for col in num_cols:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.histplot(df[col], bins=20, kde=True, color='skyblue', edgecolor='black', linewidth=1.0, ax=ax)
                    ax.set_xlabel(col + ' (thousands)', fontsize=12)
                    ax.set_ylabel('Frequency', fontsize=12)
                    ax.set_title(f'Distribution of {col} (thousands)', fontsize=16)
                    ax.grid(True)

                    # Rotate x-axis labels for better visibility
                    plt.xticks(rotation=45)

                    st.pyplot(fig)

            # Get categorical columns
            cat_cols = df[selected_columns].select_dtypes(include=['object']).columns.tolist()

            # For each categorical column, compute and display proportions
            if cat_cols:
                
                for col in cat_cols:
                    st.write(f"Proportions for {col}:")
    
                    proportions = df[col].value_counts(normalize=True)

                    proportions_df = proportions.to_frame()

                    proportions_df.columns = ['Proportion']

                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.bar(proportions_df.index, proportions_df['Proportion'], color='seagreen')
                    ax.set_xlabel(col, fontsize=12)
                    ax.set_ylabel('Proportion', fontsize=12)
                    ax.set_title(f'Proportions for {col}', fontsize=16)
                    ax.grid(True)

                    st.table(proportions_df)

                    st.pyplot(fig)



if __name__ == "__main__":
    main()
