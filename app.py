import streamlit as st
import pandas as pd
import requests
from datetime import datetime, date
import io
import re
from urllib.parse import urlparse

# Page configuration
st.set_page_config(
    page_title="Academic Articles Browser",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_csv_from_github(url):
    """
    Load CSV data from GitHub URL with multiple encoding attempts
    """
    try:
        # Convert GitHub URL to raw URL if needed
        if 'github.com' in url and '/blob/' in url:
            url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Try multiple encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
        
        for encoding in encodings:
            try:
                content = response.content.decode(encoding)
                df = pd.read_csv(io.StringIO(content), sep=';')
                return df, None
            except (UnicodeDecodeError, pd.errors.EmptyDataError):
                continue
        
        return None, "Unable to decode the CSV file with any supported encoding"
        
    except requests.exceptions.RequestException as e:
        return None, f"Error fetching data from URL: {str(e)}"
    except Exception as e:
        return None, f"Error processing CSV data: {str(e)}"

def clean_and_validate_data(df):
    """
    Clean and validate the loaded data
    """
    try:
        # Expected columns (case-insensitive matching)
        expected_cols = ['title', 'date', 'abstract', 'journal', 'qualis_classification', 'link']
        
        # Create mapping of actual columns to expected columns
        col_mapping = {}
        df_cols_lower = [col.lower().strip() for col in df.columns]
        
        for expected in expected_cols:
            for i, actual in enumerate(df_cols_lower):
                if expected == 'qualis_classification':
                    # Special handling for Qualis Classification
                    if 'qualis classification' in actual or 'qualis' in actual:
                        col_mapping[df.columns[i]] = expected
                        break
                elif expected.replace('_', ' ') in actual or expected in actual:
                    col_mapping[df.columns[i]] = expected
                    break
        
        # Rename columns
        df_renamed = df.rename(columns=col_mapping)
        
        # Check if we have the essential columns
        essential_cols = ['title', 'journal']
        missing_cols = [col for col in essential_cols if col not in df_renamed.columns]
        
        if missing_cols:
            st.warning(f"Missing essential columns: {', '.join(missing_cols)}. Using available columns.")
        
        # Convert date column if it exists
        if 'date' in df_renamed.columns:
            df_renamed['date'] = pd.to_datetime(df_renamed['date'], errors='coerce', dayfirst=True, utc=True)
        
        # Fill missing values
        for col in df_renamed.columns:
            if col in ['title', 'journal', 'abstract', 'qualis_classification']:
                df_renamed[col] = df_renamed[col].fillna('Not available')
            elif col == 'link':
                df_renamed[col] = df_renamed[col].fillna('')
        
        return df_renamed, None
        
    except Exception as e:
        return None, f"Error cleaning data: {str(e)}"

def filter_articles(df, search_term, date_range, journal_classifications, sort_by, sort_order):
    """
    Filter and sort articles based on user inputs
    """
    filtered_df = df.copy()
    
    # Search filter
    if search_term:
        search_cols = ['title', 'abstract', 'journal']
        search_cols = [col for col in search_cols if col in filtered_df.columns]
        
        mask = pd.Series([False] * len(filtered_df))
        for col in search_cols:
            mask |= filtered_df[col].astype(str).str.contains(search_term, case=False, na=False)
        filtered_df = filtered_df[mask]
    
    # Date range filter
    if 'date' in filtered_df.columns and date_range[0] is not None and date_range[1] is not None:
        start_date = pd.Timestamp(date_range[0], tz='UTC')
        end_date = pd.Timestamp(date_range[1], tz='UTC')
        mask = (filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)
        filtered_df = filtered_df[mask]
    
    # QUALIS classification filter
    if journal_classifications and 'qualis_classification' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['qualis_classification'].isin(journal_classifications)]
    
    # Sorting
    if sort_by in filtered_df.columns:
        ascending = (sort_order == "Ascending")
        if sort_by == 'date':
            # Handle NaT values in date sorting
            filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending, na_position='last')
        else:
            filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)
    
    return filtered_df

def display_article_card(article, index):
    """
    Display an individual article in a card format
    """
    with st.container():
        # Title with link
        title = article.get('title', 'No Title')
        if 'link' in article and article['link'] and article['link'] != 'Not available' and article['link'].strip():
            st.markdown(f"### [{title}]({article['link']})")
        else:
            st.markdown(f"### {title}")
        
        # Journal
        journal = article.get('journal', 'Unknown Journal')
        st.markdown(f"**Journal:** {journal}")
        
        # Date and Classification
        if 'date' in article and pd.notna(article['date']):
            date_str = article['date'].strftime('%B %d, %Y') if hasattr(article['date'], 'strftime') else str(article['date'])
            st.markdown(f"**Date:** {date_str}")
        
        if 'qualis_classification' in article:
            classification = article.get('qualis_classification', 'Not classified')
            st.markdown(f"**QUALIS:** {classification}")
        
        # Abstract
        if 'abstract' in article and article['abstract'] and article['abstract'] != 'Not available':
            with st.expander("Abstract"):
                st.write(article['abstract'])
        
        st.divider()

def main():
    st.title("📚 Academic Articles Browser")
    st.markdown("Browse and search academic articles from your GitHub-hosted CSV data")
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'error' not in st.session_state:
        st.session_state.error = None
    
    # Sidebar for data loading
    with st.sidebar:
        st.header("📁 Data Source")
        
        # URL input
        github_url = st.text_input(
            "GitHub CSV URL:",
            placeholder="https://github.com/user/repo/blob/main/data.csv",
            help="Enter the GitHub URL to your CSV file"
        )
        
        load_button = st.button("Load Data", type="primary")
        
        if load_button and github_url:
            with st.spinner("Loading data from GitHub..."):
                data, error = load_csv_from_github(github_url)
                
                if error:
                    st.session_state.error = error
                    st.session_state.data = None
                else:
                    cleaned_data, clean_error = clean_and_validate_data(data)
                    if clean_error:
                        st.session_state.error = clean_error
                        st.session_state.data = None
                    else:
                        st.session_state.data = cleaned_data
                        st.session_state.error = None
                        if cleaned_data is not None and hasattr(cleaned_data, '__len__'):
                            st.success(f"Successfully loaded {len(cleaned_data)} articles!")
                        else:
                            st.success("Successfully loaded data!")
    
    # Display error if any
    if st.session_state.error:
        st.error(f"Error: {st.session_state.error}")
        st.markdown("""
        **Troubleshooting tips:**
        - Make sure the URL is accessible and points to a CSV file
        - Ensure the CSV contains columns like: title, author, journal, date, abstract, etc.
        - Try using the raw GitHub URL (replace 'github.com' with 'raw.githubusercontent.com' and remove '/blob/')
        """)
        return
    
    # Main content
    if st.session_state.data is None:
        st.info("👆 Please load your CSV data using the sidebar to get started.")
        st.markdown("""
        ### Expected CSV Format
        Your CSV should contain the following columns (case-insensitive):
        - **title**: Article title
        - **journal**: Journal name
        - **date**: Publication date
        - **abstract**: Article abstract
        - **qualis classification**: QUALIS classification
        - **link**: Link to the article
        """)
        return
    
    df = st.session_state.data
    
    # Filtering sidebar
    with st.sidebar:
        st.header("🔍 Filters & Search")
        
        # Search
        search_term = st.text_input("Search articles:", placeholder="Enter keywords...")
        
        # Date range filter
        if 'date' in df.columns:
            date_col = df['date'].dropna()
            if not date_col.empty:
                min_date = date_col.min().date() if hasattr(date_col.min(), 'date') else date(2000, 1, 1)
                max_date = date_col.max().date() if hasattr(date_col.max(), 'date') else date.today()
                
                st.subheader("Date Range")
                date_range = st.date_input(
                    "Select date range:",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                if len(date_range) != 2:
                    date_range = (None, None)
            else:
                date_range = (None, None)
        else:
            date_range = (None, None)
        
        # QUALIS classification filter
        if 'qualis_classification' in df.columns:
            unique_classifications = df['qualis_classification'].dropna().unique()
            if len(unique_classifications) > 0:
                st.subheader("QUALIS Classification")
                journal_classifications = st.multiselect(
                    "Select QUALIS:",
                    options=sorted(unique_classifications),
                    default=[]
                )
            else:
                journal_classifications = []
        else:
            journal_classifications = []
        
        # Sorting options
        st.subheader("Sorting")
        sort_options = []
        if 'date' in df.columns:
            sort_options.append('date')
        if 'qualis_classification' in df.columns:
            sort_options.append('qualis_classification')
        sort_options.extend(['title', 'journal'])
        
        sort_by = st.selectbox("Sort by:", options=sort_options)
        sort_order = st.radio("Order:", options=["Descending", "Ascending"])
    
    # Filter and display articles
    filtered_df = filter_articles(df, search_term, date_range, journal_classifications, sort_by, sort_order)
    
    # Results summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Articles", len(df))
    with col2:
        st.metric("Filtered Results", len(filtered_df))
    with col3:
        if len(df) > 0:
            percentage = (len(filtered_df) / len(df)) * 100
            st.metric("Match Rate", f"{percentage:.1f}%")
    
    st.markdown("---")
    
    # Display articles
    if len(filtered_df) == 0:
        st.warning("No articles match your current filters. Try adjusting your search criteria.")
    else:
        st.markdown(f"### Showing {len(filtered_df)} articles")
        
        # Pagination setup
        articles_per_page = 10
        total_pages = (len(filtered_df) - 1) // articles_per_page + 1
        
        # Initialize page in session state if not exists
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 1
        
        # Determine current page
        if total_pages > 1:
            current_page = st.session_state.current_page
            start_idx = (current_page - 1) * articles_per_page
            end_idx = min(start_idx + articles_per_page, len(filtered_df))
            page_df = filtered_df.iloc[start_idx:end_idx]
        else:
            page_df = filtered_df
            current_page = 1
        
        # Display articles
        for idx, (_, article) in enumerate(page_df.iterrows()):
            display_article_card(article, idx)
        
        # Pagination controls (vertical layout below articles)
        if total_pages > 1:
            st.markdown("---")
            st.markdown("### Navigation")
            
            # Create columns for pagination buttons
            col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
            
            with col1:
                if st.button("« First", disabled=(current_page == 1)):
                    st.session_state.current_page = 1
                    st.rerun()
            
            with col2:
                if st.button("‹ Previous", disabled=(current_page == 1)):
                    st.session_state.current_page = current_page - 1
                    st.rerun()
            
            with col3:
                st.markdown(f"<div style='text-align: center; padding: 8px;'>Page {current_page} of {total_pages}</div>", unsafe_allow_html=True)
            
            with col4:
                if st.button("Next ›", disabled=(current_page == total_pages)):
                    st.session_state.current_page = current_page + 1
                    st.rerun()
            
            with col5:
                if st.button("Last »", disabled=(current_page == total_pages)):
                    st.session_state.current_page = total_pages
                    st.rerun()
            
            # Page selector dropdown below buttons
            st.markdown("**Go to page:**")
            selected_page = st.selectbox(
                "Select page:",
                options=list(range(1, total_pages + 1)),
                index=current_page - 1,
                key="page_selector"
            )
            
            if selected_page != current_page:
                st.session_state.current_page = selected_page
                st.rerun()

if __name__ == "__main__":
    main()
