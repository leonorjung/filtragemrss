import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
import re
import subprocess
import os
import html

# Language translations
TRANSLATIONS = {
    'en': {
        'title': 'Articles about games classified in Communications QUALIS',
        'subtitle': 'A monthly updated compilation of articles about games published in journals classified in the 2017–2020 QUALIS ranking for Communication. The search is conducted through RSS feeds and filtered by AI, so some relevant articles may be missing.',
        'language': 'Language',
        'filters_search': '🔍 Filters & Search',
        'search_placeholder': 'Enter keywords...',
        'date_range': 'Date Range',
        'select_date_range': 'Select date range:',
        'qualis_classification': 'QUALIS Classification',
        'select_qualis': 'Select QUALIS:',
        'sorting': 'Sorting',
        'sort_by': 'Sort by:',
        'order': 'Order:',
        'descending': 'Descending',
        'ascending': 'Ascending',
        'total_articles': 'Total Articles',
        'filtered_results': 'Filtered Results',
        'match_rate': 'Match Rate',
        'showing_articles': 'Showing {count} articles',
        'no_matches': 'No articles match your current filters. Try adjusting your search criteria.',
        'navigation': 'Navigation',
        'go_to_page': 'Go to page:',
        'select_page': 'Select page:',
        'first': '« First',
        'previous': '‹ Previous',
        'next': 'Next ›',
        'last': 'Last »',
        'page_of': 'Page {current} of {total}',
        'journal': 'Journal',
        'date': 'Date',
        'qualis': 'QUALIS',
        'abstract': 'Abstract',
        'no_title': 'No Title',
        'unknown_journal': 'Unknown Journal',
        'not_classified': 'Not classified',
        'not_available': 'Not available',
        'loading_data': 'Loading academic articles database...',
        'loaded_articles': '📊 Loaded {count} academic articles',
        'error_loading': 'Error loading data: {error}',
        'data_requirements': 'Data Requirements:',
        'csv_location': '- The CSV file should be named \'data.csv\' and located in the repository',
        'expected_columns': '- Expected columns: title, journal, date, abstract, qualis classification, link',
        'separator_info': '- The file should use semicolon (;) as separator',
        'recently_added': 'Recently Added',
        'show_recent_only': 'Show only recently added articles',
        'days_threshold': 'Consider articles added in the last {days} days as recent',
        'recent_badge': '🆕 NEW'
    },
    'pt': {
        'title': 'Artigos sobre jogos no QUALIS de Comunicação',
        'subtitle': 'Compilado atualizado mensalmente de artigos sobre jogos publicados em periódicos classificados no QUALIS 2017-2020 de Comunicação. A busca é feita em feeds de RSS e filtrada por IA, então é possível que os resultados deixem alguns artigos de fora.',
        'language': 'Idioma',
        'filters_search': '🔍 Filtros e Pesquisa',
        'search_placeholder': 'Digite palavras-chave...',
        'date_range': 'Intervalo de Datas',
        'select_date_range': 'Selecione o intervalo de datas:',
        'qualis_classification': 'Classificação QUALIS',
        'select_qualis': 'Selecione QUALIS:',
        'sorting': 'Ordenação',
        'sort_by': 'Ordenar por:',
        'order': 'Ordem:',
        'descending': 'Decrescente',
        'ascending': 'Crescente',
        'total_articles': 'Total de Artigos',
        'filtered_results': 'Resultados Filtrados',
        'match_rate': 'Taxa de Correspondência',
        'showing_articles': 'Mostrando {count} artigos',
        'no_matches': 'Nenhum artigo corresponde aos seus filtros atuais. Tente ajustar seus critérios de pesquisa.',
        'navigation': 'Navegação',
        'go_to_page': 'Ir para página:',
        'select_page': 'Selecionar página:',
        'first': '« Primeira',
        'previous': '‹ Anterior',
        'next': 'Próxima ›',
        'last': 'Última »',
        'page_of': 'Página {current} de {total}',
        'journal': 'Revista',
        'date': 'Data',
        'qualis': 'QUALIS',
        'abstract': 'Resumo',
        'no_title': 'Sem Título',
        'unknown_journal': 'Revista Desconhecida',
        'not_classified': 'Não classificado',
        'not_available': 'Não disponível',
        'loading_data': 'Carregando base de dados de artigos acadêmicos...',
        'loaded_articles': '📊 Carregados {count} artigos acadêmicos',
        'error_loading': 'Erro ao carregar dados: {error}',
        'data_requirements': 'Requisitos dos Dados:',
        'csv_location': '- O arquivo CSV deve se chamar \'data.csv\' e estar localizado no repositório',
        'expected_columns': '- Colunas esperadas: title, journal, date, abstract, qualis classification, link',
        'separator_info': '- O arquivo deve usar ponto e vírgula (;) como separador',
        'recently_added': 'Adicionados Recentemente',
        'show_recent_only': 'Mostrar apenas artigos adicionados recentemente',
        'days_threshold': 'Considerar artigos adicionados nos últimos {days} dias como recentes',
        'recent_badge': '🆕 NOVO'
    }
}

def get_text(key, lang='en', **kwargs):
    """Get translated text for the given key and language"""
    text = TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)
    if kwargs:
        return text.format(**kwargs)
    return text

def sanitize_html(text):
    """
    Remove HTML tags from text and decode HTML entities
    """
    if not text or pd.isna(text):
        return text
    
    # Convert to string if needed
    text = str(text)
    
    # Remove HTML tags using regex
    clean_text = re.sub(r'<[^>]+>', '', text)
    
    # Decode HTML entities (like &amp;, &lt;, etc.)
    clean_text = html.unescape(clean_text)
    
    # Clean up extra whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    return clean_text

def get_file_modification_dates():
    """Get the modification dates from Git history for data.csv"""
    try:
        # Check if we're in a git repository
        if not os.path.exists('.git'):
            return None
        
        # Get the Git log for data.csv to see when it was last modified
        cmd = ['git', 'log', '--follow', '--format=%ct', '--', 'data.csv']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and result.stdout.strip():
            # Get the most recent modification timestamp
            timestamps = result.stdout.strip().split('\n')
            if timestamps:
                latest_timestamp = int(timestamps[0])
                return datetime.fromtimestamp(latest_timestamp)
        return None
        
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return None

def is_recently_added(days_threshold=7):
    """Check if the data.csv file was modified recently"""
    modification_date = get_file_modification_dates()
    if modification_date is None:
        return False
    
    current_date = datetime.now()
    time_diff = current_date - modification_date
    return time_diff.days <= days_threshold

def add_recent_indicators(df, days_threshold=7):
    """Add indicators for recently added articles based on Git history"""
    recent_modification = is_recently_added(days_threshold)
    
    # For simplicity, if the file was recently modified, mark all articles as potentially recent
    # In a more sophisticated implementation, you could track individual row additions
    df['is_recent'] = recent_modification
    
    return df

# Page configuration
st.set_page_config(
    page_title="Academic Articles Browser",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_csv_data():
    """
    Load CSV data from local repository file with multiple encoding attempts
    """
    csv_file_path = "data.csv"
    
    try:
        # Try multiple encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_file_path, sep=';', encoding=encoding)
                return df, None
            except (UnicodeDecodeError, pd.errors.EmptyDataError):
                continue
        
        return None, "Unable to decode the CSV file with any supported encoding"
        
    except FileNotFoundError:
        return None, f"CSV file not found: {csv_file_path}"
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
        
        # Fill missing values and sanitize HTML
        for col in df_renamed.columns:
            if col in ['title', 'journal', 'abstract', 'qualis_classification']:
                df_renamed[col] = df_renamed[col].fillna('Not available')
                # Sanitize HTML tags from text fields
                df_renamed[col] = df_renamed[col].apply(sanitize_html)
            elif col == 'link':
                df_renamed[col] = df_renamed[col].fillna('')
        
        return df_renamed, None
        
    except Exception as e:
        return None, f"Error cleaning data: {str(e)}"

def filter_articles(df, search_term, date_range, journal_classifications, sort_by, sort_order, show_recent_only=False):
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
    
    # Recent articles filter
    if show_recent_only and 'is_recent' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['is_recent'] == True]
    
    # Sorting
    if sort_by in filtered_df.columns:
        ascending = (sort_order == "Ascending")
        if sort_by == 'date':
            # Handle NaT values in date sorting
            filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending, na_position='last')
        else:
            filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)
    
    return filtered_df

def display_article_card(article, index, lang='en'):
    """
    Display an individual article in a card format
    """
    with st.container():
        # Title with link and recent badge
        title = article.get('title', get_text('no_title', lang))
        recent_badge = ""
        if article.get('is_recent', False):
            recent_badge = f" {get_text('recent_badge', lang)}"
        
        if 'link' in article and article['link'] and article['link'] != get_text('not_available', lang) and article['link'].strip():
            st.markdown(f"### [{title}]({article['link']}){recent_badge}")
        else:
            st.markdown(f"### {title}{recent_badge}")
        
        # Journal
        journal = article.get('journal', get_text('unknown_journal', lang))
        st.markdown(f"**{get_text('journal', lang)}:** {journal}")
        
        # Date and Classification
        if 'date' in article and pd.notna(article['date']):
            date_str = article['date'].strftime('%B %d, %Y') if hasattr(article['date'], 'strftime') else str(article['date'])
            st.markdown(f"**{get_text('date', lang)}:** {date_str}")
        
        if 'qualis_classification' in article:
            classification = article.get('qualis_classification', get_text('not_classified', lang))
            st.markdown(f"**{get_text('qualis', lang)}:** {classification}")
        
        # Abstract
        if 'abstract' in article and article['abstract'] and article['abstract'] != get_text('not_available', lang):
            with st.expander(get_text('abstract', lang)):
                clean_abstract = sanitize_html(article['abstract'])
                st.write(clean_abstract)
        
        st.divider()

def main():
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'error' not in st.session_state:
        st.session_state.error = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'language' not in st.session_state:
        st.session_state.language = 'en'
    
    # Language selector in sidebar
    with st.sidebar:
        st.session_state.language = st.selectbox(
            get_text('language', st.session_state.language),
            options=['en', 'pt'],
            format_func=lambda x: 'English' if x == 'en' else 'Português',
            index=0 if st.session_state.language == 'en' else 1
        )
    
    lang = st.session_state.language
    
    st.title(get_text('title', lang))
    st.markdown(get_text('subtitle', lang))
    
    # Auto-load data on first run
    if not st.session_state.data_loaded:
        with st.spinner(get_text('loading_data', lang)):
            data, error = load_csv_data()
            
            if error:
                st.session_state.error = error
                st.session_state.data = None
            else:
                cleaned_data, clean_error = clean_and_validate_data(data)
                if clean_error:
                    st.session_state.error = clean_error
                    st.session_state.data = None
                else:
                    # Add recent indicators based on Git history
                    cleaned_data = add_recent_indicators(cleaned_data)
                    st.session_state.data = cleaned_data
                    st.session_state.error = None
            
            st.session_state.data_loaded = True
    
    # Display error if any
    if st.session_state.error:
        st.error(get_text('error_loading', lang, error=st.session_state.error))
        st.markdown(f"""
        **{get_text('data_requirements', lang)}**
        {get_text('csv_location', lang)}
        {get_text('expected_columns', lang)}
        {get_text('separator_info', lang)}
        """)
        return
    
    # Main content
    if st.session_state.data is None:
        st.info(get_text('loading_data', lang))
        return
    
    df = st.session_state.data
    
    # Display data summary
    st.success(get_text('loaded_articles', lang, count=len(df)))
    
    # Filtering sidebar
    with st.sidebar:
        st.header(get_text('filters_search', lang))
        
        # Search
        search_term = st.text_input(
            f"{get_text('filters_search', lang).replace('🔍 ', '')}:",
            placeholder=get_text('search_placeholder', lang)
        )
        
        # Date range filter
        if 'date' in df.columns:
            date_col = df['date'].dropna()
            if not date_col.empty:
                min_date = date_col.min().date() if hasattr(date_col.min(), 'date') else date(2000, 1, 1)
                max_date = date_col.max().date() if hasattr(date_col.max(), 'date') else date.today()
                
                st.subheader(get_text('date_range', lang))
                date_range = st.date_input(
                    get_text('select_date_range', lang),
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
                st.subheader(get_text('qualis_classification', lang))
                journal_classifications = st.multiselect(
                    get_text('select_qualis', lang),
                    options=sorted(unique_classifications),
                    default=[]
                )
            else:
                journal_classifications = []
        else:
            journal_classifications = []
        
        # Recently added filter
        st.subheader(get_text('recently_added', lang))
        days_threshold = st.slider(
            "Days threshold:",
            min_value=1, max_value=30, value=7,
            help=get_text('days_threshold', lang, days=7)
        )
        show_recent_only = st.checkbox(get_text('show_recent_only', lang))
        
        # Sorting options
        st.subheader(get_text('sorting', lang))
        sort_options = []
        if 'date' in df.columns:
            sort_options.append('date')
        if 'qualis_classification' in df.columns:
            sort_options.append('qualis_classification')
        sort_options.extend(['title', 'journal'])
        
        sort_by = st.selectbox(get_text('sort_by', lang), options=sort_options)
        sort_order = st.radio(
            get_text('order', lang), 
            options=[get_text('descending', lang), get_text('ascending', lang)]
        )
        
        # Convert back to English for processing
        sort_order_en = "Descending" if sort_order == get_text('descending', lang) else "Ascending"
    
    # Update recent indicators with current threshold
    if 'is_recent' not in df.columns:
        df = add_recent_indicators(df, days_threshold)
        st.session_state.data = df
    else:
        # Refresh recent indicators if threshold changed
        df = add_recent_indicators(df, days_threshold)
        st.session_state.data = df
    
    # Filter and display articles
    filtered_df = filter_articles(df, search_term, date_range, journal_classifications, sort_by, sort_order_en, show_recent_only)
    
    # Results summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(get_text('total_articles', lang), len(df))
    with col2:
        st.metric(get_text('filtered_results', lang), len(filtered_df))
    with col3:
        if len(df) > 0:
            percentage = (len(filtered_df) / len(df)) * 100
            st.metric(get_text('match_rate', lang), f"{percentage:.1f}%")
    
    st.markdown("---")
    
    # Display articles
    if len(filtered_df) == 0:
        st.warning(get_text('no_matches', lang))
    else:
        st.markdown(f"### {get_text('showing_articles', lang, count=len(filtered_df))}")
        
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
            display_article_card(article, idx, lang)
        
        # Pagination controls (vertical layout below articles)
        if total_pages > 1:
            st.markdown("---")
            st.markdown(f"### {get_text('navigation', lang)}")
            
            # Create columns for pagination buttons
            col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
            
            with col1:
                if st.button(get_text('first', lang), disabled=(current_page == 1)):
                    st.session_state.current_page = 1
                    st.rerun()
            
            with col2:
                if st.button(get_text('previous', lang), disabled=(current_page == 1)):
                    st.session_state.current_page = current_page - 1
                    st.rerun()
            
            with col3:
                page_text = get_text('page_of', lang, current=current_page, total=total_pages)
                st.markdown(f"<div style='text-align: center; padding: 8px;'>{page_text}</div>", unsafe_allow_html=True)
            
            with col4:
                if st.button(get_text('next', lang), disabled=(current_page == total_pages)):
                    st.session_state.current_page = current_page + 1
                    st.rerun()
            
            with col5:
                if st.button(get_text('last', lang), disabled=(current_page == total_pages)):
                    st.session_state.current_page = total_pages
                    st.rerun()
            
            # Page selector dropdown below buttons
            st.markdown(f"**{get_text('go_to_page', lang)}**")
            selected_page = st.selectbox(
                get_text('select_page', lang),
                options=list(range(1, total_pages + 1)),
                index=current_page - 1,
                key="page_selector"
            )
            
            if selected_page != current_page:
                st.session_state.current_page = selected_page
                st.rerun()

if __name__ == "__main__":
    main()
