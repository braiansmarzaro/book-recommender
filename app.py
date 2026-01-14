from chromadb.config import Settings
import chromadb
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = os.getenv("CHROMA_PORT")
AUTH_TOKEN = os.getenv("CHROMA_AUTH_TOKEN")  # Ensure this environment variable is set
AUTH_HEADER = os.getenv("AUTH_HEADER")  # Must match CHROMA_SERVER_AUTH_TOKEN_TRANSPORT_HEADER

# Page configuration
st.set_page_config(
    page_title="Book Recommender System",
    page_icon="📚",
    layout="wide"
)

# Initialize session state for caching
@st.cache_resource
def load_embeddings():
    """Load the HuggingFace embeddings model"""
    return HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

@st.cache_resource
def load_vector_store(_embeddings):
    """Load the Chroma vector store"""
    # Create the ChromaDB HTTP client
    chroma_client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=int(CHROMA_PORT),
        settings=Settings(
            chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
            chroma_client_auth_credentials=AUTH_TOKEN,
        ),
        headers={AUTH_HEADER: AUTH_TOKEN} if AUTH_TOKEN else None,
    )
    
    # Return LangChain Chroma wrapper for MMR support
    return Chroma(
        client=chroma_client,
        collection_name="books",
        embedding_function=_embeddings
    )

@st.cache_data
def load_dataframe():
    """Load the books dataset"""
    dataset_path = os.getenv("DATASET_PATH")
    if not dataset_path or not os.path.exists(dataset_path):
        st.error("Dataset path not found. Please check your .env file.")
        return pd.DataFrame()
    
    df = pd.read_csv(dataset_path)
    # Clean the Authors column
    if df['Authors'].str.startswith("By ").any():
        df['Authors'] = df['Authors'].str[3:]
    return df

# Load data
embeddings = load_embeddings()
vector_store = load_vector_store(embeddings)
df = load_dataframe()

# Title
st.title("📚 Book Recommender System")
st.markdown("Search for books and apply filters to find your next great read!")
st.markdown("Key technologies used: `LangChain`, `ChromaDB`, `HuggingFace Embeddings`")
st.markdown("Deployed with `Docker compose`, `Streamlit` and `Nginx`")

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Category filter
categories = ['All'] + sorted(df['Category'].dropna().unique().tolist())
selected_category = st.sidebar.selectbox("Category", categories)

# Price range filter
if not df.empty:
    min_price = float(df['Price Starting With ($)'].min())
    max_price = float(df['Price Starting With ($)'].max())
    price_range = st.sidebar.slider(
        "Price Range ($)",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price)
    )
else:
    price_range = (0.0, 100.0)

# Year range filter
if not df.empty:
    min_year = int(df['Publish Date (Year)'].min())
    max_year = int(df['Publish Date (Year)'].max())
    year_range = st.sidebar.slider(
        "Publication Year",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
else:
    year_range = (1900, 2024)

# Publisher filter
publishers = ['All'] + sorted(df['Publisher'].dropna().unique().tolist())
selected_publisher = st.sidebar.selectbox("Publisher", publishers)

# Author search
author_search = st.sidebar.text_input("Author contains (optional)")

# Main search area
st.header("Search for Books")

# Use a form to allow Enter key to trigger search
with st.form(key='search_form'):
    col1, col2 = st.columns([4, 1])
    
    with col1:
        search_query = st.text_input(
            "Enter your search query",
            placeholder="e.g., science fiction adventure in space...",
            help="Describe the book you're looking for or enter keywords"
        )
    
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        search_button = st.form_submit_button("🔎 Search", type="primary", width='stretch')

# Number of results (outside form so it doesn't reset)
num_results = st.number_input("Number of results to show", min_value=5, max_value=50, value=10, step=5)

# Apply filters to dataframe
def apply_filters(dataframe):
    """Apply selected filters to the dataframe"""
    filtered_df = dataframe.copy()
    
    # Category filter
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == selected_category]
    
    # Price filter
    filtered_df = filtered_df[
        (filtered_df['Price Starting With ($)'] >= price_range[0]) &
        (filtered_df['Price Starting With ($)'] <= price_range[1])
    ]
    
    # Year filter
    filtered_df = filtered_df[
        (filtered_df['Publish Date (Year)'] >= year_range[0]) &
        (filtered_df['Publish Date (Year)'] <= year_range[1])
    ]
    
    # Publisher filter
    if selected_publisher != 'All':
        filtered_df = filtered_df[filtered_df['Publisher'] == selected_publisher]
    
    # Author filter
    if author_search:
        filtered_df = filtered_df[
            filtered_df['Authors'].str.contains(author_search, case=False, na=False)
        ]
    
    return filtered_df

# Perform search
if search_button and search_query:
    with st.spinner("Searching for similar books..."):
        try:
            # Perform vector similarity search
            #results = vector_store.similarity_search(search_query, k=num_results * 3)  # Get more results to filter
            results = vector_store.max_marginal_relevance_search(search_query, k=num_results, fetch_k=num_results*4)
            # Convert results to dataframe
            results_data = []
            for result in results:
                results_data.append({
                    'Title': result.metadata['title'],
                    'Authors': result.metadata['authors'],
                    'Category': result.metadata.get('category', ''),
                    'Publisher': result.metadata.get('publisher', ''),
                    'Price Starting With ($)': result.metadata['price'],
                    'Publish Date (Year)': result.metadata['publish_year']
                })
            
            results_df = pd.DataFrame(results_data)
            
            # Apply filters to results
            filtered_results = apply_filters(results_df)
            
            # Limit to requested number of results
            filtered_results = filtered_results.head(num_results)
            
            # Display results
            st.success(f"Found {len(filtered_results)} matching books")
            
            if len(filtered_results) > 0:
                # Display as a styled dataframe
                st.dataframe(
                    filtered_results,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Title": st.column_config.TextColumn("Title", width="large"),
                        "Authors": st.column_config.TextColumn("Authors", width="medium"),
                        "Category": st.column_config.TextColumn("Category", width="medium"),
                        "Publisher": st.column_config.TextColumn("Publisher", width="medium"),
                        "Price Starting With ($)": st.column_config.NumberColumn(
                            "Price ($)",
                            format="$%.2f"
                        ),
                        "Publish Date (Month)": st.column_config.TextColumn("Month"),
                        "Publish Date (Year)": st.column_config.NumberColumn("Year", format="%d")
                    }
                )
                
                # Download button
                csv = filtered_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="book_recommendations.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No books found matching your filters. Try adjusting your filters.")
                
        except Exception as e:
            st.error(f"An error occurred during search: {str(e)}")

elif search_button and not search_query:
    st.warning("Please enter a search query.")

# Display browse option
st.markdown("---")
st.header("Browse All Books")

if st.checkbox("Show filtered dataset (without semantic search)"):
    filtered_browse_df = apply_filters(df)
    
    st.info(f"Showing {len(filtered_browse_df)} books matching your filters")
    
    if len(filtered_browse_df) > 0:
        # Display first N rows
        display_count = st.slider("Number of books to display", 10, 100, 20, key="browse_slider")
        
        st.dataframe(
            filtered_browse_df.head(display_count),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Title": st.column_config.TextColumn("Title", width="large"),
                "Authors": st.column_config.TextColumn("Authors", width="medium"),
                "Category": st.column_config.TextColumn("Category", width="medium"),
                "Publisher": st.column_config.TextColumn("Publisher", width="medium"),
                "Price Starting With ($)": st.column_config.NumberColumn(
                    "Price ($)",
                    format="$%.2f"
                ),
                "Publish Date (Month)": st.column_config.TextColumn("Month"),
                "Publish Date (Year)": st.column_config.NumberColumn("Year", format="%d")
            }
        )

# Footer
st.markdown("---")
st.markdown("""
    Built with ❤️ by Deivid Smarzaro 
    <a href="https://www.linkedin.com/in/deividsmarzaro/">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Deivid Braian LinkedIn" style="max-width: 100%;"></a>
""", unsafe_allow_html=True)

