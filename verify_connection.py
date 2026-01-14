import chromadb
from chromadb.config import Settings
import dotenv
import os

dotenv.load_dotenv()

# Define the connection details and credentials
CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = os.getenv("CHROMA_PORT")
AUTH_TOKEN = os.getenv("CHROMA_AUTH_TOKEN")  # Ensure this environment variable is set
AUTH_HEADER = os.getenv("AUTH_HEADER")  # Must match CHROMA_SERVER_AUTH_TOKEN_TRANSPORT_HEADER

# Initialize the Chroma client with authentication settings
# Note: Settings does not accept `chroma_client_auth_token_transport_header` (extra field)
# Pass the transport header via the HttpClient `headers` parameter instead.
chroma_client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT,
    settings=Settings(
        chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
        chroma_client_auth_credentials=AUTH_TOKEN,
    ),
    headers={AUTH_HEADER: AUTH_TOKEN} if AUTH_TOKEN else None,
)

# You can now interact with your secured Chroma DB
# For example, listing collections
try:
    collections = chroma_client.list_collections()
    print("Successfully connected to Chroma DB. Collections:", collections)
except Exception as e:
    print(f"Connection failed: {e}")