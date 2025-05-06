import pandas as pd
import numpy as np
import logging
import os
import json
import faiss
import pickle
from dotenv import load_dotenv
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential

# Çevre değişkenlerini yükle
load_dotenv()

logger = logging.getLogger(__name__)

class RagSystem:
    """
    Implements a RAG system using Azure AI Inference embeddings and FAISS for retrieval.
    Loads data from a JSON file, creates embeddings, and allows retrieval.
    """
    def __init__(self, json_path="data/db_data.json", vector_store_path="data/vector_store.pkl"):
        self.json_path = json_path
        self.vector_store_path = vector_store_path
        self.data = None
        self.embeddings = None
        self.index = None
        self.client = None
        self.last_modified_time = None
        
        # Azure AI Inference yapılandırması
        self.endpoint = os.getenv("AZURE_EMBEDDING_ENDPOINT", "https://egeme-m13lrihj-eastus.openai.azure.com/openai/deployments/text-embedding-3-large")
        self.model_name = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
        self.api_version = os.getenv("AZURE_EMBEDDING_API_VERSION", "2024-02-15")

        
        
        self.api_key = os.getenv("AZURE_TEXT_EMBEDDING_API_KEY")
        
        self._initialize_azure_client()
        self._load_or_create_vector_store()

    def _initialize_azure_client(self):
        """Azure AI Inference client'ını başlatır"""
        try:
            if not self.api_key:
                logger.error("Azure Text Embedding API key not found in environment variables.")
                return
            
            logger.info(f"Attempting to initialize Azure AI Inference client with endpoint: {self.endpoint}")
            logger.info(f"Using API key: {self.api_key[:4]}...{self.api_key[-4:] if self.api_key else 'None'}")
                
            self.client = EmbeddingsClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.api_key)
            )
            # Test call yaparak bağlantıyı kontrol et
            try:
                test_response = self.client.embed(
                    input=["Test connection to Azure AI Inference API"],
                    model=self.model_name
                )
                logger.info(f"Test connection successful! Embedding dimension: {len(test_response.data[0].embedding)}")
            except Exception as e:
                logger.error(f"Test connection failed: {e}")
                logger.error(f"API returned: {str(e)}")
                self.client = None
                return
                
            logger.info(f"Azure AI Inference embeddings client initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing Azure AI Inference client: {e}", exc_info=True)
            self.client = None

    def _get_embeddings(self, texts):
        """
        Creates embeddings for texts using Azure AI Inference service
        """
        if not self.client:
            logger.error("Cannot generate embeddings: Azure AI Inference client not initialized")
            return np.array([])
            
        try:
            # Process texts in batches of 16 (for Azure limits)
            all_embeddings = []
            batch_size = 16  # Azure API batch_size limit
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                # We don't need to specify the model parameter as it's already defined in the deployment endpoint
                response = self.client.embed(
                    input=batch_texts,
                    model=self.model_name
                )
                
                # Extract embedding for each text
                for item in response.data:
                    all_embeddings.append(item.embedding)
                    
                logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
            return np.array(all_embeddings)
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}", exc_info=True)
            return np.array([])

    def _get_file_modified_time(self, file_path):
        """Returns the last modified time of the file"""
        if os.path.exists(file_path):
            return os.path.getmtime(file_path)
        return None

    def _load_or_create_vector_store(self):
        """
        Loads the local vector store file or creates a new vector store from db_data.json.
        """
        # Check the last modified time of the db_data.json file
        json_modified_time = self._get_file_modified_time(self.json_path)
        
        # Check existence and last modified time of the vector store file
        vector_store_exists = os.path.exists(self.vector_store_path)
        vector_store_modified_time = self._get_file_modified_time(self.vector_store_path)
        
        # Create a new vector store if it doesn't exist or if JSON is newer
        if not vector_store_exists or (json_modified_time and vector_store_modified_time and json_modified_time > vector_store_modified_time):
            logger.info("Vector store file not found or JSON file is newer. Creating a new vector store.")
            self._create_vector_store_from_json()
        else:
            # Load the vector store
            logger.info(f"Loading existing vector store: {self.vector_store_path}")
            try:
                with open(self.vector_store_path, 'rb') as f:
                    store_data = pickle.load(f)
                    
                self.data = store_data.get('data')
                self.embeddings = store_data.get('embeddings')
                self.last_modified_time = store_data.get('last_modified_time', json_modified_time)
                
                # Recreate the FAISS index
                if self.embeddings is not None and len(self.embeddings) > 0:
                    dimension = self.embeddings.shape[1]
                    self.index = faiss.IndexFlatL2(dimension)
                    self.index.add(self.embeddings.astype('float32'))
                    logger.info(f"Vector store loaded successfully. {self.embeddings.shape[0]} embeddings, {dimension} dimensions.")
                else:
                    logger.warning("Vector store loaded but embeddings not found or empty.")
                    self._create_vector_store_from_json()
            except Exception as e:
                logger.error(f"Error loading vector store: {e}", exc_info=True)
                self._create_vector_store_from_json()

    def _create_vector_store_from_json(self):
        """Creates and saves vector store from JSON data."""
        logger.info(f"Creating vector store from JSON data: {self.json_path}")
        if not os.path.exists(self.json_path):
            logger.error(f"JSON file not found: {self.json_path}")
            # Create an empty DataFrame to prevent errors
            self.data = pd.DataFrame(columns=['question_id', 'answer', 'category', 'language_id', 'hints', 'combined_text'])
            self.embeddings = np.array([])
            # Create an empty index placeholder
            dummy_dim = 3072  # dimensions for text-embedding-3-large
            self.index = faiss.IndexFlatL2(dummy_dim) 
            logger.warning("Initialized with empty data and index because JSON file was not found.")
            return
            
        try:
            # Load JSON data
            with open(self.json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Convert JSON to DataFrame
            self.data = pd.DataFrame(json_data)
            
            # Fill missing values appropriately
            self.data.fillna({'answer': '', 'hints': '', 'category': 'unknown'}, inplace=True)
            
            # Process hints - they may be stored as JSON strings in the JSON file
            self.data['hints'] = self.data['hints'].apply(
                lambda x: (json.loads(x) if isinstance(x, str) and x.startswith('[') else 
                          (x.split('#') if isinstance(x, str) else 
                           [''] if pd.isna(x) else x))
            )
            
            # Convert hint lists to strings with spaces between them
            hint_texts = self.data['hints'].apply(
                lambda x: ' '.join(x) if isinstance(x, list) else str(x)
            )
            
            # Combine relevant text fields for embedding
            self.data['combined_text'] = self.data['answer'] + " " + hint_texts
            logger.info(f"Loaded {len(self.data)} records from JSON.")

            if not self.client:
                logger.error("Cannot create vector store: Azure AI Inference client not initialized")
                return
                
            logger.info(f"Creating embeddings using Azure AI Inference {self.model_name}")
            # Ensure combined_text is a list of strings
            texts_to_embed = self.data['combined_text'].astype(str).tolist()
            self.embeddings = self._get_embeddings(texts_to_embed)
            
            if len(self.embeddings) == 0:
                logger.error("Could not create embeddings, cannot create index")
                return
                
            logger.info(f"Created {self.embeddings.shape[0]} embeddings, dimensions: {self.embeddings.shape[1]}.")

            # Create FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)  # Using L2 distance
            self.index.add(self.embeddings.astype('float32')) # FAISS requires float32
            logger.info(f"FAISS index created successfully, contains {self.index.ntotal} vectors.")
            
            # Save vector store
            self._save_vector_store()

        except Exception as e:
            logger.error(f"Error during RAG initialization with JSON: {e}", exc_info=True)
            # Return to empty state in case of error
            self.data = pd.DataFrame(columns=['question_id', 'answer', 'category', 'language_id', 'hints', 'combined_text'])
            self.embeddings = np.array([])
            dummy_dim = 3072  # dimensions for text-embedding-3-large
            self.index = faiss.IndexFlatL2(dummy_dim) 
            logger.warning("Initialized with empty data and index due to setup error.")

    def _save_vector_store(self):
        """
        Saves embeddings and data to a local file.
        """
        # Check if the data directory exists and create it if not
        os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
        
        try:
            # Get the last modified time of the JSON file
            self.last_modified_time = self._get_file_modified_time(self.json_path)
            
            # Prepare the data to be saved
            store_data = {
                'data': self.data,
                'embeddings': self.embeddings,
                'last_modified_time': self.last_modified_time
            }
            
            # Save to file
            with open(self.vector_store_path, 'wb') as f:
                pickle.dump(store_data, f)
                
            logger.info(f"Vector store saved successfully: {self.vector_store_path}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {e}", exc_info=True)

    def update_vector_store(self):
        """
        Recreates the vector store from JSON data.
        Can be called from a Gradio button.
        """
        logger.info("Updating vector store...")
        try:
            # Create new vector store from JSON
            self._create_vector_store_from_json()
            return True, f"Vector store updated successfully. {len(self.data)} records, embeddings with {self.embeddings.shape[1] if self.embeddings is not None and len(self.embeddings) > 0 else 0} dimensions."
        except Exception as e:
            logger.error(f"Error updating vector store: {e}", exc_info=True)
            return False, f"Could not update vector store: {str(e)}"

    def retrieve(self, query_text: str, target_categories: list[str], language: str, k: int = 3) -> str:
        """
        Retrieves the top-k most relevant results for the given query.
        
        Args:
            query_text: The query text
            target_categories: List of categories to filter by
            language: Language code to filter by
            k: Number of results to return
            
        Returns:
            Formatted context text for the selected task
        """
        if not self.client:
            logger.error("Cannot perform retrieval: Azure AI Inference client not initialized")
            return "Error: RAG system not properly initialized"
            
        if not isinstance(language, str):
            logger.warning(f"Invalid language type: {type(language)}. Defaulting to 'en'.")
            language = 'en'
        
        # Trim whitespace and convert to lowercase for consistent comparison
        language = language.strip().lower()
        
        # Validate language is supported
        if language not in ['en', 'tr', 'pt']:
            logger.warning(f"Unsupported language requested: {language}. Defaulting to 'en'.")
            language = 'en'
            
        if not self.data is not None or len(self.data) == 0:
            logger.error("Cannot perform retrieval: No data available")
            return "Error: No data in RAG system"
            
        # Create query vector
        try:
            query_embedding = self._get_embeddings([query_text])[0]
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}", exc_info=True)
            return "Error: Could not generate embedding for query"
        
        # Filter data
        filtered_indices = []
        filtered_data = []
        
        # Check if language_id column exists
        if 'language_id' in self.data.columns:
            logger.info(f"Filtering data by language: {language}")
            # First filter by language
            language_mask = self.data['language_id'] == language
            logger.info(f"Language filter matches: {language_mask.sum()}/{len(self.data)} records")
            
            # Filter by categories
            if target_categories and 'category' in self.data.columns:
                # Category column exists, check each record for categories
                category_mask = self.data['category'].apply(
                    lambda cat: any(c.lower().strip() in str(cat).lower() for c in target_categories)
                )
                logger.info(f"Category filter matches: {category_mask.sum()}/{len(self.data)} records")
                
                # Combine filters
                combined_mask = language_mask & category_mask
            else:
                combined_mask = language_mask
            
            if combined_mask.sum() == 0:
                logger.warning("No data found matching language and category filters.")
                return ""
                
            # Filtered records
            filtered_data = self.data[combined_mask]
            filtered_indices = filtered_data.index.tolist()
            filtered_embeddings = self.embeddings[filtered_indices]
            
            logger.info(f"Found {len(filtered_indices)} records matching filters")
        else:
            logger.warning("No language_id column found in data, skipping language filtering")
            filtered_data = self.data
            filtered_indices = list(range(len(self.data)))
            filtered_embeddings = self.embeddings
        
        # Return empty if no results
        if len(filtered_indices) == 0:
            logger.warning("No records found matching filters")
            return ""
        
        # Create FAISS index with filtered indices
        dimension = filtered_embeddings.shape[1]
        tmp_index = faiss.IndexFlatL2(dimension)
        tmp_index.add(filtered_embeddings.astype('float32'))
        
        # Perform similarity search
        D, I = tmp_index.search(np.array([query_embedding]).astype('float32'), min(k, len(filtered_indices)))
        
        # Format results
        results = []
        for idx in I[0]:
            if idx >= len(filtered_indices):
                continue
                
            answer = filtered_data.iloc[idx]['answer']
            hints = filtered_data.iloc[idx]['hints']
            category = filtered_data.iloc[idx].get('category', '')
            
            # Prepare hints in a formatted way
            hints_str = ""
            if isinstance(hints, list):
                hints_str = "#".join(hints)
            elif isinstance(hints, str):
                if hints.startswith('[') and hints.endswith(']'):
                    try:
                        hints_list = json.loads(hints)
                        hints_str = "#".join(hints_list)
                    except:
                        hints_str = hints
                else:
                    hints_str = hints
            
            format_str = f"Existing Answer: {answer}\nHints: {hints_str}"
            if category:
                format_str += f"\nCategory: {category}"
            
            results.append(format_str)
            
        return "\n---\n".join(results)

# Example Usage (for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Example JSON file creation for testing if needed
    if not os.path.exists('data/db_data.json') and not os.path.exists('data/db_data.csv'):
        if not os.path.exists('data'):
            os.makedirs('data')
            
        # Create a minimal test dataset
        test_data = [
            {
                'question_id': 1,
                'answer': 'Apple',
                'category': 'fruit',
                'language_id': 'en',
                'hints': '["Red", "Tree", "Keeps doctor away", "Pie", "Newton"]'
            },
            {
                'question_id': 2,
                'answer': 'Banana',
                'category': 'fruit',
                'language_id': 'en',
                'hints': '["Yellow", "Monkey", "Peel", "Split", "Smoothie"]'
            },
            {
                'question_id': 3,
                'answer': 'Elma',
                'category': 'meyve',
                'language_id': 'tr',
                'hints': '["Kırmızı", "Ağaç", "Doktoru uzak tutar", "Turta", "Newton"]'
            },
            {
                'question_id': 4,
                'answer': 'Muz',
                'category': 'meyve',
                'language_id': 'tr',
                'hints': '["Sarı", "Maymun", "Soymak", "Bölmek", "Smoothie"]'
            }
        ]
        
        with open('data/db_data.json', 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
            
        logger.info("Created test data/db_data.json for testing as no data files were found.")

    rag = RagSystem()
    if rag.index and rag.index.ntotal > 0:
        print("\n--- English Retrieval Example ---")
        retrieved_content_en = rag.retrieve(query_text="healthy snack", target_categories=['fruit'], language='en', k=2)
        print(retrieved_content_en)
        
        print("\n--- Turkish Retrieval Example ---")
        retrieved_content_tr = rag.retrieve(query_text="sağlıklı atıştırmalık", target_categories=['meyve'], language='tr', k=2)
        print(retrieved_content_tr)

        print("\n--- Food Retrieval Example (EN) ---")
        retrieved_content_food = rag.retrieve(query_text="italian food", target_categories=['food & beverage'], language='en', k=2)
        print(retrieved_content_food)
    else:
        print("RAG system could not be initialized properly, skipping retrieval examples.") 