import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
import joblib
import os


# Path dataset
DATA_PATH = os.path.join('dataset', 'preprocessed_data.xlsx')

# Path model BERT
BERT_MODEL_DIR = os.path.join('bert_model')
MODEL_BERT_PATH = os.path.join(BERT_MODEL_DIR, 'bert_model')
DOC_EMBEDDINGS_PATH = os.path.join(BERT_MODEL_DIR, 'doc_embeddings.npy')

# Path model Autoencoder
AUTOENCODER_DIR = os.path.join('autoencoder_model')
AUTOENCODER_MODEL_PATH = os.path.join(AUTOENCODER_DIR, 'autoencoder_model.h5')
ENCODER_MODEL_PATH = os.path.join(AUTOENCODER_DIR, 'encoder_model.h5')
SCALER_PATH = os.path.join(AUTOENCODER_DIR, 'scaler.pkl')


# --- Load Data dan Model (Sekali saja saat modul diimpor) ---
try:
    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.str.replace(" ", "_")
except FileNotFoundError as e:
    print(f"Error: File not found at {DATA_PATH}")
    print(f"Specific error: {e}")
    df = pd.DataFrame()
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    print(f"Specific error: {e}")
    df = pd.DataFrame()


notes_list = [
    'rubber', 'violet', 'citrus', 'foresty', 'vinyl', 'green', 'camphor',
    'clay', 'ozonic', 'bitter', 'lactonic', 'wine', 'chocolate', 'paper',
    'lavender', 'musky', 'sour', 'earthy', 'terpenic', 'Pear', 'oud',
    'cannabis', 'salty', 'coffee', 'metallic', 'almond', 'oily', 'herbal',
    'smoky', 'balsamic', 'conifer', 'tennis ball', 'alcohol', 'aquatic',
    'woody', 'leather', 'tropical', 'industrial glue', 'vanilla', 'beeswax',
    'amber', 'rose', 'coca-cola', 'cherry', 'nutty', 'tobacco', 'whiskey',
    'soapy', 'fresh', 'brown scotch tape', 'asphault', 'rum', 'mineral',
    'aldehydic', 'iris', 'plastic', 'warm spicy', 'marine', 'anis',
    'tuberose', 'aromatic', 'Champagne', 'savory', 'fresh spicy', 'sand',
    'powdery', 'cinnamon', 'cacao', 'yellow floral', 'coconut', 'sweet',
    'vodka', 'honey', 'white floral', 'sake', 'fruity', 'patchouli',
    'soft spicy', 'floral', 'mossy', 'caramel', 'animalic'
]
notes = pd.DataFrame({'Notes': notes_list})
notes['percentage'] = 0


# Load BERT model and embeddings
try:
    model_bert = SentenceTransformer(MODEL_BERT_PATH)
    doc_embeddings = np.load(DOC_EMBEDDINGS_PATH)
except FileNotFoundError as e:
    print(f"Error: BERT model or embeddings not found at {MODEL_BERT_PATH} or {DOC_EMBEDDINGS_PATH}")
    print(f"Specific error: {e}")
    model_bert = None  
    doc_embeddings = None
except Exception as e:
    print("Error loading BERT components", e)
    print(f"Specific error: {e}")
    model_bert, doc_embeddings = None, None


# Load Autoencoder and scaler
try:
    autoencoder = load_model(AUTOENCODER_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Pre-calculate perfume encodings (Lakukan ini sekali saja)
    numerical_columns = df.select_dtypes(include=['float64', 'int64', 'int32']).columns
    columns_to_exclude = ['Rating_Value', 'Best_Rating', 'Votes']
    features = numerical_columns.drop(columns_to_exclude)
    X = df[features].values
    X_scaled = scaler.transform(X)
    perfume_encodings = autoencoder.predict(X_scaled)

except FileNotFoundError as e:
    print(f"Error: Autoencoder model or scaler not found at {AUTOENCODER_MODEL_PATH} or {SCALER_PATH}")
    print(f"Specific error: {e}")
    autoencoder = None  
    scaler = None
    perfume_encodings = None
except Exception as e:
    print("Error loading Autoencoder components", e)
    print(f"Specific error: {e}")
    autoencoder, scaler, perfume_encodings = None, None, None


# --- Fungsi Rekomendasi ---

def bert_recommend(query):
    if model_bert is None or doc_embeddings is None:  # Check if BERT components are loaded
        return np.array([]) # Return empty array

    query_embedding = model_bert.encode([query])
    similarities = cosine_similarity(query_embedding, doc_embeddings)
    return similarities[0]


def recommend_autoencoder(input_encoding):
    if autoencoder is None or perfume_encodings is None:
        return np.array([]) 

    similarities = cosine_similarity(input_encoding, perfume_encodings)
    return similarities[0]


# models.py
def get_combined_recommendations(description, percentages):
    print("--- Inside get_combined_recommendations ---")
    print(f"Description: {description}")
    print(f"Percentages: {percentages}")

    if model_bert is None or autoencoder is None or scaler is None:
        print(model_bert)
        print(autoencoder)
        print(model_bert)
        print("Warning: One or more models failed to load.")
        return pd.DataFrame()

    bert_similarities = bert_recommend(description)
    print(f"BERT Similarities: {bert_similarities}")
    print(f"BERT Similarities Size: {bert_similarities.size}")
    if bert_similarities.size > 0:
        print(f"BERT Max Similarity: {np.max(bert_similarities)}") # Tambahkan ini
        print(f"BERT Min Similarity: {np.min(bert_similarities)}") # Tambahkan ini

    input_data = np.array(percentages).reshape(1, -1)
    if input_data.size == 0:
        print("Error: input_data is empty!")
        return pd.DataFrame()

    input_scaled = scaler.transform(input_data)
    input_encoding = autoencoder.predict(input_scaled)
    autoencoder_similarities = recommend_autoencoder(input_encoding)
    print(f"Autoencoder Similarities: {autoencoder_similarities}")
    print(f"Autoencoder Similarities Size: {autoencoder_similarities.size}")
    if autoencoder_similarities.size > 0:
        print(f"Autoencoder Max Similarity: {np.max(autoencoder_similarities)}") # Tambahkan ini
        print(f"Autoencoder Min Similarity: {np.min(autoencoder_similarities)}") # Tambahkan ini

    if bert_similarities.size > 0 and autoencoder_similarities.size > 0:
        combined_similarity = (bert_similarities + autoencoder_similarities) / 2
        print(f"Combined Similarity (Both): {combined_similarity}")
        print(f"Combined Max Similarity: {np.max(combined_similarity)}") # Tambah
        print(f"Combined Min Similarity: {np.min(combined_similarity)}") # Tambah
    elif bert_similarities.size > 0:
        combined_similarity = bert_similarities
        print(f"Combined Similarity (BERT only): {combined_similarity}")
        print(f"Combined Max Similarity (BERT only): {np.max(combined_similarity)}") # Tambah
        print(f"Combined Min Similarity (BERT only): {np.min(combined_similarity)}") # Tambah
    elif autoencoder_similarities.size > 0:
        combined_similarity = autoencoder_similarities
        print(f"Combined Similarity (Autoencoder only): {combined_similarity}")
        print(f"Combined Max Similarity (Autoencoder only): {np.max(combined_similarity)}")
        print(f"Combined Min Similarity (Autoencoder only): {np.min(combined_similarity)}")
    else:
        print("Error: Both BERT and Autoencoder similarities are empty.")
        return pd.DataFrame()

    top_indices = np.argsort(combined_similarity)[::-1][:10]
    print(f"Top Indices: {top_indices}")

    recommendations_df = df.iloc[top_indices].copy()
    recommendations_df['Combined_Similarity_Score'] = combined_similarity[top_indices]
    return recommendations_df[['Perfume_Name', 'Brand', 'URL', 'Image', 'Combined_Similarity_Score', 'Gender']]

def convert_df_to_list(df):
    """Konversi DataFrame ke list of dictionaries."""
    return df.to_dict(orient='records')