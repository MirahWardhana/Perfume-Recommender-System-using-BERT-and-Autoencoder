import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os


# Path dataset
DATA_PATH = os.path.join('dataset', 'preprocessed_data.xlsx')

# Path model BERT
MODEL_BERT_PATH = os.path.join('bert_model', 'bert_model')
DOC_EMBEDDINGS_PATH = os.path.join('bert_model', 'doc_embeddings.npy')

# Path model Autoencoder
AUTOENCODER_MODEL_PATH = os.path.join('autoencoder_model', 'autoencoder_model.h5')
SCALER_PATH = os.path.join('autoencoder_model', 'scaler.pkl')


# --- Load Data dan Model (Sekali saja saat modul diimpor) ---
try:
    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.str.replace(" ", "_")
    print(df.columns)
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


# Initialize all model components to None
df = pd.DataFrame()
notes = pd.DataFrame({'Notes': notes_list})
notes['percentage'] = 0
model_bert = None
doc_embeddings = None
autoencoder = None
scaler = None
perfume_encodings = None
combined_features_matrix = None

try:
    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.str.replace(" ", "_")
except FileNotFoundError as e:
    print(f"Error: File not found at {DATA_PATH}. {e}") 
except Exception as e:
    print(f"An error occurred while loading the data: {e}")

# Load BERT model and embeddings
model_bert = SentenceTransformer(MODEL_BERT_PATH)
doc_embeddings = np.load(DOC_EMBEDDINGS_PATH)
print("BERT components loaded successfully.")


# Load Autoencoder and scaler

if os.path.exists(AUTOENCODER_MODEL_PATH) and os.path.exists(SCALER_PATH):
    autoencoder = load_model(AUTOENCODER_MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    print("Autoencoder components loaded successfully.")

    # Pre-calculate perfume encodings if df and scaler are loaded
    if not df.empty and scaler is not None:
        numerical_columns = df.select_dtypes(include=['float64', 'int64', 'int32']).columns
        # Ensure 'Description' is handled correctly if it's numerical (it shouldn't be)
        columns_to_exclude = ['Rating_Value', 'Best_Rating', 'Votes'] 
        # Filter out columns that might not exist
        features = numerical_columns.drop(columns_to_exclude)
        print(features)

        if features: # Proceed only if there are features to scale
                # Ensure all feature columns actually exist in df
            features_present = [f for f in features if f in df.columns]
            if len(features_present) == len(features): # Check if all expected features are present
                X = df[features_present].fillna(0).values # Fill NaN with 0 before scaling
                X_scaled = scaler.transform(X)
                perfume_encodings = autoencoder.predict(X_scaled)
                print("Perfume encodings calculated.")
            else:
                print(f"Warning: Missing feature columns for autoencoder: {set(features) - set(features_present)}")
        else:
                print("Warning: No numerical features found for autoencoder after exclusions.")

else:
    print(f"Error: Autoencoder model or scaler not found at {AUTOENCODER_MODEL_PATH} or {SCALER_PATH}")



# Pre-calculate combined features matrix
try:
    if perfume_encodings is not None and doc_embeddings is not None:
        # Ensure consistent number of samples
        if perfume_encodings.shape[0] == doc_embeddings.shape[0]:
            print("Calculating combined features matrix...")
            v1 = perfume_encodings  # Shape typically (num_perfumes, 82)
            v2 = doc_embeddings    # Shape typically (num_perfumes, 384)

            # Get dimensions
            num_perfumes, dim_v1 = v1.shape
            _, dim_v2 = v2.shape

            # Pad v1 to match v2 dimension
            padding_width = dim_v2 - dim_v1
            if padding_width < 0:
                raise ValueError(f"BERT embedding dimension ({dim_v2}) is smaller than Autoencoder dimension ({dim_v1}). Padding logic needs adjustment.")
            elif padding_width > 0:
                 v1_padded = np.pad(v1, ((0, 0), (0, padding_width)), mode='constant')
            else:
                 v1_padded = v1 # No padding needed if dimensions match

            # Calculate components using TensorFlow for potential GPU acceleration and compatibility
            tf_v1_padded = tf.constant(v1_padded, dtype=tf.float32)
            tf_v2 = tf.constant(v2, dtype=tf.float32)

            concat = tf.concat([tf_v1_padded, tf_v2], axis=1)
            difference = tf.norm(tf_v1_padded - tf_v2, axis=1, keepdims=True)
            hadamard_product = tf_v1_padded * tf_v2

            # Stack results horizontally
            combined_features_matrix_tf = tf.concat([concat, difference, hadamard_product], axis=1)
            combined_features_matrix = combined_features_matrix_tf.numpy() # Convert back to numpy array
            print(f"Combined features matrix calculated. Shape: {combined_features_matrix.shape}")
        else:
            print(f"Error: Mismatch in number of samples between perfume_encodings ({perfume_encodings.shape[0]}) and doc_embeddings ({doc_embeddings.shape[0]})")
            combined_features_matrix = None
    else:
        print("Skipping combined features matrix calculation due to missing perfume or doc embeddings.")
        combined_features_matrix = None
except Exception as e:
    print(f"Error calculating combined features matrix: {e}")
    combined_features_matrix = None


# --- Fungsi Rekomendasi ---

# Remove bert_recommend and recommend_autoencoder as they are replaced

# models.py
def get_combined_recommendations(description, percentages):
    print("--- Inside get_combined_recommendations (New Method) ---")
    print(f"Description: {description}")
    print(f"Percentages: {percentages}")

    # Check if all necessary components are loaded
    if model_bert is None or autoencoder is None or scaler is None or combined_features_matrix is None or df.empty:
        print("Warning: One or more models/data components failed to load or calculate. Cannot generate recommendations.")
        print(f"model_bert loaded: {model_bert is not None}")
        print(f"autoencoder loaded: {autoencoder is not None}")
        print(f"scaler loaded: {scaler is not None}")
        print(f"combined_features_matrix calculated: {combined_features_matrix is not None}")
        print(f"df loaded: {not df.empty}")
        return pd.DataFrame()

    try:
        # --- Autoencoder Part (Input) ---
        input_data = np.array(percentages).reshape(1, -1)
        if input_data.shape[1] != perfume_encodings.shape[1]: # Check dimension consistency
             # Attempt to pad/truncate if notes list changed? Or raise error?
             # For now, raise error or return empty if dimensions mismatch critically.
             # This assumes 'features' list used for training matches 'percentages' length
             print(f"Error: Input percentages length ({input_data.shape[1]}) does not match expected features length ({perfume_encodings.shape[1]})")
             # Adjust the expected length check if needed based on 'features' variable definition
             num_expected_features = len(features) # Use the 'features' list defined during loading
             if input_data.shape[1] != num_expected_features:
                 print(f"Error: Input percentages length ({input_data.shape[1]}) does not match expected features length ({num_expected_features})")
                 return pd.DataFrame()


        input_scaled = scaler.transform(input_data)
        v1_input = autoencoder.predict(input_scaled) # Shape (1, 82)

        # --- BERT Part (Input) ---
        v2_input = model_bert.encode([description]) # Shape (1, 384)

        # --- Combine Input Features ---
        dim_v1_input = v1_input.shape[1]
        dim_v2_input = v2_input.shape[1]

        # Pad v1_input to match v2_input dimension
        padding_width_input = dim_v2_input - dim_v1_input
        if padding_width_input < 0:
             raise ValueError(f"Input BERT embedding dimension ({dim_v2_input}) is smaller than Input Autoencoder dimension ({dim_v1_input}).")
        elif padding_width_input > 0:
            v1_input_padded = np.pad(v1_input, ((0, 0), (0, padding_width_input)), mode='constant')
        else:
            v1_input_padded = v1_input


        # Calculate components for the input
        tf_v1_input_padded = tf.constant(v1_input_padded, dtype=tf.float32)
        tf_v2_input = tf.constant(v2_input, dtype=tf.float32)

        concat_input = tf.concat([tf_v1_input_padded, tf_v2_input], axis=1)
        difference_input = tf.norm(tf_v1_input_padded - tf_v2_input, axis=1, keepdims=True)
        hadamard_product_input = tf_v1_input_padded * tf_v2_input

        # Stack input results horizontally
        result_input_tf = tf.concat([concat_input, difference_input, hadamard_product_input], axis=1)
        result_input = result_input_tf.numpy() # Shape (1, 1153)

        print(f"Input combined feature shape: {result_input.shape}")
        print(f"Dataset combined features shape: {combined_features_matrix.shape}")

        # --- Calculate Similarity ---
        # Ensure shapes are compatible for cosine similarity
        if result_input.shape[1] != combined_features_matrix.shape[1]:
            print(f"Error: Shape mismatch between input features ({result_input.shape[1]}) and dataset features ({combined_features_matrix.shape[1]})")
            return pd.DataFrame()

        similarities = cosine_similarity(combined_features_matrix, result_input).flatten()
        print(f"Similarities calculated. Shape: {similarities.shape}")
        if similarities.size > 0:
             print(f"Max Similarity: {np.max(similarities)}")
             print(f"Min Similarity: {np.min(similarities)}")


        # --- Get Top Recommendations ---
        # Ensure similarities array size matches DataFrame index
        if similarities.size != len(df):
             print(f"Error: Number of similarities ({similarities.size}) does not match DataFrame length ({len(df)})")
             # This might happen if df was filtered after combined_features_matrix calculation
             return pd.DataFrame()

        top_indices = np.argsort(similarities)[::-1][:10]
        print(f"Top Indices: {top_indices}")

        recommendations_df = df.iloc[top_indices].copy()
        recommendations_df['Combined_Similarity_Score'] = similarities[top_indices]

        # Ensure required columns are present
        required_columns = ['Perfume_Name', 'Brand', 'URL', 'Image', 'Combined_Similarity_Score', 'Gender', 'Description']
        available_columns = [col for col in required_columns if col in recommendations_df.columns]

        # Add missing columns with default values if needed (optional)
        for col in required_columns:
            if col not in recommendations_df.columns:
                recommendations_df[col] = 'N/A' # Or some other default

        return recommendations_df[available_columns]

    except Exception as e:
        print(f"An error occurred during recommendation generation: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        return pd.DataFrame()


def convert_df_to_list(df):
    """Konversi DataFrame ke list of dictionaries."""
    return df.to_dict(orient='records')