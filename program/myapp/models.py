import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from keras.models import load_model
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
    'woody',
 'citrus',
 'sweet',
 'powdery',
 'floral',
 'fruity',
 'aromatic',
 'white floral',
 'warm spicy',
 'fresh spicy',
 'amber',
 'musky',
 'vanilla',
 'fresh',
 'green',
 'rose',
 'earthy',
 'patchouli',
 'balsamic',
 'soft spicy',
 'aquatic',
 'animalic',
 'leather',
 'lavender',
 'iris',
 'violet',
 'tropical',
 'herbal',
 'yellow floral',
 'oud',
 'ozonic',
 'lactonic',
 'smoky',
 'mossy',
 'tuberose',
 'marine',
 'cinnamon',
 'aldehydic',
 'caramel',
 'almond',
 'coconut',
 'nutty',
 'honey',
 'tobacco',
 'salty',
 'cherry',
 'anis',
 'coffee',
 'cacao',
 'metallic',
 'chocolate',
 'rum',
 'soapy',
 'sour',
 'conifer',
 'mineral',
 'camphor',
 'savory',
 'beeswax',
 'sand',
 'whiskey',
 'Champagne',
 'bitter',
 'terpenic',
 'wine',
 'alcohol',
 'cannabis',
 'vodka',
 'coca-cola',
 'oily',
 'clay',
 'vinyl',
 'asphault',
 'tennis ball',
 'industrial glue',
 'plastic',
 'brown scotch tape',
 'rubber',
 'foresty',
 'sake',
 'paper',
 'Pear'
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
features = [] # Inisialisasi features sebagai list kosong

try:
    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.str.replace(" ", "_")
except FileNotFoundError as e:
    print(f"Error: File not found at {DATA_PATH}. {e}")
except Exception as e:
    print(f"An error occurred while loading the data: {e}")

# Load BERT model and embeddings
try: # Tambahkan try-except block untuk BERT juga
    model_bert = SentenceTransformer(MODEL_BERT_PATH)
    doc_embeddings = np.load(DOC_EMBEDDINGS_PATH)
    print("BERT components loaded successfully.")
except Exception as e:
    print(f"Error loading BERT components: {e}")
    model_bert = None
    doc_embeddings = None


# Load Autoencoder and scaler
try: # Tambahkan try-except block
    if os.path.exists(AUTOENCODER_MODEL_PATH) and os.path.exists(SCALER_PATH):
        autoencoder = load_model(AUTOENCODER_MODEL_PATH, compile=False)
        scaler = joblib.load(SCALER_PATH) # Sudah benar menggunakan joblib
        print("Autoencoder components loaded successfully.")

        # Pre-calculate perfume encodings if df and scaler are loaded
        if not df.empty and scaler is not None and autoencoder is not None:
            numerical_columns = df.select_dtypes(include=np.number).columns # Gunakan np.number untuk mencakup semua tipe numerik
            columns_to_exclude = ['Rating_Value', 'Best_Rating', 'Votes', 'Unnamed:_0'] # Tambahkan 'Unnamed:_0' jika ada
            # Filter out columns that might not exist AND are in the exclusion list
            features = [col for col in numerical_columns if col not in columns_to_exclude]
            print(f"Features identified for scaling: {features}")

            if features: # Proceed only if there are features to scale
                # Ensure all feature columns actually exist in df
                features_present = [f for f in features if f in df.columns]
                if len(features_present) == len(features): # Check if all expected features are present
                    X = df[features_present].copy() # Ambil data fitur
                    # --- PERBAIKAN: Handle NaN dan Infinity ---
                    # Ganti infinity dengan NaN, lalu isi semua NaN dengan 0
                    X.replace([np.inf, -np.inf], np.nan, inplace=True)
                    X.fillna(0, inplace=True)
                    # Pastikan tipe data adalah float32 atau float64 yang sesuai
                    X = X.astype(np.float64) # Atau np.float32 jika sesuai
                    # -----------------------------------------
                    print(f"Shape of X before scaling: {X.shape}")
                    print(f"Data types of X: {X.dtypes.unique()}")
                    print(f"Any NaN in X after fillna? {X.isnull().values.any()}")
                    print(f"Any Inf in X after replace? {np.isinf(X).values.any()}")


                    X_scaled = scaler.transform(X.values) # Lakukan scaling
                    print("Data scaling successful.")
                    perfume_encodings = autoencoder.predict(X_scaled)
                    print(f"Perfume encodings calculated. Shape: {perfume_encodings.shape}")
                else:
                    print(f"Warning: Missing feature columns for autoencoder: {set(features) - set(features_present)}")
                    perfume_encodings = None # Set ke None jika fitur tidak lengkap
            else:
                print("Warning: No numerical features found for autoencoder after exclusions.")
                perfume_encodings = None # Set ke None jika tidak ada fitur
        else:
             # Kondisi jika df kosong atau scaler/autoencoder gagal load
             print("Skipping perfume encoding calculation due to missing df, scaler, or autoencoder.")
             perfume_encodings = None


    else:
        print(f"Error: Autoencoder model or scaler not found at {AUTOENCODER_MODEL_PATH} or {SCALER_PATH}")
        autoencoder = None
        scaler = None
        perfume_encodings = None

except Exception as e:
    print(f"Error loading Autoencoder/Scaler or calculating encodings: {e}")
    import traceback
    traceback.print_exc()
    autoencoder = None
    scaler = None
    perfume_encodings = None


# Pre-calculate combined features matrix
try:
    print("\n--- Attempting Combined Features Matrix Calculation ---") # Added Header
    if perfume_encodings is not None and doc_embeddings is not None:
        print(f"Shape of perfume_encodings: {perfume_encodings.shape}") # Added Print
        print(f"Shape of doc_embeddings: {doc_embeddings.shape}")     # Added Print

        # Ensure consistent number of samples (rows)
        if perfume_encodings.shape[0] == doc_embeddings.shape[0]:
            print("Number of samples match between perfume and doc embeddings.") # Added Print
            v1 = perfume_encodings  # Shape e.g., (num_perfumes, 82) -> or latent dim
            v2 = doc_embeddings    # Shape e.g., (num_perfumes, 384)

            # Get dimensions
            num_perfumes, dim_v1 = v1.shape
            _, dim_v2 = v2.shape
            print(f"Autoencoder embedding dimension (dim_v1): {dim_v1}") # Added Print
            print(f"BERT embedding dimension (dim_v2): {dim_v2}")       # Added Print

            # Pad v1 to match v2 dimension
            padding_width = dim_v2 - dim_v1
            print(f"Padding width required for v1: {padding_width}") # Added Print

            if padding_width < 0:
                 # This case means BERT dim < Autoencoder dim. Need to decide how to handle.
                 # Option 1: Raise error (current)
                 # Option 2: Pad v2 instead
                 # Option 3: Truncate v1
                 print(f"Error Condition: BERT dimension ({dim_v2}) is smaller than Autoencoder dimension ({dim_v1}).") # Added Print
                 raise ValueError(f"BERT embedding dimension ({dim_v2}) is smaller than Autoencoder dimension ({dim_v1}). Padding logic needs adjustment or model dimensions must be aligned.")
            elif padding_width > 0:
                 v1_padded = np.pad(v1, ((0, 0), (0, padding_width)), mode='constant', constant_values=0) # Ensure padding with 0
                 print(f"Shape of v1 after padding: {v1_padded.shape}") # Added Print
            else:
                 v1_padded = v1 # No padding needed if dimensions match
                 print("Dimensions match, no padding needed for v1.") # Added Print


            # --- Inner Try-Except for TensorFlow operations ---
            try:
                print("Attempting TensorFlow operations for combination...") # Added Print
                # Calculate components using TensorFlow
                tf_v1_padded = tf.constant(v1_padded, dtype=tf.float32)
                tf_v2 = tf.constant(v2, dtype=tf.float32)

                concat = tf.concat([tf_v1_padded, tf_v2], axis=1)
                difference = tf.norm(tf_v1_padded - tf_v2, axis=1, keepdims=True)
                hadamard_product = tf_v1_padded * tf_v2

                # Stack results horizontally
                combined_features_matrix_tf = tf.concat([concat, difference, hadamard_product], axis=1)
                # Convert back to numpy array BEFORE assigning to the main variable
                temp_combined_matrix = combined_features_matrix_tf.numpy()
                print(f"Successfully calculated combined matrix with TensorFlow. Shape: {temp_combined_matrix.shape}") # Added Print
                combined_features_matrix = temp_combined_matrix # Assign only on success

            except Exception as tf_error:
                print(f"!!! Error during TensorFlow combination: {tf_error}") # Added Specific Error Print
                import traceback
                traceback.print_exc()
                combined_features_matrix = None # Ensure it's None on TF error
            # --- End Inner Try-Except ---

        else:
            print(f"!!! Error: Mismatch in number of samples between perfume_encodings ({perfume_encodings.shape[0]}) and doc_embeddings ({doc_embeddings.shape[0]})") # Enhanced Print
            combined_features_matrix = None
    else:
        print("!!! Skipping combined features matrix calculation due to missing prerequisites:") # Enhanced Print
        if perfume_encodings is None:
            print("  - perfume_encodings is None")
        if doc_embeddings is None:
            print("  - doc_embeddings is None")
        combined_features_matrix = None # Ensure it's None if prerequisites missing

except Exception as e:
    print(f"!!! Error calculating combined features matrix (Outer Try-Except): {e}") # Enhanced Print
    import traceback
    traceback.print_exc()
    combined_features_matrix = None # Ensure it's None on any outer error

# --- Final Check and Log ---
if combined_features_matrix is not None:
    print(f"--- Combined Features Matrix Calculation Successful ---")
    print(f"Final shape: {combined_features_matrix.shape}")
else:
    print(f"--- Combined Features Matrix Calculation Failed ---")
    # The specific reason should have been printed above


# --- Fungsi Rekomendasi ---

# Remove bert_recommend and recommend_autoencoder as they are replaced

# Fungsi get_combined_recommendations
# -----------------------------------
# 🌟 Fungsi utama untuk menghasilkan rekomendasi parfum gabungan.
# Menerima deskripsi teks dari pengguna dan daftar persentase untuk berbagai fitur (notes).
# Proses:
# 1. Memeriksa apakah semua komponen model (BERT, Autoencoder, Scaler, matriks fitur gabungan, DataFrame) telah dimuat.
# 2. Mengubah input persentase menjadi array numpy dan melakukan scaling.
# 3. Menghasilkan embedding dari input persentase menggunakan Autoencoder (v1_input).
# 4. Menghasilkan embedding dari deskripsi teks menggunakan model BERT (v2_input).
# 5. Menggabungkan v1_input dan v2_input (setelah padding jika perlu untuk menyamakan dimensi)
#    menjadi satu vektor fitur input tunggal melalui konkatenasi, perbedaan (norma), dan produk Hadamard.
# 6. Menghitung similaritas kosinus antara vektor fitur input gabungan dengan matriks fitur gabungan
#    dari semua parfum dalam dataset.
# 7. Mengembalikan 10 parfum teratas dengan skor similaritas tertinggi.
# Jika ada komponen yang hilang atau terjadi error, fungsi akan mengembalikan DataFrame kosong.
def get_combined_recommendations(description, percentages):
    print("--- Inside get_combined_recommendations (New Method) ---")
    print(f"Description: {description}")
    print(f"Percentages: {percentages}")

    # Check if all necessary components are loaded
    if model_bert is None or autoencoder is None or scaler is None or combined_features_matrix is None or df.empty or not features: # Use 'not features' for empty list check
        print("Warning: One or more models/data components failed to load or calculate. Cannot generate recommendations.")
        print(f"  model_bert loaded: {model_bert is not None}")
        print(f"  doc_embeddings loaded: {doc_embeddings is not None}") # Add check for doc_embeddings here too
        print(f"  autoencoder loaded: {autoencoder is not None}")
        print(f"  scaler loaded: {scaler is not None}")
        print(f"  perfume_encodings calculated: {perfume_encodings is not None}") # Add check for perfume_encodings
        print(f"  combined_features_matrix calculated: {combined_features_matrix is not None}") # This should now reflect the actual status
        print(f"  df loaded: {not df.empty}")
        print(f"  features identified: {bool(features)}") # Check if features list is populated
        return pd.DataFrame()

    try:
        # --- Autoencoder Part (Input) ---
        input_data = np.array(percentages).reshape(1, -1)

        # --- PERBAIKAN: Gunakan panjang 'features' yang dihitung saat load ---
        num_expected_features = len(features)
        if input_data.shape[1] != num_expected_features:
             print(f"Error: Input percentages length ({input_data.shape[1]}) does not match expected features length ({num_expected_features}) based on loaded data.")
             # Opsional: Coba pad/truncate jika memungkinkan, tapi lebih aman error
             # if input_data.shape[1] < num_expected_features:
             #     input_data = np.pad(input_data, ((0,0), (0, num_expected_features - input_data.shape[1])), 'constant')
             # elif input_data.shape[1] > num_expected_features:
             #     input_data = input_data[:, :num_expected_features]
             # else: # Jika tidak ingin pad/truncate:
             return pd.DataFrame() # Kembalikan kosong jika dimensi tidak cocok

        # Handle NaN/inf pada input juga (meskipun seharusnya sudah bersih dari view)
        input_data = np.nan_to_num(input_data)

        # --- Lakukan scaling pada input ---
        input_scaled = scaler.transform(input_data)
        v1_input = autoencoder.predict(input_scaled) # Shape (1, num_features) -> (1, latent_dim)

        # --- BERT Part (Input) ---
        v2_input = model_bert.encode([description]) # Shape (1, 384)

        # --- Combine Input Features ---
        # Dapatkan dimensi latent autoencoder dari outputnya
        dim_v1_input = v1_input.shape[1]
        dim_v2_input = v2_input.shape[1]

        # Pad v1_input to match v2_input dimension
        padding_width_input = dim_v2_input - dim_v1_input
        if padding_width_input < 0:
             # Ini bisa terjadi jika latent dim > bert dim
             # Anda mungkin perlu mem-pad v2 atau memotong v1, tergantung logika yang diinginkan
             print(f"Warning: Input BERT embedding dimension ({dim_v2_input}) is smaller than Input Autoencoder dimension ({dim_v1_input}). Adjusting padding logic.")
             # Contoh: Pad v2 instead
             v2_input_padded = np.pad(v2_input, ((0, 0), (0, -padding_width_input)), mode='constant')
             v1_input_padded = v1_input
             # Recalculate difference and hadamard using adjusted shapes if necessary
        elif padding_width_input > 0:
            v1_input_padded = np.pad(v1_input, ((0, 0), (0, padding_width_input)), mode='constant')
            v2_input_padded = v2_input # No padding needed for v2
        else:
            v1_input_padded = v1_input
            v2_input_padded = v2_input


        # Calculate components for the input using padded versions
        tf_v1_input_padded = tf.constant(v1_input_padded, dtype=tf.float32)
        tf_v2_input_padded = tf.constant(v2_input_padded, dtype=tf.float32) # Gunakan v2 yang mungkin sudah dipad

        concat_input = tf.concat([tf_v1_input_padded, tf_v2_input_padded], axis=1)
        # Hitung difference dengan dimensi yang sudah disamakan
        difference_input = tf.norm(tf_v1_input_padded - tf_v2_input_padded, axis=1, keepdims=True)
        # Hitung hadamard product dengan dimensi yang sudah disamakan
        hadamard_product_input = tf_v1_input_padded * tf_v2_input_padded

        # Stack input results horizontally
        result_input_tf = tf.concat([concat_input, difference_input, hadamard_product_input], axis=1)
        result_input = result_input_tf.numpy()

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

# Fungsi convert_df_to_list
# -------------------------
# 🔄 Mengkonversi DataFrame Pandas menjadi list of dictionaries.
# Setiap dictionary dalam list mewakili satu baris dari DataFrame,
# dengan keys adalah nama kolom dan values adalah nilai sel.
def convert_df_to_list(df):
    """Konversi DataFrame ke list of dictionaries."""
    return df.to_dict(orient='records')