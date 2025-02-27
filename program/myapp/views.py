from django.core.paginator import Paginator
from django.shortcuts import render
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
import joblib

# Ambil path file dataset secara dinamis
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'dataset/preprocessed_data.xlsx')

df = pd.read_excel(DATA_PATH)
df.columns = df.columns.str.replace(" ", "_")

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


def convert_df_to_list(df):
    """Konversi DataFrame ke list of dictionaries."""
    return df.to_dict(orient='records')


# BERT MODEL
def bert_recommend(query, model_bert, doc_embeddings, df):
    # Encode query
    query_embedding = model_bert.encode([query])

    # Hitung kesamaan cosine
    similarities = cosine_similarity(query_embedding, doc_embeddings)

    # Tambahkan skor kesamaan ke DataFrame tanpa mengurutkan
    results = df.copy()  
    results['Similarity'] = similarities[0]
    return results


def bert_model(query): 
    # Load model yang sudah disimpan
    model_bert = SentenceTransformer('bert_model/bert_model')

    # Load embeddings dari file .npy
    doc_embeddings = np.load('bert_model/doc_embeddings.npy')

    if (query == ""):
        print("Query is empty. Please insert a text...")

    else: 
        result_bert = bert_recommend(query, model_bert, doc_embeddings, df)
        
    # Tampilkan hasil pencarian
    return result_bert.sort_values(by='Similarity', ascending=False)


# AUTOENCODER MODEL
# Recommendation function
def recommend_autoencoder(input_encoding, perfume_encodings):
    similarities = cosine_similarity(input_encoding, perfume_encodings)

    recommendations_df = pd.DataFrame({
        'Similarity': similarities[0]
    })
    
    # Concatenate the similarity scores with the original DataFrame
    recommendations_df = pd.concat([recommendations_df, df], axis=1)
    
    return recommendations_df

def autoencoder_model(query_encoding ):
    numerical_columns = df.select_dtypes(include=['float64', 'int64', 'int32']).columns
    columns_to_exclude = ['Rating Value', 'Best Rating', 'Votes']
    features = numerical_columns.drop(columns_to_exclude)
    
    # Load autoencoder dan encoder
    autoencoder = load_model('autoencoder_model/autoencoder_model.h5')
    scaler = joblib.load('autoencoder_model/scaler.pkl')
    
    X = df[features].values
    X_scaled = scaler.transform(X)
    perfume_encodings = autoencoder.predict(X_scaled)

    # Dapatkan rekomendasi
    result_autoencoder = recommend_autoencoder(query_encoding, perfume_encodings)

    # Tampilkan 10 rekomendasi teratas
    return result_autoencoder.sort_values(by='Similarity', ascending=False)



# Views
def index(request):
    return render(request, "myapp/index.html")


def feature1(request):
    notes["Notes"] = notes["Notes"].str.replace("_", " ").str.title()
    
    df_notes = convert_df_to_list(notes) 
    
    return render(request, "myapp/feature1.html", {
        "df_notes": df_notes
    })


def feature2(request):
    df_list = convert_df_to_list(df)

    # Ambil parameter halaman dari request
    page = request.GET.get('page', 1)

    # Paginasi dengan 100 item per halaman
    paginator = Paginator(df_list, 100)
    page_obj = paginator.get_page(page)

    return render(request, "myapp/feature2.html", {
        "datas": page_obj.object_list,
        "page_obj": page_obj  
    })
    