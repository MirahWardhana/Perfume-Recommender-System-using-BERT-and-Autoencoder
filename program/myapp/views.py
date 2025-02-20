from django.shortcuts import render
import os
import pandas as pd

# Ambil path file dataset secara dinamis
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'dataset/preprocessed_data.xlsx')

# Fungsi untuk membaca dataset dengan aman
def load_dataset():
    try:
        df = pd.read_excel(DATA_PATH)
        # Ganti spasi dengan underscore (_) dalam nama kolom
        df.columns = df.columns.str.replace(" ", "_")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()  


def convert_df_to_list(df):
    """Konversi DataFrame ke list of dictionaries."""
    return df.to_dict(orient='records')

# Views
def index(request):
    return render(request, "myapp/index.html")

def feature1(request):
    return render(request, 'myapp/feature1.html')

def feature2(request):
    if request.method == 'GET':
        df = load_dataset()  # Muat ulang dataset
        df_list = convert_df_to_list(df)
        return render(request, "myapp/feature2.html", {
            "datas": df_list[:100]  # Kirim 100 data pertama ke template
        })
