from django.core.paginator import Paginator
from django.shortcuts import render
import os
import pandas as pd

# Ambil path file dataset secara dinamis
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'dataset/preprocessed_data.xlsx')

df = pd.read_excel(DATA_PATH)
df.columns = df.columns.str.replace(" ", "_")

def convert_df_to_list(df):
    """Konversi DataFrame ke list of dictionaries."""
    return df.to_dict(orient='records')


# Views
def index(request):
    return render(request, "myapp/index.html")

def feature1(request):
    return render(request, 'myapp/feature1.html')

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