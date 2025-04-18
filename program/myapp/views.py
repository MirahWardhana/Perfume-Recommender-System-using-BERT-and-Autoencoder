from django.shortcuts import render, redirect
from django.views.decorators.http import require_POST, require_GET
from django.views.decorators.csrf import csrf_protect
from django.core.paginator import Paginator
from django.http import JsonResponse
import json
import pandas as pd
from .models import notes, get_combined_recommendations, convert_df_to_list, df, model_bert, autoencoder, scaler
from django.urls import reverse

# Views
def index(request):
    return render(request, "myapp/index.html")

def feature1(request):
    notes_df = notes.copy()
    notes_df["Notes"] = notes_df["Notes"].str.replace("_", " ").str.title()
    df_notes = convert_df_to_list(notes_df)

    if request.method == 'POST':
        description = request.POST.get('description', '')
        percentages = []
        num_notes = len(notes)

        # Debugging: Cetak semua data POST
        print("--- POST Data ---")
        for key, value in request.POST.items():
            print(f"{key}: {value}")
        print("------------------")


        for i in range(1, num_notes + 1):
            percentage_key = f'percentage_{i}'
            percentage = request.POST.get(percentage_key)  # Hapus default value

            # Periksa apakah percentage ada dan merupakan angka
            if percentage: 
                try:
                    percentages.append(int(percentage))
                except ValueError:
                    print(f"Warning: Invalid percentage value for {percentage_key}: {percentage}")
                    percentages.append(0)  # Atau berikan nilai default lain, atau lewati
            else:
                print(f"Warning: Missing percentage value for {percentage_key}")  
                percentages.append(0) # Masukkan nilai default jika tidak ada
                

        # Dapatkan rekomendasi
        recommendations_df = get_combined_recommendations(description, percentages)
        recommendations = recommendations_df.to_dict(orient='records')
        
        print(recommendations)

        context = {
            'description': description,
            'percentages': percentages,
            'notes': notes.Notes.tolist(),
            "df_notes": df_notes,
            'recommendations': recommendations,
        }

        return render(request, "myapp/feature1.html", context)
    else:
        return render(request, "myapp/feature1.html", {"df_notes": df_notes})



@csrf_protect
@require_POST
def process_url(request):
    try:
        data = json.loads(request.body)
        url = data.get('url')
        if url is None:
             return JsonResponse({'status': 'error', 'message': 'URL key is missing.'}, status=400)

        input_df, numeric_df = insight(url)

        if input_df.empty:
            return JsonResponse({'status': 'error', 'message': 'URL not found.'}, status=404)

        request.session['numeric_df'] = numeric_df.to_dict(orient='records')
        request.session['input_df'] = input_df.to_dict(orient='records')
        return JsonResponse({'status': 'success', 'message': 'URL diterima.'})

    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON.'}, status=400)
    except KeyError:
        return JsonResponse({'status': 'error', 'message': 'URL key is missing.'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


def insight(url):
    input_df = df[df['URL'] == url]
    numeric_cols = input_df.select_dtypes(include=['number']).columns
    non_zero_mask = (input_df[numeric_cols] != 0)
    filtered_numeric = input_df[numeric_cols].loc[:, non_zero_mask.any()]  
    result_df = pd.concat([input_df.drop(columns=numeric_cols), filtered_numeric], axis=1)
    result_df = result_df.drop(columns=['Unnamed:_0','Rating_Value', 'Best_Rating', 'Votes'], errors='ignore')
    numeric_df = result_df.select_dtypes(include=['number'])
    return input_df, numeric_df
    
def feature2(request):
    if request.method == 'POST':
        return redirect('feature2')

    df_list = convert_df_to_list(df)
    page_number = request.GET.get('page', 1)
    paginator = Paginator(df_list, 100)
    page_obj = paginator.get_page(page_number)

    # Calculate elided page range in the view
    elided_page_range = paginator.get_elided_page_range(number=page_obj.number, 
                                                         on_each_side=2, 
                                                         on_ends=2)

    # Retrieve numeric_df data from the session (if it exists)
    numeric_df_data = request.session.get('numeric_df')
    input_df_data = request.session.get('input_df')
    numeric_df = None
    input_df = None

    if numeric_df_data:
        numeric_df = pd.DataFrame(numeric_df_data)
        # Modify the column names *before* passing to the template
        new_cols = []
        for col in numeric_df.columns:
             new_cols.append(col.replace("_", " ").title())
        numeric_df.columns = new_cols
    if input_df_data:
        input_df = pd.DataFrame(input_df_data)

    # Add elided_page_range to the context
    context = {
        "datas": page_obj.object_list,
        "page_obj": page_obj,
        "numeric_df": numeric_df,
        "input_df": input_df,
        "elided_page_range": elided_page_range,
    }
    return render(request, "myapp/feature2.html", context)