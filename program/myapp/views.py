from django.shortcuts import render, redirect
from django.views.decorators.http import require_POST, require_GET
from django.views.decorators.csrf import csrf_protect, csrf_exempt
from django.core.paginator import Paginator
from django.http import JsonResponse
import json
import pandas as pd
from .models import notes as notes_df_global, get_combined_recommendations, convert_df_to_list, df, model_bert, autoencoder, scaler, features
from django.urls import reverse
import traceback
import numpy as np

# Views
@csrf_exempt
def index(request):
    return render(request, "myapp/index.html")

@csrf_exempt
def feature1(request):
    notes_name_list = []
    if notes_df_global is not None and 'Notes' in notes_df_global.columns:
        notes_name_list = notes_df_global["Notes"].str.replace("_", " ").str.title().tolist()
    context = {"df_notes": notes_df_global.to_dict(orient='records') if notes_df_global is not None else [],
               "notes_name_list": notes_name_list}

    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

    if request.method == 'POST':
        try:
            description = ''
            percentages_list = []

            if is_ajax:
                data = json.loads(request.body)
                description = data.get('description', '')
                percentages_input = data.get('percentages', [])
                if not isinstance(percentages_input, list):
                    return JsonResponse({'status': 'error', 'message': 'Invalid percentages format. Expected array.'}, status=400)
                percentages_list = [float(val) for val in percentages_input]
            else:
                description = request.POST.get('description', '')
                percentages_json = request.POST.get('percentages', '[]')
                try:
                    percentages_list = json.loads(percentages_json)
                    if not isinstance(percentages_list, list):
                        raise ValueError("Percentages must be an array")
                    percentages_list = [float(val) for val in percentages_list]
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error parsing percentages: {e}")
                    percentages_list = []

            print(f"Received percentages (raw): {percentages_list}")

            if features:
                num_expected_features = len(features)
                if len(percentages_list) != num_expected_features:
                    print(f"Warning: Received {len(percentages_list)} percentages but expected {num_expected_features}")
                    # Pad or truncate the list to match expected length
                    if len(percentages_list) < num_expected_features:
                        percentages_list.extend([0.0] * (num_expected_features - len(percentages_list)))
                    else:
                        percentages_list = percentages_list[:num_expected_features]
            else:
                print("Error: 'features' list from models.py is empty or not loaded.")
                if is_ajax:
                    return JsonResponse({'status': 'error', 'message': 'Model features configuration error.'}, status=500)
                else:
                    context['error_message'] = 'Model features configuration error.'
                    return render(request, "myapp/feature1.html", context)

            recommendations_df = get_combined_recommendations(description, percentages_list)

            recommendations = []
            if recommendations_df is not None and not recommendations_df.empty:
                # Get exactly 10 recommendations
                recommendations = recommendations_df.head(10).to_dict(orient='records')
                print(f"Generated {len(recommendations)} recommendations.")
            else:
                print("No recommendations generated.")

            if is_ajax:
                return JsonResponse({'recommendations': recommendations})
            else:
                context['description'] = description
                context['input_percentages'] = percentages_list
                context['recommendations'] = recommendations
                return render(request, "myapp/feature1.html", context)

        except json.JSONDecodeError:
            if is_ajax:
                return JsonResponse({'status': 'error', 'message': 'Invalid JSON data.'}, status=400)
            else:
                context['error_message'] = 'Invalid data submitted.'
                return render(request, "myapp/feature1.html", context)
        except Exception as e:
            print(f"Error processing feature1 request: {e}")
            traceback.print_exc()
            if is_ajax:
                return JsonResponse({'status': 'error', 'message': 'An internal server error occurred.'}, status=500)
            else:
                context['error_message'] = 'An error occurred while generating recommendations.'
                return render(request, "myapp/feature1.html", context)

    else:
        return render(request, "myapp/feature1.html", context)

@csrf_exempt
@require_POST
def process_url(request):
    try:
        data = json.loads(request.body)
        url = data.get('url')
        if url is None:
             return JsonResponse({'status': 'error', 'message': 'URL key is missing.'}, status=400)

        input_df, numeric_df = insight(url)

        if input_df.empty:
            return JsonResponse({'status': 'error', 'message': 'Perfume data not found for the given URL.'}, status=404)

        numeric_data_formatted = []
        if not numeric_df.empty:
            first_numeric_record = numeric_df.iloc[0].to_dict()
            for col, val in first_numeric_record.items():
                 formatted_col = col.replace("_", " ").title()
                 serializable_val = float(val) if isinstance(val, (np.number, np.floating, np.integer)) else val
                 numeric_data_formatted.append({'col': formatted_col, 'val': serializable_val})

        input_data_dict = {}
        if not input_df.empty:
            required_cols = ['Perfume_Name', 'Brand', 'Gender', 'Description', 'Image', 'URL']
            existing_cols = [col for col in required_cols if col in input_df.columns]
            input_data_dict = input_df[existing_cols].iloc[0].to_dict()
            for key, value in input_data_dict.items():
                if pd.isna(value):
                    input_data_dict[key] = None
                elif isinstance(value, (np.datetime64, pd.Timestamp)):
                     input_data_dict[key] = value.isoformat()

        return JsonResponse({
            'status': 'success',
            'message': 'Data retrieved successfully.',
            'numeric_data': numeric_data_formatted,
            'input_data': input_data_dict
        })

    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON payload.'}, status=400)
    except Exception as e:
        print(f"Error in process_url: {type(e).__name__} - {e}")
        traceback.print_exc()
        return JsonResponse({'status': 'error', 'message': 'An internal server error occurred while processing the request.'}, status=500)


def insight(url):
    if df is None or df.empty:
         print("Error in insight: Main DataFrame 'df' is not loaded.")
         return pd.DataFrame(), pd.DataFrame()

    try:
        input_df = df[df['URL'] == url].copy()
        if input_df.empty:
            print(f"Insight: URL '{url}' not found in DataFrame.")
            return pd.DataFrame(), pd.DataFrame()

        numeric_cols = input_df.select_dtypes(include=np.number).columns
        cols_to_exclude = ['Rating_Value', 'Best_Rating', 'Votes', 'Unnamed:_0']
        valid_numeric_cols = [col for col in numeric_cols if col not in cols_to_exclude and col in input_df.columns]

        numeric_subset = input_df[valid_numeric_cols]
        numeric_df_final = input_df[valid_numeric_cols]

        return input_df, numeric_df_final

    except KeyError as e:
         print(f"Error in insight: DataFrame key error - {e}. Available columns: {df.columns.tolist()}")
         return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
         print(f"Unexpected error in insight: {type(e).__name__} - {e}")
         traceback.print_exc()
         return pd.DataFrame(), pd.DataFrame()


@csrf_exempt
def feature2(request):
    df_list = []
    if df is not None and not df.empty:
        df_list = convert_df_to_list(df)

    page_number = request.GET.get('page', 1)
    items_per_page = 100
    paginator = Paginator(df_list, items_per_page)
    page_obj = paginator.get_page(page_number)

    elided_page_range = paginator.get_elided_page_range(number=page_obj.number,
                                                         on_each_side=2,
                                                         on_ends=1)

    numeric_df_processed = []
    input_data_dict = None

    all_notes_list = []
    if notes_df_global is not None and 'Notes' in notes_df_global.columns:
         all_notes_list = notes_df_global["Notes"].tolist()

    context = {
        "datas": page_obj.object_list,
        "page_obj": page_obj,
        "numeric_df": numeric_df_processed,
        "input_df": input_data_dict,
        "elided_page_range": elided_page_range,
        "notes_name_list_json": json.dumps(all_notes_list),
    }

    return render(request, "myapp/feature2.html", context)