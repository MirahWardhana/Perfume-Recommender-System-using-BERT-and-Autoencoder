from django.shortcuts import render, redirect
from django.views.decorators.http import require_POST, require_GET
from django.views.decorators.csrf import csrf_protect, csrf_exempt
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.http import JsonResponse
import json
import pandas as pd
from .models import notes as notes_df_global, get_combined_recommendations, convert_df_to_list, df, model_bert, autoencoder, scaler, features
from django.urls import reverse
import traceback
import numpy as np
from rapidfuzz import process, fuzz

# Views
def index(request):
    return render(request, "myapp/index.html")

@csrf_protect
def feature1(request):
    notes_name_list = []
    if notes_df_global is not None and 'Notes' in notes_df_global.columns:
        notes_name_list = notes_df_global["Notes"].str.replace("_", " ").str.title().tolist()
    context = {"df_notes": notes_df_global.to_dict(orient='records') if notes_df_global is not None else [],
               "notes_name_list": notes_name_list}

    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

    if request.method == 'POST':
        # Debug: print raw request data and AJAX flag
        print("=== FEATURE1 POST RECEIVED (Feature2 AJAX) ===")
        print("AJAX flag:", is_ajax)
        print("Raw request body:", request.body)
        print("Request headers:", dict(request.headers))
        try:
            description = ''
            percentages_dict = {}

            if is_ajax:
                data = json.loads(request.body)
                description = data.get('description', '')
                percentages_input = data.get('percentages', {})
                if not isinstance(percentages_input, dict):
                     return JsonResponse({'status': 'error', 'message': 'Invalid percentages format.'}, status=400)
<<<<<<< Updated upstream
                percentages_dict = {k: float(v) for k, v in percentages_input.items() if isinstance(v, (int, float))}
=======
                # Normalize the keys in percentages_input
                percentages_dict = {k.replace("_", " ").title(): v for k, v in percentages_input.items() if isinstance(v, (int, float))}

                # DEBUG: Log incoming data from feature2 when Get Recommendations is clicked
                print("--- FEATURE2 → FEATURE1 AJAX Received ---")
                print(f"Description: {description}")
                print(f"Selected Percentages: {percentages_dict}")
>>>>>>> Stashed changes

            else:
                description = request.POST.get('description', '')
                for note_item in notes_df_global.to_dict(orient='records'):
                    note_name = note_item['Notes']
                    percentage_val_str = request.POST.get(note_name)
                    try:
<<<<<<< Updated upstream
                        percentages_dict[note_name] = float(percentage_val_str) if percentage_val_str else 0.0
                    except (ValueError, TypeError):
                        percentages_dict[note_name] = 0.0
=======
                        percentages_dict[note_name.replace("_", " ").title()] = int(percentage_val_str) if percentage_val_str else 0
                    except (ValueError, TypeError):
                        percentages_dict[note_name.replace("_", " ").title()] = 0
>>>>>>> Stashed changes

            ordered_percentages = []
            if features:
                num_expected_features = len(features)
                print(f"Expected features order (models.py): {features}")
                for feature_name in features:
<<<<<<< Updated upstream
                    percentage_value = min(1.0, max(0.0, float(percentages_dict.get(feature_name, 0.0))))
=======
                    normalized_feature_name = feature_name.replace("_", " ").title()
                    percentage_value = percentages_dict.get(normalized_feature_name, 0)
>>>>>>> Stashed changes
                    ordered_percentages.append(percentage_value)

                print(f"Input Percentages Dict (from request): {percentages_dict}")
                print(f"Ordered Percentages List (to model): {ordered_percentages}")
                if len(ordered_percentages) != num_expected_features:
                    print(f"Warning: Mismatch ordered percentages ({len(ordered_percentages)}) vs expected features ({num_expected_features}).")

            else:
                print("Error: 'features' list from models.py is empty or not loaded.")
                if is_ajax:
                    return JsonResponse({'status': 'error', 'message': 'Model features configuration error.'}, status=500)
                else:
                    context['error_message'] = 'Model features configuration error.'
                    return render(request, "myapp/feature1.html", context)

            recommendations_df = get_combined_recommendations(description, ordered_percentages)

            recommendations = []
            if recommendations_df is not None and not recommendations_df.empty:
                recommendations = recommendations_df.to_dict(orient='records')
                print(f"Generated {len(recommendations)} recommendations.")
            else:
                print("No recommendations generated.")

            if is_ajax:
                 return JsonResponse({'recommendations': recommendations})
            else:
                 context['description'] = description
                 context['input_percentages'] = percentages_dict
                 context['recommendations'] = recommendations
                 return render(request, "myapp/feature1.html", context)

        except json.JSONDecodeError:
             print("Error decoding JSON from AJAX request.")
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
        non_zero_mask = (numeric_subset != 0) & (numeric_subset.notna())
        filtered_numeric_cols = numeric_subset.loc[:, non_zero_mask.any(axis=0)].columns
        numeric_df_final = input_df[filtered_numeric_cols]

        return input_df, numeric_df_final

    except KeyError as e:
         print(f"Error in insight: DataFrame key error - {e}. Available columns: {df.columns.tolist()}")
         return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
         print(f"Unexpected error in insight: {type(e).__name__} - {e}")
         traceback.print_exc()
         return pd.DataFrame(), pd.DataFrame()


def feature2(request):
    search_query = request.GET.get('search', '').strip()
    print(f"--- Feature 2 View ---")
    print(f"Search Query Received: '{search_query}'")

    current_df = df.copy() if df is not None and not df.empty else pd.DataFrame()
    df_list = [] # Start with an empty list

    if not current_df.empty:
        print(f"DataFrame loaded. Shape: {current_df.shape}")
        if search_query:
            print("--- Starting Fuzzy Search ---")
            search_columns = ['Perfume_Name', 'Brand', 'Description']
            existing_search_cols = [col for col in search_columns if col in current_df.columns]
            if existing_search_cols:
                 try:
                    current_df['Searchable_Text'] = current_df[existing_search_cols].fillna('').astype(str).agg(' '.join, axis=1).str.lower()
                    query_lower = search_query.lower()
                    cutoff_score = 55 # Keep lower cutoff for now, adjust later if needed
                    results = process.extract(
                        query_lower,
                        current_df['Searchable_Text'],
                        scorer=fuzz.WRatio,
                        limit=len(current_df),
                        score_cutoff=cutoff_score
                    )
                    matched_indices = [res[2] for res in results]
                    print(f"Number of matches found above cutoff: {len(matched_indices)}")
                    if matched_indices:
                        filtered_df = current_df.iloc[matched_indices].copy()
                        df_list = convert_df_to_list(filtered_df)
                    else:
                        df_list = [] # Explicitly empty if no matches
                 except Exception as e:
                    print(f"Error during fuzzy search processing: {e}")
                    df_list = [] # Empty list on error
            else:
                 print("Error: Search columns not found.")
                 df_list = []
        else:
            print("No search query provided, using all data.")
            df_list = convert_df_to_list(current_df) # Use all data if no search
    else:
        print("Error: DataFrame 'df' is not loaded or empty.")
        df_list = [] # Ensure df_list is empty if df fails to load

    print(f"Length of df_list before pagination: {len(df_list)}")

    # --- Pagination (Applied AFTER filtering - df_list might be empty) ---
    page_number = request.GET.get('page', 1)
    items_per_page = 100
    paginator = Paginator(df_list, items_per_page)

    try:
        page_obj = paginator.get_page(page_number)
    except PageNotAnInteger:
        # If page is not an integer, deliver first page.
        page_obj = paginator.get_page(1)
    except EmptyPage:
        # If page is out of range (e.g., 9999), deliver last page of results.
        page_obj = paginator.get_page(paginator.num_pages)

    # This should work even if df_list is empty (page_obj.number will be 1)
    elided_page_range = paginator.get_elided_page_range(number=page_obj.number,
                                                         on_each_side=2,
                                                         on_ends=1)

    # --- Notes List for Feature 2 AJAX ---
    all_notes_list = []
    if notes_df_global is not None and 'Notes' in notes_df_global.columns:
         all_notes_list = notes_df_global["Notes"].str.replace("_", " ").str.title().tolist()

    # --- Context ---
    context = {
        "datas": page_obj.object_list, # List of items on the current page (might be empty)
        "page_obj": page_obj,         # The Page object itself
        "elided_page_range": elided_page_range, # The calculated range for display
        "notes_name_list_json": json.dumps(all_notes_list),
        "search_query": search_query,
    }
    print("--- Feature 2 View END ---")
    return render(request, "myapp/feature2.html", context)