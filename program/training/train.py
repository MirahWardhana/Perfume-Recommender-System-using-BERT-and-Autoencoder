# %% [markdown]
# # **Import Library**

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from keras.models import Model
from keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import joblib

# %% [markdown]
# # **Get and Preprocessed The Dataset**

# %%
df = pd.read_excel('D:/KULIAH/Tugas-Tugas Kuliah Semester 7/Tugas Akhir/dataset/data.xlsx')

unique_accords = set()
for accords_str in df['Main Accords']:
    accords = accords_str.split('\n')
    for accord in accords:
        if ':' in accord:
            unique_accords.add(accord.split(':')[0].strip())

# 2. Create new columns for each unique accord
df = df.assign(**{accord: 0.0 for accord in unique_accords}) 

# 3. Populate the accord columns with percentages
for index, row in df.iterrows():
    accords_str = row['Main Accords']
    accords = accords_str.split('\n')
    for accord in accords:
        if ':' in accord:
            accord_name = accord.split(':')[0].strip()
            percentage = float(accord.split(':')[1].strip().rstrip('%')) / 100
            df.loc[index, accord_name] = percentage

# 4. Remove the 'Main Accords' column (optional)
df = df.drop('Main Accords', axis=1) 
df

# %% [markdown]
# # **Feature Selection**

# %%
numerical_columns = df.select_dtypes(include=['float64', 'int64', 'int32']).columns
numerical_columns

# %%
columns_to_exclude = ['Rating Value', 'Best Rating', 'Votes']
features = numerical_columns.drop(columns_to_exclude)
features

# %% [markdown]
# # **Build Autoecoder Model**

# %%
# Encoder
encoder_input = layers.Input(shape=(len(features),))
encoder = layers.Dense(128, activation='relu')(encoder_input)
encoder = layers.Dense(64, activation='relu')(encoder)
encoder_output = layers.Dense(32, activation='relu')(encoder)

# Decoder
decoder = layers.Dense(64, activation='relu')(encoder_output)
decoder = layers.Dense(128, activation='relu')(decoder)
decoder_output = layers.Dense(len(features), activation='sigmoid')(decoder)

# Autoencoder Model
autoencoder = models.Model(inputs=encoder_input, outputs=decoder_output)
autoencoder.compile(optimizer='adam', loss=MeanSquaredError())

# %%
n_splits = 10 

# K-Fold Cross-Validation
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
X = df[features].values

# Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    mode='min',
    min_delta=0.0001
)

# %% [markdown]
# # **Extract Features and Iterate through the folds**

# %%
# List untuk menyimpan hasil
scalers = []
training_losses = []
validation_losses = []

# Membuat subplot sesuai jumlah fold
fig, axes = plt.subplots(n_splits, 1, figsize=(15, 4 * n_splits))
fig.tight_layout(pad=3.0)

# Jika hanya ada 1 fold, ubah menjadi list untuk konsistensi
if n_splits == 1:
    axes = [axes]

# Training dan evaluasi untuk setiap fold
for fold_idx, (train_index, val_index) in enumerate(kfold.split(X)):
    X_train, X_val = X[train_index], X[val_index]
    
    # Normalisasi data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    scalers.append(scaler)
    
    # Melatih model
    history = autoencoder.fit(
        X_train_scaled, X_train_scaled,
        epochs=100,
        batch_size=32,
        validation_data=(X_val_scaled, X_val_scaled),
        verbose=0,
        callbacks=[early_stopping]
    )
    
    # Menyimpan riwayat loss untuk plotting
    axes[fold_idx].plot(history.history['loss'], label='Training Loss')
    axes[fold_idx].plot(history.history['val_loss'], label='Validation Loss')
    axes[fold_idx].set_title(f"Fold {fold_idx + 1}")
    axes[fold_idx].set_xlabel('Epochs')
    axes[fold_idx].set_ylabel('Loss')
    axes[fold_idx].legend()
    
    # Evaluasi data training
    train_loss = autoencoder.evaluate(X_train_scaled, X_train_scaled, verbose=0)
    training_losses.append(train_loss)
    
    # Evaluasi data validasi
    val_loss = autoencoder.evaluate(X_val_scaled, X_val_scaled, verbose=0)
    validation_losses.append(val_loss)
    
    print(f'Fold {fold_idx + 1}: Training MSE = {train_loss:.4f}, Validation MSE = {val_loss:.4f}')

# Menampilkan plot
plt.show()

# Menyimpan plot ke file
fig.savefig("training_kfold_results.png", dpi=300, bbox_inches='tight')
print("\nTraining and validation loss telah disimpan di training_kfold_results.png")

# %%
# Generate encodings after scaling
X_scaled = np.zeros_like(X, dtype=np.float64)

for fold_idx, (train_index, val_index) in enumerate(kfold.split(X)):
    X_fold = X[val_index]
    scaler = scalers[fold_idx]
    X_scaled[val_index] = scaler.transform(X_fold)

perfume_encodings = autoencoder.predict(X_scaled) 

# %%
df[features]

# %%
# Simpan autoencoder
autoencoder.save('D:/KULIAH/Tugas-Tugas Kuliah Semester 7/Tugas Akhir/autoencoder/autoencoder_model.h5')

# Simpan scaler dari fold tertentu, misalnya fold 0
joblib.dump(scalers[0], 'D:/KULIAH/Tugas-Tugas Kuliah Semester 7/Tugas Akhir/autoencoder/scaler.pkl')


# %% [markdown]
# # **Compute Similarity**

# %%
# Recommendation function
def recommend_perfumes(input_encoding):
    similarities = cosine_similarity(input_encoding, perfume_encodings)

    recommendations_df = pd.DataFrame({
        'Similarity Score': similarities[0]
    })
    
    # Concatenate the similarity scores with the original DataFrame
    recommendations_df = pd.concat([recommendations_df, df.reset_index()], axis=1) 
    return recommendations_df.sort_values(by='Similarity Score', ascending=False)

# %% [markdown]
# # **Get Recommendations**

# %%
# Load autoencoder dan encoder
autoencoder = load_model('D:/KULIAH/Tugas-Tugas Kuliah Semester 7/Tugas Akhir/autoencoder/autoencoder_model.h5')
scaler = joblib.load('D:/KULIAH/Tugas-Tugas Kuliah Semester 7/Tugas Akhir/autoencoder/scaler.pkl')

perfume_encodings = autoencoder.predict(X_scaled)
perfume_encodings

# %%
# Test the function and encodings
query_perfume_index = 15704
recommendations = recommend_perfumes(perfume_encodings[query_perfume_index].reshape(1, -1))
print(f"Encoding for perfume at index {query_perfume_index}:\n{perfume_encodings[query_perfume_index]}")
print("\nTop 10 recommendations:")
recommendations.head(10)

# %%
data_for_heatmap = recommendations[features].head(11)

plt.figure(figsize=(16, 4))
sns.heatmap(
    data_for_heatmap,
    cmap="coolwarm",     
    annot=False,          
    cbar=True,             
    xticklabels=True,     
    yticklabels=True     
)

plt.title("Heatmap of Feature Values")
plt.xlabel("Features")
plt.ylabel("Samples")
plt.show()

# %%
recommendations[features]

# %%
row = df.iloc[7193]
print(", ".join(row[features].astype(str)))

# %%
# Test the function and encodings
query_perfume_index = 15704
recommendations = recommend_perfumes(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.64999919500072, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.85999935600058, 0.0, 0.0, 0.0]).reshape(1, -1))
print(f"Encoding for perfume at index {query_perfume_index}:\n{perfume_encodings[query_perfume_index]}")
print("\nTop 10 recommendations:")
recommendations.head(10)

# %%
data_for_heatmap = recommendations[features].head(11)

plt.figure(figsize=(16, 4))
sns.heatmap(
    data_for_heatmap,
    cmap="coolwarm",     
    annot=False,          
    cbar=True,             
    xticklabels=True,     
    yticklabels=True     
)

plt.title("Heatmap of Feature Values")
plt.xlabel("Features")
plt.ylabel("Samples")
plt.show()

# %%
recommendations[features].head(10).to_csv("hasil_print.csv")

# %%


# %% [markdown]
# # **Save The Model**

# %%
# Save encodings
np.save('D:/KULIAH/Tugas-Tugas Kuliah Semester 7/Tugas Akhir/model_encodings.npy', perfume_encodings)

# Load encodings
perfume_encodings = np.load('D:/KULIAH/Tugas-Tugas Kuliah Semester 7/Tugas Akhir/model_encodings.npy')

# %% [markdown]
# # **Finding Relevant Index**

# %%
df_recommendation = recommendations.head(30)

# %%
row = df.iloc[query_perfume_index]
input = np.array(row[features])
input_df = pd.DataFrame([input], columns=recommendations[features].columns)
input_df

# %%
df_recommendation[features]

# %%
df_features_subtracted = df_recommendation[features].sub(input_df.iloc[0], axis=1).abs()
df_features_subtracted.insert(0, 'Total', df_features_subtracted.sum(axis=1))
df_features_subtracted = df_features_subtracted.sort_values(by='Total', ascending=True)
df_features_subtracted

# %%
index_list = df_features_subtracted.index.to_list()
df_selected_rows = df.iloc[index_list]
df_selected_rows

# %%
df_selected_rows.index.to_list

# %%
data_for_heatmap = df_selected_rows[features].head(11)

plt.figure(figsize=(16, 4))
sns.heatmap(
    data_for_heatmap,
    cmap="coolwarm",     
    annot=False,          
    cbar=True,             
    xticklabels=True,     
    yticklabels=True     
)

plt.title("Heatmap of Feature Values")
plt.xlabel("Features")
plt.ylabel("Samples")
plt.show()

# %% [markdown]
# # **Evaluate Result**

# %%
# --- Evaluation Metrics ---
def precision_at_k(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]
    
    true_positives = sum(1 for p in predicted if p in actual)
    return true_positives / min(len(predicted), k)

def average_precision(actual, predicted, k=10):
    if not actual:
        return 0.0
    
    precision_values = []
    num_rel = 0
    for i, p in enumerate(predicted):
      if i >= k:
        break
      if p in actual:
        num_rel += 1
        precision_values.append(num_rel / (i + 1))
    
    if not precision_values:
        return 0.0
    return sum(precision_values) / min(len(actual), k)

def reciprocal_rank(actual, predicted, k=10):
    for i, item in enumerate(predicted):
        if i >= k:
            break
        if item in actual:
            return 1 / (i + 1)
    return 0.0

# Calculate Mean metrics
def calculate_mean_metrics(perfume_encodings, df, relevant_docs, k=10):
    mean_precision = 0.0
    mean_ap = 0.0
    mean_rr = 0.0
    num_queries = len(relevant_docs)
    
    for query_id, relevant_items in relevant_docs.items():
        if query_id >= len(perfume_encodings):
            continue  
            
        query_encoding = perfume_encodings[query_id].reshape(1, -1)
        recommendations = recommend_perfumes(query_encoding)
        recommended_items = recommendations['index'].tolist()

        # Calculate metrics for this query
        prec_k = precision_at_k(relevant_items, recommended_items, k)
        avg_prec = average_precision(relevant_items, recommended_items, k)
        recip_rank = reciprocal_rank(relevant_items, recommended_items, k)

        mean_precision += prec_k
        mean_ap += avg_prec
        mean_rr += recip_rank
        
        print(f"Query ID: {query_id}")
        print(f"  Precision@{k}: {prec_k:.4f}")
        print(f"  Average Precision: {avg_prec:.4f}")
        print(f"  Reciprocal Rank: {recip_rank:.4f}")

    if num_queries == 0:
      return 0.0, 0.0, 0.0
    return mean_precision/num_queries, mean_ap/num_queries, mean_rr/num_queries


# %%
relevant_docs = {
    10:[10, 7522, 20135, 196, 10059, 15392, 10113, 17137, 6456, 7417],
    1143:[247,101,248], 
    100:[1143, 15695,3962,7298, 4678]
}
# Evaluate the model and print the results
mean_precision, mean_ap, mean_rr = calculate_mean_metrics(perfume_encodings, df, relevant_docs, k=10)
print("\n------------------------------------------\n")
print(f"Mean Precision@10: {mean_precision:.4f}")
print(f"Mean Average Precision (MAP): {mean_ap:.4f}")
print(f"Mean Reciprocal Rank (MRR): {mean_rr:.4f}")

# %% [markdown]
# # **Create Customizable Notes Percentage**

# %%
# --- GUI Part ---
root = tk.Tk()
root.title("Features Table with Sliders")

frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)

canvas = tk.Canvas(frame)
scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

# Tempatkan scrollbar dan canvas
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Daftar untuk menyimpan slider
sliders = []

# Tambahkan data ke tabel
for i, feature in enumerate(features):
    # Label untuk kolom pertama
    label = tk.Label(scrollable_frame, text=feature, anchor="w", width=10)
    label.grid(row=i, column=0, padx=0, pady=0, sticky="w")

    # Slider untuk kolom kedua
    slider = tk.Scale(scrollable_frame, from_=0, to=100, orient=tk.HORIZONTAL, length=200)
    slider.grid(row=i, column=1, padx=0, pady=0)
    sliders.append(slider)

# Fungsi untuk mengambil nilai slider
def get_slider_values():
    slider_values = {}
    for i, slider in enumerate(sliders):
        slider_values[features[i]] = slider.get()

    # Konversi nilai slider ke NumPy array
    input_array = np.array([slider_values[feature] for feature in features], dtype=float)

    # Normalisasi jika diperlukan (contoh: rentang slider 0-100 ke 0-1)
    input_array = input_array / 100

    # Transformasi dengan scaler (gunakan scaler dari fold 0)
    input_scaled = scalers[0].transform([input_array])

    # Dapatkan encoding dari model encoder
    input_encoding = autoencoder.predict(input_scaled)

    # Hitung similarity dan tampilkan rekomendasi
    recommendations = recommend_perfumes(input_encoding)

    # # Tampilkan 10 rekomendasi teratas
    print(recommendations['URL'].head(10).to_list())
    print(recommendations.index[:10].tolist())


    data_for_heatmap = recommendations[features].head(11)

    plt.figure(figsize=(16, 4))
    sns.heatmap(
        data_for_heatmap,
        cmap="coolwarm",
        annot=False,
        cbar=True,
        xticklabels=True,
        yticklabels=True
    )

    plt.title("Heatmap of Feature Values")
    plt.xlabel("Features")
    plt.ylabel("Samples")
    plt.show()


# Tombol untuk menampilkan nilai slider
btn = tk.Button(root, text="Get Recommendations", command=get_slider_values)
btn.pack(pady=10)

root.mainloop()

# %% [markdown]
# # **Save Model**

# %%
# Simpan autoencoder
autoencoder.save('D:/KULIAH/Tugas-Tugas Kuliah Semester 7/Tugas Akhir/autoencoder/autoencoder_model.h5')

# Simpan encoder
encoder_model.save('D:/KULIAH/Tugas-Tugas Kuliah Semester 7/Tugas Akhir/autoencoder/encoder_model.h5')


# %%
import joblib


# Simpan scaler dari fold tertentu, misalnya fold 0
joblib.dump(scalers[0], 'D:/KULIAH/Tugas-Tugas Kuliah Semester 7/Tugas Akhir/autoencoder/scaler.pkl')


# %%



