import joblib
import numpy as np
from tensorflow.keras.models import load_model

# --- Ganti dengan path file Anda ---
scaler_path = './scaler.pkl'  # Ganti dengan path ke file scaler.pkl Anda
model_path = './autoencoder_model.h5'    # Ganti dengan path ke file model.h5 Anda

# --- Muat Scaler ---
try:
    scaler = joblib.load(scaler_path)
    print("Scaler berhasil dimuat menggunakan joblib.")
except FileNotFoundError:
    print(f"Error: File scaler tidak ditemukan di {scaler_path}")
    exit()
except Exception as e:
    print(f"Error saat memuat scaler dengan joblib: {e}")
    exit()

# --- Muat Model Keras ---
try:
    model = load_model(model_path)
    print("Model Keras berhasil dimuat.")
    # Anda mungkin ingin melihat ringkasan model untuk memastikan input shape
    # model.summary()
except FileNotFoundError:
    print(f"Error: File model tidak ditemukan di {model_path}")
    exit()
except Exception as e:
    print(f"Error saat memuat model: {e}")
    exit()

# --- Siapkan Data Input Contoh ---
# Ganti data ini dengan data input Anda yang sebenarnya.
# Pastikan shape dan tipe datanya sesuai dengan yang diharapkan oleh model Anda.
# Contoh: Jika model Anda mengharapkan 1 sampel dengan 10 fitur:
contoh_input_data = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, -1)  

# --- Lakukan Preprocessing dengan Scaler ---
try:
    # Pastikan data di-reshape jika scaler mengharapkannya (misalnya, jika dilatih pada data 2D)
    # Jika scaler dilatih pada data 1D, Anda mungkin tidak perlu reshape
    # Sesuaikan '.shape[1]' jika scaler dilatih pada data dengan jumlah fitur yang berbeda
    if len(contoh_input_data.shape) == 1:
         contoh_input_data_reshaped = contoh_input_data.reshape(1, -1) # Ubah ke 2D jika perlu
    else:
         contoh_input_data_reshaped = contoh_input_data

    input_data_scaled = scaler.transform(contoh_input_data_reshaped)
    print("Data input berhasil di-scale.")
    print("Data setelah di-scale:", input_data_scaled)

except AttributeError:
     print("Error: Objek yang dimuat dari scaler.pkl sepertinya bukan scaler Scikit-learn (tidak ada metode 'transform').")
     exit()
except ValueError as e:
     print(f"Error saat scaling data: {e}. Pastikan jumlah fitur data input ({contoh_input_data.shape[-1]}) cocok dengan yang diharapkan scaler.")
     exit()
except Exception as e:
     print(f"Error saat scaling data: {e}")
     exit()


# --- Lakukan Prediksi ---
try:
    prediksi = model.predict(input_data_scaled)
    print("\n--- Hasil Prediksi ---")
    print(prediksi)
except ValueError as e:
    print(f"\nError saat melakukan prediksi: {e}. Periksa input shape yang diharapkan model.")
    # Anda bisa coba print model.summary() di atas untuk melihat input shape
except Exception as e:
    print(f"\nError saat melakukan prediksi: {e}")
    exit()