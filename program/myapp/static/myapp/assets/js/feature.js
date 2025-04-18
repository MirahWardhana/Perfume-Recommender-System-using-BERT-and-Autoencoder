// Mengonversi nilai menjadi kode warna heksadesimal
function intToHex(value) {
    var hex = value.toString(16);
    return hex.length == 1 ? "0" + hex : hex;
}

// Membuat warna berdasarkan nilai persentase
function makeColor(value) {
    value = Math.min(Math.max(0, value), 100);
    var redValue = 255 - Math.round(value * 2.55);
    var greenValue = Math.round(value * 2.55);
    return "#" + intToHex(redValue) + intToHex(greenValue) + "00";
}

// Memperbarui nilai dan warna slider secara dinamis
function updateSlider(id) {
    const slider = document.getElementById(`slider${id}`);
    const percentage = document.getElementById(`sliderPercentage${id}`);
    const color = makeColor(slider.value);

    percentage.textContent = slider.value + "%"; // Memperbarui teks
    slider.style.background = `linear-gradient(to right, ${color} ${slider.value}%, #ddd ${slider.value}%)`; // Warna latar slider
}

// Inisialisasi warna slider saat memuat halaman
document.querySelectorAll(".custom-slider").forEach((slider, index) => {
    updateSlider(index + 1);
});

function scrollToInsight() {
    document.getElementById("insightDiv").scrollIntoView({ behavior: "smooth" });
}

function scrollToTarget() {
    document.getElementById("targetDiv").scrollIntoView({ behavior: "smooth" });
}

function scrollToInsight() {
    document.getElementById("insightDiv").scrollIntoView({ behavior: "smooth" });
}

function processCardClick(url) {
    console.log("URL yang diklik (di browser): " + url);
    sendURLToServer(url);
    scrollToInsight();
}

function sendURLToServer(url) {
    fetch('/process-url/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken')
        },
        body: JSON.stringify({ url: url })
    })
    .then(response => response.json())  // Ubah menjadi response.json()
    .then(data => {
        if (data.status === 'success') {
            console.log('URL berhasil dikirim ke server.');
            updateInsightDiv(data.insight_data); // Panggil fungsi untuk update tampilan
        } else {
            console.error('Gagal mengirim URL ke server:', data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}


function updateInsightDiv(insightData) {
    if (insightData && insightData.length > 0) {
        const data = insightData[0]; // Ambil data pertama (karena insight() mengembalikan list)

        // Update gambar
        document.getElementById('insightImage').src = data.Image;

        // Update tabel (contoh sederhana, sesuaikan dengan struktur data Anda)
        const tableBody = document.getElementById('insightTableBody');
        tableBody.innerHTML = `
            <tr><td>Name</td><td>${data.Perfume_Name}</td></tr>
            <tr><td>Brand</td><td>${data.Brand}</td></tr>
             <tr><td>URL</td><td>${data.URL}</td></tr>
             
        `; //Ganti data sesuain yang ingin di tampilkan

    } else {
        console.error("Data insight kosong.");
    }
}

function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            let cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}