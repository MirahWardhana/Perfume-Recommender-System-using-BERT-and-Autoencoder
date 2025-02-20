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
