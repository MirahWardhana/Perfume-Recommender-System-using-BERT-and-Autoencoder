'use strict';


// navbar variables
const nav = document.querySelector('.navbar-nav');
const navLinks = document.querySelectorAll('.nav-link');
const cartToggleBtn = document.querySelector('.shopping-cart-btn');
const navToggleBtn = document.querySelector('.menu-toggle-btn');
const shoppingCart = document.querySelector('.cart-box');



// nav toggle function
const navToggleFunc = function () {
  nav.classList.toggle('active');
  navToggleBtn.classList.toggle('active');
}

// shopping cart toggle function
const cartToggleFunc = function () { shoppingCart.classList.toggle('active') }



// add event on nav-toggle-btn
navToggleBtn.addEventListener('click', function () {

  // If the shopping-cart has an `active` class, it will be removed.
  if (shoppingCart.classList.contains('active')) cartToggleFunc();

  navToggleFunc();

});

// add event on cart-toggle-btn
cartToggleBtn.addEventListener('click', function () {

  // If the navbar-nav has an `active` class, it will be removed.
  if (nav.classList.contains('active')) navToggleFunc();

  cartToggleFunc();

});

// add event on all nav-link
for (let i = 0; i < navLinks.length; i++) {

  navLinks[i].addEventListener('click', navToggleFunc);

}

function setSelectorPosition() {
  var activeItem = $('#navbarSupportedContent .active');
  var selector = $('.hori-selector');
  if (activeItem.length) {
      selector.css({
          top: activeItem.position().top + 'px',
          left: activeItem.position().left + 'px',
          width: activeItem.innerWidth() + 'px',
          height: activeItem.innerHeight() + 'px'
      });
  }
}

$(document).ready(function () {
  // Set default active section ke LSTM
  $('#navbarSupportedContent ul li').removeClass('active');
  $('[data-target="LSTM"]').addClass('active');

  // Sembunyikan semua section dan tampilkan hanya LSTM
  $('.content-section').hide();
  $('#LSTM').show();

  // Fungsi untuk navigasi antar tab
  $('#navbarSupportedContent ul li').on('click', function () {
      var target = $(this).data('target');
      $('.content-section').hide();
      $('#' + target).show();

      $('#navbarSupportedContent ul li').removeClass('active');
      $(this).addClass('active');
      setSelectorPosition();
  });

  setSelectorPosition();
});


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
