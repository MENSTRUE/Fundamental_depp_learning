import re

teks_dengan_angka = "Ada 3 mobil dan 10 motor di parkiran pada tahun 2025."

# Proses menghapus angka menggunakan regular expression (regex)
teks_tanpa_angka = re.sub(r"\d+", "", teks_dengan_angka)

print("Teks Asli:", teks_dengan_angka)
print("Hasil Menghapus Angka:", teks_tanpa_angka)