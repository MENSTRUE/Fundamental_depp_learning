teks_dengan_spasi = "   Ini adalah   kalimat   dengan   spasi   berlebih.   "

# Menghapus spasi di awal dan akhir
teks_strip = teks_dengan_spasi.strip()

# Mengganti spasi ganda di tengah kalimat menjadi spasi tunggal
teks_bersih = re.sub(r'\s+', ' ', teks_strip)

print("Teks Asli:", teks_dengan_spasi)
print("Hasil Akhir:", teks_bersih)