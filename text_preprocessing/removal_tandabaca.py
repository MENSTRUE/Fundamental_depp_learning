import string

teks_dengan_tanda_baca = "Apa kabarmu hari ini? Semoga baik-baik saja!"

# Proses menghapus tanda baca
# string.punctuation berisi semua tanda baca: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
teks_tanpa_tanda_baca = teks_dengan_tanda_baca.translate(str.maketrans("","",string.punctuation))

print("Teks Asli:", teks_dengan_tanda_baca)
print("Hasil Menghapus Tanda Baca:", teks_tanpa_tanda_baca)