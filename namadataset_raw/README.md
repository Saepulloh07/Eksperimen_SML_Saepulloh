# namadataset_raw

Folder ini berisi data mentah (raw) yang menjadi input pipeline preprocessing.

Letakkan 3 file Excel berikut di folder ini (nama & sheet dapat disesuaikan lewat
argumen CLI pada `automate_nama-siswa.py` bila periode datanya berbeda):

| File | Sheet yang dipakai | Keterangan |
|---|---|---|
| `Laporan kasir.xlsx` | `TO`, `TRANSFERAN` | Data kasir tunai & transfer harian |
| `data sistem 1 - 31 mei.xlsx` | `PendapatanPerAkunClosing` | Data closing sistem kasir |
| `ARUS KAS 2026.xlsx` | `Mei` (nama bulan berjalan) | Mutasi rekening / arus kas bank |

> Catatan: karena data ini bersifat sensitif (data keuangan internal), file
> aslinya **tidak** disertakan di repository publik. Commit file Excel Anda
> sendiri ke folder ini (atau unggah lewat UI GitHub) agar GitHub Actions
> dapat menjalankan preprocessing secara otomatis setiap kali ada perubahan.
