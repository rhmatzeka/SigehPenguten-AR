# AR Tari Sigeh Penguten — Rancangan

Tanggal: 2026-10-13 · Deadline: Jumat 2026-10-16 · Tujuan: tugas kuliah/sekolah

## Hasil akhir

Poster A4 (desain user dari Canva) dengan kode QR di pojok kanan bawah, di samping kotak "Makna".
1. Orang scan QR → web AR terbuka di browser HP (tanpa install aplikasi).
2. Web minta izin kamera → kamera diarahkan ke poster.
3. Penari 3D berkostum Lampung (siger, baju putih, tapis merah, kuku tanggai) muncul berdiri di atas poster
   dan menari berulang (loop) mengikuti gerakan dari video tari asli.

Sukses = jalan di HP Android dan iPhone biasa, penari jelas terlihat sebagai penari Lampung, gerakan
mengikuti video. Model boleh agak kasar (bukan kualitas lomba). Tanpa musik.

## Bahan dari user

- Foto penari: `~/Downloads/WhatsApp Image 2026-10-13 at 18.23.11.jpeg` (sumber model 3D)
- Poster draf: `~/Downloads/WhatsApp Image 2026-10-13 at 18.28.47.jpeg` (versi final resolusi asli menyusul)
- Video tari: `~/Downloads/WhatsApp Video 2026-10-13 at 18.23.12.mp4` (57 dtk, 1280x720, 5 penari,
  penari tengah terlihat penuh; 0-8 dtk berjalan masuk berdiri bawa tepak, sisanya berlutut di belakang tepak
  dengan gerakan tangan; kamera diam tapi sempat zoom)

## Alur pembuatan

| # | Tahap | Alat | Keluaran |
|---|-------|------|----------|
| 1 | Hapus background foto | rembg (lokal) | PNG penari tanpa latar |
| 2 | Foto → model 3D | Tripo / Meshy / Hunyuan3D (gratis, user login) | GLB bertekstur |
| 3 | Rigging | Mixamo atau auto-rig Meshy/Tripo | FBX/GLB dengan tulang humanoid |
| 4 | Siapkan video | ffmpeg: crop penari tengah, perbesar, pilih segmen | MP4 satu penari |
| 5 | Video → gerakan | Rokoko Vision / DeepMotion (gratis, user login) | BVH/FBX animasi |
| 6 | Pasang gerakan ke model | Blender (headless, skrip Python) | GLB beranimasi |
| 7 | Optimasi | gltf-transform (meshopt, tekstur ≤1024px) | GLB ≤ ~8 MB |
| 8 | Web AR | MindAR.js (image tracking) + Three.js, versi terkunci di CDN | `index.html` + `poster.mind` |
| 9 | Hosting | GitHub Pages repo `rhmatzeka/tari-sigeh-penguten` (HTTPS wajib untuk kamera) | link publik |
| 10 | QR | QR PNG resolusi tinggi ke link Pages | ditempel user di Canva |

Link dan QR dibuat paling awal (tahap 9-10 duluan dengan halaman sementara), supaya user bisa
menyelesaikan poster sambil model dikerjakan. Target scan (`poster.mind`) dikompilasi dari poster final
yang sudah ada QR-nya; bila poster berubah, target dikompilasi ulang.

## Gerakan: full badan persis video (pilihan user)

- Rok tapis: bobot skinning rok dipindah ke pinggul/panggul dan dihaluskan antar paha, supaya rok tidak
  melar atau terbelah saat berlutut.
- Tepak (kotak sirih) tidak dimodelkan; tangan bergerak seperti di video.
- Kaki tertutup rok sehingga hasil mocap kaki bisa kacau. Cadangan: kaki dikunci (pose berlutut/berdiri)
  dan badan atas tetap ikut video. Cadangan hanya dipakai setelah ditunjukkan ke user.
- Zoom kamera bisa membuat posisi akar (root) melompat; posisi akar dihaluskan atau dikunci di tempat
  supaya penari tidak bergeser dari poster.

## Web AR

- Satu halaman statis: tombol "Mulai" → izin kamera → tracking poster → model muncul + animasi loop.
- Saat poster hilang dari kamera, model disembunyikan; muncul lagi saat poster terlihat.
- Indikator memuat (model beberapa MB), pesan bila kamera ditolak atau browser tidak didukung.
- Uji: kompilasi target, cek di desktop dengan webcam/gambar poster di layar, lalu user uji di HP.

## Risiko

- Kuku tanggai dan siger bisa menyatu/tumpul hasil image-to-3D.
- Kuota gratis layanan AI terbatas: pilih segmen video ± 20-30 dtk terbaik bila perlu.
- Blender belum terpasang; RAM laptop 7 GB, jadi Blender dijalankan headless saja.

## Di luar cakupan

Musik/gamelan, aplikasi Android, filter IG/TikTok, model kualitas lomba.
