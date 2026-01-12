# 3D-Globe-Using-Handgesture
Project ini membangun sistem visualisasi globe 3D yang bisa dikontrol langsung dengan gestur tangan. Sistem memakai webcam, MediaPipe Hands, dan OpenCV. Bisa memutar, menggeser, dan melakukan zoom pada globe secara real time. Gerakan terasa halus karena sistem memakai momentum dan damping.

TEKNOLOGI
• Python
• OpenCV
• MediaPipe Hands
• NumPy

FITUR UTAMA
• Rotasi 3 sumbu X, Y, Z berbasis gestur tangan.
• Zoom in dan zoom out dengan jarak jempol dan telunjuk.
• Geser posisi globe dengan gesture cubit.
• Momentum dan inersia untuk gerakan berkelanjutan.
• Depth shading untuk efek 3D.
• Kontrol dua tangan secara independen.

KONTROL GESTUR
Tangan kanan.
• Tangan terbuka. Gerakkan untuk rotasi X dan Y.
• Kepalan tangan. Putar pergelangan untuk rotasi Z.

Tangan kiri.
• Cubit jempol dan telunjuk. Geser posisi globe.
• Tangan terbuka. Atur zoom.
• Kepalan tangan. Reset posisi ke tengah.

CARA MENJALANKAN
• Pastikan Python sudah terpasang.
• Install dependensi.
– opencv-python
– mediapipe
– numpy
• Jalankan file Python.
• Arahkan tangan ke kamera.
• Tekan tombol q untuk keluar.

KEGUNAAN PROJECT
• Contoh Human Computer Interaction.
• Media pembelajaran transformasi 3D.
• Demo interaksi real time berbasis gestur.

CATATAN
• Gunakan pencahayaan yang cukup agar deteksi tangan stabil.
• Webcam dengan resolusi HD memberi hasil lebih halus.
