import cv2
import mediapipe as mp
import numpy as np
# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
static_image_mode=False,
max_num_hands=2,
min_detection_confidence=0.5,
min_tracking_confidence=0.5
)
# Buat globe (sphere) dengan latitude dan longitude
def create_globe(radius=1.0, lat_res=20, lon_res=20):
vertices = []
edges = []
# Generate vertices
for i in range(lat_res + 1):
lat = np.pi * (-0.5 + i / lat_res)
for j in range(lon_res):
lon = 2 * np.pi * j / lon_res
x = radius * np.cos(lat) * np.cos(lon)
y = radius * np.sin(lat)
z = radius * np.cos(lat) * np.sin(lon)
vertices.append([x, y, z])
vertices = np.array(vertices, dtype=np.float32)
# Generate edges (latitude lines)
for i in range(lat_res + 1):
for j in range(lon_res):
idx = i * lon_res + j
next_idx = i * lon_res + (j + 1) % lon_res
edges.append([idx, next_idx])
# Generate edges (longitude lines)
for i in range(lat_res):
for j in range(lon_res):
idx = i * lon_res + j
next_idx = (i + 1) * lon_res + j
edges.append([idx, next_idx])
return vertices, edges
# Buat globe dengan lebih detail
vertices, edges = create_globe(radius=1.0, lat_res=12, lon_res=24)
# State untuk kontrol yang lebih smooth
rotation_x, rotation_y, rotation_z = 0, 0, 0
rotation_velocity_x, rotation_velocity_y, rotation_velocity_z = 0, 0, 0
scale = 1.0
scale_velocity = 0
position_x, position_y = 0, 0
position_velocity_x, position_velocity_y = 0, 0
# Parameter untuk smooth motion
DAMPING = 0.85 # Perlambatan gerakan
SENSITIVITY = 0.15 # Sensitivitas pergerakan
MOMENTUM = 0.3 # Momentum untuk gerakan berkelanjutan
# Buka webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
def calculate_distance(p1, p2):
return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
def rotate_x(vertices, angle):
cos_a, sin_a = np.cos(angle), np.sin(angle)
rotation_matrix = np.array([
[1, 0, 0],
[0, cos_a, -sin_a],
[0, sin_a, cos_a]
], dtype=np.float32)
return np.dot(vertices, rotation_matrix.T)
def rotate_y(vertices, angle):
cos_a, sin_a = np.cos(angle), np.sin(angle)
rotation_matrix = np.array([
[cos_a, 0, sin_a],
[0, 1, 0],
[-sin_a, 0, cos_a]
], dtype=np.float32)
return np.dot(vertices, rotation_matrix.T)
def rotate_z(vertices, angle):
cos_a, sin_a = np.cos(angle), np.sin(angle)
rotation_matrix = np.array([
[cos_a, -sin_a, 0],
[sin_a, cos_a, 0],
[0, 0, 1]
], dtype=np.float32)
return np.dot(vertices, rotation_matrix.T)
def project_3d_to_2d(vertices, frame_width, frame_height, scale_factor=150):
"""Proyeksikan vertices 3D ke koordinat 2D layar dengan perspektif"""
projected = []
depths = []
for vertex in vertices:
# Perspektif dengan depth
z = vertex[2] + 5
if z == 0:
z = 0.1
factor = scale_factor / z
x = int(vertex[0] * factor + frame_width / 2 + position_x)
y = int(vertex[1] * factor + frame_height / 2 + position_y)
projected.append([x, y])
depths.append(z)
return np.array(projected), np.array(depths)
def draw_globe(frame, vertices, edges):
"""Gambar globe dengan efek depth"""
height, width = frame.shape[:2]
projected, depths = project_3d_to_2d(vertices, width, height, scale * 150)
# Normalisasi depth untuk coloring
min_depth, max_depth = depths.min(), depths.max()
if max_depth - min_depth > 0:
normalized_depths = (depths - min_depth) / (max_depth - min_depth)
else:
normalized_depths = np.zeros_like(depths)
# Gambar edges dengan warna berdasarkan depth
for i, edge in enumerate(edges):
pt1 = tuple(projected[edge[0]].astype(int))
pt2 = tuple(projected[edge[1]].astype(int))
# Hitung depth rata-rata untuk edge ini
avg_depth = (normalized_depths[edge[0]] + normalized_depths[edge[1]]) / 2
# Warna dari biru gelap (jauh) ke cyan terang (dekat)
color_intensity = int(100 + avg_depth * 155)
color = (color_intensity, color_intensity, int(50 + avg_depth * 205))
# Thickness berdasarkan depth
thickness = max(1, int(1 + avg_depth * 2))
# Cek apakah edge visible (dalam frame)
if (0 <= pt1[0] < width and 0 <= pt1[1] < height and
0 <= pt2[0] < width and 0 <= pt2[1] < height):
cv2.line(frame, pt1, pt2, color, thickness)
# Gambar vertices highlight pada posisi tertentu
for i, (point, depth) in enumerate(zip(projected, normalized_depths)):
if i % 8 == 0: # Hanya gambar beberapa vertices
pt = tuple(point.astype(int))
if 0 <= pt[0] < width and 0 <= pt[1] < height:
color_intensity = int(100 + depth * 155)
color = (color_intensity // 2, color_intensity // 2, color_intensity)
cv2.circle(frame, pt, max(2, int(3 + depth * 3)), color, -1)
def is_pinching(thumb_tip, index_tip, threshold=0.05):
"""Deteksi gesture mencubit"""
return calculate_distance(thumb_tip, index_tip) < threshold
def is_fist(hand_landmarks):
"""Deteksi gesture kepalan tangan"""
# Cek apakah semua jari tertutup
finger_tips = [8, 12, 16, 20] # Index, Middle, Ring, Pinky tips
finger_mcp = [5, 9, 13, 17] # Knuckles
closed_fingers = 0
for tip, mcp in zip(finger_tips, finger_mcp):
if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[mcp].y:
closed_fingers += 1
return closed_fingers >= 3
def get_palm_center(hand_landmarks):
"""Hitung pusat telapak tangan"""
palm_indices = [0, 1, 5, 9, 13, 17]
x = sum([hand_landmarks.landmark[i].x for i in palm_indices]) / len(palm_indices)
y = sum([hand_landmarks.landmark[i].y for i in palm_indices]) / len(palm_indices)
z = sum([hand_landmarks.landmark[i].z for i in palm_indices]) / len(palm_indices)
return x, y, z
print("=" * 60)
print("🌍GLOBE 3D HAND GESTURE CONTROL")
print("=" * 60)
print("\n📋 INSTRUKSI KONTROL:")
print("\n🤚 TANGAN KANAN - Rotasi Globe:")
print(" • Tangan Terbuka: Gerakkan untuk rotasi smooth X & Y")
print(" • Kepalan Tangan: Putar pergelangan untuk rotasi Z")
print(" • Momentum: Globe akan terus berputar dengan inersia")
print("\n✋ TANGAN KIRI - Posisi & Zoom:")
print(" • Cubit (Jempol + Telunjuk): Geser posisi globe")
print(" • Buka/Tutup Cubit: Zoom in/out dengan smooth")
print(" • Kepalan: Reset posisi ke tengah")
print("\n🎮 FITUR ADVANCED:")
print(" • Gerakan memiliki momentum dan inersia")
print(" • Transisi smooth antar gesture")
print(" • Depth shading untuk efek 3D realistis")
print(" • Kontrol independen kedua tangan")
print("\n⌨️ Tekan 'q' untuk keluar")
print("=" * 60)
prev_right_palm = None
prev_left_palm = None
is_rotating = False
is_moving = False
while cap.isOpened():
ret, frame = cap.read()
if not ret:
break
frame = cv2.flip(frame, 1)
height, width = frame.shape[:2]
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = hands.process(rgb_frame)
left_hand_detected = False
right_hand_detected = False
current_right_palm = None
current_left_palm = None
if results.multi_hand_landmarks and results.multi_handedness:
for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
# Gambar landmark tangan
mp_drawing.draw_landmarks(
frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
mp_drawing.DrawingSpec(color=(0, 255, 100), thickness=2, circle_radius=2),
mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
)
hand_label = handedness.classification[0].label
thumb_tip = hand_landmarks.landmark[4]
index_tip = hand_landmarks.landmark[8]
wrist = hand_landmarks.landmark[0]
middle_mcp = hand_landmarks.landmark[9]
palm_x, palm_y, palm_z = get_palm_center(hand_landmarks)
if hand_label == "Right": # Tangan kanan untuk rotasi
right_hand_detected = True
current_right_palm = (palm_x, palm_y, palm_z)
# Deteksi gesture
is_fist_gesture = is_fist(hand_landmarks)
if is_fist_gesture:
# Kepalan tangan: rotasi Z dengan pergelangan
wrist_to_middle = np.array([middle_mcp.x - wrist.x, middle_mcp.y - wrist.y])
target_rotation_z = np.arctan2(wrist_to_middle[0], -wrist_to_middle[1])
rotation_velocity_z += (target_rotation_z - rotation_z) * SENSITIVITY
cv2.putText(frame, "✊ FIST: Z-Rotation", (1000, 3000),
cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 255), 2)
else:
# Tangan terbuka: rotasi dengan momentum
if prev_right_palm is not None:
delta_x = (palm_y - prev_right_palm[1]) * SENSITIVITY * 100
delta_y = (palm_x - prev_right_palm[0]) * SENSITIVITY * 100
rotation_velocity_x += delta_x
rotation_velocity_y += delta_y
is_rotating = True
cv2.putText(frame, "🖐️ OPEN: XY-Rotation", (10, 30),
cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
prev_right_palm = current_right_palm
# Info
cv2.putText(frame, f"Rot X: {np.degrees(rotation_x):.1f}°", (10, 60),
cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
cv2.putText(frame, f"Rot Y: {np.degrees(rotation_y):.1f}°", (10, 85),
cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
cv2.putText(frame, f"Rot Z: {np.degrees(rotation_z):.1f}°", (10, 110),
cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
elif hand_label == "Left": # Tangan kiri untuk zoom dan posisi
left_hand_detected = True
current_left_palm = (palm_x, palm_y, palm_z)
# Deteksi gesture
is_pinch = is_pinching(thumb_tip, index_tip)
is_fist_gesture = is_fist(hand_landmarks)
if is_fist_gesture:
# Kepalan: Reset posisi
position_velocity_x += (0 - position_x) * 0.1
position_velocity_y += (0 - position_y) * 0.1
cv2.putText(frame, "✊ FIST: Reset Position", (10, height - 120),
cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
elif is_pinch:
# Cubit: geser dengan smooth motion
if prev_left_palm is not None:
delta_x = (palm_x - prev_left_palm[0]) * width * 5
delta_y = (palm_y - prev_left_palm[1]) * height * 5
position_velocity_x += delta_x * SENSITIVITY
position_velocity_y += delta_y * SENSITIVITY
is_moving = True
cv2.putText(frame, "🤏 PINCH: Moving", (100, height - 1200),
cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
else:
# Tangan terbuka: zoom dengan smooth
distance = calculate_distance(thumb_tip, index_tip)
target_scale = max(0.3, min(3.0, distance * 10))
scale_velocity += (target_scale - scale) * 0.1
cv2.putText(frame, "🖐️ OPEN: Zooming", (100, height - 1200),
cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
prev_left_palm = current_left_palm
# Info
cv2.putText(frame, f"Scale: {scale:.2f}x", (10, height - 60),
cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
cv2.putText(frame, f"Pos: ({position_x:.0f}, {position_y:.0f})", (10, height - 30),
cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
# Update dengan momentum dan damping
rotation_x += rotation_velocity_x * MOMENTUM
rotation_y += rotation_velocity_y * MOMENTUM
rotation_z += rotation_velocity_z * MOMENTUM
scale += scale_velocity * MOMENTUM
position_x += position_velocity_x * MOMENTUM
position_y += position_velocity_y * MOMENTUM
# Apply damping untuk smooth stop
rotation_velocity_x *= DAMPING
rotation_velocity_y *= DAMPING
rotation_velocity_z *= DAMPING
scale_velocity *= DAMPING
position_velocity_x *= DAMPING
position_velocity_y *= DAMPING
# Clamp values
scale = max(0.3, min(3.0, scale))
# Status
status_text = "🌍GLOBE | Hands: "
if right_hand_detected:
status_text += "RIGHT ✓ "
if left_hand_detected:
status_text += "LEFT ✓"
if not right_hand_detected and not left_hand_detected:
status_text += "NONE"
cv2.putText(frame, status_text, (width - 500, 30),
cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
# Aplikasikan transformasi ke globe
transformed_vertices = rotate_x(vertices, rotation_x)
transformed_vertices = rotate_y(transformed_vertices, rotation_y)
transformed_vertices = rotate_z(transformed_vertices, rotation_z)
transformed_vertices = transformed_vertices * scale
# Gambar globe
draw_globe(frame, transformed_vertices, edges)
# Tampilkan frame
cv2.imshow('🌍Globe 3D Hand Gesture Control', frame)
if cv2.waitKey(1) & 0xFF == ord('q'):
break
# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()
print("\n✅ Program selesai!")
