import numpy as np
from itertools import combinations
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# ============================================================
# 1. Корни E8
# ============================================================
def e8_roots():
    roots = []
    for i, j in combinations(range(8), 2):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                r = np.zeros(8)
                r[i] = s1
                r[j] = s2
                roots.append(r)
    for signs in range(256):
        bits = [(signs >> k) & 1 for k in range(8)]
        if sum(bits) % 2 == 0:
            r = np.array([(-1 if b else 1) / 2 for b in bits])
            roots.append(r)
    return np.array(roots)

roots = e8_roots()
print(f"Корней E8: {len(roots)}")

# ============================================================
# 2. Гексагональная проекция
# ============================================================
# Стандартная проекция E8 на плоскость, где корни ложатся
# в гексагональную решётку. Используем два вектора:
# v1 = (1, -1, 0, 0, 0, 0, 0, 0) / sqrt(2)
# v2 = (0, 0, 1, -1, 0, 0, 0, 0) / sqrt(2)
# Но это даёт квадратную решётку. Для гексагональной нужна
# специальная комбинация. Известная проекция E8 -> 2D:
# через векторы (1,1,0,0,0,0,0,0) и (1/2, 1/2, sqrt(3)/2, ...) — нет.

# Используем подход: проекция на плоскость, натянутую на
# два вектора, которые дают гексагональную решётку.
# Известно: E8 содержит подрешётку A2 (гексагональную).
# Векторы A2: (1, -1, 0, 0, 0, 0, 0, 0) и (0, 1, -1, 0, 0, 0, 0, 0)
# Но это даёт треугольную решётку, а не гексагональную.

# Более правильный подход: проекция на плоскость, перпендикулярную
# вектору (1,1,1,1,1,1,1,1). Это даёт кольца, но не гексагональную.
# Для гексагональной используем проекцию на плоскость,
# натянутую на (1,1,0,0,0,0,0,0) и (0,0,1,1,0,0,0,0) с поворотом.

# Практически: используем проекцию на плоскость,
# где E8 даёт гексагональную решётку с кратностью 8.
# Это проекция на плоскость, натянутую на:
v1 = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=float) / np.sqrt(2)
v2 = np.array([0, 0, 1, 1, 0, 0, 0, 0], dtype=float) / np.sqrt(2)
# Но это даёт квадрат. Для гексагональной нужно добавить поворот.

# Используем известный факт: E8 -> 2D гексагональная решётка
# через проекцию на плоскость, натянутую на векторы:
# v1 = (1, -1, 0, 0, 0, 0, 0, 0) / sqrt(2)
# v2 = (0, 1, -1, 0, 0, 0, 0, 0) / sqrt(2)
# Это даёт треугольную решётку (гексагональную упаковку).

v1 = np.array([1, -1, 0, 0, 0, 0, 0, 0], dtype=float) / np.sqrt(2)
v2 = np.array([0, 1, -1, 0, 0, 0, 0, 0], dtype=float) / np.sqrt(2)

print(f"v1 = {np.round(v1, 4)}")
print(f"v2 = {np.round(v2, 4)}")
print(f"v1·v2 = {np.dot(v1, v2):.6f} (должно быть не 0 для гексагональной)")
# Для гексагональной нужно, чтобы угол был 60°, а не 90°.
# Проверим: v1·v2 = -1/2 (если не нормировать). После нормировки:
print(f"Угол между v1 и v2: {np.degrees(np.arccos(np.dot(v1, v2))):.2f}°")

# Если угол 90°, это не гексагональная. Нужно скорректировать v2.
# Сделаем v2 так, чтобы угол был 60°:
# v2 = (1/2, 1/2, -1, 0, 0, 0, 0, 0) нормированный
v2_hex = np.array([1, 1, -2, 0, 0, 0, 0, 0], dtype=float)
v2_hex = v2_hex / np.linalg.norm(v2_hex)
print(f"\nv2_hex = {np.round(v2_hex, 4)}")
print(f"v1·v2_hex = {np.dot(v1, v2_hex):.6f}")
print(f"Угол v1-v2_hex: {np.degrees(np.arccos(np.dot(v1, v2_hex))):.2f}° (должно быть 60°)")

# Ортогонализуем v2_hex к v1
v2_orth = v2_hex - np.dot(v2_hex, v1) * v1
v2_orth = v2_orth / np.linalg.norm(v2_orth)
print(f"\nv2_orth (ортогонализованный) = {np.round(v2_orth, 4)}")
print(f"v1·v2_orth = {np.dot(v1, v2_orth):.6f} (должно быть 0)")

# ============================================================
# 3. Проекция корней
# ============================================================
proj = np.array([[np.dot(r, v1), np.dot(r, v2_orth)] for r in roots])
print(f"\nПроекция: {proj.shape}")

# ============================================================
# 4. Три аспекта
# ============================================================
print("\n" + "=" * 60)
print("ТРИ АСПЕКТА (гексагональная проекция)")
print("=" * 60)

# [1] Масштаб
radii = np.linalg.norm(proj, axis=1)
unique_radii = np.unique(np.round(radii, 3))
unique_radii = unique_radii[unique_radii > 0.01]
print(f"\n[1] МАСШТАБ (Φ)")
print(f"    Радиусы: {unique_radii[:10]}")
if len(unique_radii) >= 2:
    ratios = unique_radii[1:] / unique_radii[:-1]
    print(f"    Отношения: {np.round(ratios[:8], 4)}")
    phi = (1 + np.sqrt(5)) / 2
    print(f"    Φ = {phi:.4f}")
    closest = ratios[np.argmin(np.abs(ratios - phi))]
    print(f"    Ближайшее к Φ: {closest:.4f} (разница {abs(closest - phi):.4f})")

# [2] Цикл
angles = np.arctan2(proj[:, 1], proj[:, 0])
unique_angles = np.unique(np.round(angles, 3))
print(f"\n[2] ЦИКЛ (π)")
print(f"    Углы (рад): {unique_angles[:15]}")
for denom in [3, 4, 5, 6]:
    target = np.pi / denom
    close = np.any(np.abs(unique_angles - target) < 0.05)
    print(f"    π/{denom} = {target:.4f}: {'ЕСТЬ' if close else 'НЕТ'}")

# [3] Упаковка
unique_proj = np.unique(np.round(proj, 3), axis=0)
print(f"\n[3] УПАКОВКА (√3)")
print(f"    Уникальных точек: {len(unique_proj)}")
if len(unique_proj) > 1:
    dm = cdist(unique_proj, unique_proj)
    dm[dm == 0] = np.inf
    min_dists = dm.min(axis=1)
    unique_dists = np.unique(np.round(min_dists, 3))
    print(f"    Ближайшие расстояния: {unique_dists[:10]}")
    if len(unique_dists) >= 2:
        ratios = unique_dists[1:] / unique_dists[:-1]
        print(f"    Отношения: {np.round(ratios[:8], 4)}")
    sqrt3 = np.sqrt(3)
    print(f"    √3 = {sqrt3:.4f}")

# ============================================================
# 5. Визуализация
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

ax = axes[0]
ax.scatter(proj[:, 0], proj[:, 1], s=5, alpha=0.5, c='steelblue')
ax.set_title("E8 → гексагональная проекция")
ax.set_xlabel("v1")
ax.set_ylabel("v2")
ax.set_aspect('equal')
ax.grid(alpha=0.3)

ax = axes[1]
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_radii)))
for i, r in enumerate(unique_radii):
    mask = np.abs(radii - r) < 0.05
    ax.scatter(proj[mask, 0], proj[mask, 1], s=10, alpha=0.6,
               color=colors[i], label=f'r={r:.2f}')
ax.set_title(f"Кольца ({len(unique_radii)} радиусов)")
ax.set_xlabel("v1")
ax.set_ylabel("v2")
ax.set_aspect('equal')
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('e8_hex.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nГотово.")
