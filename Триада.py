import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.spatial import distance_matrix

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
# 2. Проекция на плоскость Коксетера
# ============================================================
axis = np.ones(8) / np.sqrt(8)

def build_plane_basis(axis):
    v1 = np.array([1, -1, 0, 0, 0, 0, 0, 0], dtype=float)
    v1 = v1 - np.dot(v1, axis) * axis
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.array([0, 0, 1, -1, 0, 0, 0, 0], dtype=float)
    v2 = v2 - np.dot(v2, axis) * axis - np.dot(v2, v1) * v1
    v2 = v2 / np.linalg.norm(v2)
    return v1, v2

v1, v2 = build_plane_basis(axis)
proj = np.array([[np.dot(r, v1), np.dot(r, v2)] for r in roots])
print(f"Проекция: {proj.shape}")

# ============================================================
# 3. Визуализация
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# --- График 1: Все корни E8 в проекции ---
ax = axes[0, 0]
ax.scatter(proj[:, 0], proj[:, 1], s=5, alpha=0.5, c='steelblue')
ax.set_title("E8 → проекция на плоскость Коксетера\n240 корней")
ax.set_xlabel("v1")
ax.set_ylabel("v2")
ax.set_aspect('equal')
ax.grid(alpha=0.3)

# --- График 2: Кольца (масштаб, Φ) ---
ax = axes[0, 1]
radii = np.linalg.norm(proj, axis=1)
unique_radii = np.unique(np.round(radii, 2))
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_radii)))
for i, r in enumerate(unique_radii):
    mask = np.abs(radii - r) < 0.1
    ax.scatter(proj[mask, 0], proj[mask, 1], s=10, alpha=0.6,
               color=colors[i], label=f'r={r:.2f}')
ax.set_title(f"Масштаб: кольца в проекции\n{len(unique_radii)} уникальных радиусов")
ax.set_xlabel("v1")
ax.set_ylabel("v2")
ax.set_aspect('equal')
ax.legend(fontsize=7, loc='upper right')
ax.grid(alpha=0.3)

# --- График 3: Углы (цикл, π) ---
ax = axes[1, 0]
angles = np.arctan2(proj[:, 1], proj[:, 0])
unique_angles = np.unique(np.round(angles, 1))
for i, a in enumerate(unique_angles):
    mask = np.abs(angles - a) < 0.1
    ax.scatter(proj[mask, 0], proj[mask, 1], s=10, alpha=0.6,
               label=f'{a:.1f} рад')
ax.set_title(f"Цикл: углы между корнями\n{len(unique_angles)} уникальных углов")
ax.set_xlabel("v1")
ax.set_ylabel("v2")
ax.set_aspect('equal')
ax.legend(fontsize=6, loc='upper right', ncol=2)
ax.grid(alpha=0.3)

# --- График 4: Упаковка (√3) ---
ax = axes[1, 1]
unique_proj = np.unique(np.round(proj, 2), axis=0)
dm = distance_matrix(unique_proj, unique_proj)
dm[dm == 0] = np.inf
min_dists = dm.min(axis=1)
unique_dists = np.unique(np.round(min_dists, 2))
for i, d in enumerate(unique_dists):
    mask = np.abs(min_dists - d) < 0.1
    ax.scatter(unique_proj[mask, 0], unique_proj[mask, 1], s=10, alpha=0.6,
               label=f'd={d:.2f}')
ax.set_title(f"Упаковка: расстояния до соседей\n{len(unique_dists)} уникальных расстояний")
ax.set_xlabel("v1")
ax.set_ylabel("v2")
ax.set_aspect('equal')
ax.legend(fontsize=6, loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('e8_triada.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 4. Числа
# ============================================================
print("\n" + "=" * 60)
print("ЧИСЛА:")
print("=" * 60)
print(f"\n[1] МАСШТАБ (Φ)")
print(f"    Радиусы: {unique_radii[:10]}")
if len(unique_radii) >= 2:
    ratios = unique_radii[1:] / unique_radii[:-1]
    print(f"    Отношения: {np.round(ratios[:5], 4)}")
    print(f"    Φ = {(1+np.sqrt(5))/2:.4f}")

print(f"\n[2] ЦИКЛ (π)")
print(f"    Углы (рад): {unique_angles[:10]}")
for denom in [3, 4, 5]:
    target = np.pi / denom
    close = np.any(np.abs(unique_angles - target) < 0.2)
    print(f"    π/{denom} = {target:.4f}: {'ЕСТЬ' if close else 'НЕТ'}")

print(f"\n[3] УПАКОВКА (√3)")
print(f"    Расстояния: {unique_dists[:10]}")
if len(unique_dists) >= 2:
    ratios = unique_dists[1:] / unique_dists[:-1]
    print(f"    Отношения: {np.round(ratios[:5], 4)}")
    print(f"    √3 = {np.sqrt(3):.4f}")
