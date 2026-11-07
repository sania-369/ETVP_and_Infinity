import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.spatial.distance import cdist

# ============================================================
# 1. Матрица Картана E8
# ============================================================
def cartan_e8():
    C = np.zeros((8, 8))
    for i in range(8):
        C[i, i] = 2
    edges = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (2,7)]
    for i, j in edges:
        C[i, j] = -1
        C[j, i] = -1
    return C

C = cartan_e8()
print("Матрица Картана E8:")
print(C.astype(int))

# ============================================================
# 2. Собственные векторы
# ============================================================
eigvals, eigvecs = np.linalg.eigh(C)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

print("\nСобственные значения (по убыванию):")
print(np.round(eigvals, 4))

# Два старших собственных вектора = плоскость
v1 = eigvecs[:, 0]
v2 = eigvecs[:, 1]
print(f"\nv1 = {np.round(v1, 4)}")
print(f"v2 = {np.round(v2, 4)}")
print(f"v1·v2 = {np.dot(v1, v2):.6f}")

# ============================================================
# 3. Корни E8
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
print(f"\nКорней E8: {len(roots)}")

# ============================================================
# 4. Проекция на плоскость (v1, v2)
# ============================================================
proj = np.array([[np.dot(r, v1), np.dot(r, v2)] for r in roots])
print(f"Проекция: {proj.shape}")

# ============================================================
# 5. Три аспекта
# ============================================================
print("\n" + "=" * 60)
print("ТРИ АСПЕКТА")
print("=" * 60)

# --- [1] Масштаб: радиусы ---
radii = np.linalg.norm(proj, axis=1)
unique_radii = np.unique(np.round(radii, 3))
unique_radii = unique_radii[unique_radii > 0.01]  # убираем 0
print(f"\n[1] МАСШТАБ (Φ)")
print(f"    Радиусы: {unique_radii[:10]}")
if len(unique_radii) >= 2:
    ratios = unique_radii[1:] / unique_radii[:-1]
    print(f"    Отношения: {np.round(ratios[:8], 4)}")
    phi = (1 + np.sqrt(5)) / 2
    print(f"    Φ = {phi:.4f}")
    closest = ratios[np.argmin(np.abs(ratios - phi))]
    print(f"    Ближайшее к Φ: {closest:.4f} (разница {abs(closest - phi):.4f})")

# --- [2] Цикл: углы ---
angles = np.arctan2(proj[:, 1], proj[:, 0])
unique_angles = np.unique(np.round(angles, 3))
print(f"\n[2] ЦИКЛ (π)")
print(f"    Углы (рад): {unique_angles[:15]}")
print(f"    π/3 = {np.pi/3:.4f}, π/4 = {np.pi/4:.4f}, π/5 = {np.pi/5:.4f}")
for denom in [3, 4, 5, 6]:
    target = np.pi / denom
    close = np.any(np.abs(unique_angles - target) < 0.1)
    print(f"    π/{denom} = {target:.4f}: {'ЕСТЬ' if close else 'НЕТ'}")

# --- [3] Упаковка: расстояния ---
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
# 6. Визуализация
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# График 1: все корни
ax = axes[0, 0]
ax.scatter(proj[:, 0], proj[:, 1], s=5, alpha=0.5, c='steelblue')
ax.set_title("E8 → проекция на плоскость старших мод")
ax.set_xlabel("v1")
ax.set_ylabel("v2")
ax.set_aspect('equal')
ax.grid(alpha=0.3)

# График 2: кольца
ax = axes[0, 1]
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_radii)))
for i, r in enumerate(unique_radii):
    mask = np.abs(radii - r) < 0.05
    ax.scatter(proj[mask, 0], proj[mask, 1], s=10, alpha=0.6,
               color=colors[i], label=f'r={r:.2f}')
ax.set_title(f"Масштаб: кольца ({len(unique_radii)} радиусов)")
ax.set_xlabel("v1")
ax.set_ylabel("v2")
ax.set_aspect('equal')
ax.legend(fontsize=7, loc='upper right')
ax.grid(alpha=0.3)

# График 3: углы
ax = axes[1, 0]
unique_angles_pos = unique_angles[unique_angles >= 0]
colors = plt.cm.plasma(np.linspace(0, 1, len(unique_angles_pos)))
for i, a in enumerate(unique_angles_pos[:20]):
    mask = np.abs(angles - a) < 0.05
    ax.scatter(proj[mask, 0], proj[mask, 1], s=10, alpha=0.6,
               color=colors[i])
ax.set_title(f"Цикл: углы ({len(unique_angles)} уникальных)")
ax.set_xlabel("v1")
ax.set_ylabel("v2")
ax.set_aspect('equal')
ax.grid(alpha=0.3)

# График 4: упаковка
ax = axes[1, 1]
colors = plt.cm.cool(np.linspace(0, 1, len(unique_dists)))
for i, d in enumerate(unique_dists):
    mask = np.abs(min_dists - d) < 0.05
    ax.scatter(unique_proj[mask, 0], unique_proj[mask, 1], s=10, alpha=0.6,
               color=colors[i], label=f'd={d:.2f}')
ax.set_title(f"Упаковка: расстояния ({len(unique_dists)} значений)")
ax.set_xlabel("v1")
ax.set_ylabel("v2")
ax.set_aspect('equal')
ax.legend(fontsize=7, loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('e8_triada_v2.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("Готово. Смотри числа и картинку.")
print("=" * 60)
