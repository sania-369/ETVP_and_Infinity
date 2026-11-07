import numpy as np
from itertools import combinations

# ============================================================
# Матрица Картана E8
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

# ============================================================
# Вектор Коксетера
# ============================================================
# В базисе фундаментальных весов вектор Коксетера = (1,1,...,1).
# Чтобы получить его в базисе простых корней, используем:
# C @ rho = 2 * rho? Нет.
# Вектор Коксетера rho удовлетворяет: <rho, alpha_i^vee> = 1 для всех i.
# В координатах простых корней: C @ rho = 2 * (1,1,...,1)? 
# Проверим: для E8, rho = sum фундаментальных весов.
# Матрица Картана C: C[i,j] = <alpha_i, alpha_j^vee>
# Вектор Коксетера в базисе простых корней: rho = sum_i (A^{-1})[i,j] * 2?
# Проще: rho = C^{-1} @ (2,2,...,2)

rho = np.linalg.solve(C, 2 * np.ones(8))
print("Вектор Коксетера rho (в базисе простых корней):")
print(np.round(rho, 4))

# Проверка: C @ rho должно быть (2,2,...,2)
check = C @ rho
print(f"\nC @ rho = {np.round(check, 4)} (должно быть 2,...,2)")

# ============================================================
# Плоскость Коксетера
# ============================================================
# Вектор Коксетера rho — ось. Плоскость Коксетера — перпендикулярна rho.
# Но нам нужна 2D-плоскость. В E8 плоскость Коксетера 2D.
# Строим базис в плоскости, перпендикулярной rho.

rho_norm = rho / np.linalg.norm(rho)

# Берём два вектора, перпендикулярных rho
def build_plane_basis(axis):
    # v1
    v1 = np.array([1, -1, 0, 0, 0, 0, 0, 0], dtype=float)
    v1 = v1 - np.dot(v1, axis) * axis
    v1 = v1 / np.linalg.norm(v1)
    # v2
    v2 = np.array([0, 0, 1, -1, 0, 0, 0, 0], dtype=float)
    v2 = v2 - np.dot(v2, axis) * axis - np.dot(v2, v1) * v1
    v2 = v2 / np.linalg.norm(v2)
    return v1, v2

v1, v2 = build_plane_basis(rho_norm)
print(f"\nv1 = {np.round(v1, 4)}")
print(f"v2 = {np.round(v2, 4)}")
print(f"v1·v2 = {np.dot(v1, v2):.6f}")
print(f"v1·rho_norm = {np.dot(v1, rho_norm):.6f} (должно быть 0)")
print(f"v2·rho_norm = {np.dot(v2, rho_norm):.6f} (должно быть 0)")

# ============================================================
# Корни E8
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
# Проекция на плоскость Коксетера
# ============================================================
proj = np.array([[np.dot(r, v1), np.dot(r, v2)] for r in roots])
print(f"Проекция: {proj.shape}")

# ============================================================
# Три аспекта
# ============================================================
print("\n" + "=" * 60)
print("ТРИ АСПЕКТА (плоскость Коксетера)")
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
    from scipy.spatial.distance import cdist
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
# Визуализация
# ============================================================
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Все корни
ax = axes[0]
ax.scatter(proj[:, 0], proj[:, 1], s=5, alpha=0.5, c='steelblue')
ax.set_title("E8 → плоскость Коксетера")
ax.set_xlabel("v1")
ax.set_ylabel("v2")
ax.set_aspect('equal')
ax.grid(alpha=0.3)

# Кольца
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
plt.savefig('e8_koxeter.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nГотово.")
