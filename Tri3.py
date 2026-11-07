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
# 2. Пятиугольная проекция
# ============================================================
# Ищем два корня E8, угол между которыми = 72° (2π/5).
# В E8 углы между корнями кратны 60°, 45°, 36° (π/3, π/4, π/5).
# 36° = π/5 — это пятиугольный угол.
# Но 72° = 2π/5 — это угол между несмежными вершинами пентагона.
# В E8 угол 36° есть (π/5). Используем два корня с углом 36°.

def angle_between(a, b):
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

# Ищем пару корней с углом 36° (π/5)
found = None
for i in range(len(roots)):
    for j in range(i+1, len(roots)):
        ang = angle_between(roots[i], roots[j])
        if abs(ang - 36) < 0.5:
            found = (i, j)
            break
    if found:
        break

if found:
    i, j = found
    r1, r2 = roots[i], roots[j]
    print(f"\nНайдена пара с углом {angle_between(r1, r2):.2f}° (π/5 = 36°)")
    print(f"r1 = {np.round(r1, 4)}")
    print(f"r2 = {np.round(r2, 4)}")
    
    # Ортогонализуем
    v1 = r1 / np.linalg.norm(r1)
    v2 = r2 - np.dot(r2, v1) * v1
    v2 = v2 / np.linalg.norm(v2)
    
    proj = np.array([[np.dot(r, v1), np.dot(r, v2)] for r in roots])
    print(f"Проекция: {proj.shape}")
    
    # Три аспекта
    print("\n" + "=" * 60)
    print("ТРИ АСПЕКТА (пятиугольная проекция, π/5)")
    print("=" * 60)
    
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
    
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    unique_angles = np.unique(np.round(angles, 3))
    print(f"\n[2] ЦИКЛ (π)")
    print(f"    Углы (рад): {unique_angles[:15]}")
    for denom in [3, 4, 5, 6]:
        target = np.pi / denom
        close = np.any(np.abs(unique_angles - target) < 0.05)
        print(f"    π/{denom} = {target:.4f}: {'ЕСТЬ' if close else 'НЕТ'}")
    
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
    
    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    ax = axes[0]
    ax.scatter(proj[:, 0], proj[:, 1], s=5, alpha=0.5, c='steelblue')
    ax.set_title("E8 → пятиугольная проекция (π/5)")
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
    plt.savefig('e8_pentagon.png', dpi=150, bbox_inches='tight')
    plt.show()

print("\nГотово.")
