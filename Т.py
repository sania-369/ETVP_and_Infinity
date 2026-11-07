import numpy as np
from itertools import combinations

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
print(f"Корней: {len(roots)}")

phi = (1 + np.sqrt(5)) / 2
print(f"Φ = {phi:.6f}")

# Перебор всех пар корней как базисов проекции
best_phi = None
best_diff = float('inf')
best_pair = None

for i in range(len(roots)):
    for j in range(i+1, len(roots)):
        v1 = roots[i] / np.linalg.norm(roots[i])
        v2 = roots[j] - np.dot(roots[j], v1) * v1
        if np.linalg.norm(v2) < 1e-6:
            continue
        v2 = v2 / np.linalg.norm(v2)
        
        proj = np.array([[np.dot(r, v1), np.dot(r, v2)] for r in roots])
        radii = np.linalg.norm(proj, axis=1)
        unique_radii = np.unique(np.round(radii, 4))
        unique_radii = unique_radii[unique_radii > 0.01]
        
        if len(unique_radii) >= 2:
            ratios = unique_radii[1:] / unique_radii[:-1]
            for ratio in ratios:
                diff = abs(ratio - phi)
                if diff < best_diff:
                    best_diff = diff
                    best_phi = ratio
                    best_pair = (i, j)

print(f"\nЛучшее совпадение с Φ:")
print(f"  Пара корней: {best_pair}")
print(f"  Отношение: {best_phi:.6f}")
print(f"  Φ = {phi:.6f}")
print(f"  Разница: {best_diff:.6f}")
print(f"  Совпадение: {'ДА' if best_diff < 0.01 else 'НЕТ'}")
