import numpy as np

def build_adjacency(dem, threshold=2.5):
    rows, cols = dem.shape
    edges = []
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            base = dem[r, c]
            for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
                nb = dem[r + dr, c + dc]
                if abs(base - nb) < threshold:
                    edges.append((r * cols + c, (r + dr) * cols + c + dc))
    return np.array(edges).T
