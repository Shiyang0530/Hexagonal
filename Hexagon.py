import math
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Use non-GUI backend
import matplotlib.pyplot as plt


# ---------- Hex grid logic ----------
def axial_to_cartesian(q, r, size=1.0):
    x = size * (3.0 / 2.0 * q)
    y = size * (math.sqrt(3) / 2.0 * q + math.sqrt(3) * r)
    return x, y


def generate_hex_points(num_points=500):
    points = []
    radius = 0
    while len(points) < num_points:
        points = []
        for q in range(-radius, radius + 1):
            r1 = max(-radius, -q - radius)
            r2 = min(radius, -q + radius)
            for r in range(r1, r2 + 1):
                points.append((q, r))
        radius += 1
    return points[:num_points]


def build_neighbors(points):
    directions = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]
    coord_to_index = {coord: i for i, coord in enumerate(points)}
    edges = []
    for i, (q, r) in enumerate(points):
        for dq, dr in directions:
            nb = (q + dq, r + dr)
            j = coord_to_index.get(nb)
            if j is not None and j > i:
                edges.append((i, j))
    return edges


# ---------- Degree adjustment ----------
def adjust_degrees_to_normal(points, edges, mean=3.5, std=1.0):
    num_nodes = len(points)
    adj = {i: set() for i in range(num_nodes)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    target = np.random.normal(loc=mean, scale=std, size=num_nodes)
    for i in range(num_nodes):
        target[i] = max(0, min(len(adj[i]), int(round(target[i]))))

    for i in range(num_nodes):
        while len(adj[i]) > target[i]:
            j = random.choice(list(adj[i]))
            adj[i].remove(j)
            adj[j].remove(i)

    new_edges = []
    for i in range(num_nodes):
        for j in adj[i]:
            if j > i:
                new_edges.append((i, j))
    return new_edges


# ---------- Color classification ----------
def classify_colors(points, edges):
    num_nodes = len(points)

    adj = {i: set() for i in range(num_nodes)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    degrees = {i: len(adj[i]) for i in range(num_nodes)}

    colors = []
    for i in range(num_nodes):
        deg_i = degrees[i]
        neighbors = adj[i]

        # Degree 0 → always red
        if deg_i == 0:
            colors.append("red")
            continue

        higher = sum(1 for j in neighbors if degrees[j] > deg_i)
        lower = sum(1 for j in neighbors if degrees[j] < deg_i)

        # Apply new rule
        if higher > lower:
            colors.append("red")
        elif lower > higher:
            colors.append("green")
        else:
            colors.append("yellow")

    return colors, degrees


# ---------- Main ----------
def main():
    print("Generating PNG...")

    num_points = 500
    axial_points = generate_hex_points(num_points)
    cart_points = [axial_to_cartesian(q, r) for (q, r) in axial_points]

    edges = build_neighbors(axial_points)
    edges = adjust_degrees_to_normal(axial_points, edges, mean=3.0, std=1.0)

    colors, degrees = classify_colors(axial_points, edges)

    # Count colors
    num_red = colors.count("red")
    num_yellow = colors.count("yellow")
    num_green = colors.count("green")

    print("Red points:", num_red)
    print("Yellow points:", num_yellow)
    print("Green points:", num_green)

    # Count degree distribution
    degree_counts = {}
    for d in degrees.values():
        degree_counts[d] = degree_counts.get(d, 0) + 1

    print("\nDegree distribution:")
    for d in sorted(degree_counts.keys()):
        print(f"Degree {d}: {degree_counts[d]} points")

    # Draw PNG
    fig, ax = plt.subplots(figsize=(8, 8))

    for i, j in edges:
        x1, y1 = cart_points[i]
        x2, y2 = cart_points[j]
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=0.5)

    for (x, y), c in zip(cart_points, colors):
        ax.scatter(x, y, s=20, color=c, zorder=3)

    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig("hex_grid.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nPNG saved!")


if __name__ == "__main__":
    main()


