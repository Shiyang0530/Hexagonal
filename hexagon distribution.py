import math
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
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
def adjust_degrees_to_normal(points, edges, mean=3.0, std=1.0):
    num_nodes = len(points)
    adj = {i: set() for i in range(num_nodes)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    target = np.random.normal(loc=mean, scale=std, size=num_nodes)

    # Clamp to [0, 6]
    for i in range(num_nodes):
        target[i] = max(0, min(6, int(round(target[i]))))

    # Randomly delete edges
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

        if higher > lower:
            colors.append("red")
        elif lower > higher:
            colors.append("green")
        else:
            colors.append("yellow")

    return colors


# ---------- One simulation ----------
def run_simulation():
    points = generate_hex_points(500)
    edges = build_neighbors(points)
    edges = adjust_degrees_to_normal(points, edges, mean=3.0, std=1.0)
    colors = classify_colors(points, edges)

    return (
        colors.count("red"),
        colors.count("yellow"),
        colors.count("green")
    )


# ---------- Run 1000 simulations ----------
def main():
    NUM_SIM = 10000

    reds = []
    yellows = []
    greens = []

    print("Running 10000 simulations...")

    for _ in range(NUM_SIM):
        r, y, g = run_simulation()
        reds.append(r)
        yellows.append(y)
        greens.append(g)

    print("Done! Now plotting distribution...")

    # Plot distributions
    plt.figure(figsize=(10, 6))
    plt.hist(reds, bins=30, alpha=0.6, color='red', label='Red count')
    plt.hist(yellows, bins=30, alpha=0.6, color='yellow', label='Yellow count')
    plt.hist(greens, bins=30, alpha=0.6, color='green', label='Green count')

    plt.xlabel("Count per simulation")
    plt.ylabel("Frequency")
    plt.title("Distribution of Red / Yellow / Green Counts over 1000 Simulations")
    plt.legend()

    plt.tight_layout()
    plt.savefig("color_distribution.png", dpi=300)
    plt.close()

    print("Distribution plot saved as color_distribution.png")


if __name__ == "__main__":
    main()
