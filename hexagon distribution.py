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


def generate_hex_points_with_radius(radius):
    points = []
    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            points.append((q, r))
    return points


def build_geometric_neighbors(points):
    directions = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    coord_to_index = {coord: i for i, coord in enumerate(points)}
    neighbors = {i: [] for i in range(len(points))}
    for i, (q, r) in enumerate(points):
        for dq, dr in directions:
            nb = (q + dq, r + dr)
            j = coord_to_index.get(nb)
            if j is not None:
                neighbors[i].append(j)
    return neighbors


def find_outer_two_rings(points, radius):
    gray_ring = set()
    for i, (q, r) in enumerate(points):
        rv = max(abs(q), abs(r), abs(q + r))
        if rv == radius or rv == radius - 1:
            gray_ring.add(i)
    return gray_ring


# ---------- Target degree generation on inner nodes ----------
def generate_inner_targets(inner_indices, mean=3.0, std=1.0, dmin=0, dmax=6):
    num_inner = len(inner_indices)
    target = np.random.normal(loc=mean, scale=std, size=num_inner)

    # Clamp and round
    for k in range(num_inner):
        target[k] = max(dmin, min(dmax, int(round(target[k]))))

    target = target.astype(int)

    # Enforce exact mean on inner nodes: sum = mean * num_inner
    desired_total = int(round(mean * num_inner))
    current_total = int(target.sum())

    while current_total != desired_total:
        if current_total > desired_total:
            candidates = np.where(target > dmin)[0]
            if len(candidates) == 0:
                break
            k = int(random.choice(candidates))
            target[k] -= 1
            current_total -= 1
        else:
            candidates = np.where(target < dmax)[0]
            if len(candidates) == 0:
                break
            k = int(random.choice(candidates))
            target[k] += 1
            current_total += 1

    # Map back to full-size array (only inner indices get targets, others = 0)
    full_target = np.zeros(len(inner_indices), dtype=int)  # temporary
    # We will return a dict index -> target_degree for inner nodes
    inner_targets = {}
    for k, idx in enumerate(inner_indices):
        inner_targets[idx] = target[k]

    return inner_targets


# ---------- Iterative degree adjustment ----------
def build_graph_with_inner_targets(points, geom_neighbors, gray_ring,
                                   inner_targets,
                                   num_passes=30):
    num_nodes = len(points)

    # Start from the full geometric graph
    adj = {i: set() for i in range(num_nodes)}
    for i in range(num_nodes):
        for j in geom_neighbors[i]:
            if j > i:
                adj[i].add(j)
                adj[j].add(i)

    # Iteratively adjust degrees for inner nodes only
    inner_nodes = list(inner_targets.keys())

    for _ in range(num_passes):
        random.shuffle(inner_nodes)
        for i in inner_nodes:
            target_deg = inner_targets[i]
            current_deg = len(adj[i])

            if current_deg > target_deg:
                # Remove one random edge from i
                if adj[i]:
                    j = random.choice(list(adj[i]))
                    adj[i].remove(j)
                    adj[j].remove(i)

            elif current_deg < target_deg:
                # Try to add an edge from i
                # Candidates: geometric neighbors not yet connected
                candidates = [j for j in geom_neighbors[i] if j not in adj[i]]
                if not candidates:
                    continue

                # Prefer gray neighbors (they are slack)
                gray_candidates = [j for j in candidates if j in gray_ring]
                inner_candidates = [j for j in candidates if j not in gray_ring]

                if gray_candidates:
                    j = random.choice(gray_candidates)
                    adj[i].add(j)
                    adj[j].add(i)
                else:
                    # Inner neighbor: we try to avoid pushing it far beyond its own target
                    j = random.choice(inner_candidates)
                    adj[i].add(j)
                    adj[j].add(i)

    # Build final edge list
    edges = []
    for i in range(num_nodes):
        for j in adj[i]:
            if j > i:
                edges.append((i, j))

    return edges, adj


# ---------- Color classification ----------
def classify_colors(points, edges, gray_ring):
    num_nodes = len(points)

    adj = {i: set() for i in range(num_nodes)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    degrees = {i: len(adj[i]) for i in range(num_nodes)}
    colors = []

    for i in range(num_nodes):
        if i in gray_ring:
            colors.append("gray")
            continue

        deg_i = degrees[i]
        neighbors = adj[i]

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

    return colors, degrees


# ---------- Single simulation ----------
def run_single_simulation_with_stats(radius=15):
    points = generate_hex_points_with_radius(radius)
    geom_neighbors = build_geometric_neighbors(points)
    gray_ring = find_outer_two_rings(points, radius)

    num_nodes = len(points)
    inner_indices = [i for i in range(num_nodes) if i not in gray_ring]

    # Generate targets only on inner nodes, with exact mean 3
    inner_targets = generate_inner_targets(inner_indices, mean=3.0, std=1.0, dmin=0, dmax=6)

    # Build graph with iterative adjustment
    edges, adj = build_graph_with_inner_targets(points, geom_neighbors, gray_ring, inner_targets)

    # Color classification
    colors, degrees = classify_colors(points, edges, gray_ring)

    # Stats on inner (non-gray) nodes only
    inner_degrees = [degrees[i] for i in inner_indices]
    red = colors.count("red")
    yellow = colors.count("yellow")
    green = colors.count("green")

    degree_counts = {d: 0 for d in range(7)}
    for d in inner_degrees:
        if 0 <= d <= 6:
            degree_counts[d] += 1

    avg_degree_inner = float(np.mean(inner_degrees)) if inner_degrees else 0.0

    print("\n--- Single Simulation Stats (inner nodes only) ---")
    print(f"Total points: {num_nodes} (radius = {radius})")
    print(f"Inner nodes: {len(inner_indices)}")
    print(f"Target mean degree (inner): 3.0 (exact on targets)")
    print(f"Realized average degree (inner): {avg_degree_inner:.3f}")
    print("Red:", red)
    print("Yellow:", yellow)
    print("Green:", green)
    print("\nDegree distribution (inner, non-gray):")
    for d in range(7):
        print(f"Degree {d}: {degree_counts[d]}")

    # Draw PNG
    cart_points = [axial_to_cartesian(q, r) for (q, r) in points]
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
    plt.savefig("single_simulation.png", dpi=300)
    plt.close()
    print("\nSaved single_simulation.png\n")


# ---------- 1000 simulations ----------
def run_simulation(radius=15):
    points = generate_hex_points_with_radius(radius)
    geom_neighbors = build_geometric_neighbors(points)
    gray_ring = find_outer_two_rings(points, radius)

    num_nodes = len(points)
    inner_indices = [i for i in range(num_nodes) if i not in gray_ring]

    inner_targets = generate_inner_targets(inner_indices, mean=3.0, std=1.0, dmin=0, dmax=6)
    edges, _ = build_graph_with_inner_targets(points, geom_neighbors, gray_ring, inner_targets)
    colors, _ = classify_colors(points, edges, gray_ring)

    return colors.count("red"), colors.count("yellow"), colors.count("green")


def run_1000_simulations(radius=15):
    reds, yellows, greens = [], [], []
    print("Running 1000 simulations...")

    for _ in range(1000):
        r, y, g = run_simulation(radius=radius)
        reds.append(r)
        yellows.append(y)
        greens.append(g)

    print("Plotting distribution...")

    plt.figure(figsize=(10, 6))
    plt.hist(reds, bins=30, alpha=0.6, color='red', label='Red count')
    plt.hist(yellows, bins=30, alpha=0.6, color='yellow', label='Yellow count')
    plt.hist(greens, bins=30, alpha=0.6, color='green', label='Green count')

    plt.xlabel("Count per simulation")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Red / Yellow / Green Counts over 1000 Simulations (radius={radius})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("color_distribution_1000.png", dpi=300)
    plt.close()
    print("Saved color_distribution_1000.png")


def main():
    radius = 15
    run_single_simulation_with_stats(radius=radius)
    run_1000_simulations(radius=radius)


if __name__ == "__main__":
    main()

