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


# ---------- NEW: Only outer ring is gray; second ring is controlled ----------
def find_rings(points, radius):
    outer_ring = set()
    second_ring = set()

    for i, (q, r) in enumerate(points):
        rv = max(abs(q), abs(r), abs(q + r))
        if rv == radius:
            outer_ring.add(i)
        elif rv == radius - 1:
            second_ring.add(i)

    return outer_ring, second_ring


# ---------- Target degree generation ----------
def generate_targets(indices, mean=3.0, std=1.0, dmin=0, dmax=6):
    n = len(indices)
    target = np.random.normal(loc=mean, scale=std, size=n)

    # Clamp and round
    for k in range(n):
        target[k] = max(dmin, min(dmax, int(round(target[k]))))

    target = target.astype(int)

    # Enforce exact mean
    desired_total = int(round(mean * n))
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

    # Map back
    targets = {}
    for k, idx in enumerate(indices):
        targets[idx] = target[k]

    return targets


# ---------- Iterative degree adjustment ----------
def build_graph(points, geom_neighbors, outer_ring, controlled_targets, num_passes=30):
    num_nodes = len(points)

    # Start from geometric graph
    adj = {i: set() for i in range(num_nodes)}
    for i in range(num_nodes):
        for j in geom_neighbors[i]:
            if j > i:
                adj[i].add(j)
                adj[j].add(i)

    controlled_nodes = list(controlled_targets.keys())

    for _ in range(num_passes):
        random.shuffle(controlled_nodes)
        for i in controlled_nodes:
            target_deg = controlled_targets[i]
            current_deg = len(adj[i])

            if current_deg > target_deg:
                if adj[i]:
                    j = random.choice(list(adj[i]))
                    adj[i].remove(j)
                    adj[j].remove(i)

            elif current_deg < target_deg:
                candidates = [j for j in geom_neighbors[i] if j not in adj[i]]
                if not candidates:
                    continue

                # Prefer outer ring slack
                outer_candidates = [j for j in candidates if j in outer_ring]
                inner_candidates = [j for j in candidates if j not in outer_ring]

                if outer_candidates:
                    j = random.choice(outer_candidates)
                else:
                    j = random.choice(inner_candidates)

                adj[i].add(j)
                adj[j].add(i)

    edges = []
    for i in range(num_nodes):
        for j in adj[i]:
            if j > i:
                edges.append((i, j))

    return edges, adj


# ---------- Color classification ----------
def classify_colors(points, edges, outer_ring, second_ring):
    num_nodes = len(points)

    adj = {i: set() for i in range(num_nodes)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    degrees = {i: len(adj[i]) for i in range(num_nodes)}
    colors = []

    for i in range(num_nodes):
        if i in outer_ring or i in second_ring:
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
    outer_ring, second_ring = find_rings(points, radius)

    num_nodes = len(points)

    # Controlled nodes = inner + second ring
    controlled_indices = [i for i in range(num_nodes) if i not in outer_ring]

    # Generate targets for controlled nodes
    controlled_targets = generate_targets(controlled_indices, mean=3.0, std=1.0)

    edges, adj = build_graph(points, geom_neighbors, outer_ring, controlled_targets)

    colors, degrees = classify_colors(points, edges, outer_ring, second_ring)

    # Inner nodes = controlled except second ring
    inner_indices = [i for i in controlled_indices if i not in second_ring]
    inner_degrees = [degrees[i] for i in inner_indices]

    # ---------- Count red/yellow/green ----------
    red = colors.count("red")
    yellow = colors.count("yellow")
    green = colors.count("green")

    print("\n--- Single Simulation Stats ---")
    print(f"Total nodes: {num_nodes}")
    print(f"Inner nodes: {len(inner_indices)}")
    print(f"Second ring nodes: {len(second_ring)}")

    print("\n--- Red / Yellow / Green Counts ---")
    print(f"Red: {red}")
    print(f"Yellow: {yellow}")
    print(f"Green: {green}")

    # ---------- Inner degree distribution ----------
    print("\n--- Inner Degree Distribution ---")
    print(f"Average degree (inner): {np.mean(inner_degrees):.3f}")

    inner_counts = {d: 0 for d in range(7)}
    for d in inner_degrees:
        inner_counts[d] += 1
    for d in range(7):
        print(f"Degree {d}: {inner_counts[d]}")

    # ---------- Second ring degree distribution ----------
    second_degrees = [degrees[i] for i in second_ring]
    second_counts = {d: 0 for d in range(7)}
    for d in second_degrees:
        second_counts[d] += 1

    print("\n--- Second Ring Degree Distribution (受控) ---")
    print(f"Average degree: {np.mean(second_degrees):.3f}")
    for d in range(7):
        print(f"Degree {d}: {second_counts[d]}")

    # ---------- Draw PNG ----------
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
    outer_ring, second_ring = find_rings(points, radius)

    num_nodes = len(points)
    controlled_indices = [i for i in range(num_nodes) if i not in outer_ring]

    controlled_targets = generate_targets(controlled_indices, mean=3.0, std=1.0)
    edges, _ = build_graph(points, geom_neighbors, outer_ring, controlled_targets)
    colors, degrees = classify_colors(points, edges, outer_ring, second_ring)

    second_degrees = [degrees[i] for i in second_ring]
    avg_second = float(np.mean(second_degrees)) if second_degrees else 0.0

    return colors.count("red"), colors.count("yellow"), colors.count("green"), avg_second


def run_1000_simulations(radius=15):
    reds, yellows, greens = [], [], []
    second_ring_avgs = []

    print("Running 1000 simulations...")

    for _ in range(1000):
        r, y, g, sec_avg = run_simulation(radius=radius)
        reds.append(r)
        yellows.append(y)
        greens.append(g)
        second_ring_avgs.append(sec_avg)

    # Plot color distribution
    plt.figure(figsize=(10, 6))
    plt.hist(reds, bins=30, alpha=0.6, color='red', label='Red count')
    plt.hist(yellows, bins=30, alpha=0.6, color='yellow', label='Yellow count')
    plt.hist(greens, bins=30, alpha=0.6, color='green', label='Green count')
    plt.legend()
    plt.tight_layout()
    plt.savefig("color_distribution_1000.png", dpi=300)
    plt.close()

    # Plot second ring average degree distribution
    plt.figure(figsize=(10, 6))
    plt.hist(second_ring_avgs, bins=30, alpha=0.7, color='blue')
    plt.tight_layout()
    plt.savefig("second_ring_degree_distribution_1000.png", dpi=300)
    plt.close()

    print("Saved second_ring_degree_distribution_1000.png")


def main():
    radius = 15
    run_single_simulation_with_stats(radius=radius)
    run_1000_simulations(radius=radius)


if __name__ == "__main__":
    main()
