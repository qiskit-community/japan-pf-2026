import random
from matplotlib import pyplot as plt, patches as patches

def plot_pareto():
    plot_pareto_hv(hv=False)


def plot_pareto_hv(points=None, ref=None, hv=False, second=False):
    if points is None:
        points = [(0.05, 1.0), (0.35, 0.85), (0.5, 0.5), (0.85, 0.35), (1.0, 0.05)]
        points_cv = [(0.05, 1.0), (0.35, 0.85), (0.85, 0.35), (1.0, 0.05)]
    if ref is None:
        ref = (0, 0)

    fig, ax = plt.subplots(figsize=(5,5))

    ax.plot([x for x, _ in points], [y for _, y in points], color='lightblue', ls='--')

    #ax.plot([x for x, _ in points_cv], [y for _, y in points_cv], color='black', ls=':')


    for point in points:
        ax.plot(point[0], point[1], 'o', color='lightblue')  # Mark the top-right corner

    if hv:
        for point in points:
            width = point[0] - ref[0]
            height = point[1] - ref[1]

            rect = patches.Rectangle(ref, width, height, linewidth=1, facecolor='lightgreen')
            ax.add_patch(rect)
        if second:
            points = [(0.18,0.62), (0.37, 0.45), (0.45, 0.35), (0.62, 0.18)]
            for point in points:
                width = point[0] - ref[0]
                height = point[1] - ref[1]

                rect = patches.Rectangle(ref, width, height, linewidth=3, ls='--', facecolor='orange')
                ax.add_patch(rect)
                for point in points:
                    ax.plot(point[0], point[1], 'x', color='orange')  # Mark the top-right corner

    if not hv:
        random_points=[]
        num_random_points=120
        for _ in range(num_random_points):
            rect_point = random.choice(points)
            x = random.uniform(ref[0], rect_point[0])
            y = random.uniform(ref[1], rect_point[1])
            random_points.append((x, y))

        # Plot random points
        for x, y in random_points:
            ax.plot(x, y, 'o', color='lightblue')  # Black dots

    plt.ylabel('$f_2$')
    plt.xlabel('$f_1$')


def plot_graphs(moo_graphs):
    import networkx as nx
    import numpy as np  
    import copy

    fig, axes = plt.subplots(1, len(moo_graphs), figsize=(8 * len(moo_graphs), 6))
    for ax, G in zip(axes, moo_graphs):
        g = copy.deepcopy(G)
        for u, v in g.edges():
            g[u][v]['weight'] = 1
        pos = nx.kamada_kawai_layout(g, weight='weight')
        nx.draw(g, pos=pos, ax=ax, with_labels=True, node_size=150, font_size=9)#, node_size=30, font_size=6, width=10)

        weights = nx.get_edge_attributes(G, 'weight')
        for k, w in weights.items():
            weights[k] = np.round(w, 2)
        if len(g.nodes) < 80:
            nx.draw_networkx_edge_labels(g, ax=ax, pos=pos, edge_labels=weights, bbox=None, font_size=7)
        else:
            nx.draw_networkx_edge_labels(g, ax=ax, pos=pos, edge_labels=weights, bbox=None, font_size=3)
        ax.set_title(f'Graph $g_{moo_graphs.index(G)}$')
    plt.tight_layout()

def plot_hypervolumes_from_batch(hvs):
    import pandas as pd

    df = pd.DataFrame(hvs, columns=["Hypervolume"]).describe()    
    plt.hist(hvs)