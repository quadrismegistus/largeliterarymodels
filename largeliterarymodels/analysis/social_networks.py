"""Social network analysis utilities.

Load SocialNetworkTask output, build networkx graphs, compute metrics,
extract character trajectories, and plot networks.

Usage:
    from largeliterarymodels.analysis.social_networks import (
        load_result, build_graph, build_dialogue_graph,
        character_trajectories, network_metrics, plot_network,
    )

    result = load_result('data/social_network_...json')
    G = build_graph(result)
    nx.degree_centrality(G)
"""

import json
from collections import Counter, defaultdict
from typing import Optional, Union

import pandas as pd

try:
    import networkx as nx
except ImportError:
    nx = None


def load_result(source: Union[str, dict]) -> dict:
    """Load a SocialNetworkTask result from a JSON path or dict."""
    if isinstance(source, str):
        with open(source) as f:
            return json.load(f)
    return source


def _char_lookup(result: dict) -> dict:
    """Build a character ID -> character dict lookup."""
    return {c['id']: c for c in result.get('characters', [])}


def build_graph(
    result: Union[str, dict],
    edge_types: Optional[list[str]] = None,
    exclude_types: Optional[list[str]] = None,
) -> 'nx.Graph':
    """Build a networkx Graph from social relations.

    Args:
        result: SocialNetworkTask output (path or dict).
        edge_types: If set, only include these relation types.
        exclude_types: Relation types to exclude (default: ['same_as']).

    Returns:
        nx.Graph with character attributes as node data and
        relation type/detail/passage as edge data.
    """
    if nx is None:
        raise ImportError("networkx is required: pip install networkx")

    result = load_result(result)
    chars = _char_lookup(result)
    exclude = set(exclude_types or ['same_as'])

    G = nx.Graph()

    for c in result.get('characters', []):
        G.add_node(c['id'], name=c.get('name', '?'), gender=c.get('gender', '?'),
                   social_class=c.get('class', '?'),
                   aliases=c.get('aliases', []))

    for r in result.get('relations', []):
        rtype = r.get('type', '')
        if rtype in exclude:
            continue
        if edge_types and rtype not in edge_types:
            continue
        a, b = r.get('a', ''), r.get('b', '')
        if not a or not b or a not in G or b not in G:
            continue
        if G.has_edge(a, b):
            G[a][b]['relations'].append(r)
            G[a][b]['weight'] += 1
        else:
            G.add_edge(a, b, relations=[r], weight=1,
                       type=rtype, passage=r.get('passage', ''))

    return G


def build_directed_graph(
    result: Union[str, dict],
    edge_types: Optional[list[str]] = None,
    exclude_types: Optional[list[str]] = None,
) -> 'nx.DiGraph':
    """Build a directed networkx graph from social relations."""
    if nx is None:
        raise ImportError("networkx is required: pip install networkx")

    result = load_result(result)
    exclude = set(exclude_types or ['same_as'])

    G = nx.DiGraph()

    for c in result.get('characters', []):
        G.add_node(c['id'], name=c.get('name', '?'), gender=c.get('gender', '?'),
                   social_class=c.get('class', '?'))

    for r in result.get('relations', []):
        rtype = r.get('type', '')
        if rtype in exclude:
            continue
        if edge_types and rtype not in edge_types:
            continue
        a, b = r.get('a', ''), r.get('b', '')
        if not a or not b or a not in G or b not in G:
            continue
        if G.has_edge(a, b):
            G[a][b]['weight'] += 1
        else:
            G.add_edge(a, b, weight=1, type=rtype)

    return G


def build_dialogue_graph(
    result: Union[str, dict],
) -> 'nx.DiGraph':
    """Build a directed graph weighted by dialogue frequency.

    Edge weight = number of times speaker addressed addressee.
    """
    if nx is None:
        raise ImportError("networkx is required: pip install networkx")

    result = load_result(result)
    G = nx.DiGraph()

    for c in result.get('characters', []):
        G.add_node(c['id'], name=c.get('name', '?'))

    for d in result.get('dialogue', []):
        s, a = d.get('speaker', ''), d.get('addressee', '')
        if not s or not a or s not in G or a not in G:
            continue
        if G.has_edge(s, a):
            G[s][a]['weight'] += 1
            G[s][a]['gists'].append(d.get('gist', ''))
        else:
            G.add_edge(s, a, weight=1, gists=[d.get('gist', '')])

    return G


def build_event_graph(
    result: Union[str, dict],
) -> 'nx.DiGraph':
    """Build a directed graph from events (who did what to whom)."""
    if nx is None:
        raise ImportError("networkx is required: pip install networkx")

    result = load_result(result)
    G = nx.DiGraph()

    for c in result.get('characters', []):
        G.add_node(c['id'], name=c.get('name', '?'))

    for e in result.get('events', []):
        who, whom = e.get('who', ''), e.get('whom', '')
        if not who or not whom or who not in G or whom not in G:
            continue
        if G.has_edge(who, whom):
            G[who][whom]['weight'] += 1
            G[who][whom]['events'].append(e.get('what', ''))
        else:
            G.add_edge(who, whom, weight=1, events=[e.get('what', '')])

    return G


def character_trajectories(result: Union[str, dict]) -> dict[str, list[dict]]:
    """Extract location sequences per character from events.

    Returns:
        dict mapping character ID to list of
        {'passage': 'P042', 'where': 'Virginia', 'what': 'arrived'}
        sorted by passage number.
    """
    result = load_result(result)
    trajs = defaultdict(list)

    for e in result.get('events', []):
        where = e.get('where')
        if not where:
            continue
        who = e.get('who', '')
        if not who:
            continue
        trajs[who].append({
            'passage': e.get('passage', ''),
            'where': where,
            'what': e.get('what', ''),
        })

    for cid in trajs:
        trajs[cid].sort(key=lambda x: x.get('passage', ''))

    return dict(trajs)


def location_summary(result: Union[str, dict]) -> pd.DataFrame:
    """Summarize locations: how many events at each, which characters appear."""
    result = load_result(result)
    rows = []
    for e in result.get('events', []):
        where = e.get('where')
        if where:
            rows.append({'location': where, 'who': e.get('who', ''),
                         'passage': e.get('passage', ''), 'what': e.get('what', '')})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    summary = df.groupby('location').agg(
        n_events=('what', 'count'),
        n_characters=('who', 'nunique'),
        characters=('who', lambda x: list(x.unique())),
        first_passage=('passage', 'min'),
        last_passage=('passage', 'max'),
    ).sort_values('n_events', ascending=False)
    return summary


def network_metrics(G: 'nx.Graph') -> dict:
    """Compute basic network metrics."""
    if nx is None:
        raise ImportError("networkx is required")

    metrics = {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'density': nx.density(G),
    }

    if G.number_of_nodes() > 0:
        deg = dict(G.degree())
        metrics['mean_degree'] = sum(deg.values()) / len(deg)
        metrics['max_degree_node'] = max(deg, key=deg.get)
        metrics['max_degree'] = deg[metrics['max_degree_node']]

    if G.number_of_nodes() > 1 and G.number_of_edges() > 0:
        try:
            metrics['clustering'] = nx.average_clustering(G)
        except Exception:
            pass
        try:
            largest_cc = max(nx.connected_components(G), key=len)
            sub = G.subgraph(largest_cc)
            metrics['diameter'] = nx.diameter(sub)
            metrics['avg_path_length'] = nx.average_shortest_path_length(sub)
        except Exception:
            pass

    return metrics


def relation_type_counts(result: Union[str, dict]) -> Counter:
    """Count relation types."""
    result = load_result(result)
    return Counter(r['type'] for r in result.get('relations', []))


def event_verb_counts(result: Union[str, dict]) -> Counter:
    """Count event verbs."""
    result = load_result(result)
    return Counter(e['what'] for e in result.get('events', []))


def plot_network(
    G: 'nx.Graph',
    title: str = '',
    figsize: tuple = (12, 10),
    node_color_attr: str = 'gender',
    save: Optional[str] = None,
):
    """Plot a social network graph with matplotlib.

    Nodes colored by attribute (default: gender), sized by degree.
    """
    import matplotlib.pyplot as plt

    if nx is None:
        raise ImportError("networkx is required")

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    pos = nx.spring_layout(G, seed=42, k=2.0)

    color_maps = {
        'gender': {'male': '#4477AA', 'female': '#CC6677', 'unknown': '#999999'},
    }
    cmap = color_maps.get(node_color_attr, {})
    colors = [cmap.get(G.nodes[n].get(node_color_attr, ''), '#999999') for n in G.nodes()]

    degrees = dict(G.degree())
    sizes = [max(100, degrees.get(n, 1) * 150) for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=0.8)

    labels = {n: G.nodes[n].get('name', n)[:15] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7)

    ax.set_title(title, fontsize=14)
    ax.axis('off')
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()
