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
    plt.rcParams['figure.dpi'] = 300

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
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=10)

    ax.set_title(title, fontsize=14)
    ax.axis('off')
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Object-oriented wrapper
# ---------------------------------------------------------------------------

_GENDER_COLORS = {'male': '#4477AA', 'female': '#CC6677', 'unknown': '#999999'}
_CLASS_COLORS = {
    'gentry': '#E69F00', 'aristocracy': '#D55E00', 'nobility': '#D55E00',
    'merchant': '#56B4E9', 'tradesman': '#56B4E9', 'middle class': '#56B4E9',
    'servant': '#009E73', 'lower class': '#009E73',
    'clergy': '#CC79A7', 'military': '#0072B2', 'lawyer': '#F0E442',
    'royalty': '#882255', 'criminal': '#AA4499', 'unknown': '#999999',
}


class SocialNetwork:
    """OOP wrapper around SocialNetworkTask results.

    Usage:
        sn = SocialNetwork('data/social_network_...json', title='Emma')
        sn.summary()
        sn.plot()                          # composite 2x2
        sn.plot_relations(color_by='class') # single graph
        sn.top_characters(10)              # DataFrame
    """

    def __init__(self, source, title=''):
        self.result = load_result(source)
        self.title = title or self._infer_title()
        self._graphs = {}

    def _infer_title(self):
        meta = self.result.get('metadata', {})
        return meta.get('source', '').split('/')[-1] or ''

    # -- data accessors ---------------------------------------------------

    @property
    def characters(self):
        return self.result.get('characters', [])

    @property
    def relations(self):
        return self.result.get('relations', [])

    @property
    def events(self):
        return self.result.get('events', [])

    @property
    def dialogue(self):
        return self.result.get('dialogue', [])

    @property
    def summaries(self):
        return self.result.get('summaries', [])

    @property
    def n_passages(self):
        return self.result.get('metadata', {}).get('n_passages', 0)

    # -- graph builders (cached) ------------------------------------------

    def rel_graph(self, directed=True, **kw):
        key = ('rel', directed, tuple(sorted(kw.items())))
        if key not in self._graphs:
            fn = build_directed_graph if directed else build_graph
            self._graphs[key] = fn(self.result, **kw)
        return self._graphs[key]

    def event_graph(self):
        if 'event' not in self._graphs:
            self._graphs['event'] = build_event_graph(self.result)
        return self._graphs['event']

    def dialogue_graph(self, active_only=True):
        key = ('dialogue', active_only)
        if key not in self._graphs:
            G = build_dialogue_graph(self.result)
            if active_only:
                active = [n for n in G.nodes() if G.degree(n) > 0]
                G = G.subgraph(active).copy()
            self._graphs[key] = G
        return self._graphs[key]

    def composite_graph(self):
        """Merge relations, events, and dialogue into a single directed graph."""
        if 'composite' not in self._graphs:
            G = nx.DiGraph()
            for c in self.characters:
                G.add_node(c['id'], name=c.get('name', '?'),
                           gender=c.get('gender', '?'),
                           social_class=c.get('class', '?'))

            def _edge(a, b):
                if not G.has_edge(a, b):
                    G.add_edge(a, b, weight=0, rel_types=[],
                               event_verbs=[], n_dialogue=0)
                return G[a][b]

            for r in self.relations:
                if r.get('type') == 'same_as':
                    continue
                a, b = r.get('a', ''), r.get('b', '')
                if not a or not b or a not in G or b not in G:
                    continue
                e = _edge(a, b)
                e['weight'] += 1
                e['rel_types'].append(r['type'])

            for ev in self.events:
                who, whom = ev.get('who', ''), ev.get('whom', '')
                if not who or not whom or who not in G or whom not in G:
                    continue
                e = _edge(who, whom)
                e['weight'] += 1
                e['event_verbs'].append(ev.get('what', ''))

            for d in self.dialogue:
                s, a = d.get('speaker', ''), d.get('addressee', '')
                if not s or not a or s not in G or a not in G:
                    continue
                e = _edge(s, a)
                e['weight'] += 1
                e['n_dialogue'] += 1

            self._graphs['composite'] = G
        return self._graphs['composite']

    # -- node styling helpers ---------------------------------------------

    def _char_attr(self, node_id, attr, default='unknown'):
        chars = _char_lookup(self.result)
        return chars.get(node_id, {}).get(attr, default)

    def _node_colors(self, G, color_by='gender'):
        if isinstance(color_by, dict):
            return [color_by.get(n, color_by.get(
                G.nodes[n].get('name', n), '#999999')) for n in G.nodes()]
        if color_by == 'gender':
            return [_GENDER_COLORS.get(
                G.nodes[n].get('gender') or self._char_attr(n, 'gender'),
                '#999999') for n in G.nodes()]
        if color_by == 'class':
            return [_CLASS_COLORS.get(
                G.nodes[n].get('social_class') or self._char_attr(n, 'class'),
                '#999999') for n in G.nodes()]
        import matplotlib.pyplot as plt
        if color_by == 'degree':
            vals = dict(G.degree())
        elif color_by == 'betweenness':
            vals = nx.betweenness_centrality(G)
        else:
            return ['#4477AA'] * G.number_of_nodes()
        if not vals:
            return ['#4477AA'] * G.number_of_nodes()
        vmin, vmax = min(vals.values()), max(vals.values())
        if vmin == vmax:
            return ['#4477AA'] * G.number_of_nodes()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cm = plt.cm.YlOrRd
        return [cm(norm(vals[n])) for n in G.nodes()]

    def _node_sizes(self, G, base=80, scale=120):
        degrees = dict(G.degree())
        return [max(base, degrees.get(n, 1) * scale) for n in G.nodes()]

    def _labels(self, G, max_len=15):
        chars = _char_lookup(self.result)
        return {n: (G.nodes[n].get('name') or
                     chars.get(n, {}).get('name', n))[:max_len]
                for n in G.nodes()}

    # -- plot methods -----------------------------------------------------

    def plot_relations(self, ax=None, color_by='gender', label_edges=True,
                       directed=True, k=0.9, seed=42, font_size=8,
                       save=None, **kw):
        import matplotlib.pyplot as plt
        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        G = self.rel_graph(directed=directed, **kw)
        pos = nx.spring_layout(G, seed=seed, k=k)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=self._node_colors(G, color_by),
                               node_size=self._node_sizes(G), alpha=0.8)
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=0.8)
        nx.draw_networkx_labels(G, pos, self._labels(G), ax=ax, font_size=font_size)
        if label_edges:
            el = {(a, b): d.get('type', '') for a, b, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, el, ax=ax, font_size=7)
        m = network_metrics(G)
        ax.set_title(f"{self.title}\n{m['n_nodes']} nodes, {m['n_edges']} edges, "
                     f"density={m['density']:.3f}", fontsize=12)
        ax.axis('off')
        if save and own_fig:
            plt.savefig(save, dpi=150, bbox_inches='tight')
        return ax

    def plot_events(self, ax=None, color_by='gender', label_edges=True,
                    k=0.9, seed=42, font_size=8, save=None):
        import matplotlib.pyplot as plt
        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        G = self.event_graph()
        active = [n for n in G.nodes() if G.degree(n) > 0]
        G = G.subgraph(active)
        pos = nx.spring_layout(G, seed=seed, k=k)
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_w = max(weights) if weights else 1
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=self._node_colors(G, color_by),
                               node_size=self._node_sizes(G), alpha=0.8)
        nx.draw_networkx_edges(G, pos, ax=ax, width=[w / max_w * 4 for w in weights],
                               alpha=0.4, arrows=True, arrowsize=12)
        nx.draw_networkx_labels(G, pos, self._labels(G), ax=ax, font_size=font_size)
        if label_edges:
            el = {}
            for a, b, d in G.edges(data=True):
                verbs = d.get('events', [])
                el[(a, b)] = Counter(verbs).most_common(1)[0][0] if verbs else ''
            nx.draw_networkx_edge_labels(G, pos, el, ax=ax, font_size=7)
        ax.set_title(f"{self.title}: events ({G.number_of_edges()} edges)", fontsize=11)
        ax.axis('off')
        if save and own_fig:
            plt.savefig(save, dpi=150, bbox_inches='tight')
        return ax

    def plot_dialogue(self, ax=None, label_edges=True, k=0.75, seed=42,
                      font_size=7, save=None):
        import matplotlib.pyplot as plt
        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        DG = self.dialogue_graph(active_only=True)
        pos = nx.spring_layout(DG, seed=seed, k=k)
        weights = [DG[u][v]['weight'] for u, v in DG.edges()]
        max_w = max(weights) if weights else 1
        nx.draw_networkx_nodes(DG, pos, ax=ax, node_size=300, node_color='#4477AA', alpha=0.7)
        nx.draw_networkx_edges(DG, pos, ax=ax, width=[w / max_w * 4 for w in weights],
                               alpha=0.5, arrows=True, arrowsize=15)
        nx.draw_networkx_labels(DG, pos, self._labels(DG, 12), ax=ax, font_size=font_size)
        if label_edges:
            el = {(a, b): str(d['weight']) for a, b, d in DG.edges(data=True)}
            nx.draw_networkx_edge_labels(DG, pos, el, ax=ax, font_size=7)
        ax.set_title(f"{self.title}: dialogue ({DG.number_of_edges()} edges)", fontsize=11)
        ax.axis('off')
        if save and own_fig:
            plt.savefig(save, dpi=150, bbox_inches='tight')
        return ax

    def plot_all(self, ax=None, color_by='gender', label_edges=True,
                 k=0.9, seed=42, font_size=8, save=None):
        """Plot composite graph merging relations, events, and dialogue."""
        import matplotlib.pyplot as plt
        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(1, 1, figsize=(14, 12))
        G = self.composite_graph()
        active = [n for n in G.nodes() if G.degree(n) > 0]
        G = G.subgraph(active)
        pos = nx.spring_layout(G, seed=seed, k=k)
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_w = max(weights) if weights else 1
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=self._node_colors(G, color_by),
                               node_size=self._node_sizes(G), alpha=0.8)
        nx.draw_networkx_edges(G, pos, ax=ax, width=[w / max_w * 4 for w in weights],
                               alpha=0.35, arrows=True, arrowsize=10)
        nx.draw_networkx_labels(G, pos, self._labels(G), ax=ax, font_size=font_size)
        if label_edges:
            el = {}
            for a, b, d in G.edges(data=True):
                lbl = []
                if d['rel_types']:
                    # el[(a, b)] = Counter(d['rel_types']).most_common(1)[0][0]
                    lbl.append(Counter(d['rel_types']).most_common(1)[0][0])
                if d['event_verbs']:
                    # el[(a, b)] = Counter(d['event_verbs']).most_common(1)[0][0]
                    lbl.append(Counter(d['event_verbs']).most_common(1)[0][0])
                # if d['n_dialogue'] and not lbl:
                    # el[(a, b)] = f"dlg:{d['n_dialogue']}"
                # else:
                    # el[(a, b)] = ''
                el[(a,b)] = ' | '.join(lbl)
            nx.draw_networkx_edge_labels(G, pos, el, ax=ax, font_size=7)
        n_rels = sum(1 for _, _, d in G.edges(data=True) if d['rel_types'])
        n_evts = sum(1 for _, _, d in G.edges(data=True) if d['event_verbs'])
        n_dlgs = sum(1 for _, _, d in G.edges(data=True) if d['n_dialogue'])
        ax.set_title(f"{self.title}: all interactions\n"
                     f"{G.number_of_nodes()} chars, {G.number_of_edges()} edges "
                     f"({n_rels} rel, {n_evts} event, {n_dlgs} dialogue)",
                     fontsize=11)
        ax.axis('off')
        if save and own_fig:
            plt.savefig(save, dpi=150, bbox_inches='tight')
        return ax

    def plot_trajectory(self, char_id=None, ax=None, save=None):
        import matplotlib.pyplot as plt
        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(1, 1, figsize=(16, 5))
        if char_id is None:
            G = self.rel_graph(directed=False)
            if G.number_of_nodes() == 0:
                ax.set_title(f"{self.title}: no characters")
                return ax
            char_id = max(dict(G.degree()), key=lambda n: G.degree(n))
        trajs = character_trajectories(self.result)
        traj = trajs.get(char_id, [])
        if not traj:
            ax.set_title(f"{self.title}: no trajectory for {char_id}")
            return ax
        passages = [int(t['passage'][1:]) for t in traj if t['passage'].startswith('P')]
        locations = [t['where'] for t in traj if t['passage'].startswith('P')]
        seen = {}
        for loc in locations:
            if loc not in seen:
                seen[loc] = len(seen)
        y = [seen[loc] for loc in locations]
        ax.scatter(passages, y, s=20, alpha=0.8, c='#4477AA')
        ax.plot(passages, y, alpha=0.3, c='#4477AA', linewidth=0.8)
        ax.set_yticks(list(seen.values()))
        ax.set_yticklabels(list(seen.keys()), fontsize=7)
        ax.set_xlabel('Passage number')
        char_name = next((c['name'] for c in self.characters
                          if c['id'] == char_id), char_id)
        ax.set_title(f"{self.title}: {char_name} trajectory "
                     f"({len(traj)} events)", fontsize=11)
        if save and own_fig:
            plt.savefig(save, dpi=150, bbox_inches='tight')
        return ax

    def plot_relation_types(self, ax=None, n=10):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        top = self.relation_counts().most_common(n)
        ax.barh([t for t, _ in reversed(top)], [c for _, c in reversed(top)],
                color='#4477AA')
        ax.set_title(f'{self.title}: relation types')
        ax.set_xlabel('count')
        return ax

    def plot_event_verbs(self, ax=None, n=12):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        top = self.event_counts().most_common(n)
        ax.barh([t for t, _ in reversed(top)], [c for _, c in reversed(top)],
                color='#CC6677')
        ax.set_title(f'{self.title}: top event verbs')
        ax.set_xlabel('count')
        return ax

    def plot(self, figsize=(24, 20), color_by='gender', label_edges=True,
             protagonist=None, save=None):
        """Composite 2x2: relations, events, dialogue, trajectory."""
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        self.plot_relations(ax=axes[0, 0], color_by=color_by,
                            label_edges=label_edges)
        self.plot_events(ax=axes[0, 1], color_by=color_by,
                         label_edges=label_edges)
        self.plot_dialogue(ax=axes[1, 0], label_edges=label_edges)
        self.plot_trajectory(char_id=protagonist, ax=axes[1, 1])
        fig.suptitle(self.title, fontsize=16, y=1.01)
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=150, bbox_inches='tight')
        return fig

    # -- statistics -------------------------------------------------------

    def metrics(self, directed=False):
        G = self.rel_graph(directed=directed)
        m = network_metrics(G)
        m['n_locations'] = len({e['where'] for e in self.events if e.get('where')})
        m['n_dialogue'] = len(self.dialogue)
        m['n_passages'] = self.n_passages
        if 'max_degree_node' in m:
            m['max_degree_name'] = (
                G.nodes[m['max_degree_node']].get('name', '?'))
        return m

    def relation_counts(self):
        counts = relation_type_counts(self.result)
        counts.pop('same_as', None)
        return counts

    def event_counts(self):
        return event_verb_counts(self.result)

    def locations(self):
        return location_summary(self.result)

    def trajectories(self):
        return character_trajectories(self.result)

    def trajectory(self, char_id):
        return character_trajectories(self.result).get(char_id, [])

    def top_characters(self, n=10, by='degree'):
        G = self.rel_graph(directed=False)
        if by == 'betweenness':
            vals = nx.betweenness_centrality(G)
        else:
            vals = dict(G.degree())
        ranked = sorted(vals.items(), key=lambda x: x[1], reverse=True)
        chars = _char_lookup(self.result)
        rows = []
        for cid, val in ranked[:n]:
            c = chars.get(cid, {})
            rows.append({
                'id': cid, 'name': c.get('name', '?'),
                'gender': c.get('gender', '?'), 'class': c.get('class', '?'),
                by: round(val, 4) if isinstance(val, float) else val,
            })
        return pd.DataFrame(rows)

    def character_df(self):
        G = self.rel_graph(directed=False)
        deg = dict(G.degree())
        rows = []
        for c in self.characters:
            rows.append({
                'id': c['id'], 'name': c.get('name', '?'),
                'gender': c.get('gender', '?'), 'class': c.get('class', '?'),
                'aliases': ', '.join(c.get('aliases', [])),
                'intro_text': c.get('intro_text', ''),
                'descriptions': ' | '.join(c.get('descriptions',[])),
                'degree': deg.get(c['id'], 0),
            })
        return pd.DataFrame(rows).sort_values('degree', ascending=False)

    # -- narrative --------------------------------------------------------

    def story(self):
        for s in self.summaries:
            print(f"\n[P{s['start']:03d}-P{s['end']:03d}]")
            print(f"  {s['text']}")

    # -- display ----------------------------------------------------------

    def __repr__(self):
        return (f"SocialNetwork('{self.title}', "
                f"{len(self.characters)} chars, {len(self.relations)} rels, "
                f"{len(self.events)} events, {len(self.dialogue)} dialogue)")

    def summary(self):
        m = self.metrics()
        print(f"{self.title}")
        print(f"  Characters: {len(self.characters)}")
        print(f"  Relations:  {len(self.relations)}")
        print(f"  Events:     {len(self.events)}")
        print(f"  Dialogue:   {len(self.dialogue)}")
        print(f"  Locations:  {m.get('n_locations', '?')}")
        print(f"  Passages:   {m.get('n_passages', '?')}")
        print(f"  Density:    {m.get('density', 0):.3f}")
        print(f"  Clustering: {m.get('clustering', 0):.3f}")


def compare(*networks: SocialNetwork) -> pd.DataFrame:
    """Side-by-side metrics table for multiple SocialNetwork objects."""
    rows = []
    for sn in networks:
        m = sn.metrics()
        m['novel'] = sn.title
        rows.append(m)
    cols = ['n_nodes', 'n_edges', 'density', 'mean_degree', 'clustering',
            'diameter', 'avg_path_length', 'max_degree_name', 'max_degree',
            'n_locations', 'n_dialogue', 'n_passages']
    df = pd.DataFrame(rows).set_index('novel')
    return df[[c for c in cols if c in df.columns]]
