"""Aggregate social network statistics across all parsed texts.

Computes per-text graph metrics across 4 network types (relation, event,
dialogue, composite), event/relation distributions, gender ratios, name
classifications, and residualized metrics controlling for cast size.

Usage:
    python scripts/analyze_social_networks.py
    python scripts/analyze_social_networks.py --parish-data data/ncumb.txt
"""

import argparse
import json
import glob
import os
import re
import sys
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from largeliterarymodels.analysis.social_networks import (
    SocialNetwork, load_result, build_graph, build_directed_graph,
    build_dialogue_graph, build_event_graph,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
SN_GLOB = os.path.expanduser(
    '~/lltk_data/corpora/chadwyck/tasks/social_network/**/*.json')


def load_parish_names(path):
    names = set()
    with open(path) as f:
        for line in f:
            if not line.strip() or not line[0].isdigit():
                continue
            parts = line.strip().split(',')
            if len(parts) >= 5:
                for idx in [3, 4]:
                    n = parts[idx].strip('"').strip()
                    if n:
                        names.add(n.lower())
    return names


def classify_name(name, parish_names=None):
    if not name:
        return 'unknown'
    if re.match(r'^(the |a |an )', name.lower()):
        return 'type'
    if re.match(r"^[A-Z][a-z]+'s ", name):
        return 'type'
    if parish_names:
        for word in name.split():
            if word.strip('.,').lower() in parish_names:
                return 'realistic'
    if re.search(r'(us|ia|issa|andra|ander|enes|oles|inda|etta|ina)$',
                 name.split()[0].lower()):
        return 'classical'
    return 'other'


def degree_gini(G):
    """Gini coefficient of degree distribution. 0=uniform, 1=star."""
    if len(G) < 2:
        return 0
    degrees = np.array(sorted([d for _, d in G.degree()]), dtype=float)
    n = len(degrees)
    if degrees.sum() == 0:
        return 0
    index = np.arange(1, n + 1)
    return float((2 * (index * degrees).sum() / (n * degrees.sum())) - (n + 1) / n)


def graph_stats(G, prefix=''):
    """Compute graph metrics. Returns dict with prefixed keys."""
    if G is None or len(G) == 0:
        return {}
    s = {}
    s['nodes'] = len(G)
    s['edges'] = G.number_of_edges()
    s['density'] = round(nx.density(G), 4)

    if isinstance(G, nx.DiGraph):
        s['reciprocity'] = round(nx.reciprocity(G), 4) if G.number_of_edges() > 0 else 0
        U = G.to_undirected()
    else:
        U = G

    s['components'] = nx.number_connected_components(U)
    largest_cc = max(nx.connected_components(U), key=len)
    s['largest_cc_frac'] = round(len(largest_cc) / len(G), 4)

    if len(U) >= 3:
        s['clustering'] = round(nx.average_clustering(U), 4)
        s['transitivity'] = round(nx.transitivity(U), 4)

    bc = nx.betweenness_centrality(U)
    s['max_betweenness'] = round(max(bc.values()), 4) if bc else 0
    s['mean_betweenness'] = round(sum(bc.values()) / len(bc), 4) if bc else 0
    s['centralization'] = round(
        max(bc.values()) - sum(bc.values()) / len(bc), 4) if len(bc) > 1 else 0

    dc = nx.degree_centrality(U)
    s['max_degree_cent'] = round(max(dc.values()), 4) if dc else 0

    s['degree_gini'] = round(degree_gini(U), 4)

    try:
        s['assortativity'] = round(nx.degree_assortativity_coefficient(U), 4)
    except (nx.NetworkXError, ValueError):
        s['assortativity'] = None

    if len(U) > 1 and nx.is_connected(U):
        s['diameter'] = nx.diameter(U)
        s['avg_path'] = round(nx.average_shortest_path_length(U), 2)
    else:
        s['diameter'] = None
        s['avg_path'] = None

    if prefix:
        s = {f'{prefix}_{k}': v for k, v in s.items()}
    return s


def event_macro_counts(events):
    verbs = Counter(e.get('what', '').lower() for e in events)
    categories = {
        'violence': ['killed', 'murdered', 'died', 'executed', 'fought', 'dueled',
                     'attacked', 'poisoned', 'wounded', 'assassinated', 'stabbed'],
        'courtship': ['married', 'courted', 'proposed', 'rejected', 'confessed love',
                      'fell in love', 'declared love', 'engaged', 'attracted_to'],
        'movement': ['arrived', 'departed', 'traveled', 'fled', 'returned', 'escaped',
                     'shipwrecked', 'embarked', 'landed', 'visited'],
        'deception': ['deceived', 'disguised', 'betrayed', 'plotted', 'conspired',
                      'forged', 'impersonated', 'feigned'],
        'legal': ['arrested', 'imprisoned', 'tried', 'convicted', 'sentenced',
                  'pardoned', 'acquitted', 'confessed', 'accused', 'executed'],
    }
    return {k: sum(verbs.get(v, 0) for v in vs) for k, vs in categories.items()}


def residualize(df, metrics, x_col='log_n_chars'):
    """Add residualized versions of metrics, controlling for cast size."""
    df[x_col] = np.log1p(df['n_chars'])
    for m in metrics:
        col = m if m in df.columns else None
        if col is None:
            continue
        valid = df[[x_col, col]].dropna()
        if len(valid) < 5:
            df[f'{col}_resid'] = np.nan
            continue
        from numpy.polynomial import polynomial as P
        coef = P.polyfit(valid[x_col], valid[col], 1)
        predicted = P.polyval(df[x_col], coef)
        df[f'{col}_resid'] = df[col] - predicted
    return df


def build_composite_graph(result):
    """Build composite graph from result dict."""
    sn = SocialNetwork(result)
    return sn.composite_graph()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parish-data', default=os.path.join(DATA_DIR, 'ncumb.txt'))
    args = parser.parse_args()

    parish_names = None
    if os.path.exists(args.parish_data):
        parish_names = load_parish_names(args.parish_data)
        print(f"Parish register: {len(parish_names)} names", file=sys.stderr)

    try:
        import clickhouse_connect
        client = clickhouse_connect.get_client(
            host='localhost', port=8123, username='lltk', password='lltk')
    except Exception:
        client = None
        print("No ClickHouse — metadata will be limited", file=sys.stderr)

    files = sorted(glob.glob(SN_GLOB, recursive=True))
    print(f"Processing {len(files)} social network files", file=sys.stderr)

    rows = []
    for i, path in enumerate(files):
        with open(path) as f:
            d = json.load(f)
        src = d.get('metadata', {}).get('source', '')
        if src == 'list' or not src:
            continue
        chars = d.get('characters', [])
        if not chars:
            continue

        n_chars = len(chars)
        n_events = len(d.get('events', []))
        n_rels = len(d.get('relations', []))
        n_dialogue = len(d.get('dialogue', []))
        n_passages = d.get('metadata', {}).get('n_passages', 0)

        # Gender
        genders = Counter(c.get('gender', '?') for c in chars)
        female_pct = round(genders.get('female', 0) / n_chars * 100, 1)

        # Names
        name_counts = Counter()
        for c in chars:
            name_counts[classify_name(c['name'], parish_names)] += 1

        # Event macros
        macros = event_macro_counts(d.get('events', []))
        macro_pcts = {f'{k}_pct': round(v / n_events * 100, 1) if n_events else 0
                      for k, v in macros.items()}

        # Graph stats — 4 network types
        all_stats = {}
        try:
            all_stats.update(graph_stats(build_composite_graph(d), prefix='comp'))
        except Exception:
            pass
        try:
            all_stats.update(graph_stats(build_directed_graph(d), prefix='rel'))
        except Exception:
            pass
        try:
            all_stats.update(graph_stats(build_event_graph(d), prefix='evt'))
        except Exception:
            pass
        try:
            all_stats.update(graph_stats(build_dialogue_graph(d), prefix='dial'))
        except Exception:
            pass

        # Metadata
        title, year, author = '?', None, '?'
        form_tags, mode_tags = '?', '?'
        if client:
            try:
                r = client.query(
                    f"SELECT title, year, author FROM lltk.texts FINAL WHERE _id = '{src}'")
                if r.result_rows:
                    title, year, author = r.result_rows[0]
                    author = str(author or '?').split(',')[0][:25]
                    title = str(title)[:50]
                r2 = client.query(f"""
                    SELECT DISTINCT t.tag, t.facet FROM lltk.text_genre_tags t
                    WHERE t._id IN (
                        SELECT _id FROM lltk.match_groups FINAL
                        WHERE group_id IN (
                            SELECT group_id FROM lltk.match_groups FINAL WHERE _id = '{src}'
                        )
                    )
                """)
                tags = defaultdict(list)
                for tag, facet in r2.result_rows:
                    tags[facet].append(tag)
                form_tags = '; '.join(sorted(tags.get('form', ['?'])))
                mode_tags = '; '.join(sorted(tags.get('mode', ['?'])))
            except Exception:
                pass

        if year is None:
            continue

        row = {
            'year': int(year), 'source': src, 'id': src.split('/')[-1],
            'title': title, 'author': author,
            'form': form_tags, 'mode': mode_tags,
            'n_passages': n_passages,
            'n_chars': n_chars, 'n_rels': n_rels,
            'n_events': n_events, 'n_dialogue': n_dialogue,
            'female_pct': female_pct,
            'realistic_pct': round(name_counts['realistic'] / n_chars * 100, 1),
            'classical_pct': round(name_counts['classical'] / n_chars * 100, 1),
            'type_pct': round(name_counts['type'] / n_chars * 100, 1),
            **macro_pcts,
            **all_stats,
        }
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(files)}]", file=sys.stderr)

    df = pd.DataFrame(rows).sort_values('year').reset_index(drop=True)

    # Residualize key metrics against log(n_chars)
    resid_metrics = [
        'comp_density', 'comp_clustering', 'comp_centralization',
        'comp_degree_gini', 'comp_reciprocity', 'comp_assortativity',
        'rel_density', 'rel_clustering', 'rel_reciprocity',
        'dial_reciprocity', 'evt_centralization',
    ]
    df = residualize(df, resid_metrics)

    out_path = os.path.join(DATA_DIR, 'social_network_analysis.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} texts to {out_path}", file=sys.stderr)

    # === SUMMARY TABLES ===

    # Select columns for display
    display_cols = [
        'n_chars', 'n_events',
        'female_pct', 'violence_pct', 'courtship_pct', 'movement_pct',
        'realistic_pct', 'classical_pct', 'type_pct',
        'comp_density', 'comp_clustering', 'comp_reciprocity',
        'comp_degree_gini', 'comp_assortativity',
        'comp_centralization',
        'rel_reciprocity', 'dial_reciprocity', 'evt_centralization',
        'comp_density_resid', 'comp_clustering_resid', 'comp_reciprocity_resid',
    ]
    short = {
        'n_chars': 'chars', 'n_events': 'evnts',
        'female_pct': '%fem', 'violence_pct': '%viol',
        'courtship_pct': '%crt', 'movement_pct': '%mov',
        'realistic_pct': '%real', 'classical_pct': '%clas', 'type_pct': '%type',
        'comp_density': 'c.dens', 'comp_clustering': 'c.clus',
        'comp_reciprocity': 'c.reci', 'comp_degree_gini': 'c.gini',
        'comp_assortativity': 'c.asor', 'comp_centralization': 'c.cntr',
        'rel_reciprocity': 'r.reci', 'dial_reciprocity': 'd.reci',
        'evt_centralization': 'e.cntr',
        'comp_density_resid': 'dens.r', 'comp_clustering_resid': 'clus.r',
        'comp_reciprocity_resid': 'reci.r',
    }

    def print_table(label, groups):
        cols = [c for c in display_cols if c in df.columns]
        header = f"{'':>22} {'n':>3}"
        for c in cols:
            header += f" {short.get(c, c[:6]):>6}"
        print(f"\n=== {label} ===")
        print(header)
        print("-" * len(header))
        for name, sub in groups:
            line = f"{str(name)[:22]:>22} {len(sub):3d}"
            for c in cols:
                val = sub[c].mean()
                if pd.notna(val):
                    if abs(val) >= 10:
                        line += f" {val:6.0f}"
                    elif abs(val) >= 1:
                        line += f" {val:6.1f}"
                    else:
                        line += f" {val:6.3f}"
                else:
                    line += f" {'':>6}"
            print(line)

    # By decade
    decade_groups = []
    for decade in range(1590, 1800, 10):
        sub = df[(df['year'] >= decade) & (df['year'] < decade + 10)]
        if len(sub):
            decade_groups.append((f"{decade}s", sub))
    print_table(f"BY DECADE ({len(df)} texts)", decade_groups)

    # By form
    form_rows = []
    for _, r in df.iterrows():
        for f in str(r['form']).split('; '):
            form_rows.append({**r.to_dict(), 'form_tag': f.strip()})
    fdf = pd.DataFrame(form_rows)
    top_forms = fdf['form_tag'].value_counts().head(10).index
    print_table("BY FORM", [(f, fdf[fdf['form_tag'] == f]) for f in top_forms])

    # By mode
    mode_rows = []
    for _, r in df.iterrows():
        for m in str(r['mode']).split('; '):
            mode_rows.append({**r.to_dict(), 'mode_tag': m.strip()})
    mdf = pd.DataFrame(mode_rows)
    top_modes = mdf['mode_tag'].value_counts().head(10).index
    print_table("BY MODE", [(m, mdf[mdf['mode_tag'] == m]) for m in top_modes])

    # Correlations with n_chars
    print(f"\n=== CORRELATION WITH log(n_chars) ===")
    corr_cols = ['comp_density', 'comp_clustering', 'comp_reciprocity',
                 'comp_centralization', 'comp_degree_gini',
                 'rel_reciprocity', 'dial_reciprocity']
    for c in corr_cols:
        if c in df.columns:
            valid = df[['log_n_chars', c]].dropna()
            if len(valid) > 5:
                r = valid['log_n_chars'].corr(valid[c])
                print(f"  {short.get(c, c):>8}  r={r:+.3f}  (n={len(valid)})")


if __name__ == '__main__':
    main()
