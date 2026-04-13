import os
# -*- coding: utf-8 -*-
from flask import Flask, request, render_template_string, jsonify
from Bio.Blast import NCBIWWW, NCBIXML
import threading
import uuid
import time
import numpy as np
from collections import defaultdict, Counter
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO
 
app = Flask(__name__)
 
# In-memory job store
jobs = {}
 
# Reference genome database
REFERENCE_GENOMES = {
    "Escherichia coli": "ATGCGTACGTAGCTAGCTAGCTAGCGTAGCTAGCTAGCGTAGCTAGCTAGCTAGCGTAGCTAGCTAGCTAGCGTAGCTAGCGTAGCTAGCTAGCTAGCGTAGCTAGCTAGCTAGCGTAGCTAGCTAGCTAGC",
    "Bacillus subtilis": "ATATATATATATATATCGCGATATATATATATCGATATATATATATATCGATATATATATATATCGATATATATATATATCGATATATATATATCGATATATATATATATCGATATATATATAT",
    "Mycobacterium tuberculosis": "GCGCGCGCGCGCGCGATATCGCGCGCGCGCGCGCGCGCGCGCGATATCGCGCGCGCGCGCGCGCGCGCGCGCGGCGCGCGCGCGCGCGATATCGCGCGCGCGCGCGCGCGCGCGCGCG",
    "Saccharomyces cerevisiae": "ATGATGATGATGCGCGATATATATATGCGCGATATATATATATGCGCGATATATATATATGCGCGATATATATATATGCGCGATATATATATGCGCGATATATATATATGCGCGATAT",
    "Homo sapiens": "ATATATCGCGATATATATATCGATATATATATATATATATCGCGATATATATATATATATATCGATATATATATATATATCGCGATATATATATATATATCGATATATATATATATAT",
    "Methanocaldococcus jannaschii": "GGGCGCGCGCGCGATATATATATATATCGCGCGCGCGCGGGCGCGCGCGCGCGATATATATATATGGGCGCGCGCGCGATATATATATATATCGCGCGCGCGCG"
}
 
# ------------------------------
# PROBABILITY-BASED ANALYSIS MODELS
# ------------------------------
 
class MarkovModel:
    """k-th order Markov model"""
    def __init__(self, k=1):
        self.k = k
        self.transitions = defaultdict(Counter)
 
    def train(self, sequence):
        sequence = sequence.upper()
        for i in range(len(sequence) - self.k):
            state = sequence[i:i+self.k]
            next_state = sequence[i+self.k]
            self.transitions[state][next_state] += 1
 
    def get_probability(self, sequence):
        sequence = sequence.upper()
        prob = 1.0
        for i in range(len(sequence) - self.k):
            state = sequence[i:i+self.k]
            next_state = sequence[i+self.k]
            total = sum(self.transitions[state].values())
            if total > 0:
                prob *= self.transitions[state][next_state] / total
            else:
                prob *= 0.25
        return prob
 
    def predict_next(self, state):
        if state in self.transitions:
            return max(self.transitions[state], key=self.transitions[state].get)
        return None
 
class KMerAnalyzer:
    """k-mer frequency analyzer"""
    def __init__(self, k=3):
        self.k = k
        self.kmers = {}
 
    def calculate_distribution(self, sequence):
        sequence = sequence.upper()
        kmers = []
        for i in range(len(sequence) - self.k + 1):
            kmers.append(sequence[i:i+self.k])
        self.kmers = Counter(kmers)
        total = sum(self.kmers.values())
        return {kmer: count/total for kmer, count in self.kmers.items()} if total > 0 else {}
 
    def compare_distributions(self, seq1, seq2):
        dist1 = self.calculate_distribution(seq1)
        dist2 = self.calculate_distribution(seq2)
        kmers1 = set(dist1.keys())
        kmers2 = set(dist2.keys())
        intersection = len(kmers1 & kmers2)
        union = len(kmers1 | kmers2)
        jaccard = intersection / union if union > 0 else 0
        return {"jaccard": jaccard, "similarity": jaccard * 100}
 
class MaximumLikelihoodEstimator:
    """MLE classifier"""
    def __init__(self):
        self.models = {}
 
    def train_model(self, label, sequence):
        model = MarkovModel(k=2)
        model.train(sequence)
        self.models[label] = model
 
    def classify(self, sequence):
        likelihoods = {}
        for label, model in self.models.items():
            likelihoods[label] = model.get_probability(sequence)
        total = sum(likelihoods.values())
        if total > 0:
            posteriors = {label: l/total for label, l in likelihoods.items()}
            best = max(posteriors, key=posteriors.get)
            return best, posteriors
        return None, likelihoods
 
# ------------------------------
# PHYLOGENETIC ANALYSIS
# ------------------------------
 
def calculate_genetic_distance(seq1, seq2):
    seq1, seq2 = seq1.upper(), seq2.upper()
    min_len = min(len(seq1), len(seq2))
    seq1, seq2 = seq1[:min_len], seq2[:min_len]
    differences = sum(a != b for a, b in zip(seq1, seq2))
    return differences / min_len if min_len > 0 else 1.0
 
def neighbor_joining(sequences, labels):
    """
    Proper Neighbor-Joining algorithm returning a tree structure
    with internal node coordinates for correct cladogram drawing.
    """
    n = len(sequences)
    if n < 2:
        return None
 
    # Build initial distance matrix as list of lists
    dist_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            dist = calculate_genetic_distance(sequences[i], sequences[j])
            dist_matrix[i][j] = dist_matrix[j][i] = dist
 
    tree_nodes = {}
    for i, lbl in enumerate(labels):
        tree_nodes[i] = {"name": lbl, "children": [], "branch_length": 0.0}
 
    active = list(range(n))
    names = list(labels)
    next_internal = n  # index for new internal nodes
 
    while len(active) > 2:
        size = len(active)
 
        # Compute Q matrix
        row_sums = {i: sum(dist_matrix[i][j] for j in active) for i in active}
        min_q = float('inf')
        min_i, min_j = -1, -1
        for ii in range(len(active)):
            for jj in range(ii+1, len(active)):
                i, j = active[ii], active[jj]
                q = (size - 2) * dist_matrix[i][j] - row_sums[i] - row_sums[j]
                if q < min_q:
                    min_q = q
                    min_i, min_j = i, j
 
        # Branch lengths from new node u to min_i and min_j
        dij = dist_matrix[min_i][min_j]
        if size > 2:
            branch_i = 0.5 * dij + (row_sums[min_i] - row_sums[min_j]) / (2 * (size - 2))
        else:
            branch_i = 0.5 * dij
        branch_j = dij - branch_i
        branch_i = max(0.0, branch_i)
        branch_j = max(0.0, branch_j)
 
        # Create new internal node
        new_idx = next_internal
        next_internal += 1
        tree_nodes[new_idx] = {
            "name": f"node_{new_idx}",
            "children": [
                {"node": min_i, "length": branch_i},
                {"node": min_j, "length": branch_j}
            ],
            "branch_length": 0.0
        }
        # Set branch lengths on children
        tree_nodes[min_i]["branch_length"] = branch_i
        tree_nodes[min_j]["branch_length"] = branch_j
 
        # Compute distances from new node to all remaining active nodes
        new_row_size = max(new_idx + 1, len(dist_matrix) + 1)
        # Expand dist_matrix to fit new_idx
        for row in dist_matrix:
            while len(row) < new_idx + 1:
                row.append(0.0)
        while len(dist_matrix) < new_idx + 1:
            dist_matrix.append([0.0] * (new_idx + 1))
 
        for k in active:
            if k == min_i or k == min_j:
                continue
            d = 0.5 * (dist_matrix[min_i][k] + dist_matrix[min_j][k] - dij)
            d = max(0.0, d)
            dist_matrix[new_idx][k] = d
            dist_matrix[k][new_idx] = d
 
        active = [x for x in active if x != min_i and x != min_j]
        active.append(new_idx)
        names.append(f"node_{new_idx}")
 
    # Connect the last two active nodes
    if len(active) == 2:
        a, b = active[0], active[1]
        d = dist_matrix[a][b]
        tree_nodes[a]["branch_length"] = d / 2
        tree_nodes[b]["branch_length"] = d / 2
        root_idx = next_internal
        tree_nodes[root_idx] = {
            "name": "root",
            "children": [
                {"node": a, "length": d / 2},
                {"node": b, "length": d / 2}
            ],
            "branch_length": 0.0
        }
        return tree_nodes, root_idx
 
    return tree_nodes, active[0]
 
 
def create_tree_visualization(tree_result, method):
    """
    Draw a proper cladogram with:
    - Correct horizontal branch lengths (proportional to genetic distance)
    - Correct vertical positioning of internal nodes (midpoint of children)
    - Clean lines: horizontal branch to child, vertical connector between siblings
    """
    if not tree_result:
        return None
 
    tree_nodes, root_idx = tree_result
 
    leaf_labels = [data["name"] for idx, data in tree_nodes.items()
                   if not data.get("children")]
    n_leaves = len(leaf_labels)
 
    # Assign y positions to leaves in DFS order
    leaf_counter = [0]
    node_y = {}
    node_x = {}
 
    def assign_positions(idx, x_pos):
        node = tree_nodes[idx]
        if not node.get("children"):
            # Leaf node
            node_y[idx] = leaf_counter[0]
            node_x[idx] = x_pos
            leaf_counter[0] += 1
        else:
            for child_info in node["children"]:
                child_idx = child_info["node"]
                child_x = x_pos + child_info["length"]
                assign_positions(child_idx, child_x)
            # Internal node y = midpoint of children y
            child_ys = [node_y[c["node"]] for c in node["children"]]
            node_y[idx] = (min(child_ys) + max(child_ys)) / 2.0
            node_x[idx] = x_pos
 
    assign_positions(root_idx, 0.0)
 
    fig, ax = plt.subplots(figsize=(12, max(6, n_leaves * 0.7)))
    fig.patch.set_facecolor('#020617')
    ax.set_facecolor('#0f172a')
 
    # Draw branches
    def draw_branches(idx):
        node = tree_nodes[idx]
        if not node.get("children"):
            return
        x_parent = node_x[idx]
        y_parent = node_y[idx]
        child_ys = [node_y[c["node"]] for c in node["children"]]
 
        # Vertical connector between topmost and bottommost child
        ax.plot(
            [x_parent, x_parent],
            [min(child_ys), max(child_ys)],
            color='#38bdf8', linewidth=1.5, solid_capstyle='round'
        )
 
        for child_info in node["children"]:
            c_idx = child_info["node"]
            x_child = node_x[c_idx]
            y_child = node_y[c_idx]
            # Horizontal branch from parent x to child x
            ax.plot(
                [x_parent, x_child],
                [y_child, y_child],
                color='#4ade80', linewidth=1.8, solid_capstyle='round'
            )
            # Branch length label (only if > 0.001)
            if child_info["length"] > 0.001:
                mid_x = (x_parent + x_child) / 2
                ax.text(mid_x, y_child + 0.15, f'{child_info["length"]:.3f}',
                        fontsize=7, ha='center', va='bottom', color='#fbbf24',
                        fontfamily='monospace')
            draw_branches(c_idx)
 
    draw_branches(root_idx)
 
    # Draw leaf labels and dots
    for idx, data in tree_nodes.items():
        if not data.get("children"):
            x = node_x[idx]
            y = node_y[idx]
            ax.plot(x, y, 'o', color='#f87171', markersize=6, zorder=5)
            name = data["name"]
            if name == "Query":
                color = '#fbbf24'
                weight = 'bold'
            else:
                color = '#e2e8f0'
                weight = 'normal'
            ax.text(x + 0.005, y, name, fontsize=9, va='center', ha='left',
                    color=color, fontweight=weight, fontfamily='monospace')
 
    # Draw root dot
    ax.plot(node_x[root_idx], node_y[root_idx], 's',
            color='#38bdf8', markersize=7, zorder=5)
 
    # Internal node dots
    for idx, data in tree_nodes.items():
        if data.get("children") and idx != root_idx:
            ax.plot(node_x[idx], node_y[idx], 'o',
                    color='#64748b', markersize=4, zorder=4)
 
    max_x = max(node_x.values()) if node_x else 1.0
    ax.set_xlim(-0.02, max_x * 1.35)
    ax.set_ylim(-0.8, leaf_counter[0] - 0.2)
    ax.set_xlabel('Genetic Distance', fontsize=10, color='#94a3b8', labelpad=8)
    ax.set_title(f'{method} Phylogenetic Tree', fontsize=13,
                 fontweight='bold', color='#38bdf8', pad=12)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_color('#334155')
    ax.tick_params(axis='x', colors='#64748b', labelsize=8)
    ax.grid(axis='x', alpha=0.15, color='#334155', linestyle='--')
 
    # Legend
    legend_elements = [
        mpatches.Patch(color='#4ade80', label='Branch'),
        mpatches.Patch(color='#fbbf24', label='Query'),
        mpatches.Patch(color='#f87171', label='Reference taxon'),
    ]
    ax.legend(handles=legend_elements, loc='lower right',
              fontsize=8, framealpha=0.2,
              labelcolor='#e2e8f0', facecolor='#1e293b')
 
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=110, bbox_inches='tight',
                facecolor='#020617')
    buffer.seek(0)
    img = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return f'data:image/png;base64,{img}'
 
 
# ------------------------------
# ALIGNMENT FORMATTER  — FIXED
# Uses <pre> blocks so every character is exactly one monospace column wide.
# Coloured spans use display:inline so they don't break column widths inside <pre>.
# ------------------------------
def format_compact_pairwise(query, match, subject, q_start, s_start, width=60):
    lines = ""
    for i in range(0, len(query), width):
        q_chunk  = query[i:i+width]
        m_chunk  = match[i:i+width]
        s_chunk  = subject[i:i+width]
        q_end    = q_start + min(i + width, len(query))
        s_end    = s_start + min(i + width, len(subject))
 
        # Build coloured match string — every character must stay ONE column
        colored_match = ""
        for char in m_chunk:
            if char == '|':
                colored_match += '<span style="color:#4ade80;">|</span>'
            elif char == ' ':
                # Replace space with a visible gap marker that is still 1 char wide
                colored_match += '<span style="color:#f87171;">·</span>'
            else:
                colored_match += char
 
        lines += (
            f'<div style="margin-bottom:10px;">'
            # Use a single <pre> so all three lines share the same monospace grid
            f'<pre style="'
            f'font-family:JetBrains Mono,Consolas,monospace;'
            f'font-size:12px;line-height:1.6;margin:0;'
            f'background:#020617;padding:8px 10px;border-radius:6px;'
            f'overflow-x:auto;white-space:pre;">'
            f'<span style="color:#38bdf8;">Query </span>'
            f'<span style="color:#94a3b8;">{q_start+i:>6}</span>  '
            f'{q_chunk}  '
            f'<span style="color:#94a3b8;">{q_end}</span>\n'
            f'             {colored_match}\n'
            f'<span style="color:#fb923c;">Sbjct </span>'
            f'<span style="color:#94a3b8;">{s_start+i:>6}</span>  '
            f'{s_chunk}  '
            f'<span style="color:#94a3b8;">{s_end}</span>'
            f'</pre>'
            f'</div>'
        )
    return lines
 
 
# ------------------------------
# HELPERS
# ------------------------------
def is_protein(seq):
    return any(c not in "ATGC" for c in seq.upper())
 
def clean_seq(raw):
    lines = raw.strip().splitlines()
    cleaned_lines = [line for line in lines if not line.strip().startswith('>')]
    joined = ''.join(cleaned_lines)
    return ''.join(c for c in joined if c.isalpha()).upper()
 
def nucleotide_stats(seq):
    A = seq.count("A")
    T = seq.count("T")
    G = seq.count("G")
    C = seq.count("C")
    length = len(seq)
    gc = (G + C) / length * 100 if length else 0
    at = (A + T) / length * 100 if length else 0
    return {"A": A, "T": T, "G": G, "C": C, "gc": gc, "at": at, "length": length}
 
# ------------------------------
# BLAST WORKER
# ------------------------------
def blast_worker(job_id, seq):
    try:
        prog = "blastp" if is_protein(seq) else "blastn"
        db = "nr" if is_protein(seq) else "nt"
        seq_type = "Protein" if is_protein(seq) else "DNA"
 
        result_handle = NCBIWWW.qblast(prog, db, seq)
        blast_record = NCBIXML.read(result_handle)
 
        table_rows = ""
        alignments_html = ""
        graphics_html = ""
 
        if not blast_record.alignments:
            jobs[job_id] = {"status": "done", "seq_type": seq_type, "db": db.upper(),
                           "seq_len": len(seq), "table_rows": "", "graphics_html": "",
                           "alignments_html": "<p style='color:#fbbf24;'>⚠️ No significant alignments found.</p>",
                           "num_hits": 0}
            return
 
        for i, alignment in enumerate(blast_record.alignments[:5]):
            best_hsp = max(alignment.hsps, key=lambda h: h.score, default=None)
            if not best_hsp:
                continue
            hsp = best_hsp
            identity = (hsp.identities / hsp.align_length) * 100
            title = alignment.title.split()[0] if alignment.title else "Unknown"
            description = " ".join(alignment.title.split()[1:])[:80] if alignment.title else ""
 
            table_rows += f'''
<tr onclick="scrollToAlignment('align{i}')" style="cursor:pointer">
  <td style="padding:10px;"><strong>{title}</strong><br><small style="color:#94a3b8;">{description}</small></td>
  <td style="padding:10px;color:#38bdf8;">{hsp.score}</td>
  <td style="padding:10px;"><span style="color:#f87171;">{identity:.1f}%</span></td>
  <td style="padding:10px;color:#fbbf24;">{hsp.expect:.2e}</td>
</tr>'''
 
            length = len(seq)
            wp = ((hsp.query_end - hsp.query_start) / length * 100) if length else 0
            graphics_html += f'''
<div style="margin:8px 0;">
  <div style="display:flex;justify-content:space-between;">
    <small>{title[:30]}</small>
    <small>{wp:.1f}%</small>
  </div>
  <div style="background:#1e293b;height:16px;border-radius:8px;overflow:hidden;">
    <div style="width:{wp}%;height:16px;background:linear-gradient(90deg,#38bdf8,#f87171);border-radius:8px;"></div>
  </div>
</div>'''
 
            formatted = format_compact_pairwise(
                hsp.query, hsp.match, hsp.sbjct,
                hsp.query_start, hsp.sbjct_start
            )
            alignments_html += f'''
<div id="align{i}" style="margin-top:20px;padding:15px;background:#0f172a;border-radius:10px;border-left:3px solid #38bdf8;">
  <h4 onclick="toggleBox('box{i}')" style="color:#38bdf8;cursor:pointer;margin:0 0 10px 0;">🧬 {title}</h4>
  <div style="display:flex;gap:15px;margin-bottom:10px;font-size:12px;">
    <span>Identity: <strong style="color:#f87171;">{identity:.1f}%</strong></span>
    <span>Score: {hsp.score}</span>
    <span>E-value: {hsp.expect:.2e}</span>
  </div>
  <div id="box{i}" style="display:none;">
    {formatted}
  </div>
</div>'''
 
        jobs[job_id] = {"status": "done", "seq_type": seq_type, "db": db.upper(),
                       "seq_len": len(seq), "table_rows": table_rows, "graphics_html": graphics_html,
                       "alignments_html": alignments_html, "num_hits": min(5, len(blast_record.alignments))}
    except Exception as e:
        jobs[job_id] = {"status": "error", "message": str(e)}
 
# ------------------------------
# PROBABILITY ANALYSIS
# ------------------------------
def analyze_probabilities(seq):
    results = {}
 
    markov_1 = MarkovModel(k=1)
    markov_2 = MarkovModel(k=2)
    markov_1.train(seq)
    markov_2.train(seq)
    results["markov"] = {
        "prob_order1": f"{markov_1.get_probability(seq):.2e}",
        "prob_order2": f"{markov_2.get_probability(seq):.2e}",
        "next_pred": markov_1.predict_next(seq[-1:]) if seq else "N/A"
    }
 
    kmers_html = ""
    for k in [2, 3]:
        analyzer = KMerAnalyzer(k=k)
        if len(seq) >= k:
            dist = analyzer.calculate_distribution(seq)
            top_kmers = sorted(analyzer.kmers.items(), key=lambda x: -x[1])[:5]
            kmers_html += f"<h4>k={k} (total: {sum(analyzer.kmers.values())}, unique: {len(dist)})</h4><ul>"
            for kmer, count in top_kmers:
                pct = count/sum(analyzer.kmers.values())*100
                kmers_html += f"<li><code>{kmer}</code>: {count} ({pct:.1f}%)</li>"
            kmers_html += "</ul>"
    results["kmers_html"] = kmers_html
 
    sim_html = "<table style='width:100%'><tr><th>Species</th><th>Similarity</th></tr>"
    for name, ref in REFERENCE_GENOMES.items():
        analyzer = KMerAnalyzer(k=3)
        sim = analyzer.compare_distributions(seq, ref)
        sim_html += f"<tr><td>{name}</td><td><div style='background:#1e293b;height:20px;border-radius:10px;width:100%'><div style='width:{sim['similarity']:.1f}%;height:20px;background:#4ade80;border-radius:10px;text-align:center;font-size:11px;'>{sim['similarity']:.1f}%</div></div></td></tr>"
    sim_html += "</table>"
    results["similarity_html"] = sim_html
 
    mle = MaximumLikelihoodEstimator()
    for name, ref in REFERENCE_GENOMES.items():
        mle.train_model(name, ref)
    best, posteriors = mle.classify(seq)
 
    if best:
        results["mle_best"] = best
        results["mle_conf"] = f"{max(posteriors.values())*100:.1f}%"
        mle_html = "<table style='width:100%'><tr><th>Species</th><th>Likelihood</th></tr>"
        for name, prob in sorted(posteriors.items(), key=lambda x: -x[1]):
            pct = prob * 100
            mle_html += f"<tr><td>{name}</td><td><div style='background:#1e293b;height:20px;border-radius:10px;width:100%'><div style='width:{pct:.1f}%;height:20px;background:#38bdf8;border-radius:10px;text-align:center;font-size:11px;'>{pct:.1f}%</div></div></td></tr>"
        mle_html += "</table>"
        results["mle_html"] = mle_html
    else:
        results["mle_best"] = "N/A"
        results["mle_conf"] = "N/A"
        results["mle_html"] = "<p>Could not classify</p>"
 
    return results
 
# ------------------------------
# PHYLOGENETIC ANALYSIS
# ------------------------------
def analyze_phylogenetics(seq):
    results = {}
    sequences = [seq] + list(REFERENCE_GENOMES.values())
    labels = ["Query"] + list(REFERENCE_GENOMES.keys())
 
    nj_result = neighbor_joining(sequences, labels)
    if nj_result:
        results["nj_image"] = create_tree_visualization(nj_result, "Neighbor-Joining")
 
    dist_html = "<table style='width:100%'><tr><th>Species</th><th>Distance</th><th>Closeness</th></tr>"
    for name, ref in REFERENCE_GENOMES.items():
        dist = calculate_genetic_distance(seq, ref)
        closeness = (1 - dist) * 100
        bar_color = "#4ade80" if closeness > 70 else "#fbbf24" if closeness > 40 else "#f87171"
        dist_html += f"<tr><td>{name}</td><td>{dist:.4f}</td><td><div style='background:#1e293b;height:20px;border-radius:10px;width:100%'><div style='width:{closeness:.1f}%;height:20px;background:{bar_color};border-radius:10px;text-align:center;font-size:11px;'>{closeness:.1f}%</div></div></td></tr>"
    dist_html += "</table>"
    results["distances_html"] = dist_html
 
    return results
 
# ------------------------------
# CLASSIFICATION
# ------------------------------
def classify_seq(seq):
    stats = nucleotide_stats(seq)
    gc, at, length = stats["gc"], stats["at"], stats["length"]
    A, T, G, C = stats["A"], stats["T"], stats["G"], stats["C"]
 
    scores = {
        "Proteobacteria (E. coli)": G + C * 0.8,
        "Firmicutes (Bacillus)": A + T * 0.6,
        "Actinobacteria": G + C * 1.2,
        "Fungi (Yeast)": A * 0.5,
        "Animal (Human)": A + T * 0.4,
        "Archaea (Methanogen)": G * 0.7,
    }
    total = sum(scores.values())
 
    if total == 0:
        return {
            "best": "Unknown", "confidence": 0.0, "gc": gc, "at": at,
            "length": length, "seq_preview": seq[:60], "bars": "<p>No recognizable nucleotides.</p>",
            "A": A, "T": T, "G": G, "C": C
        }
 
    best = max(scores, key=scores.get)
    conf = scores[best] / total * 100
 
    bars = ""
    for k, v in sorted(scores.items(), key=lambda x: -x[1]):
        pct = v / total * 100
        bars += f'''
<div style="margin-bottom:8px;">
  <div style="display:flex;justify-content:space-between;font-size:12px;">
    <span style="color:{'#38bdf8' if k==best else '#94a3b8'};">{k}</span>
    <span>{pct:.1f}%</span>
  </div>
  <div style="background:#1e293b;height:6px;border-radius:3px;">
    <div style="width:{pct}%;height:6px;background:{'#38bdf8' if k==best else '#334155'};border-radius:3px;"></div>
  </div>
</div>'''
 
    return {"best": best, "confidence": conf, "gc": gc, "at": at, "length": length,
            "seq_preview": seq[:60], "bars": bars, "A": A, "T": T, "G": G, "C": C}
 
# ------------------------------
# HTML TEMPLATES
# ------------------------------
HOME = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>GenomeScope Pro — Advanced Genome Analysis</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0;}
:root{--bg:#020617;--surface:#0f172a;--surface2:#1e293b;--accent:#38bdf8;--accent2:#f87171;--accent3:#4ade80;--text:#e2e8f0;--muted:#64748b;}
body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:2rem;}
.wrap{width:100%;max-width:800px;}
.logo{font-family:'JetBrains Mono',monospace;font-size:2rem;font-weight:700;margin-bottom:.3rem;}
.logo span{color:var(--accent);}
.logo small{font-size:.7rem;color:var(--accent2);}
.sub{color:var(--muted);margin-bottom:2rem;}
textarea{width:100%;height:180px;background:var(--surface);border:1px solid var(--surface2);border-radius:12px;padding:14px;font-family:'JetBrains Mono',monospace;font-size:13px;color:var(--text);resize:vertical;}
textarea:focus{outline:none;border-color:var(--accent);}
.btn{margin-top:1rem;width:100%;padding:14px;border:none;border-radius:12px;background:linear-gradient(135deg,var(--accent),var(--accent2));color:#020617;font-weight:600;cursor:pointer;}
.btn:hover{transform:translateY(-2px);opacity:.9;}
.chip-group{display:flex;gap:8px;margin-top:12px;}
.chip{padding:6px 14px;background:var(--surface2);border-radius:20px;font-size:12px;cursor:pointer;border:1px solid transparent;color:var(--text);}
.chip:hover{border-color:var(--accent);color:var(--accent);}
.note{margin-top:1.5rem;padding:16px;background:var(--surface);border-radius:12px;border-left:3px solid var(--accent);}
.feature-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:12px;}
.feature{font-size:11px;color:var(--accent3);}
</style>
<script>
const examples={dna:"ATGAAAGCAATTTTCGTACTGAAAGGTTTTGTTGGTTTTTTTTCTTGTGGTTTTGTTGGAAGTTTCTATGGCTAACGAAATCATCAAAGAAATCGATGAAGGTGTTAAGAAAGCGACTGATGTTTTACGTGCTAAAGACTTCAATCAAGTTGGTAAAGCTACTGGTGATCGTCCTGGTGCATTTGAAGAAATTCGTAAAGCTA",prot:"MKALFILKLVVFFLVFLVAYMANDMIKEIDESGVKKATLMLRAKDFNQVGKATGDRPGAFEEIRKA"};
function fill(t){document.getElementById('seq').value=examples[t];}
</script>
</head>
<body>
<div class="wrap">
  <div class="logo">Genome<span>Scope</span><small> Pro</small></div>
  <p class="sub">BLAST + Probability Models + Phylogenetics + Classification</p>
  <form action="/analyze" method="POST">
    <textarea id="seq" name="sequence" placeholder="Paste DNA or protein sequence (FASTA format accepted)..."></textarea>
    <div class="chip-group">
      <button type="button" class="chip" onclick="fill('dna')">📊 Load DNA</button>
      <button type="button" class="chip" onclick="fill('prot')">🧬 Load Protein</button>
    </div>
    <button class="btn" type="submit">🔬 Analyze Genome</button>
  </form>
  <div class="note">
    <strong>🧪 Analysis Modules:</strong>
    <div class="feature-grid">
      <div class="feature">🔬 BLAST Search</div>
      <div class="feature">📊 k-mer Analysis</div>
      <div class="feature">🧬 Markov Models</div>
      <div class="feature">🌳 Phylogenetics</div>
    </div>
  </div>
</div>
</body>
</html>"""
 
RESULT_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Results — GenomeScope Pro</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0;}
:root{--bg:#020617;--surface:#0f172a;--surface2:#1e293b;--accent:#38bdf8;--accent2:#f87171;--accent3:#4ade80;--text:#e2e8f0;--muted:#64748b;}
body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;padding:2rem;}
.wrap{max-width:1200px;margin:0 auto;}
.logo{font-family:'JetBrains Mono',monospace;font-size:1.4rem;font-weight:700;margin-bottom:2rem;}
.logo span{color:var(--accent);}
.tabs{display:flex;gap:8px;margin-bottom:1.5rem;flex-wrap:wrap;}
.tab{padding:10px 24px;background:var(--surface);border-radius:10px;cursor:pointer;transition:all .2s;}
.tab.active,.tab:hover{background:var(--accent);color:#020617;}
.panel{display:none;animation:fadein .3s;}
.panel.active{display:block;}
@keyframes fadein{from{opacity:0}to{opacity:1}}
.card{background:var(--surface);border-radius:12px;padding:1.5rem;margin-bottom:1rem;}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:1.5rem;}
.stat{background:var(--surface2);padding:1rem;border-radius:10px;}
.stat-label{font-size:.7rem;text-transform:uppercase;color:var(--muted);}
.stat-val{font-size:1.5rem;font-weight:600;}
table{width:100%;border-collapse:collapse;}
th{text-align:left;padding:8px;color:var(--muted);border-bottom:1px solid var(--surface2);}
td{padding:8px;border-bottom:1px solid var(--surface2);}
.blast-spinner{display:flex;align-items:center;gap:12px;padding:24px;}
.spin{width:20px;height:20px;border:2px solid var(--surface2);border-top-color:var(--accent);border-radius:50%;animation:spin 1s linear infinite;}
@keyframes spin{to{transform:rotate(360deg)}}
.back{display:inline-block;margin-top:2rem;color:var(--muted);text-decoration:none;}
.back:hover{color:var(--accent);}
</style>
<script>
const JOB_ID = "{{ job_id }}";
function showTab(id){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.getElementById('tab-'+id).classList.add('active');
  document.getElementById('panel-'+id).classList.add('active');
}
function toggleBox(id){let b=document.getElementById(id);b.style.display=(b.style.display==='none'||!b.style.display)?'block':'none';}
function scrollToAlignment(id){showTab('blast');setTimeout(()=>{let el=document.getElementById(id);if(el)el.scrollIntoView({behavior:'smooth',block:'start'});},100);}
function pollBlast(){
  fetch('/blast_status/'+JOB_ID).then(r=>r.json()).then(data=>{
    if(data.status==='done'){renderBlast(data);}
    else if(data.status==='error'){document.getElementById('blast-content').innerHTML='<div style="padding:20px;color:#f87171;">❌ BLAST error: '+data.message+'</div>';}
    else{setTimeout(pollBlast,3000);}
  }).catch(()=>setTimeout(pollBlast,5000));
}
function renderBlast(d){
  let el=document.getElementById('blast-content');
  if(!d.table_rows){el.innerHTML=d.alignments_html||'<p>No alignments found.</p>';return;}
  el.innerHTML=`<p>Type: ${d.seq_type} | DB: ${d.db} | Length: ${d.seq_len} bp</p>
    <h3>Top ${d.num_hits} hits</h3>
    <div style="overflow-x:auto;"><table><thead><tr><th>Description</th><th>Score</th><th>Identity</th><th>E-value</th></tr></thead><tbody>${d.table_rows}</tbody></table></div>
    <h3 style="margin-top:1rem;">Coverage</h3><div>${d.graphics_html}</div>
    <h3 style="margin-top:1rem;">Alignments</h3>${d.alignments_html}`;
}
window.onload=function(){showTab('overview');pollBlast();};
</script>
</head>
<body>
<div class="wrap">
  <div class="logo">Genome<span>Scope</span> Pro</div>
  <div class="tabs">
    <div class="tab" id="tab-overview" onclick="showTab('overview')">📊 Overview</div>
    <div class="tab" id="tab-probability" onclick="showTab('probability')">📈 Probability Models</div>
    <div class="tab" id="tab-phylogeny" onclick="showTab('phylogeny')">🌳 Phylogenetics</div>
    <div class="tab" id="tab-blast" onclick="showTab('blast')">🔬 BLAST</div>
  </div>
 
  <div id="panel-overview" class="panel">
    <div class="stat-grid">
      <div class="stat"><div class="stat-label">Best match</div><div class="stat-val" style="font-size:1rem;color:#38bdf8;">{{ cl.best }}</div></div>
      <div class="stat"><div class="stat-label">Confidence</div><div class="stat-val" style="color:#fbbf24;">{{ "%.1f"|format(cl.confidence) }}%</div></div>
      <div class="stat"><div class="stat-label">GC content</div><div class="stat-val" style="color:#38bdf8;">{{ "%.1f"|format(cl.gc) }}%</div></div>
      <div class="stat"><div class="stat-label">Length</div><div class="stat-val" style="color:#f87171;">{{ cl.length }} bp</div></div>
    </div>
    <div class="card"><p style="margin-bottom:12px;">Classification scores</p>{{ cl.bars|safe }}</div>
    <div class="card"><span style="color:var(--accent);">preview</span> {{ cl.seq_preview }}…</div>
  </div>
 
  <div id="panel-probability" class="panel">
    <div class="card">
      <h3>🎲 Markov Models</h3>
      <div class="stat-grid">
        <div class="stat"><div class="stat-label">1st Order Prob</div><div class="stat-val" style="font-size:1rem;">{{ prob.markov.prob_order1 }}</div></div>
        <div class="stat"><div class="stat-label">2nd Order Prob</div><div class="stat-val" style="font-size:1rem;">{{ prob.markov.prob_order2 }}</div></div>
        <div class="stat"><div class="stat-label">Next Prediction</div><div class="stat-val" style="font-size:1rem;">{{ prob.markov.next_pred }}</div></div>
      </div>
    </div>
    <div class="card">
      <h3>🔤 k-mer Frequency Analysis</h3>
      {{ prob.kmers_html|safe }}
    </div>
    <div class="card">
      <h3>📊 Maximum Likelihood Classification</h3>
      <p><strong>Best match:</strong> {{ prob.mle_best }} ({{ prob.mle_conf }})</p>
      {{ prob.mle_html|safe }}
    </div>
    <div class="card">
      <h3>🔄 Similarity with Reference Genomes (k=3)</h3>
      {{ prob.similarity_html|safe }}
    </div>
  </div>
 
  <div id="panel-phylogeny" class="panel">
    <div class="card">
      <h3>🌳 Neighbor-Joining Tree</h3>
      {% if phylo.nj_image %}
      <img src="{{ phylo.nj_image }}" style="max-width:100%; border-radius:10px;">
      {% else %}
      <p>Tree could not be generated</p>
      {% endif %}
    </div>
    <div class="card">
      <h3>📏 Pairwise Genetic Distances</h3>
      {{ phylo.distances_html|safe }}
    </div>
  </div>
 
  <div id="panel-blast" class="panel">
    <div id="blast-content"><div class="blast-spinner"><div class="spin"></div><span>Running BLAST against NCBI...</span></div></div>
  </div>
 
  <a href="/" class="back">← New search</a>
</div>
</body>
</html>"""
 
# ------------------------------
# ROUTES
# ------------------------------
@app.route("/")
def home():
    return render_template_string(HOME)
 
@app.route("/analyze", methods=["POST"])
def analyze():
    raw = request.form.get("sequence", "")
    seq = clean_seq(raw)
 
    if len(seq) < 10:
        return render_template_string(HOME + "<script>alert('Sequence too short (min 10 characters)');</script>")
 
    cl = classify_seq(seq)
    prob_results = analyze_probabilities(seq)
    phylo_results = analyze_phylogenetics(seq)
 
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending"}
    t = threading.Thread(target=blast_worker, args=(job_id, seq), daemon=True)
    t.start()
 
    return render_template_string(RESULT_PAGE, cl=cl, prob=prob_results, phylo=phylo_results, job_id=job_id)
 
@app.route("/blast_status/<job_id>")
def blast_status(job_id):
    job = jobs.get(job_id, {"status": "error", "message": "Job not found"})
    return jsonify(job)
 
if __name__ == "__main__":
    app.run(debug=True, threaded=True)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
