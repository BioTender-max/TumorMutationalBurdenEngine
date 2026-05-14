import numpy as np; np.random.seed(42)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── Simulate 200 tumor samples ──────────────────────────────────────────────
N = 200
cancer_types = ['LUAD', 'SKCM', 'COAD', 'BRCA', 'BLCA', 'HNSC', 'STAD', 'GBM']
sample_cancer = np.random.choice(cancer_types, N)

# Somatic mutation counts (SNV + indel) — log-normal distribution
snv_counts = np.random.lognormal(mean=4.5, sigma=1.2, size=N).astype(int) + 1
indel_counts = (snv_counts * np.random.uniform(0.05, 0.20, N)).astype(int) + 1
total_mutations = snv_counts + indel_counts

# TMB = mutations / Mb (exome ~38 Mb)
EXOME_MB = 38.0
tmb = total_mutations / EXOME_MB

# MSI scoring from repeat loci (25 loci)
n_loci = 25
msi_scores = np.zeros(N)
for i in range(N):
    base_instability = 0.05 if tmb[i] < 10 else 0.45
    locus_instability = np.random.beta(base_instability * 5 + 0.5, (1 - base_instability) * 5 + 0.5, n_loci)
    msi_scores[i] = locus_instability.mean()

msi_high = msi_scores > 0.3
msi_label = np.where(msi_high, 'MSI-H', 'MSS')

# Mutational signatures (SBS1/2/3/4/13)
sig_names = ['SBS1', 'SBS2', 'SBS3', 'SBS4', 'SBS13']
sig_weights_raw = np.random.dirichlet(alpha=[2, 1.5, 1, 1.5, 1], size=N)
sig_exposures = sig_weights_raw  # N x 5

# Immunotherapy response (TMB-high threshold = 10 mut/Mb)
tmb_high = tmb >= 10
response_prob = np.where(tmb_high, 0.45, 0.15)
response = np.random.binomial(1, response_prob)

# Mutation type spectrum (6 substitution types)
mut_types = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
mut_spectrum = np.random.dirichlet([1.5, 0.5, 3.0, 0.8, 2.0, 0.7]) * total_mutations.mean()

# ── Dashboard ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('Tumor Mutational Burden & MSI Analysis Dashboard', fontsize=18,
             color='white', fontweight='bold', y=0.98)

DARK = '#161b22'
ACCENT = '#58a6ff'
ACCENT2 = '#f78166'
ACCENT3 = '#3fb950'
TEXT = 'white'

def style_ax(ax, title):
    ax.set_facecolor(DARK)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=8)
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

# 1. TMB distribution
ax = axes[0, 0]
style_ax(ax, '1. TMB Distribution')
ax.hist(tmb, bins=40, color=ACCENT, edgecolor='#0d1117', alpha=0.85)
ax.axvline(10, color=ACCENT2, lw=2, ls='--', label='TMB-high (10 mut/Mb)')
ax.set_xlabel('TMB (mutations/Mb)', color=TEXT, fontsize=9)
ax.set_ylabel('Count', color=TEXT, fontsize=9)
ax.legend(fontsize=8, facecolor='#21262d', labelcolor=TEXT)
ax.text(0.97, 0.95, f'Median: {np.median(tmb):.1f}\nTMB-H: {tmb_high.sum()}',
        transform=ax.transAxes, ha='right', va='top', color=TEXT, fontsize=8,
        bbox=dict(boxstyle='round', facecolor='#21262d', alpha=0.8))

# 2. MSI score vs TMB
ax = axes[0, 1]
style_ax(ax, '2. MSI Score vs TMB')
colors_msi = [ACCENT2 if m else ACCENT for m in msi_high]
ax.scatter(tmb, msi_scores, c=colors_msi, alpha=0.6, s=20, edgecolors='none')
ax.axhline(0.3, color='yellow', lw=1.5, ls='--', label='MSI-H threshold')
ax.axvline(10, color=ACCENT2, lw=1.5, ls='--', alpha=0.7)
ax.set_xlabel('TMB (mut/Mb)', color=TEXT, fontsize=9)
ax.set_ylabel('MSI Score', color=TEXT, fontsize=9)
patches = [mpatches.Patch(color=ACCENT2, label=f'MSI-H (n={msi_high.sum()})'),
           mpatches.Patch(color=ACCENT, label=f'MSS (n={(~msi_high).sum()})')]
ax.legend(handles=patches, fontsize=8, facecolor='#21262d', labelcolor=TEXT)

# 3. Signature contribution pie
ax = axes[0, 2]
style_ax(ax, '3. Mutational Signature Contributions')
mean_exposures = sig_exposures.mean(axis=0)
colors_pie = ['#58a6ff', '#f78166', '#3fb950', '#d2a8ff', '#ffa657']
wedges, texts, autotexts = ax.pie(mean_exposures, labels=sig_names, autopct='%1.1f%%',
                                   colors=colors_pie, textprops={'color': TEXT, 'fontsize': 8},
                                   startangle=90)
for at in autotexts:
    at.set_color('white')
    at.set_fontsize(7)

# 4. Immunotherapy response by TMB
ax = axes[1, 0]
style_ax(ax, '4. Immunotherapy Response by TMB')
tmb_bins = [0, 5, 10, 20, 50, 200]
bin_labels = ['0-5', '5-10', '10-20', '20-50', '>50']
resp_rates = []
bin_ns = []
for i in range(len(tmb_bins) - 1):
    mask = (tmb >= tmb_bins[i]) & (tmb < tmb_bins[i+1])
    if mask.sum() > 0:
        resp_rates.append(response[mask].mean() * 100)
        bin_ns.append(mask.sum())
    else:
        resp_rates.append(0)
        bin_ns.append(0)
bar_colors = [ACCENT2 if r > 30 else ACCENT for r in resp_rates]
bars = ax.bar(bin_labels, resp_rates, color=bar_colors, edgecolor='#0d1117', alpha=0.85)
ax.set_xlabel('TMB Category (mut/Mb)', color=TEXT, fontsize=9)
ax.set_ylabel('Response Rate (%)', color=TEXT, fontsize=9)
ax.set_ylim(0, 70)
for bar, n in zip(bars, bin_ns):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'n={n}', ha='center', va='bottom', color=TEXT, fontsize=7)

# 5. Mutation type spectrum
ax = axes[1, 1]
style_ax(ax, '5. Mutation Type Spectrum')
spectrum_colors = ['#58a6ff', '#f78166', '#3fb950', '#d2a8ff', '#ffa657', '#79c0ff']
ax.bar(mut_types, mut_spectrum, color=spectrum_colors, edgecolor='#0d1117', alpha=0.85)
ax.set_xlabel('Substitution Type', color=TEXT, fontsize=9)
ax.set_ylabel('Mean Count', color=TEXT, fontsize=9)

# 6. TMB by cancer type
ax = axes[1, 2]
style_ax(ax, '6. TMB by Cancer Type')
ct_tmbs = {ct: tmb[sample_cancer == ct] for ct in cancer_types}
ct_medians = {ct: np.median(v) for ct, v in ct_tmbs.items() if len(v) > 0}
sorted_cts = sorted(ct_medians, key=ct_medians.get, reverse=True)
bp = ax.boxplot([ct_tmbs[ct] for ct in sorted_cts], labels=sorted_cts,
                patch_artist=True, medianprops=dict(color='white', lw=2))
for patch in bp['boxes']:
    patch.set_facecolor(ACCENT)
    patch.set_alpha(0.7)
for element in ['whiskers', 'caps', 'fliers']:
    for item in bp[element]:
        item.set_color('#8b949e')
ax.set_xlabel('Cancer Type', color=TEXT, fontsize=9)
ax.set_ylabel('TMB (mut/Mb)', color=TEXT, fontsize=9)
ax.tick_params(axis='x', rotation=45)

# 7. MSI locus instability
ax = axes[2, 0]
style_ax(ax, '7. MSI Locus Instability')
locus_names = [f'L{i+1}' for i in range(n_loci)]
# Compute per-locus instability for MSI-H vs MSS
msi_h_loci = []
mss_loci = []
for i in range(n_loci):
    base_h = np.random.beta(2.5, 1.5, msi_high.sum()).mean()
    base_s = np.random.beta(0.5, 4.5, (~msi_high).sum()).mean()
    msi_h_loci.append(base_h)
    mss_loci.append(base_s)
x = np.arange(n_loci)
ax.bar(x - 0.2, msi_h_loci, 0.4, label='MSI-H', color=ACCENT2, alpha=0.85)
ax.bar(x + 0.2, mss_loci, 0.4, label='MSS', color=ACCENT, alpha=0.85)
ax.set_xlabel('Locus', color=TEXT, fontsize=9)
ax.set_ylabel('Instability Score', color=TEXT, fontsize=9)
ax.set_xticks(x[::5])
ax.set_xticklabels([locus_names[i] for i in range(0, n_loci, 5)], fontsize=7)
ax.legend(fontsize=8, facecolor='#21262d', labelcolor=TEXT)

# 8. Signature correlation heatmap
ax = axes[2, 1]
style_ax(ax, '8. Signature Correlation Matrix')
corr_matrix = np.corrcoef(sig_exposures.T)
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels(sig_names, color=TEXT, fontsize=8)
ax.set_yticklabels(sig_names, color=TEXT, fontsize=8)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center',
                color='white', fontsize=7)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors=TEXT, labelsize=7)

# 9. Summary panel
ax = axes[2, 2]
style_ax(ax, '9. Analysis Summary')
ax.axis('off')
summary_lines = [
    ('Total Samples', f'{N}'),
    ('Median TMB', f'{np.median(tmb):.1f} mut/Mb'),
    ('Mean TMB', f'{np.mean(tmb):.1f} mut/Mb'),
    ('TMB-High (≥10)', f'{tmb_high.sum()} ({tmb_high.mean()*100:.1f}%)'),
    ('MSI-H Samples', f'{msi_high.sum()} ({msi_high.mean()*100:.1f}%)'),
    ('MSS Samples', f'{(~msi_high).sum()} ({(~msi_high).mean()*100:.1f}%)'),
    ('Overall Response Rate', f'{response.mean()*100:.1f}%'),
    ('TMB-H Response Rate', f'{response[tmb_high].mean()*100:.1f}%'),
    ('TMB-L Response Rate', f'{response[~tmb_high].mean()*100:.1f}%'),
    ('Dominant Signature', f'{sig_names[np.argmax(mean_exposures)]}'),
]
y_pos = 0.95
for label, value in summary_lines:
    ax.text(0.05, y_pos, label + ':', color='#8b949e', fontsize=9, transform=ax.transAxes)
    ax.text(0.65, y_pos, value, color=ACCENT3, fontsize=9, fontweight='bold', transform=ax.transAxes)
    y_pos -= 0.09

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('/mnt/shared-workspace/shared/tumor_mutational_burden_engine_dashboard.png',
            dpi=100, bbox_inches='tight', facecolor='#0d1117')
plt.close()

import shutil, os
try:
    shutil.copy(__file__, '/mnt/shared-workspace/shared/tumor_mutational_burden_engine.py')
except shutil.SameFileError:
    pass  # already in destination

# Key numerical results
print("=== TumorMutationalBurdenEngine Results ===")
print(f"N samples: {N}")
print(f"Median TMB: {np.median(tmb):.2f} mut/Mb")
print(f"Mean TMB: {np.mean(tmb):.2f} mut/Mb")
print(f"TMB-High (>=10): {tmb_high.sum()} ({tmb_high.mean()*100:.1f}%)")
print(f"MSI-H: {msi_high.sum()} ({msi_high.mean()*100:.1f}%)")
print(f"MSS: {(~msi_high).sum()} ({(~msi_high).mean()*100:.1f}%)")
print(f"Overall immunotherapy response: {response.mean()*100:.1f}%")
print(f"TMB-High response rate: {response[tmb_high].mean()*100:.1f}%")
print(f"TMB-Low response rate: {response[~tmb_high].mean()*100:.1f}%")
print(f"Mean signature exposures: {dict(zip(sig_names, mean_exposures.round(3)))}")
print(f"Dashboard saved: /mnt/shared-workspace/shared/tumor_mutational_burden_engine_dashboard.png")
