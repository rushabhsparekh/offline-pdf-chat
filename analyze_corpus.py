import pdfplumber
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import json
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

# Your document categories and paths
CORPUS = {
    "arxiv": [
        "resources/test_docs/arxiv/01_arxiv.pdf",
        "resources/test_docs/arxiv/02_arxiv.pdf",
        "resources/test_docs/arxiv/03_arxiv.pdf",
    ],
    "wiki": [
        "resources/test_docs/wiki/01_wiki.pdf",
        "resources/test_docs/wiki/02_wiki.pdf",
        "resources/test_docs/wiki/03_wiki.pdf",
        "resources/test_docs/wiki/04_wiki.pdf",
    ],
    "research": [
        "resources/test_docs/research/01_research.pdf",
        "resources/test_docs/research/02_research.pdf",
        "resources/test_docs/research/03_research.pdf",
        "resources/test_docs/research/04_research.pdf",
    ],
    "legal": [
        "resources/test_docs/legal/01_legal.pdf",
        "resources/test_docs/legal/02_legal.pdf",
        "resources/test_docs/legal/03_legal.pdf",
    ],
    "finance": [
        "resources/test_docs/finance/01_finance.pdf",
        "resources/test_docs/finance/02_finance.pdf",
        "resources/test_docs/finance/03_finance.pdf",
        "resources/test_docs/finance/04_finance.pdf",
    ]
}

OUTPUT_DIR = "resources/analysis"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────
# SIGNAL EXTRACTION
# ─────────────────────────────────────────

def extract_signals(pdf_path, max_pages=30):
    """
    Extracts all statistical signals from a PDF.
    Caps at max_pages for large documents — 
    first 30 pages representative enough.
    
    Returns dict with all raw signal arrays.
    """
    signals = {
        "gaps": [],           # vertical gaps between lines
        "font_sizes": [],     # font size of every character
        "x_positions": [],    # x start position of every line
        "line_lengths": [],   # character count of every line
        "page_count": 0,
        "has_tables": False,
        "path": pdf_path
    }
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            signals["page_count"] = len(pdf.pages)
            pages_to_scan = min(max_pages, len(pdf.pages))
            
            for page_num in range(pages_to_scan):
                page = pdf.pages[page_num]
                
                # Check for tables
                tables = page.extract_tables()
                if tables:
                    signals["has_tables"] = True
                
                # Extract character-level font sizes
                for char in page.chars:
                    size = round(char['size'], 1)
                    if 4 < size < 72:
                        signals["font_sizes"].append(size)
                
                # Extract word-level positions
                words = page.extract_words(
                    keep_blank_chars=False,
                    extra_attrs=["size"]
                )
                if not words:
                    continue
                
                # Group words into lines
                lines = {}
                for word in words:
                    y_key = round(word['top'] / 2) * 2
                    if y_key not in lines:
                        lines[y_key] = {
                            "top": word['top'],
                            "x_start": word['x0'],
                            "text": "",
                            "word_count": 0
                        }
                    lines[y_key]["text"] += word['text'] + " "
                    lines[y_key]["word_count"] += 1
                    # Track leftmost x position of line
                    lines[y_key]["x_start"] = min(
                        lines[y_key]["x_start"], 
                        word['x0']
                    )
                
                sorted_lines = sorted(lines.items())
                
                for i, (y_pos, line_data) in enumerate(sorted_lines):
                    
                    # X position signal
                    signals["x_positions"].append(
                        round(line_data["x_start"], 1)
                    )
                    
                    # Line length signal
                    signals["line_lengths"].append(
                        len(line_data["text"].strip())
                    )
                    
                    # Gap signal
                    if i > 0:
                        prev_y = sorted_lines[i-1][1]["top"]
                        gap = round(line_data["top"] - prev_y, 1)
                        if 3 < gap < 150:
                            signals["gaps"].append(gap)
    
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    
    return signals


# ─────────────────────────────────────────
# SINGLE DOCUMENT VISUALIZATION
# ─────────────────────────────────────────

def plot_document_analysis(signals, doc_name, save_path):
    """
    Creates a 2x2 grid of plots for one document.
    Each plot shows one signal distribution.
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Document Analysis: {doc_name}", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    # ── Plot 1: Gap Distribution ──
    ax1 = fig.add_subplot(gs[0, 0])
    if signals["gaps"]:
        gaps = np.array(signals["gaps"])
        ax1.hist(gaps, bins=50, color='steelblue', 
                 edgecolor='white', alpha=0.8)
        ax1.axvline(np.percentile(gaps, 50), color='red', 
                    linestyle='--', linewidth=1.5,
                    label=f'Median: {np.percentile(gaps, 50):.1f}')
        ax1.axvline(np.percentile(gaps, 75), color='orange',
                    linestyle='--', linewidth=1.5,
                    label=f'P75: {np.percentile(gaps, 75):.1f}')
        ax1.axvline(np.percentile(gaps, 90), color='green',
                    linestyle='--', linewidth=1.5,
                    label=f'P90: {np.percentile(gaps, 90):.1f}')
        ax1.set_xlabel("Gap size (pixels)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Line Gap Distribution\n(red=para boundary, green=section boundary)")
        ax1.legend(fontsize=8)
        ax1.set_xlim(0, min(100, max(gaps)))
    
    # ── Plot 2: Font Size Distribution ──
    ax2 = fig.add_subplot(gs[0, 1])
    if signals["font_sizes"]:
        fonts = np.array(signals["font_sizes"])
        unique_sizes = sorted(set(round(f) for f in fonts))
        counts = [sum(1 for f in fonts if round(f) == s) 
                  for s in unique_sizes]
        
        bars = ax2.bar(unique_sizes, counts, 
                       color='coral', edgecolor='white', alpha=0.8)
        
        # Highlight the most common size (body text)
        max_idx = counts.index(max(counts))
        bars[max_idx].set_color('darkred')
        bars[max_idx].set_label(f'Body: {unique_sizes[max_idx]}pt')
        
        ax2.set_xlabel("Font size (points)")
        ax2.set_ylabel("Character count")
        ax2.set_title("Font Size Distribution\n(dark red = body text)")
        ax2.legend(fontsize=8)
    
    # ── Plot 3: X-Position Distribution ──
    ax3 = fig.add_subplot(gs[1, 0])
    if signals["x_positions"]:
        x_pos = np.array(signals["x_positions"])
        ax3.hist(x_pos, bins=60, color='mediumseagreen',
                 edgecolor='white', alpha=0.8)
        ax3.set_xlabel("X position (pixels from left)")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Line Start X-Position Distribution\n(peaks = column left margins)")
        
        # Mark obvious peaks
        ax3.axvline(np.percentile(x_pos, 10), color='red',
                    linestyle='--', linewidth=1,
                    label=f'P10: {np.percentile(x_pos, 10):.0f}px')
        ax3.legend(fontsize=8)
    
    # ── Plot 4: Line Length Distribution ──
    ax4 = fig.add_subplot(gs[1, 1])
    if signals["line_lengths"]:
        lengths = np.array(signals["line_lengths"])
        ax4.hist(lengths, bins=40, color='mediumpurple',
                 edgecolor='white', alpha=0.8)
        ax4.axvline(np.median(lengths), color='red',
                    linestyle='--', linewidth=1.5,
                    label=f'Median: {np.median(lengths):.0f} chars')
        ax4.set_xlabel("Line length (characters)")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Line Length Distribution\n(short lines = headers/bullets/captions)")
        ax4.legend(fontsize=8)
    
    # Add document stats as text
    stats_text = (
        f"Pages: {signals['page_count']}  |  "
        f"Lines analyzed: {len(signals['gaps'])}  |  "
        f"Has tables: {signals['has_tables']}"
    )
    fig.text(0.5, 0.01, stats_text, ha='center', 
             fontsize=9, color='gray')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────
# CATEGORY COMPARISON VISUALIZATION  
# ─────────────────────────────────────────

def plot_category_comparison(category_signals, save_path):
    """
    Overlays gap distributions for all documents
    in a category on one plot.
    Shows how consistent or varied a category is.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Category Comparison — Gap | Font | X-Position",
                 fontsize=13, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(category_signals)))
    
    for i, (doc_name, signals) in enumerate(category_signals.items()):
        color = colors[i]
        label = Path(doc_name).stem
        
        # Gap overlay
        if signals["gaps"]:
            gaps = np.array(signals["gaps"])
            axes[0].hist(gaps, bins=40, alpha=0.4,
                        color=color, label=label,
                        density=True)
        
        # Font overlay
        if signals["font_sizes"]:
            fonts = np.array(signals["font_sizes"])
            axes[1].hist(fonts, bins=30, alpha=0.4,
                        color=color, label=label,
                        density=True)
        
        # X-position overlay
        if signals["x_positions"]:
            x_pos = np.array(signals["x_positions"])
            axes[2].hist(x_pos, bins=40, alpha=0.4,
                        color=color, label=label,
                        density=True)
    
    axes[0].set_title("Gap Distribution")
    axes[0].set_xlabel("Gap (pixels)")
    axes[0].set_xlim(0, 80)
    axes[0].legend(fontsize=7)
    
    axes[1].set_title("Font Size Distribution")
    axes[1].set_xlabel("Font size (points)")
    axes[1].legend(fontsize=7)
    
    axes[2].set_title("X-Position Distribution")
    axes[2].set_xlabel("X position (pixels)")
    axes[2].legend(fontsize=7)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────
# CUMULATIVE CORPUS HEATMAP
# ─────────────────────────────────────────

def plot_corpus_heatmap(all_signals, save_path):
    """
    Shows key statistics for every document as a heatmap.
    Makes cross-document comparison easy at a glance.
    """
    doc_names = []
    metrics = {
        "Median gap": [],
        "P90 gap": [],
        "Font variety": [],
        "Median font": [],
        "X-spread": [],
        "Median line len": [],
    }
    
    for doc_path, signals in all_signals.items():
        doc_names.append(Path(doc_path).stem[:15])
        
        gaps = np.array(signals["gaps"]) if signals["gaps"] else np.array([0])
        fonts = np.array(signals["font_sizes"]) if signals["font_sizes"] else np.array([0])
        x_pos = np.array(signals["x_positions"]) if signals["x_positions"] else np.array([0])
        lengths = np.array(signals["line_lengths"]) if signals["line_lengths"] else np.array([0])
        
        metrics["Median gap"].append(round(float(np.median(gaps)), 1))
        metrics["P90 gap"].append(round(float(np.percentile(gaps, 90)), 1))
        metrics["Font variety"].append(len(set(round(f) for f in fonts)))
        metrics["Median font"].append(round(float(np.median(fonts)), 1))
        metrics["X-spread"].append(round(float(np.std(x_pos)), 1))
        metrics["Median line len"].append(round(float(np.median(lengths)), 1))
    
    # Normalize each metric 0-1 for heatmap
    matrix = []
    for metric_name, values in metrics.items():
        arr = np.array(values, dtype=float)
        if arr.max() > arr.min():
            normalized = (arr - arr.min()) / (arr.max() - arr.min())
        else:
            normalized = np.zeros_like(arr)
        matrix.append(normalized)
    
    matrix = np.array(matrix)
    
    fig, ax = plt.subplots(figsize=(max(12, len(doc_names)), 6))
    
    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=doc_names,
        yticklabels=list(metrics.keys()),
        cmap="YlOrRd",
        annot=False,
        linewidths=0.5,
        linecolor='white'
    )
    
    ax.set_title("Corpus Heatmap — Normalized Signal Intensity\n"
                 "(darker = higher relative value)",
                 fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=9)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────
# MAIN — RUN EVERYTHING
# ─────────────────────────────────────────

if __name__ == "__main__":
    
    all_signals = {}
    
    print("="*60)
    print("Extracting signals from all documents...")
    print("="*60)
    
    # Process each category
    for category, pdf_paths in CORPUS.items():
        
        print(f"\nCategory: {category.upper()}")
        category_signals = {}
        
        category_dir = f"{OUTPUT_DIR}/{category}"
        Path(category_dir).mkdir(parents=True, exist_ok=True)
        
        for pdf_path in pdf_paths:
            
            if not Path(pdf_path).exists():
                print(f"  Skipping (not found): {pdf_path}")
                continue
            
            doc_name = Path(pdf_path).stem
            print(f"  Analyzing: {doc_name}...", end=" ")
            
            signals = extract_signals(pdf_path)
            
            if not signals["gaps"]:
                print("FAILED — no signals extracted")
                continue
            
            print(f"OK ({signals['page_count']} pages, "
                  f"{len(signals['gaps'])} gaps)")
            
            # Individual document plot
            plot_path = f"{category_dir}/{doc_name}_analysis.png"
            plot_document_analysis(signals, doc_name, plot_path)
            
            category_signals[pdf_path] = signals
            all_signals[pdf_path] = signals
        
        # Category comparison plot
        if len(category_signals) > 1:
            comparison_path = f"{category_dir}/_category_comparison.png"
            plot_category_comparison(category_signals, comparison_path)
    
    # Corpus-wide heatmap
    if all_signals:
        print(f"\nGenerating corpus heatmap...")
        heatmap_path = f"{OUTPUT_DIR}/_corpus_heatmap.png"
        plot_corpus_heatmap(all_signals, heatmap_path)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("CORPUS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Document':<20} {'Pages':>5} {'Gaps':>6} "
          f"{'Med gap':>8} {'P90 gap':>8} "
          f"{'Fonts':>6} {'X-std':>7}")
    print("-" * 60)
    
    for doc_path, signals in all_signals.items():
        doc_name = Path(doc_path).stem[:18]
        gaps = np.array(signals["gaps"]) if signals["gaps"] else np.array([0])
        fonts = np.array(signals["font_sizes"]) if signals["font_sizes"] else np.array([0])
        x_pos = np.array(signals["x_positions"]) if signals["x_positions"] else np.array([0])
        
        print(f"{doc_name:<20} "
              f"{signals['page_count']:>5} "
              f"{len(signals['gaps']):>6} "
              f"{np.median(gaps):>8.1f} "
              f"{np.percentile(gaps, 90):>8.1f} "
              f"{len(set(round(f) for f in fonts)):>6} "
              f"{np.std(x_pos):>7.1f}")
    
    print(f"\nAll plots saved to: {OUTPUT_DIR}/")
    print("Open the PNG files to analyze visually.")
