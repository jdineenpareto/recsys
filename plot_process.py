"""
Visualize the entire skill extraction and matching pipeline.
Creates a static vector graphic (SVG) showing all stages from resume to final score.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import argparse


def create_pipeline_visualization(output_file='pipeline.svg'):
    """Create comprehensive pipeline visualization"""

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Title
    ax.text(8, 11.5, 'Skill Extraction & Matching Pipeline',
            fontsize=20, fontweight='bold', ha='center')

    # Colors
    color_input = '#E8F4F8'
    color_extract = '#B8E6F0'
    color_verify = '#7ECCE8'
    color_rag = '#4AB3D4'
    color_llm = '#2980B9'
    color_graph = '#1A5276'
    color_output = '#16A085'

    # Stage 1: Input (Resume)
    stage1_y = 9.5
    box1 = FancyBboxPatch((0.5, stage1_y), 2, 1.2,
                          boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor=color_input, linewidth=2)
    ax.add_patch(box1)
    ax.text(1.5, stage1_y + 0.6, 'Resume', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.5, stage1_y + 0.2, 'resume.txt', fontsize=9, ha='center', style='italic')

    # Arrow to Stage 2
    arrow1 = FancyArrowPatch((2.5, stage1_y + 0.6), (3.5, stage1_y + 0.6),
                            arrowstyle='->', lw=2, color='black',
                            mutation_scale=20)
    ax.add_artist(arrow1)

    # Stage 2: Skill Extraction (extract.py)
    box2 = FancyBboxPatch((3.5, stage1_y - 0.3), 2.5, 1.8,
                          boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor=color_extract, linewidth=2)
    ax.add_patch(box2)
    ax.text(4.75, stage1_y + 1.1, 'Skill Extraction', fontsize=12, fontweight='bold', ha='center')
    ax.text(4.75, stage1_y + 0.7, 'extract.py', fontsize=9, ha='center', style='italic')
    ax.text(4.75, stage1_y + 0.4, '• Iterative LLM prompting', fontsize=8, ha='center')
    ax.text(4.75, stage1_y + 0.1, '• Adaptive masking', fontsize=8, ha='center')
    ax.text(4.75, stage1_y - 0.2, 'N skills extracted', fontsize=8, ha='center', fontweight='bold')

    # Arrow to Stage 3
    arrow2 = FancyArrowPatch((6.0, stage1_y + 0.6), (7.0, stage1_y + 0.6),
                            arrowstyle='->', lw=2, color='black',
                            mutation_scale=20)
    ax.add_artist(arrow2)
    ax.text(6.5, stage1_y + 0.9, 'optional', fontsize=7, ha='center', style='italic', color='gray')

    # Stage 3: Evidence Verification (optional LLM gate in extract.py)
    box3 = FancyBboxPatch((7.0, stage1_y - 0.3), 2.5, 1.8,
                          boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor=color_verify, linewidth=2)
    ax.add_patch(box3)
    ax.text(8.25, stage1_y + 1.1, 'Evidence Filter', fontsize=12, fontweight='bold', ha='center')
    ax.text(8.25, stage1_y + 0.7, '--verify-evidence', fontsize=9, ha='center', style='italic')
    ax.text(8.25, stage1_y + 0.4, '• LLM verification gate', fontsize=8, ha='center')
    ax.text(8.25, stage1_y + 0.1, '• Direct vs inferred', fontsize=8, ha='center')
    ax.text(8.25, stage1_y - 0.2, 'extracted_skills.json', fontsize=8, ha='center', fontweight='bold')

    # Arrow to Stage 4
    arrow3 = FancyArrowPatch((8.25, stage1_y - 0.4), (8.25, stage1_y - 1.5),
                            arrowstyle='->', lw=2, color='black',
                            mutation_scale=20)
    ax.add_artist(arrow3)

    # Stage 4: RAG Matching (taskmatch.py)
    stage4_y = 6.5
    box4 = FancyBboxPatch((6.5, stage4_y), 3.5, 1.5,
                          boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor=color_rag, linewidth=2)
    ax.add_patch(box4)
    ax.text(8.25, stage4_y + 1.1, 'RAG Matching', fontsize=12, fontweight='bold', ha='center')
    ax.text(8.25, stage4_y + 0.8, 'taskmatch.py', fontsize=9, ha='center', style='italic')
    ax.text(8.25, stage4_y + 0.5, '• Semantic embeddings (sentence-transformers)', fontsize=8, ha='center')
    ax.text(8.25, stage4_y + 0.2, '• Cosine similarity → [0,1]', fontsize=8, ha='center')

    # Input nodes count (static text)
    ax.text(8.25, stage4_y - 0.1, 'M input nodes × N skills',
            fontsize=8, ha='center', fontweight='bold')

    # Ontology enhancement (side branch)
    box_ontology = FancyBboxPatch((10.5, stage4_y + 0.2), 2.5, 1.1,
                                  boxstyle="round,pad=0.05",
                                  edgecolor='gray', facecolor='#F0F0F0',
                                  linewidth=1.5, linestyle='dashed')
    ax.add_patch(box_ontology)
    ax.text(11.75, stage4_y + 1.0, 'Ontology Fuzzy', fontsize=10, fontweight='bold', ha='center')
    ax.text(11.75, stage4_y + 0.7, '(optional)', fontsize=8, ha='center', style='italic')
    ax.text(11.75, stage4_y + 0.4, '--use-ontologies', fontsize=7, ha='center')

    # Arrow from RAG to Ontology
    arrow_ont = FancyArrowPatch((10.0, stage4_y + 0.75), (10.5, stage4_y + 0.75),
                               arrowstyle='<->', lw=1.5, color='gray',
                               linestyle='dashed', mutation_scale=15)
    ax.add_artist(arrow_ont)

    # Arrow to Stage 5
    arrow4 = FancyArrowPatch((8.25, stage4_y), (8.25, stage4_y - 1.0),
                            arrowstyle='->', lw=2, color='black',
                            mutation_scale=20)
    ax.add_artist(arrow4)
    ax.text(8.7, stage4_y - 0.5, 'similarity', fontsize=7, ha='left', style='italic')
    ax.text(8.7, stage4_y - 0.7, 'scores', fontsize=7, ha='left', style='italic')

    # Stage 5: LLM Homonym Filter
    stage5_y = 4.5
    box5 = FancyBboxPatch((6.5, stage5_y), 3.5, 1.3,
                          boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor=color_llm, linewidth=2)
    ax.add_patch(box5)
    ax.text(8.25, stage5_y + 0.9, 'Homonym Filter', fontsize=12, fontweight='bold', ha='center', color='white')
    ax.text(8.25, stage5_y + 0.6, '--use-llm-gate', fontsize=9, ha='center', style='italic', color='white')
    ax.text(8.25, stage5_y + 0.3, '• LLM verification (synonym vs homonym)', fontsize=8, ha='center', color='white')
    ax.text(8.25, stage5_y + 0.0, '• Early stopping on confirmed match', fontsize=8, ha='center', color='white')

    # Arrow to Stage 6
    arrow5 = FancyArrowPatch((8.25, stage5_y), (8.25, stage5_y - 1.0),
                            arrowstyle='->', lw=2, color='black',
                            mutation_scale=20)
    ax.add_artist(arrow5)
    ax.text(8.7, stage5_y - 0.5, 'filtered', fontsize=7, ha='left', style='italic')
    ax.text(8.7, stage5_y - 0.7, 'scores', fontsize=7, ha='left', style='italic')

    # Stage 6: Dependency Graph Evaluation
    stage6_y = 1.5
    box6 = FancyBboxPatch((6.0, stage6_y), 4.5, 1.8,
                          boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor=color_graph, linewidth=2)
    ax.add_patch(box6)
    ax.text(8.25, stage6_y + 1.4, 'Dependency Graph', fontsize=12, fontweight='bold', ha='center', color='white')
    ax.text(8.25, stage6_y + 1.1, 'TASK_SKILL_GRAPH', fontsize=9, ha='center', style='italic', color='white')
    ax.text(8.25, stage6_y + 0.8, '• MIN (AND gate) - both required', fontsize=8, ha='center', color='white')
    ax.text(8.25, stage6_y + 0.5, '• MAX (OR gate) - either sufficient', fontsize=8, ha='center', color='white')
    ax.text(8.25, stage6_y + 0.2, '• AVG - weighted combination', fontsize=8, ha='center', color='white')

    # Graph structure visualization (small)
    graph_x = 11.5
    graph_y = stage6_y + 0.9
    circle_radius = 0.15

    # Root node
    root_circle = Circle((graph_x, graph_y), circle_radius, color='#E74C3C', ec='white', linewidth=1.5, zorder=10)
    ax.add_patch(root_circle)
    ax.text(graph_x, graph_y, 'root', fontsize=6, ha='center', va='center', color='white', fontweight='bold')

    # Child nodes
    for i, offset_x in enumerate([-0.4, 0, 0.4]):
        child_circle = Circle((graph_x + offset_x, graph_y - 0.5), circle_radius * 0.8,
                             color='#3498DB', ec='white', linewidth=1, zorder=10)
        ax.add_patch(child_circle)
        # Arrow from root to child
        ax.plot([graph_x, graph_x + offset_x], [graph_y - circle_radius, graph_y - 0.5 + circle_radius * 0.8],
               'w-', linewidth=1, alpha=0.7)

    # Arrow to Stage 7
    arrow6 = FancyArrowPatch((8.25, stage6_y), (8.25, stage6_y - 0.8),
                            arrowstyle='->', lw=2, color='black',
                            mutation_scale=20)
    ax.add_artist(arrow6)

    # Stage 7: Final Output
    stage7_y = 0.2
    box7 = FancyBboxPatch((6.5, stage7_y), 3.5, 0.8,
                          boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor=color_output, linewidth=2)
    ax.add_patch(box7)
    ax.text(8.25, stage7_y + 0.5, 'Compatibility Score [0, 1]', fontsize=12, fontweight='bold',
            ha='center', color='white')
    ax.text(8.25, stage7_y + 0.15, 'task_skill_matches.json', fontsize=9, ha='center',
            style='italic', color='white')

    # Left side: Data flow legend
    legend_x = 0.3
    legend_y = 6.5
    ax.text(legend_x, legend_y + 1.5, 'Data Files:', fontsize=10, fontweight='bold')
    ax.text(legend_x, legend_y + 1.1, '→ resume.txt', fontsize=8)
    ax.text(legend_x, legend_y + 0.8, '→ extracted_skills.json', fontsize=8)
    ax.text(legend_x, legend_y + 0.5, '→ sorted_skills.json', fontsize=8, color='gray')
    ax.text(legend_x + 0.1, legend_y + 0.3, '(optional from sort.py)', fontsize=7, style='italic', color='gray')
    ax.text(legend_x, legend_y, '→ task_skill_matches.json', fontsize=8)

    # Right side: Key features
    key_x = 13.5
    key_y = 9.5
    ax.text(key_x, key_y + 0.8, 'Key Features:', fontsize=10, fontweight='bold')

    # Feature boxes
    features = [
        ('GPU Acceleration', 'CUDA/PyTorch'),
        ('FAISS Search', 'Ontology APIs'),
        ('Embedding Cache', 'Disk persistence'),
        ('Match Threshold', 'Global filtering'),
    ]

    for i, (feature, detail) in enumerate(features):
        y_pos = key_y + 0.4 - i * 0.35
        feature_box = FancyBboxPatch((key_x - 0.2, y_pos - 0.12), 2.0, 0.22,
                                    boxstyle="round,pad=0.03",
                                    edgecolor='gray', facecolor='#ECF0F1', linewidth=1)
        ax.add_patch(feature_box)
        ax.text(key_x + 0.8, y_pos + 0.03, feature, fontsize=8, ha='center', fontweight='bold')
        ax.text(key_x + 0.8, y_pos - 0.05, detail, fontsize=6, ha='center', style='italic', color='gray')

    # Bottom: Score interpretation
    score_y = 0.1
    score_text = 'Score: 0.7+ Excellent | 0.5-0.7 Good | 0.3-0.5 Moderate | 0.15-0.3 Partial | <0.15 Low'
    ax.text(8, score_y, score_text, fontsize=8, ha='center', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8F9F9', edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(output_file, format='svg', dpi=300, bbox_inches='tight')
    print(f"Pipeline visualization saved to {output_file}")

    # Also save as PNG for convenience
    png_file = output_file.replace('.svg', '.png')
    plt.savefig(png_file, format='png', dpi=300, bbox_inches='tight')
    print(f"PNG version saved to {png_file}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize skill extraction and matching pipeline')
    parser.add_argument('--output', type=str, default='pipeline.svg',
                       help='Output file path (default: pipeline.svg)')

    args = parser.parse_args()

    print("Creating static pipeline visualization...")
    create_pipeline_visualization(args.output)


if __name__ == '__main__':
    main()