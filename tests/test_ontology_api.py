"""
Quick test script for Ontology APIs

Tests basic functionality of each API to ensure everything works.
"""

import sys
from pathlib import Path

# Add project root to path so imports work from tests directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ontologies.api import (
    GeneOntologyAPI,
    ESCOAPI,
    CellOntologyAPI,
    MeSHAPI,
    OntologyManager
)


def test_gene_ontology():
    """Test GO API"""
    print("\n" + "="*60)
    print("Testing Gene Ontology API...")
    print("="*60)

    try:
        go = GeneOntologyAPI(use_basic=True)
        print(f"✓ Loaded {len(go.get_all_nodes())} GO terms")

        # Test search
        results = go.search_by_name("apoptosis")
        print(f"✓ Found {len(results)} results for 'apoptosis'")

        # Test RAG matching
        matches = go.match_term_to_nodes("cell death", top_k=3)
        print(f"✓ RAG matching found {len(matches)} matches")
        if matches:
            print(f"  Top match: {matches[0].node.name} ({matches[0].similarity_score:.3f})")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_esco():
    """Test ESCO API"""
    print("\n" + "="*60)
    print("Testing ESCO API...")
    print("="*60)

    try:
        esco = ESCOAPI()
        print(f"✓ Loaded {len(esco.get_all_nodes())} ESCO concepts")

        # Test skills
        skills = esco.get_skills()
        print(f"✓ Found {len(skills)} skills")

        # Test RAG matching
        matches = esco.match_term_to_nodes("python programming", top_k=3)
        print(f"✓ RAG matching found {len(matches)} matches")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_cell_ontology():
    """Test Cell Ontology API"""
    print("\n" + "="*60)
    print("Testing Cell Ontology API...")
    print("="*60)

    try:
        cell = CellOntologyAPI()
        print(f"✓ Loaded {len(cell.get_all_nodes())} cell types")

        # Test search
        results = cell.search_by_name("neuron")
        print(f"✓ Found {len(results)} results for 'neuron'")

        # Test RAG matching
        matches = cell.match_term_to_nodes("immune cell", top_k=3)
        print(f"✓ RAG matching found {len(matches)} matches")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_mesh():
    """Test MeSH API"""
    print("\n" + "="*60)
    print("Testing MeSH API...")
    print("="*60)

    try:
        mesh = MeSHAPI(use_rdf=False)
        print(f"✓ Loaded {len(mesh.get_all_nodes())} MeSH descriptors")

        # Test RAG matching
        matches = mesh.match_term_to_nodes("heart disease", top_k=3)
        print(f"✓ RAG matching found {len(matches)} matches")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_ontology_manager():
    """Test Ontology Manager"""
    print("\n" + "="*60)
    print("Testing Ontology Manager...")
    print("="*60)

    try:
        manager = OntologyManager()

        # Load ontologies
        manager.load_go(use_basic=True)
        manager.load_esco()

        # Test cross-ontology matching
        matches = manager.match_term_across_ontologies(
            "protein synthesis",
            top_k_per_ontology=2
        )
        print(f"✓ Cross-ontology matching found results in {len(matches)} ontologies")

        # Test best match
        best = manager.get_best_match_across_ontologies("cell division")
        if best:
            print(f"✓ Best match: {best.node.name} ({best.similarity_score:.3f})")

        # Test stats
        stats = manager.get_stats()
        print(f"✓ Loaded ontologies: {', '.join(stats['loaded_ontologies'])}")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("ONTOLOGY API TEST SUITE")
    print("="*60)

    results = {
        'Gene Ontology': test_gene_ontology(),
        'ESCO': test_esco(),
        'Cell Ontology': test_cell_ontology(),
        'MeSH': test_mesh(),
        'Ontology Manager': test_ontology_manager()
    }

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ {total - passed} test(s) failed")


if __name__ == "__main__":
    main()
