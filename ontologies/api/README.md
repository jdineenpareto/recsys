# Ontology API Documentation

Python APIs for accessing biomedical and skills ontologies with RAG-based term matching capabilities.

## Features

- **Node Navigation**: Retrieve nodes and traverse relationships
- **RAG-Based Matching**: Semantic similarity search using sentence transformers
- **Unified Interface**: Consistent API across all ontologies
- **Cross-Ontology Search**: Match terms across multiple ontologies simultaneously
- **Lazy Loading**: Load ontologies on-demand for efficiency

## Supported Ontologies

| Ontology | Description | Size | Format |
|----------|-------------|------|--------|
| **Gene Ontology (GO)** | Gene functions and biological processes | 30-186 MB | OBO |
| **ChEBI** | Chemical entities | 247-773 MB | OBO |
| **ESCO** | Skills, competences, occupations | 177 KB | RDF |
| **Cell Ontology (CL)** | Cell types | 16-60 MB | OBO |
| **MeSH** | Medical terminology | 130-216 MB | XML/RDF |

## Quick Start

### Basic Usage

```python
from ontologies.api import GeneOntologyAPI

# Load Gene Ontology
go = GeneOntologyAPI(use_basic=True)

# Get a node by ID
node = go.get_node("GO:0008150")
print(f"Name: {node.name}")
print(f"Definition: {node.definition}")

# Search by name
results = go.search_by_name("apoptosis")
for node in results:
    print(f"{node.name} ({node.id})")

# Get neighbors (relationships)
neighbors = go.get_neighbors("GO:0006915", relation_types=['is_a'])
for neighbor, rel_type in neighbors:
    print(f"{neighbor.name} [{rel_type}]")

# RAG-based term matching
matches = go.match_term_to_nodes("cell death", top_k=5)
for match in matches:
    print(f"{match.similarity_score:.3f} - {match.node.name}")
```

### Unified Ontology Manager

```python
from ontologies.api import OntologyManager

# Create manager
manager = OntologyManager()

# Load multiple ontologies
manager.load_go(use_basic=True)
manager.load_esco()
manager.load_cell_ontology()

# Match term across all loaded ontologies
matches = manager.match_term_across_ontologies(
    "protein synthesis",
    top_k_per_ontology=3,
    min_similarity=0.3
)

for ontology_name, ontology_matches in matches.items():
    print(f"\n{ontology_name}:")
    for match in ontology_matches:
        print(f"  {match.similarity_score:.3f} - {match.node.name}")

# Get best match across all ontologies
best = manager.get_best_match_across_ontologies("immune response")
print(f"Best: {best.node.name} (score: {best.similarity_score:.3f})")
```

## API Reference

### Base Classes

#### OntologyNode
Represents a node/term in an ontology.

**Attributes:**
- `id`: Unique identifier
- `name`: Term name/label
- `definition`: Text definition
- `synonyms`: List of alternative names
- `metadata`: Dictionary of additional properties

#### TermMatch
Represents a match between a query term and an ontology node.

**Attributes:**
- `node`: The matched OntologyNode
- `similarity_score`: Float between 0-1
- `match_type`: 'exact', 'synonym', or 'semantic'
- `matched_text`: The text that was matched

### Core Methods

All ontology APIs implement these methods:

#### `get_node(node_id: str) -> Optional[OntologyNode]`
Retrieve a node by its ID.

#### `get_neighbors(node_id, relation_types=None, direction='both') -> List[Tuple[OntologyNode, str]]`
Get neighboring nodes.

**Parameters:**
- `node_id`: ID of the source node
- `relation_types`: Filter by relation types (e.g., `['is_a', 'part_of']`)
- `direction`: 'outgoing', 'incoming', or 'both'

**Returns:** List of (node, relation_type) tuples

#### `search_by_name(name: str, exact=False) -> List[OntologyNode]`
Search for nodes by name.

#### `match_term_to_nodes(query_term, top_k=10, min_similarity=0.3) -> List[TermMatch]`
Match a term using RAG (semantic similarity).

**Parameters:**
- `query_term`: Term to match
- `top_k`: Number of top matches
- `min_similarity`: Minimum similarity threshold (0-1)

**Returns:** List of TermMatch objects sorted by similarity

#### `batch_match_terms(terms, top_k=5, min_similarity=0.3) -> Dict[str, List[TermMatch]]`
Match multiple terms efficiently.

#### `get_node_context(node_id, depth=1, relation_types=None) -> Dict`
Get a node with its neighborhood context.

**Parameters:**
- `node_id`: Central node ID
- `depth`: Number of hops to traverse
- `relation_types`: Filter by relation types

**Returns:** Dictionary with node and multi-level neighbors

## Ontology-Specific Features

### Gene Ontology (GO)

```python
from ontologies.api import GeneOntologyAPI

go = GeneOntologyAPI(use_basic=True)

# Get terms by namespace
bp_terms = go.get_terms_by_namespace('biological_process')
mf_terms = go.get_terms_by_namespace('molecular_function')
cc_terms = go.get_terms_by_namespace('cellular_component')

# Get ancestors (parents, grandparents, etc.)
ancestors = go.get_ancestors("GO:0006915")

# Get descendants (children, grandchildren, etc.)
descendants = go.get_descendants("GO:0008150")
```

### ChEBI

```python
from ontologies.api import ChEBIAPI

chebi = ChEBIAPI()

# Search by molecular formula
water = chebi.search_by_formula("H2O")

# Get compound roles
roles = chebi.get_compound_roles("CHEBI:15377")  # water

# Get parent/child compounds
parents = chebi.get_parent_compounds("CHEBI:15377")
children = chebi.get_child_compounds("CHEBI:15377")
```

### ESCO

```python
from ontologies.api import ESCOAPI

esco = ESCOAPI()

# Get all skills
skills = esco.get_skills()

# Get all occupations
occupations = esco.get_occupations()

# Get all qualifications
qualifications = esco.get_qualifications()

# Get broader/narrower concepts
broader = esco.get_broader_concepts("concept_id")
narrower = esco.get_narrower_concepts("concept_id")
```

### Cell Ontology

```python
from ontologies.api import CellOntologyAPI

cell = CellOntologyAPI()

# Get cell lineage (developmental ancestors)
lineage = cell.get_cell_lineage("CL:0000540")  # neuron

# Get derived cell types
derived = cell.get_derived_cell_types("CL:0000540")

# Get cell parts
parts = cell.get_cell_parts("CL:0000540")
```

### MeSH

```python
from ontologies.api import MeSHAPI

mesh = MeSHAPI(use_rdf=False)

# Get descriptor by tree number
desc = mesh.get_descriptor_by_tree_number("C01.539")

# Get broader/narrower terms
broader = mesh.get_broader_terms("D000001")
narrower = mesh.get_narrower_terms("D000001")
```

## Examples

See the `examples/` directory for comprehensive examples:

- **example_basic_usage.py**: Basic operations for each ontology
- **example_unified_manager.py**: Cross-ontology operations

Run examples:
```bash
cd ontologies/api/examples
python example_basic_usage.py
python example_unified_manager.py
```

## Use Cases

### 1. Skill Extraction Enhancement

```python
from ontologies.api import OntologyManager

manager = OntologyManager()
manager.load_esco()
manager.load_go(use_basic=True)

# Extract skills from resume
extracted_skills = ["Python", "machine learning", "data analysis"]

# Match to ontology concepts
for skill in extracted_skills:
    matches = manager.match_term_across_ontologies(skill, top_k=3)
    print(f"'{skill}' maps to:")
    for ont, ont_matches in matches.items():
        for match in ont_matches:
            print(f"  [{ont}] {match.node.name} ({match.similarity_score:.2f})")
```

### 2. Job Requirement Matching

```python
from ontologies.api import ESCOAPI

esco = ESCOAPI()

# Job requirements
requirements = [
    "Python programming",
    "Machine learning",
    "Team collaboration"
]

# Match requirements to ESCO skills
for req in requirements:
    matches = esco.match_term_to_nodes(req, top_k=5, min_similarity=0.4)
    print(f"\n'{req}':")
    for match in matches:
        print(f"  {match.similarity_score:.3f} - {match.node.name}")
        if match.node.definition:
            print(f"    {match.node.definition[:100]}...")
```

### 3. Biomedical Text Annotation

```python
from ontologies.api import OntologyManager

manager = OntologyManager()
manager.load_go(use_basic=True)
manager.load_cell_ontology()
manager.load_mesh()

# Terms extracted from research paper
terms = ["T cell activation", "apoptosis", "inflammation"]

# Annotate with ontology IDs
annotations = manager.batch_match_across_ontologies(
    terms,
    top_k_per_ontology=2,
    min_similarity=0.5
)

for term, matches in annotations.items():
    print(f"\n'{term}':")
    for ontology, ont_matches in matches.items():
        for match in ont_matches:
            print(f"  {ontology}:{match.node.id} - {match.node.name}")
```

## Performance Considerations

### Memory Usage
- **GO (basic)**: ~500 MB loaded
- **ChEBI**: ~2-3 GB loaded (large!)
- **ESCO**: ~50 MB loaded
- **Cell Ontology**: ~300 MB loaded
- **MeSH**: ~500 MB loaded

### Loading Times
- GO: 2-5 seconds
- ChEBI: 30-60 seconds (large!)
- ESCO: 1-2 seconds
- Cell Ontology: 3-5 seconds
- MeSH: 5-10 seconds

### Optimization Tips

1. **Use `go-basic.obo` instead of full GO**: Smaller, faster
2. **Load ontologies selectively**: Only load what you need
3. **Use lazy loading**: OntologyManager loads on-demand
4. **Cache embeddings**: First RAG query is slow, subsequent ones are fast
5. **Batch operations**: Use `batch_match_terms()` for multiple queries

```python
# Good: Load only what you need
manager = OntologyManager()
manager.load_go(use_basic=True)  # Smaller version
manager.load_esco()

# Avoid: Loading everything unnecessarily
manager.load_all()  # Loads ChEBI too (slow!)

# Good: Skip large ontologies
manager.load_all(skip_large=True)  # Skips ChEBI
```

## Dependencies

```bash
pip install sentence-transformers numpy
```

Or use the project's pyproject.toml:
```bash
uv sync
```

## Troubleshooting

### "File not found" errors
- Ensure ontologies are downloaded in `ontologies/` directory
- Check paths in API constructors
- See `ontologies/README.md` for download instructions

### Out of memory errors
- Skip ChEBI if memory constrained (it's huge)
- Use `go-basic.obo` instead of full `go.obo`
- Process terms in smaller batches

### Slow first query
- First RAG query computes embeddings (slow)
- Subsequent queries use cached embeddings (fast)
- This is expected behavior

## Architecture

```
ontologies/api/
├── base_ontology.py      # Base API class
├── go_api.py             # Gene Ontology
├── chebi_api.py          # ChEBI
├── esco_api.py           # ESCO
├── cell_ontology_api.py  # Cell Ontology
├── mesh_api.py           # MeSH
├── ontology_manager.py   # Unified manager
├── __init__.py           # Package exports
├── examples/             # Usage examples
│   ├── example_basic_usage.py
│   └── example_unified_manager.py
└── README.md             # This file
```

## Contributing

To add a new ontology:

1. Create `new_ontology_api.py` extending `BaseOntologyAPI`
2. Implement required methods: `load_ontology()`, `get_node()`, `get_neighbors()`, `search_by_name()`, `get_all_nodes()`
3. Add to `ontology_manager.py`
4. Update `__init__.py` exports
5. Add examples

## License

See individual ontology licenses in `ontologies/*/README.md` files.
