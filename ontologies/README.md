# Ontologies Collection

This directory contains various biomedical and skills ontologies for use in the escoSkill2 project.

## Download Date
2025-10-13

## Directory Structure

```
ontologies/
├── gene_ontology/     # Gene Ontology (GO)
├── chebi/             # Chemical Entities of Biological Interest
├── esco/              # European Skills/Competences/Qualifications/Occupations
├── cell_ontology/     # Cell Ontology (CL)
├── snomed/            # SNOMED CT (requires separate download)
├── mesh/              # Medical Subject Headings
└── download_errors.txt # Log of download issues
```

## Successfully Downloaded Ontologies

### 1. Gene Ontology (GO)
- **Files**: 3 (go.owl, go-basic.obo, go.obo)
- **Total Size**: 186 MB
- **Format**: OWL, OBO
- **License**: CC BY 4.0
- **Status**: ✓ Complete

### 2. ChEBI (Chemical Entities of Biological Interest)
- **Files**: 2 (chebi.owl, chebi.obo)
- **Total Size**: 1020 MB
- **Format**: OWL, OBO
- **License**: CC BY 4.0
- **Status**: ✓ Complete

### 3. Cell Ontology (CL)
- **Files**: 2 (cl.owl, cl.obo)
- **Total Size**: 76 MB
- **Format**: OWL, OBO
- **License**: CC BY 4.0
- **Status**: ✓ Complete

### 4. MeSH (Medical Subject Headings)
- **Files**: 2 (desc2025.xml, mesh.nt.gz)
- **Total Size**: 346 MB
- **Format**: XML, RDF N-Triples (compressed)
- **License**: Public Domain
- **Status**: ✓ Complete

### 5. ESCO (European Skills/Competences)
- **Files**: 1 (model.rdf)
- **Total Size**: 177 KB
- **Format**: RDF
- **License**: CC BY 4.0
- **Status**: ⚠ Partial (model only)
- **Note**: Full classification requires manual download with registration

## Not Downloaded

### 6. SNOMED CT
- **Status**: ✗ Requires registration
- **Reason**: Requires SNOMED International license agreement and account creation
- **Action**: Visit https://www.snomed.org/snomed-ct/get-snomed to register

## Total Downloaded
- **Files**: 10 ontology files
- **Total Size**: ~1.6 GB
- **Formats**: OWL, OBO, RDF, XML

## Usage Notes

### Compressed Files
The `mesh.nt.gz` file is compressed. To extract:
```bash
gunzip ontologies/mesh/mesh.nt.gz
```

### File Formats

- **OWL**: Web Ontology Language - Standard format for semantic web ontologies
- **OBO**: Open Biomedical Ontologies format - Human-readable text format
- **RDF**: Resource Description Framework - Standard model for data interchange
- **XML**: eXtensible Markup Language - Structured hierarchical format

### Integration with escoSkill2

These ontologies can be used to enhance the skill extraction and matching system:

1. **ESCO** - Primary source for skills, competences, and occupations taxonomy
2. **GO** - For bioinformatics and life sciences positions
3. **ChEBI** - For chemistry and pharmaceutical roles
4. **Cell Ontology** - For biomedical and research positions
5. **MeSH** - For medical and healthcare terminology
6. **SNOMED CT** - For clinical healthcare positions (when available)

## References

Each subdirectory contains a detailed README.md with:
- Ontology description
- Source URLs
- File specifications
- License information
- Documentation links

## Error Log

See `download_errors.txt` for detailed information about:
- Failed downloads
- Partial downloads
- Manual download requirements
- Troubleshooting steps
