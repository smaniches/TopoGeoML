"""
Core mathematical objects.

Layout:
    filtrations.py  Rips filtration via ripser
    cubical.py      binary-mask topology diagnostic
    complexes.py    SimplicialComplex, boundary operators, Hodge Laplacian
    diagrams.py     PersistenceDiagram with provenance
    vectorizers.py  persistence image, Betti curve

Diagram distances (bottleneck, Wasserstein) are not implemented.
"""

from topogeoml.core.complexes import (
    SimplicialComplex,
    betti_numbers,
    hodge_laplacian,
    is_chain_complex,
)
from topogeoml.core.cubical import CubicalDiagnostic, cubical_mask_diagnostic
from topogeoml.core.diagrams import DiagramProvenance, PersistenceDiagram
from topogeoml.core.filtrations import RipsFiltration
from topogeoml.core.vectorizers import (
    BettiCurveVectorizer,
    PersistenceImageVectorizer,
)

__all__ = [
    "BettiCurveVectorizer",
    "CubicalDiagnostic",
    "DiagramProvenance",
    "PersistenceDiagram",
    "PersistenceImageVectorizer",
    "RipsFiltration",
    "SimplicialComplex",
    "betti_numbers",
    "cubical_mask_diagnostic",
    "hodge_laplacian",
    "is_chain_complex",
]
