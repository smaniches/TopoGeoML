"""
Topological audits of learned representations.

    embedding_audit.py — heuristic audit of an embedding matrix's topology
                         via Rips persistence.
"""

from topogeoml.audits.embedding_audit import (
    EmbeddingTopologyAudit,
    audit_embedding,
)

__all__ = ["EmbeddingTopologyAudit", "audit_embedding"]
