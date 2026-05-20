"""
Topological audits of learned representations.

v0.0.1: embedding_audit (item 9) — prototype audit of an embedding matrix's
                                   topology via Rips persistence.
"""

from topogeoml.audits.embedding_audit import (
    EmbeddingTopologyAudit,
    audit_embedding,
)

__all__ = ["EmbeddingTopologyAudit", "audit_embedding"]
