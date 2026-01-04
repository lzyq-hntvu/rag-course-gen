"""
Chunk Entity Linker for Ingestion

Links text chunks to knowledge graph entities during document ingestion.
Stores entity_ids in ChromaDB metadata for later graph-enhanced retrieval.
"""

from typing import List, Set, Dict, Optional
from difflib import SequenceMatcher
from collections import defaultdict
import re

from langchain_core.documents import Document


class ChunkEntityLinker:
    """
    Links document chunks to knowledge graph entities for ingestion.

    Uses fuzzy matching between chunk content and graph node names
    to identify relevant entities.
    """

    def __init__(self, graph_builder, similarity_threshold: float = 0.85):
        """
        Initialize chunk entity linker.

        Args:
            graph_builder: VocationalGraphBuilder instance
            similarity_threshold: Threshold for fuzzy matching (default: 0.85)
        """
        self.graph = graph_builder
        self.similarity_threshold = similarity_threshold
        self._entity_name_index: Optional[Dict[str, List[Dict]]] = None

    def _build_entity_name_index(self) -> None:
        """Build an index of entity names to node data."""
        self._entity_name_index = defaultdict(list)

        for node_id, node_data in self.graph.graph.nodes(data=True):
            name = node_data.get("name", "")
            if name:
                self._entity_name_index[name].append({
                    "id": node_id,
                    "type": node_data.get("entity_type", ""),
                    "data": node_data,
                })

    def link_chunks(
        self,
        chunks: List[Document],
        build_index: bool = True,
    ) -> List[Document]:
        """
        Link chunks to entities and add entity_ids to metadata.

        Args:
            chunks: List of document chunks
            build_index: Whether to build the entity name index

        Returns:
            List of chunks with updated metadata
        """
        if build_index:
            self._build_entity_name_index()

        linked_chunks = []

        for chunk in chunks:
            entity_ids, entity_types = self._link_chunk(chunk)
            # Convert to JSON string for ChromaDB compatibility
            import json
            chunk.metadata["entity_ids"] = json.dumps(list(entity_ids), ensure_ascii=False)
            chunk.metadata["entity_types"] = json.dumps(list(entity_types), ensure_ascii=False)
            linked_chunks.append(chunk)

        return linked_chunks

    def _link_chunk(self, chunk: Document) -> tuple[Set[str], Set[str]]:
        """
        Link a single chunk to entities.

        Args:
            chunk: Document chunk

        Returns:
            Tuple of (entity_ids, entity_types)
        """
        if self._entity_name_index is None:
            self._build_entity_name_index()

        entity_ids = set()
        entity_types = set()

        # Extract potential entity names from chunk
        terms = self._extract_terms(chunk.page_content)

        # Match against graph entities
        for term in terms:
            matches = self._find_matching_entities(term)
            for match in matches:
                entity_ids.add(match["id"])
                entity_types.add(match["type"])

        return entity_ids, entity_types

    def _extract_terms(self, text: str) -> List[str]:
        """
        Extract potential entity names from text.

        Args:
            text: Input text

        Returns:
            List of potential entity names
        """
        # Split by common delimiters
        terms = re.split(r'[,，、；;\s]+', text)

        # Filter: keep terms with 2+ characters
        filtered = [
            term.strip()
            for term in terms
            if len(term.strip()) >= 2  # Minimum length
            and not term.strip().isdigit()  # Not pure numbers
            and not term.strip().startswith('(')  # Skip parenthetical
        ]

        return filtered

    def _find_matching_entities(self, term: str) -> List[Dict]:
        """
        Find entities matching the given term.

        Args:
            term: Term to match

        Returns:
            List of matching entity data
        """
        matches = []
        term_lower = term.lower()

        for entity_name, entities in self._entity_name_index.items():
            # Calculate similarity
            similarity = SequenceMatcher(
                None,
                term_lower,
                entity_name.lower()
            ).ratio()

            if similarity >= self.similarity_threshold:
                matches.extend(entities)

        return matches

    def get_statistics(self, chunks: List[Document]) -> Dict:
        """
        Get statistics about entity linking.

        Args:
            chunks: List of chunks with entity metadata

        Returns:
            Statistics dictionary
        """
        import json

        chunks_with_entities = 0
        total_entity_links = 0
        entity_type_counts = defaultdict(int)

        for chunk in chunks:
            entity_ids_json = chunk.metadata.get("entity_ids", "[]")
            entity_types_json = chunk.metadata.get("entity_types", "[]")

            # Parse JSON strings
            try:
                entity_ids = json.loads(entity_ids_json) if entity_ids_json else []
                entity_types = json.loads(entity_types_json) if entity_types_json else []
            except (json.JSONDecodeError, TypeError):
                entity_ids = []
                entity_types = []

            if entity_ids:
                chunks_with_entities += 1
                total_entity_links += len(entity_ids)

                for etype in entity_types:
                    entity_type_counts[etype] += 1

        return {
            "total_chunks": len(chunks),
            "chunks_with_entities": chunks_with_entities,
            "chunks_with_entities_ratio": chunks_with_entities / len(chunks) if chunks else 0,
            "total_entity_links": total_entity_links,
            "avg_links_per_chunk": total_entity_links / len(chunks) if chunks else 0,
            "entity_type_distribution": dict(entity_type_counts),
        }
