"""
Entity Linker for Hybrid Retrieval

Links retrieved text chunks to knowledge graph nodes using
metadata-based lookup and fuzzy matching.
"""

from typing import List, Tuple, Set, Dict, Optional, TYPE_CHECKING
from difflib import SequenceMatcher
from collections import defaultdict
import re

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from ..graph.graph_builder import VocationalGraphBuilder
    from .config import RetrieverConfig


class EntityLinker:
    """
    Links retrieved chunks to graph nodes.

    Uses a two-stage approach:
    1. Metadata-based lookup (fast path) - if entity_ids are stored in metadata
    2. Fuzzy matching (fallback) - match entity names in chunk content
    """

    def __init__(
        self,
        graph_builder: "VocationalGraphBuilder",
        config: Optional["RetrieverConfig"] = None,
    ):
        """
        Initialize entity linker.

        Args:
            graph_builder: The graph builder containing the knowledge graph
            config: Retriever configuration
        """
        self.graph = graph_builder
        self.config = config
        self._entity_name_index: Optional[Dict[str, Set[str]]] = None

    def link_from_results(
        self,
        results: List[Tuple["Document", float]],
    ) -> List[Set[str]]:
        """
        Extract entity IDs from retrieved documents.

        Args:
            results: List of (document, score) tuples from vector search

        Returns:
            List of entity ID sets (one set per document)
        """
        chunk_entity_ids = []

        for doc, score in results:
            entity_ids = set()

            # Fast path: metadata lookup (now stored as JSON string)
            if "entity_ids" in doc.metadata:
                import json
                stored_ids = doc.metadata["entity_ids"]
                if isinstance(stored_ids, str):
                    # Parse JSON string
                    try:
                        parsed_ids = json.loads(stored_ids)
                        if isinstance(parsed_ids, list):
                            entity_ids.update(parsed_ids)
                    except (json.JSONDecodeError, TypeError):
                        pass
                elif isinstance(stored_ids, list):
                    entity_ids.update(stored_ids)

            # Fallback: fuzzy matching
            if self.config and self.config.enable_fuzzy_match:
                fuzzy_ids = self._fuzzy_match_entities(doc.page_content)
                entity_ids.update(fuzzy_ids)

            chunk_entity_ids.append(entity_ids)

        return chunk_entity_ids

    def link_from_chunks(
        self,
        chunks: List[str],
    ) -> List[Set[str]]:
        """
        Extract entity IDs from text chunks.

        Args:
            chunks: List of text chunks

        Returns:
            List of entity ID sets (one set per chunk)
        """
        return [self._fuzzy_match_entities(chunk) for chunk in chunks]

    def _fuzzy_match_entities(
        self,
        text: str,
    ) -> Set[str]:
        """
        Find entities by fuzzy matching names in text.

        Args:
            text: Text to search for entity names

        Returns:
            Set of matching entity IDs
        """
        if self._entity_name_index is None:
            self._build_entity_name_index()

        threshold = self.config.fuzzy_match_threshold if self.config else 0.85
        matching_ids = set()

        # Extract key terms from text (simple approach)
        terms = self._extract_terms(text)

        for term in terms:
            for entity_name, entity_ids in self._entity_name_index.items():
                # Calculate similarity
                similarity = SequenceMatcher(None, term.lower(), entity_name.lower()).ratio()

                if similarity >= threshold:
                    matching_ids.update(entity_ids)

        return matching_ids

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

        # Filter: keep terms with 2+ characters (Chinese or English)
        filtered = [
            term.strip()
            for term in terms
            if len(term.strip()) >= 2  # Minimum length
            and not term.strip().isdigit()  # Not pure numbers
        ]

        return filtered

    def _build_entity_name_index(self) -> None:
        """
        Build an index of entity names to node IDs.

        Index structure: {entity_name: {node_id1, node_id2, ...}}
        """
        self._entity_name_index = defaultdict(set)

        for node_id, node_data in self.graph.graph.nodes(data=True):
            name = node_data.get("name", "")
            if name:
                self._entity_name_index[name].add(node_id)

    def get_entity_by_id(
        self,
        entity_id: str,
    ) -> Optional[Dict]:
        """
        Get entity data by ID.

        Args:
            entity_id: Entity ID

        Returns:
            Entity data dict or None if not found
        """
        return self.graph.get_node_by_id(entity_id)

    def get_entities_by_type(
        self,
        entity_type: str,
    ) -> List[Dict]:
        """
        Get all entities of a specific type.

        Args:
            entity_type: Entity type (e.g., "action_skill", "standard_task")

        Returns:
            List of entity data dicts
        """
        return self.graph.get_nodes_by_type(entity_type)

    def find_entities_by_name_pattern(
        self,
        pattern: str,
        entity_type: Optional[str] = None,
    ) -> List[Tuple[str, Dict]]:
        """
        Find entities by name pattern.

        Args:
            pattern: Regex pattern to match entity names
            entity_type: Optional filter by entity type

        Returns:
            List of (node_id, node_data) tuples matching the pattern
        """
        regex = re.compile(pattern, re.IGNORECASE)
        matches = []

        for node_id, node_data in self.graph.graph.nodes(data=True):
            # Filter by type if specified
            if entity_type and node_data.get("entity_type") != entity_type:
                continue

            # Match name
            name = node_data.get("name", "")
            if regex.search(name):
                matches.append((node_id, node_data))

        return matches
