"""Knowledge graph implementation using NetworkX with cosine similarity and visualization."""

import json
import sqlite3
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
import pickle

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from .config import FinderConfig
from .indexer import DocumentIndexer


class KnowledgeGraph:
    """Knowledge graph for document relationships using cosine similarity."""
    
    def __init__(self, config: FinderConfig, indexer: DocumentIndexer):
        self.config = config
        self.indexer = indexer
        self.graph = nx.Graph()
        self.embedding_model = SentenceTransformer(config.embedding_model)
        
        # Knowledge graph storage  
        data_dir = config.index_path.parent
        self.kg_db_path = data_dir / "knowledge_graph.db"
        self.graph_file_path = data_dir / "knowledge_graph.pkl"
        
        self.init_kg_database()
        self.load_graph()
    
    def init_kg_database(self):
        """Initialize SQLite database for knowledge graph metadata."""
        self.kg_conn = sqlite3.connect(str(self.kg_db_path))
        self.kg_conn.execute("""
            CREATE TABLE IF NOT EXISTS graph_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE,
                node_id TEXT,
                embedding TEXT,
                metadata TEXT,
                created_time REAL,
                updated_time REAL
            )
        """)
        
        self.kg_conn.execute("""
            CREATE TABLE IF NOT EXISTS graph_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_node TEXT,
                target_node TEXT,
                similarity_score REAL,
                edge_type TEXT,
                metadata TEXT,
                created_time REAL
            )
        """)
        self.kg_conn.commit()
    
    def save_graph(self):
        """Save NetworkX graph to disk."""
        with open(self.graph_file_path, 'wb') as f:
            pickle.dump(self.graph, f)
    
    def load_graph(self):
        """Load NetworkX graph from disk."""
        if self.graph_file_path.exists():
            try:
                with open(self.graph_file_path, 'rb') as f:
                    self.graph = pickle.load(f)
                print(f"Loaded knowledge graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            except Exception as e:
                print(f"Error loading graph: {e}. Creating new graph.")
                self.graph = nx.Graph()
        else:
            self.graph = nx.Graph()
    
    def _get_document_embedding(self, file_path: str) -> Optional[np.ndarray]:
        """Get document embedding by averaging chunk embeddings."""
        cursor = self.indexer.conn.execute("""
            SELECT c.content, c.embedding_index
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE d.file_path = ?
        """, (file_path,))
        
        chunks = cursor.fetchall()
        if not chunks:
            return None
        
        # Get embeddings from FAISS index
        embeddings = []
        for _, embedding_idx in chunks:
            try:
                # Get embedding from FAISS index
                vector = self.indexer.index.reconstruct(int(embedding_idx))
                embeddings.append(vector)
            except Exception as e:
                print(f"Error getting embedding for index {embedding_idx}: {e}")
                continue
        
        if not embeddings:
            return None
        
        # Average embeddings to get document-level embedding
        doc_embedding = np.mean(embeddings, axis=0)
        return doc_embedding / np.linalg.norm(doc_embedding)  # Normalize
    
    def _calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        return float(similarity)
    
    def add_document_node(self, file_path: str, metadata: Dict[str, Any] = None) -> bool:
        """Add a document as a node to the knowledge graph."""
        try:
            # Get document embedding
            embedding = self._get_document_embedding(file_path)
            if embedding is None:
                return False
            
            # Create unique node ID
            node_id = f"doc_{hash(file_path) % 1000000}"
            
            # Add node to graph
            node_attrs = {
                'file_path': file_path,
                'embedding': embedding.tolist(),
                'type': 'document',
                'metadata': metadata or {}
            }
            self.graph.add_node(node_id, **node_attrs)
            
            # Store in database
            embedding_json = json.dumps(embedding.tolist())
            metadata_json = json.dumps(metadata or {})
            
            self.kg_conn.execute("""
                INSERT OR REPLACE INTO graph_nodes 
                (file_path, node_id, embedding, metadata, created_time, updated_time)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                file_path,
                node_id,
                embedding_json,
                metadata_json,
                datetime.now().timestamp(),
                datetime.now().timestamp()
            ))
            self.kg_conn.commit()
            
            return True
            
        except Exception as e:
            print(f"Error adding document node {file_path}: {e}")
            return False
    
    def add_similarity_edge(self, node1: str, node2: str, similarity_threshold: float = 0.7) -> bool:
        """Add similarity edge between two nodes if similarity exceeds threshold."""
        try:
            if node1 not in self.graph.nodes or node2 not in self.graph.nodes:
                return False
            
            # Get embeddings
            embedding1 = np.array(self.graph.nodes[node1]['embedding'])
            embedding2 = np.array(self.graph.nodes[node2]['embedding'])
            
            # Calculate similarity
            similarity = self._calculate_cosine_similarity(embedding1, embedding2)
            
            if similarity >= similarity_threshold:
                # Add edge to graph
                self.graph.add_edge(node1, node2, 
                                  weight=similarity,
                                  edge_type='similarity',
                                  similarity_score=similarity)
                
                # Store in database
                self.kg_conn.execute("""
                    INSERT OR REPLACE INTO graph_edges
                    (source_node, target_node, similarity_score, edge_type, metadata, created_time)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    node1,
                    node2,
                    similarity,
                    'similarity',
                    json.dumps({'threshold': similarity_threshold}),
                    datetime.now().timestamp()
                ))
                self.kg_conn.commit()
                
                return True
            
            return False
            
        except Exception as e:
            print(f"Error adding similarity edge between {node1} and {node2}: {e}")
            return False
    
    async def build_knowledge_graph(self, similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """Build knowledge graph from all indexed documents."""
        print("Building knowledge graph...")
        
        stats = {"nodes_added": 0, "edges_added": 0, "errors": 0}
        
        # Get all indexed documents
        cursor = self.indexer.conn.execute("SELECT file_path FROM documents")
        documents = [row[0] for row in cursor.fetchall()]
        
        # Add document nodes
        for file_path in documents:
            try:
                if self.add_document_node(file_path):
                    stats["nodes_added"] += 1
            except Exception as e:
                stats["errors"] += 1
                print(f"Error adding node for {file_path}: {e}")
        
        # Add similarity edges
        nodes = list(self.graph.nodes())
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                try:
                    if self.add_similarity_edge(node1, node2, similarity_threshold):
                        stats["edges_added"] += 1
                except Exception as e:
                    stats["errors"] += 1
                    print(f"Error adding edge between {node1} and {node2}: {e}")
        
        self.save_graph()
        print(f"Knowledge graph built: {stats}")
        return stats
    
    def find_similar_documents(self, file_path: str, k: int = 5) -> List[Dict[str, Any]]:
        """Find documents similar to the given file."""
        results = []
        
        # Find node for this file
        target_node = None
        for node_id, data in self.graph.nodes(data=True):
            if data.get('file_path') == file_path:
                target_node = node_id
                break
        
        if not target_node:
            return results
        
        # Get all neighbors with similarity scores
        neighbors = []
        for neighbor in self.graph.neighbors(target_node):
            edge_data = self.graph.edges[target_node, neighbor]
            similarity = edge_data.get('similarity_score', 0)
            file_path_neighbor = self.graph.nodes[neighbor].get('file_path', '')
            
            neighbors.append({
                'file_path': file_path_neighbor,
                'similarity_score': similarity,
                'node_id': neighbor
            })
        
        # Sort by similarity and return top k
        neighbors.sort(key=lambda x: x['similarity_score'], reverse=True)
        return neighbors[:k]
    
    def get_document_clusters(self, min_similarity: float = 0.8) -> List[List[str]]:
        """Find clusters of highly similar documents."""
        # Create subgraph with high similarity edges only
        high_sim_edges = [(u, v) for u, v, d in self.graph.edges(data=True) 
                         if d.get('similarity_score', 0) >= min_similarity]
        
        subgraph = self.graph.edge_subgraph(high_sim_edges)
        
        # Find connected components (clusters)
        clusters = []
        for component in nx.connected_components(subgraph):
            cluster_files = [self.graph.nodes[node].get('file_path', '') 
                           for node in component]
            clusters.append([f for f in cluster_files if f])
        
        return clusters
    
    def visualize_graph(self, output_path: Optional[str] = None, 
                       layout: str = 'spring', 
                       node_size_factor: int = 300,
                       figsize: Tuple[int, int] = (12, 8)) -> str:
        """Visualize the knowledge graph using matplotlib."""
        plt.figure(figsize=figsize)
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout == 'random':
            pos = nx.random_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)
        
        # Draw nodes
        node_colors = ['lightblue' if self.graph.nodes[node].get('type') == 'document' 
                      else 'lightgreen' for node in self.graph.nodes()]
        
        nx.draw_networkx_nodes(self.graph, pos, 
                              node_color=node_colors,
                              node_size=node_size_factor,
                              alpha=0.7)
        
        # Draw edges with varying thickness based on similarity
        edges = self.graph.edges(data=True)
        edge_weights = [d.get('similarity_score', 0.5) * 3 for _, _, d in edges]
        
        nx.draw_networkx_edges(self.graph, pos,
                              width=edge_weights,
                              alpha=0.5,
                              edge_color='gray')
        
        # Add labels for nodes (file names)
        labels = {}
        for node in self.graph.nodes():
            file_path = self.graph.nodes[node].get('file_path', '')
            if file_path:
                labels[node] = Path(file_path).name[:15] + '...' if len(Path(file_path).name) > 15 else Path(file_path).name
            else:
                labels[node] = node[:10]
        
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        
        plt.title(f"Document Knowledge Graph\n{self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        plt.axis('off')
        plt.tight_layout()
        
        # Save or show
        if output_path is None:
            data_dir = self.config.index_path.parent
            output_path = str(data_dir / f"knowledge_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return output_path
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        if self.graph.number_of_nodes() == 0:
            return {"nodes": 0, "edges": 0, "density": 0, "avg_degree": 0}
        
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "avg_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
            "connected_components": nx.number_connected_components(self.graph),
            "avg_clustering": nx.average_clustering(self.graph) if self.graph.number_of_edges() > 0 else 0
        }
    
    def find_central_documents(self, centrality_type: str = 'betweenness', k: int = 5) -> List[Dict[str, Any]]:
        """Find most central documents in the graph."""
        if self.graph.number_of_nodes() == 0:
            return []
        
        # Calculate centrality
        if centrality_type == 'betweenness':
            centrality = nx.betweenness_centrality(self.graph)
        elif centrality_type == 'closeness':
            centrality = nx.closeness_centrality(self.graph)
        elif centrality_type == 'degree':
            centrality = nx.degree_centrality(self.graph)
        elif centrality_type == 'eigenvector':
            try:
                centrality = nx.eigenvector_centrality(self.graph, max_iter=1000)
            except:
                centrality = nx.degree_centrality(self.graph)
        else:
            centrality = nx.degree_centrality(self.graph)
        
        # Sort by centrality and get top k
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for node_id, centrality_score in sorted_nodes[:k]:
            file_path = self.graph.nodes[node_id].get('file_path', '')
            results.append({
                'file_path': file_path,
                'centrality_score': centrality_score,
                'centrality_type': centrality_type,
                'node_id': node_id
            })
        
        return results
    
    def close(self):
        """Close database connections and save graph."""
        self.save_graph()
        if hasattr(self, 'kg_conn'):
            self.kg_conn.close()


class KnowledgeGraphAgent:
    """Agent for managing knowledge graph operations."""
    
    def __init__(self, config: FinderConfig, indexer: DocumentIndexer):
        self.config = config
        self.indexer = indexer
        self.kg = KnowledgeGraph(config, indexer)
    
    async def build_graph(self, similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """Build the complete knowledge graph."""
        return await self.kg.build_knowledge_graph(similarity_threshold)
    
    def find_related_documents(self, file_path: str, k: int = 5) -> List[Dict[str, Any]]:
        """Find documents related to the given file."""
        return self.kg.find_similar_documents(file_path, k)
    
    def analyze_document_clusters(self, min_similarity: float = 0.8) -> List[List[str]]:
        """Analyze clusters of similar documents."""
        return self.kg.get_document_clusters(min_similarity)
    
    def visualize_knowledge_graph(self, output_path: Optional[str] = None, **kwargs) -> str:
        """Create a visualization of the knowledge graph."""
        return self.kg.visualize_graph(output_path, **kwargs)
    
    def get_central_documents(self, centrality_type: str = 'betweenness', k: int = 5) -> List[Dict[str, Any]]:
        """Find the most central/important documents."""
        return self.kg.find_central_documents(centrality_type, k)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive knowledge graph statistics."""
        return self.kg.get_graph_stats()