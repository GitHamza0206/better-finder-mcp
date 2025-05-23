"""MCP Server implementation for Enhanced Finder."""

import asyncio
import json
from typing import Any, Sequence, Dict, List
from pathlib import Path

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
)

from .config import FinderConfig
from .indexer import DocumentIndexer
from .simple_agents import SimpleSearchAgent, SimpleIndexingAgent
from .knowledge_graph import KnowledgeGraphAgent


class EnhancedFinderMCPServer:
    """MCP Server for Enhanced Finder functionality."""
    
    def __init__(self):
        self.config = FinderConfig()
        self.indexer = None
        self.search_agent = None
        self.indexing_agent = None
        self.kg_agent = None
        self.server = Server("enhanced-finder")
        
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup MCP server handlers."""
        
        @self.server.list_resources()
        async def handle_list_resources() -> list[Resource]:
            """List available resources."""
            return [
                Resource(
                    uri="finder://stats",
                    name="Indexing Statistics",
                    description="Current indexing statistics and status",
                    mimeType="application/json",
                ),
                Resource(
                    uri="finder://config",
                    name="Configuration",
                    description="Current finder configuration",
                    mimeType="application/json",
                ),
            ]
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Read a specific resource."""
            if uri == "finder://stats":
                if self.indexer:
                    stats = self.indexer.get_stats()
                    return json.dumps(stats, indent=2)
                else:
                    return json.dumps({"status": "not_initialized"})
            
            elif uri == "finder://config":
                config_dict = {
                    "supported_extensions": list(self.config.supported_extensions),
                    "scan_paths": [str(p) for p in self.config.scan_paths],
                    "max_file_size_mb": self.config.max_file_size_mb,
                    "max_search_results": self.config.max_search_results,
                    "embedding_model": self.config.embedding_model
                }
                return json.dumps(config_dict, indent=2)
            
            else:
                raise ValueError(f"Unknown resource: {uri}")
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="search_files",
                    description="Search for files using intelligent semantic and filename matching",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (can be keywords, filename, or description)"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 10
                            },
                            "file_type": {
                                "type": "string",
                                "description": "Filter by file type (pdf, excel, text, etc.)",
                                "enum": ["pdf", "excel", "word", "text", "csv", "any"]
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="index_files",
                    description="Index files in the specified directory or perform full reindex",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Directory path to index (optional for full reindex)"
                            },
                            "full_reindex": {
                                "type": "boolean",
                                "description": "Perform full reindex of all configured paths",
                                "default": False
                            }
                        }
                    }
                ),
                Tool(
                    name="get_file_content",
                    description="Get the content of a specific file",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to read"
                            },
                            "max_length": {
                                "type": "integer",
                                "description": "Maximum content length to return",
                                "default": 5000
                            }
                        },
                        "required": ["file_path"]
                    }
                ),
                Tool(
                    name="get_stats",
                    description="Get current indexing statistics",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="configure_paths",
                    description="Add or remove paths from scanning configuration",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["add", "remove", "list"],
                                "description": "Action to perform"
                            },
                            "path": {
                                "type": "string",
                                "description": "Path to add or remove"
                            }
                        },
                        "required": ["action"]
                    }
                ),
                Tool(
                    name="build_knowledge_graph",
                    description="Build a knowledge graph from indexed documents using cosine similarity",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "similarity_threshold": {
                                "type": "number",
                                "description": "Minimum similarity threshold for connecting documents (0.0-1.0)",
                                "default": 0.7
                            }
                        }
                    }
                ),
                Tool(
                    name="find_related_documents",
                    description="Find documents related to a specific file using the knowledge graph",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to find related documents for"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of related documents to return",
                                "default": 5
                            }
                        },
                        "required": ["file_path"]
                    }
                ),
                Tool(
                    name="analyze_document_clusters",
                    description="Analyze clusters of similar documents in the knowledge graph",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "min_similarity": {
                                "type": "number",
                                "description": "Minimum similarity for documents to be in the same cluster",
                                "default": 0.8
                            }
                        }
                    }
                ),
                Tool(
                    name="get_central_documents",
                    description="Find the most central/important documents in the knowledge graph",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "centrality_type": {
                                "type": "string",
                                "enum": ["betweenness", "closeness", "degree", "eigenvector"],
                                "description": "Type of centrality measure to use",
                                "default": "betweenness"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of central documents to return",
                                "default": 5
                            }
                        }
                    }
                ),
                Tool(
                    name="get_knowledge_graph_stats",
                    description="Get statistics about the knowledge graph",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
            """Handle tool calls."""
            try:
                if name == "search_files":
                    return await self._handle_search_files(arguments)
                elif name == "index_files":
                    return await self._handle_index_files(arguments)
                elif name == "get_file_content":
                    return await self._handle_get_file_content(arguments)
                elif name == "get_stats":
                    return await self._handle_get_stats(arguments)
                elif name == "configure_paths":
                    return await self._handle_configure_paths(arguments)
                elif name == "build_knowledge_graph":
                    return await self._handle_build_knowledge_graph(arguments)
                elif name == "find_related_documents":
                    return await self._handle_find_related_documents(arguments)
                elif name == "analyze_document_clusters":
                    return await self._handle_analyze_document_clusters(arguments)
                elif name == "get_central_documents":
                    return await self._handle_get_central_documents(arguments)
                elif name == "get_knowledge_graph_stats":
                    return await self._handle_get_knowledge_graph_stats(arguments)
                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
            
            except Exception as e:
                return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def _initialize_components(self):
        """Initialize indexer and agents."""
        if self.indexer is None:
            self.indexer = DocumentIndexer(self.config)
            self.indexer.load_or_create_index()
            
            self.search_agent = SimpleSearchAgent(self.config, self.indexer)
            self.indexing_agent = SimpleIndexingAgent(self.config, self.indexer)
            self.kg_agent = KnowledgeGraphAgent(self.config, self.indexer)
    
    async def _handle_search_files(self, arguments: dict) -> list[TextContent]:
        """Handle file search requests."""
        await self._initialize_components()
        
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 10)
        file_type = arguments.get("file_type", "any")
        
        # Add file type filter to query if specified
        if file_type != "any":
            type_extensions = {
                "pdf": [".pdf"],
                "excel": [".xlsx", ".xls", ".csv"],
                "word": [".docx", ".doc"],
                "text": [".txt", ".md"],
                "csv": [".csv"]
            }
            query += f" {' '.join(type_extensions.get(file_type, []))}"
        
        results = await self.search_agent.search(query)
        results = results[:max_results]
        
        if not results:
            return [TextContent(type="text", text="No files found matching your query.")]
        
        response_text = f"Found {len(results)} files:\n\n"
        
        for i, result in enumerate(results, 1):
            file_path = result.get("file_path", "")
            file_name = result.get("file_name", "")
            score = result.get("similarity_score", 0)
            snippet = result.get("content_snippet", "")
            search_type = result.get("search_type", "")
            
            response_text += f"{i}. **{file_name}**\n"
            response_text += f"   Path: `{file_path}`\n"
            response_text += f"   Score: {score:.3f} ({search_type})\n"
            
            if snippet:
                response_text += f"   Preview: {snippet}\n"
            
            response_text += "\n"
        
        return [TextContent(type="text", text=response_text)]
    
    async def _handle_index_files(self, arguments: dict) -> list[TextContent]:
        """Handle file indexing requests."""
        await self._initialize_components()
        
        full_reindex = arguments.get("full_reindex", False)
        path = arguments.get("path")
        
        if full_reindex:
            response_text = "Starting full reindex of all configured paths...\n"
            stats = await self.indexing_agent.full_reindex()
        elif path:
            path_obj = Path(path)
            if not path_obj.exists():
                return [TextContent(type="text", text=f"Path does not exist: {path}")]
            
            response_text = f"Indexing directory: {path}\n"
            stats = await self.indexer.index_directory(path_obj)
        else:
            response_text = "Starting incremental indexing...\n"
            stats = await self.indexing_agent.incremental_index()
        
        response_text += f"\nIndexing completed:\n"
        response_text += f"- Files processed: {stats['processed']}\n"
        response_text += f"- Files indexed: {stats['indexed']}\n"
        response_text += f"- Errors: {stats['errors']}\n"
        
        return [TextContent(type="text", text=response_text)]
    
    async def _handle_get_file_content(self, arguments: dict) -> list[TextContent]:
        """Handle file content requests."""
        await self._initialize_components()
        
        file_path = arguments.get("file_path", "")
        max_length = arguments.get("max_length", 5000)
        
        path_obj = Path(file_path)
        if not path_obj.exists():
            return [TextContent(type="text", text=f"File does not exist: {file_path}")]
        
        file_data = self.indexer.processor_manager.process_file(path_obj)
        
        if file_data.get("error"):
            return [TextContent(type="text", text=f"Error reading file: {file_data['error']}")]
        
        content = file_data.get("content", "")
        if len(content) > max_length:
            content = content[:max_length] + f"\n\n... (truncated, showing first {max_length} characters)"
        
        response = f"**File:** {path_obj.name}\n"
        response += f"**Path:** {file_path}\n"
        response += f"**Size:** {file_data.get('file_size', 0)} bytes\n\n"
        response += f"**Content:**\n```\n{content}\n```"
        
        return [TextContent(type="text", text=response)]
    
    async def _handle_get_stats(self, arguments: dict) -> list[TextContent]:
        """Handle statistics requests."""
        await self._initialize_components()
        
        stats = self.indexer.get_stats()
        
        response = "**Enhanced Finder Statistics**\n\n"
        response += f"- Total documents indexed: {stats['total_documents']}\n"
        response += f"- Total text chunks: {stats['total_chunks']}\n"
        response += f"- Vector store size: {stats['index_size_mb']:.2f} MB\n"
        response += f"- Supported file types: {len(self.config.supported_extensions)}\n"
        response += f"- Configured scan paths: {len(self.config.scan_paths)}\n"
        
        return [TextContent(type="text", text=response)]
    
    async def _handle_configure_paths(self, arguments: dict) -> list[TextContent]:
        """Handle path configuration requests."""
        action = arguments.get("action", "")
        path = arguments.get("path", "")
        
        if action == "list":
            response = "**Configured scan paths:**\n\n"
            for i, scan_path in enumerate(self.config.scan_paths, 1):
                exists = "✓" if scan_path.exists() else "✗"
                response += f"{i}. {exists} {scan_path}\n"
            return [TextContent(type="text", text=response)]
        
        elif action == "add":
            if not path:
                return [TextContent(type="text", text="Path is required for add action")]
            
            path_obj = Path(path).expanduser().resolve()
            if not path_obj.exists():
                return [TextContent(type="text", text=f"Path does not exist: {path}")]
            
            if path_obj not in self.config.scan_paths:
                self.config.scan_paths.append(path_obj)
                return [TextContent(type="text", text=f"Added path: {path_obj}")]
            else:
                return [TextContent(type="text", text=f"Path already configured: {path_obj}")]
        
        elif action == "remove":
            if not path:
                return [TextContent(type="text", text="Path is required for remove action")]
            
            path_obj = Path(path).expanduser().resolve()
            if path_obj in self.config.scan_paths:
                self.config.scan_paths.remove(path_obj)
                return [TextContent(type="text", text=f"Removed path: {path_obj}")]
            else:
                return [TextContent(type="text", text=f"Path not found in configuration: {path_obj}")]
        
        else:
            return [TextContent(type="text", text="Invalid action. Use 'add', 'remove', or 'list'")]
    
    async def _handle_build_knowledge_graph(self, arguments: dict) -> list[TextContent]:
        """Handle knowledge graph building requests."""
        await self._initialize_components()
        
        similarity_threshold = arguments.get("similarity_threshold", 0.7)
        
        response_text = f"Building knowledge graph with similarity threshold {similarity_threshold}...\n\n"
        
        try:
            stats = await self.kg_agent.build_graph(similarity_threshold)
            
            response_text += "**Knowledge Graph Built Successfully!**\n\n"
            response_text += f"- Nodes added: {stats['nodes_added']}\n"
            response_text += f"- Edges added: {stats['edges_added']}\n"
            response_text += f"- Errors: {stats['errors']}\n"
            response_text += f"- Similarity threshold: {similarity_threshold}\n"
            
        except Exception as e:
            response_text += f"Error building knowledge graph: {str(e)}"
        
        return [TextContent(type="text", text=response_text)]
    
    async def _handle_find_related_documents(self, arguments: dict) -> list[TextContent]:
        """Handle finding related documents requests."""
        await self._initialize_components()
        
        file_path = arguments.get("file_path", "")
        max_results = arguments.get("max_results", 5)
        
        if not file_path:
            return [TextContent(type="text", text="File path is required")]
        
        try:
            related = self.kg_agent.find_related_documents(file_path, max_results)
            
            if not related:
                return [TextContent(type="text", text=f"No related documents found for '{file_path}'")]
            
            response_text = f"**Related Documents for '{Path(file_path).name}':**\n\n"
            
            for i, doc in enumerate(related, 1):
                file_name = Path(doc['file_path']).name
                similarity = doc['similarity_score']
                
                response_text += f"{i}. **{file_name}**\n"
                response_text += f"   Path: `{doc['file_path']}`\n"
                response_text += f"   Similarity: {similarity:.3f}\n\n"
            
        except Exception as e:
            response_text = f"Error finding related documents: {str(e)}"
        
        return [TextContent(type="text", text=response_text)]
    
    async def _handle_analyze_document_clusters(self, arguments: dict) -> list[TextContent]:
        """Handle document clustering analysis requests."""
        await self._initialize_components()
        
        min_similarity = arguments.get("min_similarity", 0.8)
        
        try:
            clusters = self.kg_agent.analyze_document_clusters(min_similarity)
            
            if not clusters:
                return [TextContent(type="text", text=f"No clusters found with minimum similarity {min_similarity}")]
            
            response_text = f"**Document Clusters (similarity ≥ {min_similarity}):**\n\n"
            response_text += f"Found {len(clusters)} clusters:\n\n"
            
            for i, cluster in enumerate(clusters, 1):
                response_text += f"**Cluster {i}** ({len(cluster)} documents):\n"
                
                for file_path in cluster:
                    file_name = Path(file_path).name
                    response_text += f"  • {file_name}\n"
                    response_text += f"    `{file_path}`\n"
                
                response_text += "\n"
            
        except Exception as e:
            response_text = f"Error analyzing clusters: {str(e)}"
        
        return [TextContent(type="text", text=response_text)]
    
    async def _handle_get_central_documents(self, arguments: dict) -> list[TextContent]:
        """Handle central documents requests."""
        await self._initialize_components()
        
        centrality_type = arguments.get("centrality_type", "betweenness")
        max_results = arguments.get("max_results", 5)
        
        try:
            central_docs = self.kg_agent.get_central_documents(centrality_type, max_results)
            
            if not central_docs:
                return [TextContent(type="text", text="No central documents found")]
            
            response_text = f"**Most Central Documents ({centrality_type.title()} Centrality):**\n\n"
            
            for i, doc in enumerate(central_docs, 1):
                file_name = Path(doc['file_path']).name
                centrality = doc['centrality_score']
                
                response_text += f"{i}. **{file_name}**\n"
                response_text += f"   Path: `{doc['file_path']}`\n"
                response_text += f"   Centrality: {centrality:.3f}\n\n"
            
        except Exception as e:
            response_text = f"Error finding central documents: {str(e)}"
        
        return [TextContent(type="text", text=response_text)]
    
    async def _handle_get_knowledge_graph_stats(self, arguments: dict) -> list[TextContent]:
        """Handle knowledge graph statistics requests."""
        await self._initialize_components()
        
        try:
            stats = self.kg_agent.get_statistics()
            
            response_text = "**Knowledge Graph Statistics:**\n\n"
            response_text += f"- Nodes: {stats['nodes']}\n"
            response_text += f"- Edges: {stats['edges']}\n"
            response_text += f"- Density: {stats['density']:.4f}\n"
            response_text += f"- Average degree: {stats['avg_degree']:.2f}\n"
            response_text += f"- Connected components: {stats['connected_components']}\n"
            response_text += f"- Average clustering: {stats['avg_clustering']:.4f}\n"
            
        except Exception as e:
            response_text = f"Error getting knowledge graph statistics: {str(e)}"
        
        return [TextContent(type="text", text=response_text)]
    
    async def run(self):
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="enhanced-finder",
                    server_version="0.1.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )