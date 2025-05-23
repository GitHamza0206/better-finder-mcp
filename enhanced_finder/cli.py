"""Command-line interface for Enhanced Finder."""

import asyncio
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm

from .config import FinderConfig
from .indexer import DocumentIndexer
from .simple_agents import SimpleSearchAgent, SimpleIndexingAgent
from .staging import StagingManager
from .knowledge_graph import KnowledgeGraphAgent

app = typer.Typer(
    help="Better Finder - Intelligent file search with MCP support",
    no_args_is_help=True,
    add_completion=False
)
console = Console()

def show_logo():
    """Display the Better Finder logo."""
    logo = """
[bold blue]╔══════════════════════════════════════╗[/bold blue]
[bold blue]║[/bold blue]  [bold cyan]🔍 Better Finder MCP[/bold cyan]             [bold blue]║[/bold blue]
[bold blue]║[/bold blue]  [dim]Intelligent file search & indexing[/dim]  [bold blue]║[/bold blue]
[bold blue]╚══════════════════════════════════════╝[/bold blue]
"""
    console.print(logo)
    console.print("")

@app.callback()
def main():
    """Better Finder - Git-like workflow for intelligent file search and MCP integration."""
    show_logo()


@app.command()
def add(
    paths: List[str] = typer.Argument(..., help="Files or directories to stage")
):
    """Stage files for indexing (like git add)."""
    config = FinderConfig()
    staging = StagingManager(config)
    
    total_added = 0
    total_ignored = 0
    total_unsupported = 0
    
    for path in paths:
        result = staging.add_path(path)
        
        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
            continue
        
        total_added += result["total_added"]
        total_ignored += result["total_ignored"]
        total_unsupported += result["total_unsupported"]
        
        if result["total_added"] > 0:
            console.print(f"[green]Added {result['total_added']} files from '{path}'[/green]")
        
        if result["total_ignored"] > 0:
            console.print(f"[yellow]Ignored {result['total_ignored']} files from '{path}' (matched .betterfinderignore)[/yellow]")
        
        if result["total_unsupported"] > 0:
            console.print(f"[dim]Skipped {result['total_unsupported']} files from '{path}' (unsupported extensions)[/dim]")
    
    # Summary
    if total_added > 0:
        console.print(f"\n[green]✓ {total_added} files staged for indexing[/green]")
        console.print("[blue]Run 'better-finder status' to see staged files[/blue]")
        console.print("[blue]Run 'better-finder index' to index staged files[/blue]")
    else:
        console.print("[yellow]No files were staged for indexing[/yellow]")
        if total_unsupported > 0:
            supported_exts = ", ".join(sorted(config.supported_extensions))
            console.print(f"[dim]Supported extensions: {supported_exts}[/dim]")
        if total_ignored > 0:
            console.print("[dim]Check .betterfinderignore file for ignore patterns[/dim]")


@app.command()
def rm(
    paths: List[str] = typer.Argument(..., help="Files or directories to unstage")
):
    """Remove files from staging."""
    config = FinderConfig()
    staging = StagingManager(config)
    
    total_removed = 0
    
    for path in paths:
        result = staging.remove_path(path)
        total_removed += result["total_removed"]
        
        if result["total_removed"] > 0:
            console.print(f"[green]Removed {result['total_removed']} files from staging for '{path}'[/green]")
        else:
            console.print(f"[yellow]No staged files found for '{path}'[/yellow]")
    
    if total_removed > 0:
        console.print(f"\n[green]✓ Total files unstaged: {total_removed}[/green]")
    else:
        console.print("[yellow]No files were unstaged[/yellow]")


@app.command()
def status():
    """Show staged files."""
    config = FinderConfig()
    staging = StagingManager(config)
    
    status_info = staging.get_status()
    
    if status_info["total_staged"] == 0:
        console.print("On branch main")
        console.print("No files staged for indexing")
        console.print("")
        console.print("Use 'better-finder add <path>' to stage files for indexing")
        return
    
    # Git-style output
    console.print("On branch main")
    console.print("Changes to be indexed:")
    console.print("")
    
    # Show staged files in git-style format
    for file_path in sorted(status_info["staged_files"]):
        try:
            # Try to make path relative to current working directory for cleaner display
            relative_path = Path(file_path).relative_to(Path.cwd())
            display_path = str(relative_path)
        except ValueError:
            # If not relative to cwd, show absolute path
            display_path = file_path
        
        console.print(f"        [green]new:[/green]   {display_path}")
    
    console.print("")
    
    # Show summary and commands
    console.print(f"[blue]{status_info['total_staged']} files staged for indexing[/blue]")
    console.print("")
    console.print("Commands:")
    console.print("  better-finder index        - Index all staged files")
    console.print("  better-finder rm <path>    - Unstage files/directories") 
    console.print("  better-finder add <path>   - Stage more files/directories")
    
    if status_info["missing_files"]:
        console.print(f"\n[yellow]Warning: {len(status_info['missing_files'])} staged files no longer exist and were removed from staging[/yellow]")


@app.command()
def search(
    query: str = typer.Argument(..., help="What to search for"),
    max_results: int = typer.Option(10, "--max", "-m", help="Maximum results"),
    file_type: Optional[str] = typer.Option(None, "--type", "-t", help="File type filter"),
    output_format: str = typer.Option("table", "--format", "-f", help="Output format")
):
    """Search indexed files."""
    asyncio.run(_search_async(query, max_results, file_type, output_format))


async def _search_async(query: str, max_results: int, file_type: Optional[str], output_format: str):
    """Async search implementation."""
    config = FinderConfig()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing search...", total=None)
        
        indexer = DocumentIndexer(config)
        indexer.load_or_create_index()
        
        if indexer.index is None or indexer.index.ntotal == 0:
            console.print("[red]No indexed files found. Please run 'enhanced-finder index' first.[/red]")
            return
        
        progress.update(task, description="Searching files...")
        
        search_agent = SimpleSearchAgent(config, indexer)
        results = await search_agent.search(query)
        results = results[:max_results]
        
        progress.update(task, description="Formatting results...")
    
    if not results:
        console.print(f"[yellow]No files found matching '{query}'[/yellow]")
        return
    
    # Display results based on format
    if output_format == "json":
        import json
        console.print(json.dumps(results, indent=2))
    elif output_format == "simple":
        for result in results:
            console.print(f"{result['file_path']}")
    else:
        _display_search_table(results, query)


def _display_search_table(results, query):
    """Display search results in a rich table."""
    table = Table(title=f"Search Results for '{query}'")
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Path", style="blue")
    table.add_column("Score", style="green", justify="center")
    table.add_column("Type", style="magenta", justify="center")
    table.add_column("Preview", style="white")
    
    for result in results:
        file_name = result.get("file_name", "")
        file_path = result.get("file_path", "")
        score = f"{result.get('similarity_score', 0):.3f}"
        search_type = result.get("search_type", "")
        preview = result.get("content_snippet", "")[:50] + "..." if result.get("content_snippet") else ""
        
        # Truncate long paths
        display_path = file_path
        if len(display_path) > 60:
            display_path = "..." + display_path[-57:]
        
        table.add_row(file_name, display_path, score, search_type, preview)
    
    console.print(table)


@app.command()
def index(
    path: Optional[str] = typer.Argument(None, help="Specific directory to index"),
    full: bool = typer.Option(False, "--full", "-f", help="Index all configured paths"),
    incremental: bool = typer.Option(False, "--incremental", "-i", help="Index only new files"),
    staged: bool = typer.Option(True, "--staged/--no-staged", help="Use staged files")
):
    """Index files for searching (uses staged files by default)."""
    asyncio.run(_index_async(path, full, incremental, staged))


async def _index_async(path: Optional[str], full: bool, incremental: bool, staged: bool = True):
    """Async indexing implementation."""
    config = FinderConfig()
    config.ensure_directories()
    
    indexer = DocumentIndexer(config)
    indexer.load_or_create_index()
    
    indexing_agent = SimpleIndexingAgent(config, indexer)
    staging = StagingManager(config)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        if full:
            task = progress.add_task("Performing full reindex...", total=None)
            stats = await indexing_agent.full_reindex()
        elif incremental:
            task = progress.add_task("Performing incremental index...", total=None)
            stats = await indexing_agent.incremental_index()
        elif path:
            path_obj = Path(path)
            if not path_obj.exists():
                console.print(f"[red]Path does not exist: {path}[/red]")
                return
            
            task = progress.add_task(f"Indexing {path}...", total=None)
            stats = await indexer.index_directory(path_obj)
        elif staged:
            # Index staged files
            staged_files = staging.get_staged_files()
            if not staged_files:
                console.print("[yellow]No files staged for indexing.[/yellow]")
                console.print("[blue]Use 'better-finder add <path>' to stage files first.[/blue]")
                return
            
            task = progress.add_task(f"Indexing {len(staged_files)} staged files...", total=None)
            stats = {"processed": 0, "indexed": 0, "errors": 0}
            
            for file_path in staged_files:
                stats["processed"] += 1
                try:
                    if await indexer.index_file(file_path):
                        stats["indexed"] += 1
                except Exception as e:
                    stats["errors"] += 1
                    console.print(f"[red]Error indexing {file_path}: {e}[/red]")
                
                # Save periodically
                if stats["indexed"] % 50 == 0:
                    indexer.save_index()
            
            indexer.save_index()
            
            # Clear staging after successful indexing
            if stats["errors"] == 0:
                staging.clear_staging()
                console.print("[green]✓ Staging cleared after successful indexing[/green]")
        else:
            console.print("[yellow]Please specify --full, --incremental, provide a path, or use staged files (default).[/yellow]")
            console.print("[blue]Use 'better-finder add <path>' to stage files for indexing.[/blue]")
            return
    
    # Display results
    panel_content = f"""
[green]✓[/green] Files processed: {stats['processed']}
[green]✓[/green] Files indexed: {stats['indexed']}
[red]✗[/red] Errors: {stats['errors']}
"""
    
    console.print(Panel(panel_content, title="Indexing Complete", border_style="green"))


@app.command()
def stats():
    """Show index statistics."""
    config = FinderConfig()
    
    try:
        indexer = DocumentIndexer(config)
        indexer.load_or_create_index()
        
        stats = indexer.get_stats()
        
        table = Table(title="Enhanced Finder Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Documents", str(stats['total_documents']))
        table.add_row("Total Chunks", str(stats['total_chunks']))
        table.add_row("Vector Count", str(stats['vector_count']))
        table.add_row("Index Size", f"{stats['index_size_mb']:.2f} MB")
        
        console.print(table)
        
        # Show configuration
        config_table = Table(title="Configuration")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="blue")
        
        config_table.add_row("Supported Extensions", str(len(config.supported_extensions)))
        config_table.add_row("Max File Size", f"{config.max_file_size_mb} MB")
        config_table.add_row("Embedding Model", config.embedding_model)
        config_table.add_row("Max Search Results", str(config.max_search_results))
        
        console.print(config_table)
        
        # Show scan paths
        paths_table = Table(title="Scan Paths")
        paths_table.add_column("Path", style="blue")
        paths_table.add_column("Status", style="green")
        
        for scan_path in config.scan_paths:
            status = "✓ Exists" if scan_path.exists() else "✗ Missing"
            paths_table.add_row(str(scan_path), status)
        
        console.print(paths_table)
        
    except Exception as e:
        console.print(f"[red]Error getting statistics: {e}[/red]")



@app.command()
def show(file_path: str = typer.Argument(..., help="File path")):
    """Display file content."""
    config = FinderConfig()
    indexer = DocumentIndexer(config)
    
    path_obj = Path(file_path)
    if not path_obj.exists():
        console.print(f"[red]File does not exist: {file_path}[/red]")
        return
    
    file_data = indexer.processor_manager.process_file(path_obj)
    
    if file_data.get("error"):
        console.print(f"[red]Error reading file: {file_data['error']}[/red]")
        return
    
    content = file_data.get("content", "")
    
    # Create info panel
    info_content = f"""
[cyan]File:[/cyan] {path_obj.name}
[cyan]Path:[/cyan] {file_path}
[cyan]Size:[/cyan] {file_data.get('file_size', 0)} bytes
[cyan]Type:[/cyan] {path_obj.suffix}
"""
    
    console.print(Panel(info_content, title="File Information", border_style="blue"))
    
    # Display content
    if len(content) > 5000:
        if Confirm.ask(f"File is large ({len(content)} characters). Show full content?"):
            console.print(Panel(content, title="File Content", border_style="green"))
        else:
            truncated = content[:5000] + "\n\n... (truncated)"
            console.print(Panel(truncated, title="File Content (Truncated)", border_style="yellow"))
    else:
        console.print(Panel(content, title="File Content", border_style="green"))


@app.command()
def clear_index(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation")
):
    """Clear all indexed data."""
    config = FinderConfig()
    
    if not confirm:
        console.print("[yellow]This will delete all indexed data and cannot be undone.[/yellow]")
        if not Confirm.ask("Are you sure you want to clear the index?"):
            console.print("[blue]Operation cancelled.[/blue]")
            return
    
    try:
        # Remove vector store files
        if config.vector_store_path.exists():
            import shutil
            shutil.rmtree(config.vector_store_path)
            console.print(f"[green]✓[/green] Removed vector store: {config.vector_store_path}")
        
        # Remove metadata database
        if config.metadata_db_path.exists():
            config.metadata_db_path.unlink()
            console.print(f"[green]✓[/green] Removed metadata database: {config.metadata_db_path}")
        
        # Remove index directory if empty
        if config.index_path.exists() and not any(config.index_path.iterdir()):
            config.index_path.rmdir()
            console.print(f"[green]✓[/green] Removed empty index directory: {config.index_path}")
        
        console.print(Panel(
            "[green]Index cleared successfully![/green]\n\nRun 'better-finder index --full' to rebuild the index.",
            title="Clear Complete",
            border_style="green"
        ))
    
    except Exception as e:
        console.print(f"[red]Error clearing index: {e}[/red]")


@app.command()
def remove_file(
    file_path: str = typer.Argument(..., help="File to remove from index"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation")
):
    """Remove file from index."""
    config = FinderConfig()
    path_obj = Path(file_path).resolve()
    
    if not confirm:
        if not Confirm.ask(f"Remove '{path_obj}' from index?"):
            console.print("[blue]Operation cancelled.[/blue]")
            return
    
    try:
        indexer = DocumentIndexer(config)
        indexer.load_or_create_index()
        
        # Check if file is in index
        if not indexer.is_file_indexed(path_obj):
            console.print(f"[yellow]File not found in index: {path_obj}[/yellow]")
            return
        
        # Remove from index (this would need to be implemented in indexer)
        removed = indexer.remove_file_from_index(path_obj)
        
        if removed:
            console.print(f"[green]✓[/green] Removed file from index: {path_obj}")
        else:
            console.print(f"[yellow]File was not found in index: {path_obj}[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error removing file from index: {e}[/red]")


@app.command("kg-build")
def knowledge_graph_build(
    similarity_threshold: float = typer.Option(0.7, "--threshold", "-t", help="Similarity threshold for edges")
):
    """Build knowledge graph from indexed documents."""
    asyncio.run(_build_knowledge_graph_async(similarity_threshold))


async def _build_knowledge_graph_async(similarity_threshold: float):
    """Async knowledge graph building implementation."""
    config = FinderConfig()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing knowledge graph...", total=None)
        
        indexer = DocumentIndexer(config)
        indexer.load_or_create_index()
        
        if indexer.index is None or indexer.index.ntotal == 0:
            console.print("[red]No indexed files found. Please run 'better-finder index' first.[/red]")
            return
        
        kg_agent = KnowledgeGraphAgent(config, indexer)
        
        progress.update(task, description="Building knowledge graph...")
        stats = await kg_agent.build_graph(similarity_threshold)
        
        progress.update(task, description="Saving knowledge graph...")
    
    # Display results
    panel_content = f"""
[green]✓[/green] Nodes added: {stats['nodes_added']}
[green]✓[/green] Edges added: {stats['edges_added']}
[red]✗[/red] Errors: {stats['errors']}
[blue]i[/blue] Similarity threshold: {similarity_threshold}
"""
    
    console.print(Panel(panel_content, title="Knowledge Graph Built", border_style="green"))


@app.command("kg-related")
def knowledge_graph_related(
    file_path: str = typer.Argument(..., help="File path to find related documents"),
    max_results: int = typer.Option(5, "--max", "-m", help="Maximum related documents")
):
    """Find documents related to a specific file using knowledge graph."""
    config = FinderConfig()
    
    try:
        indexer = DocumentIndexer(config)
        indexer.load_or_create_index()
        kg_agent = KnowledgeGraphAgent(config, indexer)
        
        related = kg_agent.find_related_documents(file_path, max_results)
        
        if not related:
            console.print(f"[yellow]No related documents found for '{file_path}'[/yellow]")
            return
        
        table = Table(title=f"Documents Related to '{Path(file_path).name}'")
        table.add_column("File Name", style="cyan")
        table.add_column("Path", style="blue")
        table.add_column("Similarity", style="green", justify="center")
        
        for doc in related:
            file_name = Path(doc['file_path']).name
            display_path = doc['file_path']
            if len(display_path) > 60:
                display_path = "..." + display_path[-57:]
            similarity = f"{doc['similarity_score']:.3f}"
            
            table.add_row(file_name, display_path, similarity)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error finding related documents: {e}[/red]")


@app.command("kg-clusters")
def knowledge_graph_clusters(
    min_similarity: float = typer.Option(0.8, "--min-sim", "-s", help="Minimum similarity for clusters")
):
    """Analyze document clusters in knowledge graph."""
    config = FinderConfig()
    
    try:
        indexer = DocumentIndexer(config)
        indexer.load_or_create_index()
        kg_agent = KnowledgeGraphAgent(config, indexer)
        
        clusters = kg_agent.analyze_document_clusters(min_similarity)
        
        if not clusters:
            console.print(f"[yellow]No clusters found with minimum similarity {min_similarity}[/yellow]")
            return
        
        console.print(f"[green]Found {len(clusters)} clusters with similarity ≥ {min_similarity}[/green]\n")
        
        for i, cluster in enumerate(clusters, 1):
            console.print(f"[bold cyan]Cluster {i}[/bold cyan] ({len(cluster)} documents):")
            for file_path in cluster:
                file_name = Path(file_path).name
                console.print(f"  • {file_name}")
                console.print(f"    [dim]{file_path}[/dim]")
            console.print()
        
    except Exception as e:
        console.print(f"[red]Error analyzing clusters: {e}[/red]")


@app.command("kg-central")
def knowledge_graph_central(
    centrality_type: str = typer.Option("betweenness", "--type", "-t", help="Centrality type"),
    max_results: int = typer.Option(5, "--max", "-m", help="Maximum results")
):
    """Find most central/important documents in knowledge graph."""
    config = FinderConfig()
    
    try:
        indexer = DocumentIndexer(config)
        indexer.load_or_create_index()
        kg_agent = KnowledgeGraphAgent(config, indexer)
        
        central_docs = kg_agent.get_central_documents(centrality_type, max_results)
        
        if not central_docs:
            console.print("[yellow]No central documents found[/yellow]")
            return
        
        table = Table(title=f"Most Central Documents ({centrality_type.title()} Centrality)")
        table.add_column("Rank", style="magenta", justify="center")
        table.add_column("File Name", style="cyan")
        table.add_column("Path", style="blue")
        table.add_column("Centrality", style="green", justify="center")
        
        for i, doc in enumerate(central_docs, 1):
            file_name = Path(doc['file_path']).name
            display_path = doc['file_path']
            if len(display_path) > 60:
                display_path = "..." + display_path[-57:]
            centrality = f"{doc['centrality_score']:.3f}"
            
            table.add_row(str(i), file_name, display_path, centrality)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error finding central documents: {e}[/red]")


@app.command("kg-visualize")
def knowledge_graph_visualize(
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    layout: str = typer.Option("spring", "--layout", "-l", help="Graph layout"),
    figsize: str = typer.Option("12,8", "--figsize", "-f", help="Figure size (width,height)")
):
    """Visualize knowledge graph."""
    config = FinderConfig()
    
    try:
        # Parse figsize
        width, height = map(int, figsize.split(','))
        
        indexer = DocumentIndexer(config)
        indexer.load_or_create_index()
        kg_agent = KnowledgeGraphAgent(config, indexer)
        
        output_path = kg_agent.visualize_knowledge_graph(
            output_path=output,
            layout=layout,
            figsize=(width, height)
        )
        
        console.print(f"[green]✓ Knowledge graph visualization saved to: {output_path}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error visualizing knowledge graph: {e}[/red]")


@app.command("kg-stats")
def knowledge_graph_stats():
    """Show knowledge graph statistics."""
    config = FinderConfig()
    
    try:
        indexer = DocumentIndexer(config)
        indexer.load_or_create_index()
        kg_agent = KnowledgeGraphAgent(config, indexer)
        
        stats = kg_agent.get_statistics()
        
        table = Table(title="Knowledge Graph Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Nodes", str(stats['nodes']))
        table.add_row("Edges", str(stats['edges']))
        table.add_row("Density", f"{stats['density']:.4f}")
        table.add_row("Average Degree", f"{stats['avg_degree']:.2f}")
        table.add_row("Connected Components", str(stats['connected_components']))
        table.add_row("Average Clustering", f"{stats['avg_clustering']:.4f}")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error getting knowledge graph statistics: {e}[/red]")


@app.command()
def server():
    """Start MCP server for Claude integration."""
    from .mcp_server import EnhancedFinderMCPServer
    
    console.print("[green]Starting Enhanced Finder MCP Server...[/green]")
    console.print("[blue]Connect your MCP client to use the server.[/blue]")
    
    server = EnhancedFinderMCPServer()
    asyncio.run(server.run())


def cli_main():
    """CLI entry point."""
    app()


if __name__ == "__main__":
    cli_main()