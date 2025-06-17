[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/githamza0206-better-finder-mcp-badge.png)](https://mseep.ai/app/githamza0206-better-finder-mcp)

# 🔍 Better Finder MCP

**Git-like workflow for intelligent file search and Claude MCP integration**

Better Finder transforms file discovery with semantic search, hybrid matching, and a familiar Git-style workflow. Index your documents, search with natural language, and integrate seamlessly with Claude Desktop.

## 🚀 Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/GitHamza0206/better-finder-mcp.git
cd better-finder-mcp
uv venv
source .venv/bin/activate
pip install -e .

# Install it globally with uv
uv tool install -e .
```

### Basic Workflow

```bash
# 1. Stage files for indexing (like git add)
better-finder add ~/Documents

# 2. Check what's staged
better-finder status

# 3. Index staged files
better-finder index

# 4. Search your files
better-finder search "quarterly financial reports"
```

## 📋 Commands

### File Staging
- `better-finder add <path>` - Stage files or Documents for indexing
- `better-finder rm <path>` - Remove files from staging  
- `better-finder status` - Show staged files
- `better-finder index` - Index staged files

### Search & Management
- `better-finder search <query>` - Search indexed files
- `better-finder stats` - Show index statistics
- `better-finder clear-index` - Clear all indexed data
- `better-finder server` - Start MCP server for Claude

### Utilities
- `better-finder show <file>` - Display file content
- `better-finder remove-file <file>` - Remove file from index

## 🎯 Key Features

### **Hybrid Search**
Combines semantic search, keyword matching, and fuzzy filename search for comprehensive results.

### **Git-like Workflow**
Familiar staging process gives you precise control over what gets indexed.

### **.betterfinderignore Support**
Use ignore patterns to exclude sensitive files:

```
# .betterfinderignore
secrets/
*.key
temp-*.pdf
node_modules/
```

### **Supported File Types**
- **Documents**: PDF, DOC, DOCX, TXT, MD, RTF, ODT
- **Spreadsheets**: XLSX, XLS, CSV, ODS  
- **Presentations**: PPTX, PPT
- **Data**: JSON, XML

### **Claude MCP Integration**
Start the MCP server to use Better Finder directly within Claude Desktop:

```bash
better-finder server
```

Add to your Claude Desktop config (`~/.cursor/mcp.json`):
```json
{
  "mcpServers": {
    "better-finder": {
      "command": "better-finder",
      "args": ["server"]
    }
  }
}
```

## 🔧 Configuration

Better Finder uses smart defaults:

- **Default scan paths**: `~/Documents`, `~/Desktop`, `~/Downloads`
- **Ignored directories**: `.git`, `node_modules`, `.venv`, cache folders
- **Chunk size**: 2000 characters for better context
- **Search threshold**: 0.4 for comprehensive results

## 💡 Examples

### Document Discovery
```bash
# Find presentations about sales
better-finder search "sales presentation Q4"

# Look for specific file types
better-finder search "budget" --type excel

# Get more results
better-finder search "meeting notes" --max 20
```

### Staging Workflow
```bash
# Stage entire project documentation
better-finder add ./docs

# Remove sensitive files
better-finder rm ./docs/secrets/

# Check what will be indexed
better-finder status

# Index everything staged
better-finder index
```

### MCP Integration
Once the server is running, ask Claude:
- "Search my documents for budget reports"
- "Find presentations about project timelines"  
- "Show me files related to client proposals"

## 🏗️ Architecture

- **File Processors**: Extract content from different formats
- **FAISS Vector Store**: Fast similarity search with sentence transformers
- **SQLite Metadata**: Efficient file metadata and chunk storage
- **Staging System**: Git-like file management with JSON persistence
- **MCP Server**: Model Context Protocol integration for Claude

## 📊 Performance

- **Indexing**: ~100-500 files per minute
- **Search**: Sub-second results
- **Memory**: ~200-500 MB for typical collections
- **Storage**: ~1-5 MB per 1000 documents


## 📝 License

MIT License - see LICENSE file for details.

---

**Made for developers who want intelligent file discovery with familiar Git-like controls.**