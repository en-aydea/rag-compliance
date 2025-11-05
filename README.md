Banking Compliance RAG Pipeline
This project is an advanced Retrieval-Augmented Generation (RAG) pipeline designed to automatically audit call center transcripts for regulatory compliance. It analyzes conversations between bank agents and customers, checking them against Turkish Banking Regulation and Supervision Agency (BDDK) documents to identify policy violations and information omissions.

🚀 The Problem & Our Solution
Manually auditing thousands of call transcripts is slow, costly, and inconsistent. This pipeline automates the process by using a "Dual-Stage Analysis" RAG architecture.

The primary challenge is that customers use conversational language (e.g., "Can I split my bill?"), while legal documents use formal language (e.g., "Debt restructuring and installment limits"). Our pipeline solves this by:

Segmenting the call to find relevant "Query/Response" pairs.

Transforming the conversational query into a formal, keyword-rich search query.

Retrieving the correct legal text from a vector database using the transformed query.

Analyzing the agent's response against the retrieved legal context to find violations or omissions.

✨ Core Features
Dual-Stage RAG Pipeline: Uses multiple, specialized LLM calls for segmentation, query transformation, and final analysis.

Query Transformation: Intelligently rewrites user queries to bridge the "semantic gap" between conversational and legal language, ensuring high-accuracy retrieval.

Local & Private Embeddings: Uses Hugging Face Sentence Transformers to run embeddings on your local CPU, keeping data private and saving on API costs.

Local Vector Store: Employs ChromaDB for a persistent, local-first vector database.

Persistent Job Queue: Uses SQLite (via SQLAlchemy) to manage a queue of calls to be processed (calls_input) and to store all structured analysis results (compliance_analysis_output).

Asynchronous Batch Processing: The main pipeline (main.py) processes multiple calls in parallel for high throughput.

🛠️ Tech Stack
Python 3.10+

Orchestration: LangChain

LLM (Reasoning): OpenAI (GPT-4 / GPT-4o)

Embeddings (Local): Hugging Face Sentence Transformers (e.g., paraphrase-multilingual-MiniLM-L12-v2)

Vector Database: ChromaDB

Data/Job Management: SQLite & SQLAlchemy

Data Loading: Pandas & openpyxl

Environment: python-dotenv