# Smart Research Assistant with Context Engineering

## Step 1: Installation (Simple & Error-Free)

### Method 1: Using pip (Recommended)
```bash
# Create a new folder for your project
mkdir smart_research_assistant
cd smart_research_assistant

# Create a virtual environment (recommended)
python -m venv research_env

# Activate virtual environment
# On Windows:
research_env\Scripts\activate
# On Mac/Linux:
source research_env/bin/activate

# Install required packages
pip install langchain
pip install langchain-openai
pip install langchain-community
pip install python-dotenv
pip install requests
pip install beautifulsoup4
pip install faiss-cpu
```

### Method 2: If you get errors with pip
```bash
# Try with --user flag
pip install --user langchain langchain-openai langchain-community python-dotenv requests beautifulsoup4 faiss-cpu

# Or try with --upgrade
pip install --upgrade langchain langchain-openai langchain-community python-dotenv requests beautifulsoup4 faiss-cpu
```

### Alternative: Using Ollama (Local Model - No API Key Needed)
```bash
# Install Ollama from https://ollama.ai
# Then run:
ollama pull llama2
# or
ollama pull mistral

# Install Python packages for Ollama
pip install langchain-ollama
```

## Step 2: Project Structure
Create these files in your project folder:
```
smart_research_assistant/
├── .env
├── main.py
├── research_assistant.py
└── requirements.txt
```

## Step 3: Environment Setup

### Create `.env` file:
```env
# If using OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# If using other APIs
SERPAPI_KEY=your_serpapi_key_here  # Optional for web search
```

### Create `requirements.txt`:
```
langchain>=0.1.0
langchain-openai>=0.0.5
langchain-community>=0.0.10
python-dotenv>=1.0.0
requests>=2.31.0
beautifulsoup4>=4.12.0
faiss-cpu>=1.7.4
langchain-ollama>=0.0.1  # If using Ollama
```

## Step 4: Core Research Assistant Code

### Create `research_assistant.py`:
```python
import os
import requests
from typing import List, Dict, Any
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from bs4 import BeautifulSoup
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SmartResearchAssistant:
    def __init__(self, use_ollama=False, model_name="gpt-3.5-turbo"):
        """
        Initialize the Research Assistant
        
        Args:
            use_ollama (bool): Whether to use Ollama local models
            model_name (str): Model name to use
        """
        self.use_ollama = use_ollama
        self.model_name = model_name
        
        # Initialize the language model
        if use_ollama:
            from langchain_ollama import OllamaLLM
            self.llm = OllamaLLM(model=model_name)
        else:
            self.llm = ChatOpenAI(
                temperature=0.7,
                model_name=model_name,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        
        # Initialize embeddings for vector storage
        if not use_ollama:
            self.embeddings = OpenAIEmbeddings()
        
        # Initialize components
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vector_store = None
        self.research_data = []
        
        # Create prompt templates
        self._create_prompts()
    
    def _create_prompts(self):
        """Create prompt templates for different tasks"""
        
        # Research query analysis prompt
        self.query_analysis_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            Analyze this research query and break it down into key components:
            
            Query: {query}
            
            Please provide:
            1. Main topic/subject
            2. Specific aspects to research
            3. Type of information needed (facts, analysis, opinions, etc.)
            4. Suggested search keywords (3-5 keywords)
            5. Research scope (broad/narrow)
            
            Format your response as JSON with these keys:
            - main_topic
            - aspects
            - info_type
            - keywords
            - scope
            """
        )
        
        # Context synthesis prompt
        self.synthesis_prompt = PromptTemplate(
            input_variables=["query", "context", "sources"],
            template="""
            Based on the research query and gathered information, provide a comprehensive answer:
            
            Query: {query}
            
            Context Information:
            {context}
            
            Sources: {sources}
            
            Please provide:
            1. A clear, well-structured answer to the query
            2. Key findings and insights
            3. Different perspectives if applicable
            4. Confidence level in the information
            5. Suggestions for further research
            
            Make sure to cite sources and provide a balanced view.
            """
        )
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze the research query to understand what information is needed"""
        
        chain = LLMChain(llm=self.llm, prompt=self.query_analysis_prompt)
        
        try:
            result = chain.run(query=query)
            # Try to parse as JSON, fallback to dict if parsing fails
            try:
                return json.loads(result)
            except:
                return {"raw_analysis": result, "main_topic": query}
        except Exception as e:
            print(f"Error analyzing query: {e}")
            return {"main_topic": query, "keywords": [query]}
    
    def search_web(self, keywords: List[str], num_results: int = 5) -> List[Dict]:
        """
        Simple web search using DuckDuckGo (no API key required)
        """
        search_results = []
        
        for keyword in keywords[:3]:  # Limit to first 3 keywords
            try:
                # Using DuckDuckGo Instant Answer API (free)
                url = f"https://api.duckduckgo.com/?q={keyword}&format=json&no_redirect=1"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract relevant information
                    if data.get('Abstract'):
                        search_results.append({
                            'title': data.get('Heading', keyword),
                            'content': data.get('Abstract', ''),
                            'source': data.get('AbstractURL', ''),
                            'keyword': keyword
                        })
                    
                    # Add related topics
                    for topic in data.get('RelatedTopics', [])[:2]:
                        if isinstance(topic, dict) and topic.get('Text'):
                            search_results.append({
                                'title': topic.get('Text', '')[:100],
                                'content': topic.get('Text', ''),
                                'source': topic.get('FirstURL', ''),
                                'keyword': keyword
                            })
            
            except Exception as e:
                print(f"Error searching for {keyword}: {e}")
                continue
        
        return search_results[:num_results]
    
    def scrape_content(self, url: str) -> str:
        """Scrape content from a URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text content
                text = soup.get_text()
                
                # Clean up text
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                return text[:5000]  # Limit to 5000 characters
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
        
        return ""
    
    def process_documents(self, texts: List[str], sources: List[str]):
        """Process and store documents in vector database"""
        documents = []
        
        for i, text in enumerate(texts):
            if text.strip():  # Only process non-empty texts
                # Split text into chunks
                chunks = self.text_splitter.split_text(text)
                
                for chunk in chunks:
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            'source': sources[i] if i < len(sources) else f"source_{i}",
                            'chunk_index': len(documents)
                        }
                    )
                    documents.append(doc)
        
        if documents and not self.use_ollama:
            # Create vector store
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            print(f"Processed {len(documents)} document chunks")
    
    def retrieve_context(self, query: str, k: int = 5) -> List[str]:
        """Retrieve relevant context from vector database"""
        if self.vector_store:
            try:
                docs = self.vector_store.similarity_search(query, k=k)
                return [doc.page_content for doc in docs]
            except:
                pass
        
        # Fallback: return recent research data
        return [item['content'] for item in self.research_data[-k:]]
    
    def conduct_research(self, query: str) -> Dict[str, Any]:
        """Main research function that coordinates all steps"""
        
        print(f"🔍 Starting research on: {query}")
        
        # Step 1: Analyze the query
        print("📊 Analyzing query...")
        analysis = self.analyze_query(query)
        keywords = analysis.get('keywords', [query])
        if isinstance(keywords, str):
            keywords = [keywords]
        
        print(f"🔑 Keywords identified: {keywords}")
        
        # Step 2: Search for information
        print("🌐 Searching web...")
        search_results = self.search_web(keywords)
        
        # Step 3: Process and store information
        texts = []
        sources = []
        
        for result in search_results:
            texts.append(result['content'])
            sources.append(result.get('source', 'Unknown'))
            self.research_data.append(result)
        
        # Scrape additional content if URLs are available
        for result in search_results[:3]:  # Limit scraping
            if result.get('source') and result['source'].startswith('http'):
                scraped_content = self.scrape_content(result['source'])
                if scraped_content:
                    texts.append(scraped_content)
                    sources.append(result['source'])
        
        print(f"📚 Processing {len(texts)} documents...")
        self.process_documents(texts, sources)
        
        # Step 4: Retrieve relevant context
        context_chunks = self.retrieve_context(query)
        context = "\n\n".join(context_chunks)
        
        # Step 5: Synthesize final answer
        print("🧠 Synthesizing answer...")
        chain = LLMChain(llm=self.llm, prompt=self.synthesis_prompt)
        
        try:
            final_answer = chain.run(
                query=query,
                context=context,
                sources="\n".join([f"- {s}" for s in sources[:5]])
            )
        except Exception as e:
            print(f"Error generating answer: {e}")
            final_answer = f"Based on the research, here's what I found about '{query}':\n\n{context[:1000]}..."
        
        return {
            'query': query,
            'analysis': analysis,
            'search_results': search_results,
            'answer': final_answer,
            'sources': sources[:5],
            'confidence': 'Medium'  # Simple confidence measure
        }
    
    def save_research(self, research_result: Dict, filename: str = None):
        """Save research results to file"""
        if not filename:
            safe_query = "".join(c for c in research_result['query'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"research_{safe_query[:30]}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(research_result, f, indent=2, ensure_ascii=False)
            print(f"💾 Research saved to {filename}")
        except Exception as e:
            print(f"Error saving research: {e}")
```

### Create `main.py`:
```python
from research_assistant import SmartResearchAssistant
import json

def main():
    print("🚀 Smart Research Assistant with Context Engineering")
    print("=" * 60)
    
    # Choose your setup
    print("Choose your setup:")
    print("1. OpenAI API (requires API key)")
    print("2. Ollama (local, free)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        print("Using Ollama (make sure you have it installed and a model pulled)")
        assistant = SmartResearchAssistant(use_ollama=True, model_name="llama2")
    else:
        print("Using OpenAI API")
        assistant = SmartResearchAssistant(use_ollama=False, model_name="gpt-3.5-turbo")
    
    print("\n" + "=" * 60)
    print("Research Assistant Ready! 🎯")
    print("Type 'quit' to exit, 'save' to save last research")
    print("=" * 60)
    
    last_research = None
    
    while True:
        query = input("\n📝 Enter your research question: ").strip()
        
        if query.lower() == 'quit':
            print("👋 Goodbye!")
            break
        elif query.lower() == 'save' and last_research:
            assistant.save_research(last_research)
            continue
        elif not query:
            continue
        
        try:
            # Conduct research
            result = assistant.conduct_research(query)
            last_research = result
            
            # Display results
            print("\n" + "=" * 60)
            print("🎯 RESEARCH RESULTS")
            print("=" * 60)
            print(f"Query: {result['query']}")
            print(f"\nAnswer:\n{result['answer']}")
            
            if result.get('sources'):
                print(f"\n📚 Sources:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"{i}. {source}")
            
            print(f"\nConfidence Level: {result.get('confidence', 'Unknown')}")
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error during research: {e}")
            print("Please try again with a different query.")

if __name__ == "__main__":
    main()
```

## Step 5: Running the Assistant

1. **Set up your API key** (if using OpenAI):
   - Get your API key from https://platform.openai.com
   - Add it to your `.env` file

2. **Run the assistant**:
   ```bash
   python main.py
   ```

3. **Example queries to try**:
   - "What are the latest developments in artificial intelligence?"
   - "How does climate change affect ocean temperatures?"
   - "What are the benefits and risks of renewable energy?"

## Features Included

✅ **Query Analysis** - Breaks down research questions intelligently
✅ **Web Search** - Searches multiple sources automatically  
✅ **Content Scraping** - Extracts information from web pages
✅ **Context Engineering** - Processes and chunks information optimally
✅ **Vector Storage** - Stores information for similarity search
✅ **Answer Synthesis** - Generates comprehensive, sourced answers
✅ **Research Saving** - Saves results for later reference
✅ **Local Model Support** - Works with Ollama (no API costs)

## Troubleshooting

### Common Issues:
1. **Import errors**: Make sure all packages are installed
2. **API key errors**: Check your `.env` file and API key
3. **Connection errors**: Check your internet connection
4. **Ollama not working**: Make sure Ollama is installed and a model is pulled

### If you get package conflicts:
```bash
pip install --force-reinstall langchain langchain-openai
```

This research assistant will help you gather, process, and synthesize information from multiple sources with intelligent context engineering!
