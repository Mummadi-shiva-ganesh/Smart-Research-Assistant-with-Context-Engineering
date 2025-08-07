import os
import requests
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from bs4 import BeautifulSoup
import json
from dotenv import load_dotenv
import time

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
            try:
                from langchain_ollama import OllamaLLM
                self.llm = OllamaLLM(model=model_name)
                print(f"✅ Using Ollama with {model_name}")
            except ImportError:
                print("❌ Ollama not installed. Install with: pip install langchain-ollama")
                print("🔄 Falling back to a simple text processor...")
                self.llm = None
        else:
            try:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    print("❌ No OpenAI API key found. Please add it to your .env file")
                    print("🔄 Switching to free mode (no AI processing)...")
                    self.llm = None
                else:
                    self.llm = ChatOpenAI(
                        temperature=0.7,
                        model_name=model_name,
                        openai_api_key=api_key
                    )
                    print(f"✅ Using OpenAI {model_name}")
            except Exception as e:
                print(f"❌ Error initializing OpenAI: {e}")
                print("🔄 Switching to free mode...")
                self.llm = None
        
        # Initialize embeddings for vector storage (only if using OpenAI)
        if not use_ollama and self.llm:
            try:
                self.embeddings = OpenAIEmbeddings()
            except:
                print("⚠️ Embeddings disabled due to API issues")
                self.embeddings = None
        else:
            self.embeddings = None
        
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
            
            Keep response concise and structured.
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
            4. Suggestions for further research
            
            Make sure to cite sources and provide a balanced view.
            """
        )
    
    def analyze_query_simple(self, query: str) -> Dict[str, Any]:
        """Simple query analysis without AI"""
        words = query.lower().split()
        keywords = []
        
        # Extract important words (simple approach)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where'}
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        
        return {
            "main_topic": query,
            "keywords": keywords[:5],  # Limit to 5 keywords
            "aspects": ["general information"],
            "info_type": "facts and analysis",
            "scope": "broad"
        }
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze the research query to understand what information is needed"""
        
        if not self.llm:
            return self.analyze_query_simple(query)
        
        try:
            chain = LLMChain(llm=self.llm, prompt=self.query_analysis_prompt)
            result = chain.run(query=query)
            
            # Simple parsing since JSON might fail
            lines = result.split('\n')
            analysis = {"main_topic": query, "keywords": [query]}
            
            for line in lines:
                if 'keywords' in line.lower() or 'keyword' in line.lower():
                    # Extract keywords from the line
                    words = line.split(':')[-1].strip()
                    keywords = [w.strip(' ",.-') for w in words.split(',')]
                    analysis["keywords"] = [k for k in keywords if len(k) > 2][:5]
                    break
            
            return analysis
            
        except Exception as e:
            print(f"⚠️ AI analysis failed: {e}")
            return self.analyze_query_simple(query)
    
    def search_web(self, keywords: List[str], num_results: int = 5) -> List[Dict]:
        """
        Simple web search using DuckDuckGo (no API key required)
        """
        search_results = []
        
        print(f"🔍 Searching for: {', '.join(keywords[:3])}")
        
        for keyword in keywords[:3]:  # Limit to first 3 keywords
            try:
                # Using DuckDuckGo Instant Answer API (free)
                url = f"https://api.duckduckgo.com/?q={keyword.replace(' ', '+')}&format=json&no_redirect=1"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract relevant information
                    if data.get('Abstract'):
                        search_results.append({
                            'title': data.get('Heading', keyword),
                            'content': data.get('Abstract', ''),
                            'source': data.get('AbstractURL', 'DuckDuckGo'),
                            'keyword': keyword
                        })
                    
                    # Add related topics
                    for topic in data.get('RelatedTopics', [])[:2]:
                        if isinstance(topic, dict) and topic.get('Text'):
                            search_results.append({
                                'title': topic.get('Text', '')[:100],
                                'content': topic.get('Text', ''),
                                'source': topic.get('FirstURL', 'DuckDuckGo'),
                                'keyword': keyword
                            })
                
                # Add a small delay to be respectful
                time.sleep(0.5)
            
            except Exception as e:
                print(f"⚠️ Error searching for {keyword}: {e}")
                continue
        
        # If no results, create a basic result
        if not search_results:
            search_results.append({
                'title': f"Information about {keywords[0] if keywords else 'your query'}",
                'content': f"Research topic: {keywords[0] if keywords else 'your query'}. Please try more specific keywords or check your internet connection.",
                'source': 'Local',
                'keyword': keywords[0] if keywords else 'query'
            })
        
        return search_results[:num_results]
    
    def scrape_content(self, url: str) -> str:
        """Scrape content from a URL"""
        try:
            if not url.startswith('http'):
                return ""
                
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
                
                return text[:3000]  # Limit to 3000 characters
            
        except Exception as e:
            print(f"⚠️ Error scraping {url}: {e}")
        
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
        
        if documents and self.embeddings:
            try:
                # Create vector store
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                print(f"📚 Processed {len(documents)} document chunks")
            except Exception as e:
                print(f"⚠️ Vector store creation failed: {e}")
                print("📝 Storing documents in simple format...")
        
        # Store in simple format as backup
        self.research_data.extend([
            {'content': doc.page_content, 'source': doc.metadata.get('source', 'Unknown')}
            for doc in documents
        ])
    
    def retrieve_context(self, query: str, k: int = 5) -> List[str]:
        """Retrieve relevant context from vector database"""
        if self.vector_store:
            try:
                docs = self.vector_store.similarity_search(query, k=k)
                return [doc.page_content for doc in docs]
            except Exception as e:
                print(f"⚠️ Vector search failed: {e}")
        
        # Fallback: return recent research data
        recent_data = self.research_data[-k:] if len(self.research_data) >= k else self.research_data
        return [item['content'] for item in recent_data]
    
    def synthesize_simple(self, query: str, context: str, sources: List[str]) -> str:
        """Simple synthesis without AI"""
        answer = f"Research Results for: {query}\n\n"
        
        if context.strip():
            # Take the most relevant parts
            context_parts = context.split('\n\n')[:3]
            answer += "Key Findings:\n"
            for i, part in enumerate(context_parts, 1):
                if part.strip():
                    answer += f"{i}. {part.strip()[:300]}...\n\n"
        else:
            answer += "Limited information found. Please try different keywords.\n\n"
        
        if sources:
            answer += "Sources:\n"
            for i, source in enumerate(sources[:3], 1):
                answer += f"• {source}\n"
        
        answer += "\n💡 Tip: Try more specific keywords for better results."
        return answer
    
    def conduct_research(self, query: str) -> Dict[str, Any]:
        """Main research function that coordinates all steps"""
        
        print(f"🔍 Starting research on: {query}")
        
        # Step 1: Analyze the query
        print("📊 Analyzing query...")
        analysis = self.analyze_query(query)
        keywords = analysis.get('keywords', [query])
        if isinstance(keywords, str):
            keywords = [keywords]
        if not keywords:
            keywords = [query]
        
        print(f"🔑 Keywords identified: {keywords}")
        
        # Step 2: Search for information
        print("🌐 Searching web...")
        search_results = self.search_web(keywords)
        print(f"📄 Found {len(search_results)} results")
        
        # Step 3: Process and store information
        texts = []
        sources = []
        
        for result in search_results:
            if result['content']:
                texts.append(result['content'])
                sources.append(result.get('source', 'Unknown'))
        
        # Scrape additional content if URLs are available
        scraped_count = 0
        for result in search_results[:2]:  # Limit scraping to 2 URLs
            if result.get('source') and result['source'].startswith('http') and scraped_count < 2:
                print(f"📖 Scraping: {result['source'][:50]}...")
                scraped_content = self.scrape_content(result['source'])
                if scraped_content:
                    texts.append(scraped_content)
                    sources.append(result['source'])
                    scraped_count += 1
        
        print(f"📚 Processing {len(texts)} documents...")
        self.process_documents(texts, sources)
        
        # Step 4: Retrieve relevant context
        context_chunks = self.retrieve_context(query)
        context = "\n\n".join(context_chunks)
        
        # Step 5: Synthesize final answer
        print("🧠 Synthesizing answer...")
        
        if self.llm:
            try:
                chain = LLMChain(llm=self.llm, prompt=self.synthesis_prompt)
                final_answer = chain.run(
                    query=query,
                    context=context[:4000],  # Limit context length
                    sources="\n".join([f"- {s}" for s in sources[:5]])
                )
            except Exception as e:
                print(f"⚠️ AI synthesis failed: {e}")
                final_answer = self.synthesize_simple(query, context, sources)
        else:
            final_answer = self.synthesize_simple(query, context, sources)
        
        return {
            'query': query,
            'analysis': analysis,
            'search_results': search_results,
            'answer': final_answer,
            'sources': sources[:5],
            'confidence': 'Medium'
        }
    
    def save_research(self, research_result: Dict, filename: str = None):
        """Save research results to file"""
        if not filename:
            # Create safe filename
            safe_query = "".join(c for c in research_result['query'] if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_query = safe_query.replace(' ', '_')[:30]
            filename = f"research_{safe_query}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(research_result, f, indent=2, ensure_ascii=False)
            print(f"💾 Research saved to {filename}")
            return filename
        except Exception as e:
            print(f"❌ Error saving research: {e}")
            return None