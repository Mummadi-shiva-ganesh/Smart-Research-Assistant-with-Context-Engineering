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