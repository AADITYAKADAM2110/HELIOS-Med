import json
from modules.engine import helios_app


if __name__ == "__main__":
    print("\n" + "="*80)
    print("HELIOS Medical Q&A System")
    print("="*80 + "\n")
    
    result = helios_app.invoke({"question": "What is the aim of this document?"})
    
    # Format and display the result
    print("\n" + "-"*80)
    print("RESULT:")
    print("-"*80)
    print(f"Question: {result.get('question')}")
    print(f"\nGeneration:\n{result.get('generation')}")
    print(f"\nSources: {result.get('sources')}")
    print(f"\nRelevant: {result.get('is_relevant')}")
    
    # Save to file for easy viewing
    output_file = "test_output.json"
    with open(output_file, "w") as f:
        # Convert documents to strings for JSON serialization
        result_copy = result.copy()
        if "documents" in result_copy:
            result_copy["documents"] = [
                {
                    "id": doc.id,
                    "page_content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                }
                for doc in result_copy["documents"]
            ]
        json.dump(result_copy, f, indent=2, default=str)
    
    print(f"\nFull result saved to: {output_file}")
    print("="*80 + "\n")
    
    # no pause so automated runs terminate cleanly
