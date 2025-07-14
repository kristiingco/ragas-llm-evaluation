import asyncio
import os
from dotenv import load_dotenv
from langsmith_integration import LangSmithRagasIntegration
from utils import load_test_data

# Load environment variables
load_dotenv()

async def main():
    """
    Example of uploading RAGAS evaluation results to LangSmith.
    """
    # Check if LangSmith API key is set
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("⚠️  Warning: LANGCHAIN_API_KEY not set. Please set it to upload to LangSmith.")
        print("You can get your API key from: https://smith.langchain.com/")
        print("Set it in your .env file: LANGCHAIN_API_KEY=your_key_here")
        return
    
    # Initialize LangSmith integration
    integration = LangSmithRagasIntegration(project_name="ragas-evaluation-results")
    
    # Load your test data
    test_data_list = load_test_data("test_5.json")
    
    print(f"Starting evaluation of {len(test_data_list)} test cases...")
    
    # Run evaluation and upload to LangSmith
    results = await integration.batch_evaluate(test_data_list)
    
    print("\n✅ Evaluation completed!")
    print(f"📊 Uploaded {len(results)} evaluation results to LangSmith")
    print(f"🔗 View results at: https://smith.langchain.com/")
    
    # Print summary of results
    for i, result in enumerate(results):
        print(f"\nTest Case {i+1}:")
        print(f"Result type: {type(result)}")
        if hasattr(result, '__dict__'):
            print(f"Result attributes: {list(result.__dict__.keys())}")
        elif isinstance(result, dict):
            print(f"Result keys: {list(result.keys())}")
        
        # Handle different result types
        if hasattr(result, '__dict__'):
            # EvaluationResult object
            for key, value in result.__dict__.items():
                if not key.startswith('_'):
                    if isinstance(value, list) and len(value) > 0:
                        try:
                            # Try to format as float if it's numeric
                            if isinstance(value[0], (int, float)):
                                print(f"  {key}: {value[0]:.3f}")
                            else:
                                print(f"  {key}: {value[0]}")
                        except (ValueError, TypeError):
                            print(f"  {key}: {value[0]}")
                    else:
                        print(f"  {key}: {value}")
        elif isinstance(result, dict):
            # Dictionary result
            for metric_name, score in result.items():
                if isinstance(score, list) and len(score) > 0:
                    try:
                        # Try to format as float if it's numeric
                        if isinstance(score[0], (int, float)):
                            print(f"  {metric_name}: {score[0]:.3f}")
                        else:
                            print(f"  {metric_name}: {score[0]}")
                    except (ValueError, TypeError):
                        print(f"  {metric_name}: {score[0]}")
                else:
                    print(f"  {metric_name}: {score}")
        else:
            # Fallback
            print(f"  Results: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 