import os
import asyncio
from typing import Dict, Any, List
from dotenv import load_dotenv
from langsmith import Client
from langsmith.run_helpers import traceable
from ragas import evaluate, EvaluationDataset
from ragas.metrics import ResponseRelevancy, FactualCorrectness, LLMContextPrecisionWithoutReference
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings
from compatible_chat_openai import CompatibleChatOpenAI
from utils import load_test_data, get_llm_response

# Load environment variables
load_dotenv()

class LangSmithRagasIntegration:
    def __init__(self, project_name: str = "ragas-evaluation"):
        """
        Initialize LangSmith integration for RAGAS evaluation results.
        
        Args:
            project_name: Name of the LangSmith project to upload results to
        """
        self.client = Client()
        self.project_name = project_name
        
        # Ensure project exists
        try:
            self.client.read_project(project_name=project_name)
        except:
            self.client.create_project(project_name=project_name)
    
    @traceable(name="ragas-evaluation", project_name="ragas-evaluation")
    async def evaluate_with_langsmith(self, 
                                    test_data: Dict[str, Any], 
                                    metrics: List = None,
                                    llm_wrapper = None) -> Any:
        """
        Run RAGAS evaluation and upload results to LangSmith.
        
        Args:
            test_data: Test data containing question, reference, etc.
            metrics: List of RAGAS metrics to evaluate
            llm_wrapper: LLM wrapper for evaluation
            
        Returns:
            Dictionary containing evaluation results
        """
        if llm_wrapper is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            llm_wrapper = CompatibleChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        
        if metrics is None:
            embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
            metrics = [
                ResponseRelevancy(llm=llm_wrapper, embeddings=embeddings), 
                FactualCorrectness(llm=llm_wrapper),
                LLMContextPrecisionWithoutReference(llm=llm_wrapper)
            ]
        
        # Get LLM response
        response_dict = get_llm_response(test_data)
        response_json = response_dict.json()
        answer = response_json.get("answer", "")
        retrieved_docs = response_json.get("retrieved_docs", [])
        
        # Extract retrieved contexts
        retrieved_contexts = []
        for i in range(min(3, len(retrieved_docs))):
            if "page_content" in retrieved_docs[i]:
                retrieved_contexts.append(retrieved_docs[i]["page_content"])
        
        # Create evaluation dataset
        from ragas import SingleTurnSample
        sample = SingleTurnSample(
            user_input=test_data["question"],
            reference=test_data.get("reference", ""),
            response=answer,
            retrieved_contexts=retrieved_contexts
        )
        
        dataset = EvaluationDataset([sample])
        
        # Run evaluation
        results = evaluate(dataset=dataset, metrics=metrics)
        
        # Upload results to LangSmith
        try:
            self._upload_to_langsmith(test_data, results, answer, retrieved_contexts)
        except Exception as e:
            print(f"Warning: Failed to upload to LangSmith: {e}")
            print("Results will still be returned locally")
        
        return results
    
    def _upload_to_langsmith(self, 
                            test_data: Dict[str, Any], 
                            results: Any,
                            answer: str,
                            retrieved_contexts: List[str]):
        """
        Upload evaluation results to LangSmith.
        
        Args:
            test_data: Original test data
            results: RAGAS evaluation results
            answer: LLM response
            retrieved_contexts: Retrieved context documents
        """
        # Convert RAGAS results to dictionary format
        results_dict = {}
        if hasattr(results, '__dict__'):
            # Handle EvaluationResult object
            for key, value in results.__dict__.items():
                if not key.startswith('_'):
                    results_dict[key] = value
        elif isinstance(results, dict):
            results_dict = results
        else:
            # Fallback: convert to string representation
            results_dict = {"results": str(results)}
        
        # Create a run with evaluation results
        run_data = {
            "name": "ragas-evaluation",
            "inputs": {
                "question": test_data["question"],
                "reference": test_data.get("reference", ""),
                "retrieved_contexts": retrieved_contexts
            },
            "outputs": {
                "answer": answer,
                "evaluation_results": results_dict
            },
            "metadata": {
                "evaluation_metrics": list(results_dict.keys()),
                "test_data_id": test_data.get("id", "unknown")
            }
        }
        
        # Create the run in LangSmith
        run = self.client.create_run(
            project_name=self.project_name,
            run_type="chain",  # Required parameter
            **run_data
        )
        
        print(f"Uploaded evaluation results to LangSmith run: {run.id}")
        return run
    
    async def batch_evaluate(self, 
                           test_data_list: List[Dict[str, Any]], 
                           llm_wrapper = None) -> List[Any]:
        """
        Run batch evaluation on multiple test cases and upload all results to LangSmith.
        
        Args:
            test_data_list: List of test data dictionaries
            llm_wrapper: LLM wrapper for evaluation
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for i, test_data in enumerate(test_data_list):
            print(f"Evaluating test case {i+1}/{len(test_data_list)}")
            result = await self.evaluate_with_langsmith(test_data, llm_wrapper=llm_wrapper)
            results.append(result)
        
        return results

# Example usage function
async def run_evaluation_with_langsmith():
    """
    Example function showing how to use the LangSmith integration.
    """
    # Initialize the integration
    integration = LangSmithRagasIntegration(project_name="ragas-evaluation-demo")
    
    # Load test data
    test_data_list = load_test_data("test_5.json")
    
    # Run batch evaluation
    results = await integration.batch_evaluate(test_data_list)
    
    print("Evaluation completed and uploaded to LangSmith!")
    print(f"Results: {results}")
    
    return results

if __name__ == "__main__":
    asyncio.run(run_evaluation_with_langsmith()) 