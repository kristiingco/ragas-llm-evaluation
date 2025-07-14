# RAGAS LLM Evaluation

This project uses RAGAS for evaluating LLM responses with context precision metrics and supports uploading results to LangSmith for tracking and visualization.

## Setup

1. **Install dependencies:**

    ```bash
    pip install pytest ragas python-dotenv langchain-openai langsmith
    ```

2. **Set up your API keys:**

    Create a `.env` file in the project root with your API keys:

    ```
    OPENAI_API_KEY=your_actual_api_key_here
    RAGAS_API_KEY=your_ragas_cloud_token_here
    LANGCHAIN_API_KEY=your_langsmith_api_key_here
    ```

    **Important:** Never commit your `.env` file to version control. It's already added to `.gitignore`.

    **To get your RAGAS API key:**

    - Visit [ragas.ai](https://ragas.ai)
    - Sign up for an account
    - Go to Account Settings → API Keys
    - Generate and copy your API token

    **To get your LangSmith API key:**

    - Visit [smith.langchain.com](https://smith.langchain.com)
    - Sign up for an account
    - Go to Settings → API Keys
    - Generate and copy your API token

3. **Run the test:**

    ```bash
    python first_test.py
    # or
    pytest first_test.py
    ```

4. **Upload results to LangSmith:**
    ```bash
    python example_langsmith_upload.py
    ```

## Security Best Practices

-   ✅ Use environment variables for API keys
-   ✅ Add `.env` to `.gitignore`
-   ✅ Never commit API keys to version control
-   ✅ Use `.env.example` to show required environment variables

## LangSmith Integration

This project includes LangSmith integration for tracking and visualizing your RAGAS evaluation results. The integration provides:

-   **Automatic Upload**: Results are automatically uploaded to LangSmith after each evaluation
-   **Batch Processing**: Support for evaluating multiple test cases and uploading all results
-   **Rich Metadata**: Each evaluation includes inputs, outputs, and metadata for detailed analysis
-   **Project Organization**: Results are organized by project name for easy management

### Key Features:

1. **`LangSmithRagasIntegration`**: Main class for integrating RAGAS with LangSmith
2. **`evaluate_with_langsmith()`**: Run single evaluation and upload to LangSmith
3. **`batch_evaluate()`**: Run batch evaluation on multiple test cases
4. **`example_langsmith_upload.py`**: Ready-to-use example script

### Viewing Results:

After running evaluations, you can view your results at [smith.langchain.com](https://smith.langchain.com) in the following ways:

-   **Runs**: Individual evaluation runs with detailed inputs/outputs
-   **Projects**: Organized collections of related evaluations
-   **Metrics**: Performance tracking over time
-   **Traces**: Detailed execution traces for debugging

## Alternative Methods for API Keys

1. **System Environment Variables:**

    ```bash
    # Windows
    set OPENAI_API_KEY=your_key_here
    set RAGAS_API_KEY=your_ragas_token_here

    # Linux/Mac
    export OPENAI_API_KEY=your_key_here
    export RAGAS_API_KEY=your_ragas_token_here
    ```

2. **IDE Environment Variables:**

    - Set in your IDE's run configuration
    - Use VS Code's `.vscode/settings.json` (add to `.gitignore`)

3. **CI/CD Secrets:**
    - Use GitHub Secrets for GitHub Actions
    - Use GitLab CI/CD variables
    - Use Azure DevOps variable groups
