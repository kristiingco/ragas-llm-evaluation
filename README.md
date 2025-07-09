# RAGAS LLM Evaluation

This project uses RAGAS for evaluating LLM responses with context precision metrics.

## Setup

1. **Install dependencies:**

    ```bash
    pip install pytest ragas python-dotenv langchain-openai
    ```

2. **Set up your API key:**

    Create a `.env` file in the project root with your OpenAI API key:

    ```
    OPENAI_API_KEY=your_actual_api_key_here
    ```

    **Important:** Never commit your `.env` file to version control. It's already added to `.gitignore`.

3. **Run the test:**
    ```bash
    python first_test.py
    # or
    pytest first_test.py
    ```

## Security Best Practices

-   ✅ Use environment variables for API keys
-   ✅ Add `.env` to `.gitignore`
-   ✅ Never commit API keys to version control
-   ✅ Use `.env.example` to show required environment variables

## Alternative Methods for API Keys

1. **System Environment Variables:**

    ```bash
    # Windows
    set OPENAI_API_KEY=your_key_here

    # Linux/Mac
    export OPENAI_API_KEY=your_key_here
    ```

2. **IDE Environment Variables:**

    - Set in your IDE's run configuration
    - Use VS Code's `.vscode/settings.json` (add to `.gitignore`)

3. **CI/CD Secrets:**
    - Use GitHub Secrets for GitHub Actions
    - Use GitLab CI/CD variables
    - Use Azure DevOps variable groups
