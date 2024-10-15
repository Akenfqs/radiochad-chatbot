   # Radio Chad Chatbot

   This is an AI-powered chatbot that answers questions about the Radio Chad podcast.

   ## Setup

   1. Clone this repository:
      ```
      git clone <your-repository-url>
      cd <repository-name>
      ```

   2. Install the required packages:
      ```
      pip install -r requirements.txt
      ```

   3. Set up your OpenAI API key:
      - Sign up for an OpenAI account and get your API key.
      - Set your API key as an environment variable:
        ```
        export OPENAI_API_KEY='your-api-key-here'
        ```

   4. Prepare your podcast transcripts:
      - Create a folder named `transcripts` in the project directory.
      - Place your podcast transcript text files in this folder.

   ## Running the Chatbot

   Run the following command in your terminal:
   ```
   streamlit run app.py
   ```

   The chatbot will open in your default web browser.

   ## Usage

   - Type your question about the Radio Chad podcast in the input field.
   - The AI will respond based on the information in the podcast transcripts.
   - Use the "Clear Chat History" button to start a new conversation.
