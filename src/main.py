from dotenv import load_dotenv
import os
import openai
from controller import app

# Load environment variables
load_dotenv()

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Run the app
if __name__ == '__main__':
    app.run(port=5000)
