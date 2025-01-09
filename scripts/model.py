from openai import OpenAI
from dotenv import load_dotenv
import os
import gradio as gr

class FineTunedModel:
    """
    A class to interact with OpenAI models for wine rating prediction.
    Users can choose between a pre-trained GPT model or a fine-tuned model.
    """

    def __init__(self, model_type="gpt-4o-mini"):
        """
        Initialize the FineTunedModel class.
        
        Args:
            model_type (str): The type of model to use. Options: "fine_tuned", "gpt-4o-mini", "gpt-4o-2024-08-06".
        """
        # Load environment variables and set up OpenAI API key
        load_dotenv()
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            self.openai_api_key = input("Please enter your OpenAI API key: ").strip()
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is required to proceed.")
        
        # Set the OpenAI API key as an environment variable
        os.environ['OPENAI_API_KEY'] = self.openai_api_key

        # Initialize OpenAI client
        self.openai = OpenAI()

        # Set the model type
        self.model_type = model_type

        # Retrieve the fine-tuned model name if applicable
        if self.model_type == "fine_tuned":
            self.model_name = self._get_fine_tuned_model_name()
        else:
            self.model_name = self.model_type  # Use the specified pre-trained model

    def _get_fine_tuned_model_name(self):
        """
        Retrieve the name of the fine-tuned model from OpenAI.
        
        Returns:
            str: The name of the fine-tuned model.
        """
        job_list = self.openai.fine_tuning.jobs.list(limit=1)
        job_id = job_list.data[0].id
        job_details = self.openai.fine_tuning.jobs.retrieve(job_id)
        return job_details.fine_tuned_model

    def predict(self, message):
        """
        Predict the wine rating and suggest a similar wine using the selected model.
        
        Args:
            message (str): The input wine description.
            
        Returns:
            str: The predicted rating and a suggestion for a similar wine.
        """
        system_message = (
            "You are a wine rating system. Your task is to predict the rating of the wine based on the description, if user ask about it or "
            "suggest at least two similar wines with their tasting notes. "
            "Your answer must be short and complete."
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": message}
        ]
        
        # Call the OpenAI API to generate a response
        response = self.openai.chat.completions.create(
            model=self.model_name,  # Use the fine-tuned model
            messages=messages,  # Prepare the messages for the API call
            seed=42,  # Set a random seed for reproducibility
            max_tokens=150  # Limit the response length
        )
        
        # Extract the model's reply
        reply = response.choices[0].message.content
        return reply

