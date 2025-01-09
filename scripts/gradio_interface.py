from model import FineTunedModel
import gradio as gr

# Initialize the FineTunedModel with a default model type
model = FineTunedModel(model_type="fine_tuned")

# Create a Gradio interface
def predict_wine_rating(message, model_type):
    """
    Predict the wine rating or suggest similar wines using the selected model.
    
    Args:
        message (str): The input wine description.
        model_type (str): The type of model to use for prediction.
        
    Returns:
        str: The model's response.
    """
    # Update the model type based on user selection
    model.model_type = model_type
    if model_type == "fine_tuned":
        model.model_name = model._get_fine_tuned_model_name()
    else:
        model.model_name = model_type
    return model.predict(message)

# Define the model options
model_options = ["fine_tuned", "gpt-4o-mini", "gpt-4o-2024-08-06"]

iface = gr.Interface(
    fn=predict_wine_rating,  # Function to call for predictions
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your wine description here..."),  # Input text box
        gr.Dropdown(choices=model_options, label="Select Model", value="gpt-4o-mini")  # Dropdown for model selection
    ],
    outputs="text",  # Output text
    title="Wine Rating Estimator",  # Title of the app
    description="Enter a description of the wine, and the model will estimate its rating or suggest similar wines."  # Description
)

# Launch the Gradio interface
iface.launch(inbrowser=True)