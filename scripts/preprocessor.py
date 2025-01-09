import os
import re
import random
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from huggingface_hub import login
import pickle

class DataPreprocessor:
    """
    A class to handle data preprocessing for the wine rating prediction project.
    """

    def __init__(self):
        """
        Initialize the DataPreprocessor class.
        """
        # Load environment variables
        load_dotenv()

        # Prompt the user to input their Hugging Face token if not already set
        self.hf_token = os.getenv('HF_TOKEN')
        if not self.hf_token:
            self.hf_token = input("Please enter your Hugging Face token: ").strip()
            if not self.hf_token:
                raise ValueError("Hugging Face token is required to proceed.")

        # Set the token as an environment variable
        os.environ['HF_TOKEN'] = self.hf_token

        # Log in to Hugging Face Hub
        login(self.hf_token, add_to_git_credential=True)

        # Load dataset
        self.dataset = load_dataset("alfredodeza/wine-ratings")
        self.full_df = None
        self.wine_df = None
        self.train = None
        self.test = None

    def load_and_combine_data(self):
        """
        Load the dataset and combine all splits into one DataFrame.
        """
        # Convert splits to pandas DataFrames
        train_df = self.dataset['train'].to_pandas()
        val_df = self.dataset['validation'].to_pandas()
        test_df = self.dataset['test'].to_pandas()

        # Combine all splits into one DataFrame
        self.full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    def fill_missing_variety(self):
        """
        Fill missing 'variety' values using a RandomForestClassifier.
        """
        # Split the data into known and unknown 'variety'
        known_data = self.full_df[self.full_df['variety'].notnull()]
        unknown_data = self.full_df[self.full_df['variety'].isnull()]

        # Define features (X) and target variable (y)
        X = known_data[['region', 'notes', 'rating']]
        y = known_data['variety']

        # Create a preprocessor to handle categorical and numerical features
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['region', 'notes']),
                ('num', 'passthrough', ['rating'])
            ])

        # Create a pipeline: preprocess data and then apply a RandomForestClassifier
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        # Train the model on the known data
        model.fit(X, y)

        # Predict missing 'variety' values for the unknown data
        unknown_data['variety'] = model.predict(unknown_data[['region', 'notes', 'rating']])

        # Combine the known and predicted data
        self.full_df = pd.concat([known_data, unknown_data], ignore_index=True)

    def clean_and_extract_features(self):
        """
        Clean the data and extract additional features.
        """
        # Drop rows with missing values
        self.wine_df = self.full_df.dropna()

        # Extract country/state from the 'region' column
        self.wine_df['country_state'] = self.wine_df['region'].str.split(', ').str[-1]

        # Extract year from the 'name' column
        self.wine_df['year'] = self.wine_df['name'].apply(self.extract_year)

    @staticmethod
    def extract_year(name):
        """
        Extract the year from the 'name' column using regex.
        
        Args:
            name (str): The wine name.
            
        Returns:
            int or None: The extracted year or None if no valid year is found.
        """
        match = re.search(r'\b(19[0-9]{2}|20[0-9]{2})\b', name)
        return int(match.group()) if match else None

    @staticmethod
    def scrub(text):
        """
        Clean up the provided text by removing unnecessary characters and whitespace.
        
        Args:
            text (str): The input text to clean.
            
        Returns:
            str: The cleaned text.
        """
        # Remove unwanted characters and extra spaces
        text = re.sub(r'[:\[\\"\]]', ' ', text)  # Replace special characters with a space
        text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with a single space
        return text

    def prepare_prompts(self):
        """
        Prepare prompts for training and testing.
        """
        items = []
        # Iterate over each row in the DataFrame
        for _, row in self.wine_df.iterrows():
            # Check if the 'year' column is not NaN
            if pd.notna(row['year']):
                # Format the string with year included
                formatted_item = (
                    "What is the wine rating to the nearest whole number in range 85 to 100?\n\n"
                    f"The {self.scrub(row['variety'])} name is {self.scrub(row['name'])} and it was produced in {self.scrub(row['region'])} in {int(row['year'])}.\n"
                    f"{self.scrub(row['notes'])}\n\n"
                    f"The rating is {int(row['rating'])}"
                )
                items.append(formatted_item)
            else:
                # Format the string without year
                formatted_item = (
                    "What is the wine rating to the nearest whole number in range 85 to 100?\n\n"
                    f"The {self.scrub(row['variety'])} name is {self.scrub(row['name'])} and it was produced in {self.scrub(row['region'])}.\n"
                    f"{self.scrub(row['notes'])}\n\n"
                    f"The rating is {int(row['rating'])}"
                )
                items.append(formatted_item)

        # Shuffle and split the data
        random.seed(42)
        random.shuffle(items)
        self.train = items[:30_000]
        self.test = items[30_000:]

    def split_data(self):
        """
        Split the data into training and testing sets.
        """
        train_prompts = [item for item in self.train]
        train_ratings = [int(item[-3:]) for item in self.train]
        test_prompts = [self.test_prompt(item) for item in self.test]
        test_ratings = [int(item[-3:]) for item in self.test]
        # Create a Dataset from the lists
        train_dataset = Dataset.from_dict({"prompt": train_prompts, "rating": train_ratings})  # Training dataset
        test_dataset = Dataset.from_dict({"prompt": test_prompts, "rating": test_ratings})  # Test dataset

        return train_dataset, test_dataset

    @staticmethod
    def test_prompt(prompt):
        """
        Return a prompt suitable for testing, with the actual rating removed.
        
        Args:
            prompt (str): The input prompt containing the wine description and rating.
            
        Returns:
            str: The prompt with the actual rating removed.
        """
        PREFIX = "The rating is "
        return prompt.split(PREFIX)[0] + PREFIX
    
    def save_datasets(self, train_dataset, test_dataset, save_dir=None):
        """
        Save the training and testing datasets to the specified directory.

        Args:
            train_dataset (Dataset): The training dataset.
            test_dataset (Dataset): The testing dataset.
            save_dir (str): The directory where the datasets should be saved.
        """
        # Use the current working directory if no directory is specified
        if save_dir is None:
            save_dir = os.getcwd()

        # Create a folder named 'data' inside the specified directory
        data_dir = os.path.join(save_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Save the datasets as pickle files in the 'data' folder
        train_path = os.path.join(data_dir, "train_wine.pkl")
        test_path = os.path.join(data_dir, "test_wine.pkl")

        # Save the training dataset
        with open(train_path, 'wb') as file:
            pickle.dump(train_dataset, file)  # Save training dataset to a pickle file

        # Save the testing dataset
        with open(test_path, 'wb') as file:
            pickle.dump(test_dataset, file)  # Save test dataset to a pickle file

        print(f"Training dataset saved to: {train_path}")
        print(f"Testing dataset saved to: {test_path}")
        
if __name__ == "__main__":
    # Initialize the DataPreprocessor
    preprocessor = DataPreprocessor()

    # Load and preprocess the data
    preprocessor.load_and_combine_data()
    preprocessor.fill_missing_variety()
    preprocessor.clean_and_extract_features()
    preprocessor.prepare_prompts()

    # Split the data into training and testing sets
    train_dataset, test_dataset = preprocessor.split_data()


    # Save the datasets to the current working directory
    preprocessor.save_datasets(train_dataset, test_dataset)