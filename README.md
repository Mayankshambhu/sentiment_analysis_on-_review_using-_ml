# Sentiment Analysis on Reviews Using Machine Learning

## Overview

This project focuses on sentiment analysis of customer reviews using machine learning techniques. The goal is to classify reviews into various sentiment categories (e.g., positive, negative, neutral) and provide visualizations and insights into the analysis results.

## Features

- **Sentiment Classification:** Uses a pre-trained BERT model fine-tuned for sentiment classification.
- **Data Processing:** Handles both review text and titles for comprehensive analysis.
- **Visualization:** Generates heatmaps, bar charts, and word clouds for detailed insights.
- **File Upload:** Supports various file types for input data and provides functionality for analyzing individual reviews.
- **Progress Indicators:** Includes progress bars for large file uploads and analysis.
- **User Interface:** Built with Streamlit for an interactive web experience.

## Installation

To get started, clone the repository and install the required dependencies.

```bash
git clone https://github.com/Mayankshambhu/sentiment_analysis_on-_review_using-_ml.git
cd sentiment_analysis_on-_review_using-_ml
pip install -r requirements.txt
```

## Usage

1. **Run the Application:**

   ```bash
   streamlit run app.py
   ```

2. **Upload File:**
   - Upload your data file (CSV, JSON, etc.) through the Streamlit app interface.
   - Ensure the file contains review data with appropriate columns (e.g., `review`, `title`).

3. **Analyze Reviews:**
   - Select the type of analysis you want to perform.
   - View the results, including sentiment classifications and visualizations.

4. **Review Analysis:**
   - For files with more than 10 rows, heatmaps, bar charts, and word clouds will be displayed.
   - For files with 10 or fewer rows, individual review analysis will be shown.

## Dataset

You can download datasets for your analysis from the following link:

- [Amazon Reviews Dataset](https://amazon-reviews-2023.github.io/)

Please ensure that the dataset you choose is preprocessed and formatted according to the requirements of the project.

## Training the Model

To train the `model2`, follow these steps:

1. **Prepare the Data:**
   - Ensure that your dataset is preprocessed and saved in the appropriate format.

2. **Train `model2`:**
   - Run the training script located in `train_model2.py` to train the model. Make sure to adjust hyperparameters and paths as needed.

   ```bash
   python train_model2.py
   ```

3. **Save the Model:**
   - After training, save the model to a specified directory for later use.

4. **Evaluate the Model:**
   - Evaluate the trained model using the provided evaluation scripts to ensure its performance.

## Contributing

Feel free to contribute to the project by opening issues or submitting pull requests. Ensure that your changes are tested and adhere to the coding standards of the project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please contact Mayank Shambhu at mayankshambhu2004@gmail.com.
