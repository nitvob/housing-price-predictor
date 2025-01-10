# Ames Housing Price Prediction

## Project Overview

I developed this project to predict housing prices based on the extensive features of properties in Ames, Iowa. By using a dataset of over 2,900 properties with 82 features, I applied data preprocessing, exploratory data analysis (EDA), and machine learning techniques to build predictive models. My primary goal was to deliver accurate housing price predictions by leveraging advanced statistical and machine learning methods.

## Key Features

The project utilizes the Ames Housing dataset, which provides detailed information about residential properties, including physical characteristics, neighborhood, and sale prices. I implemented various regression models, such as Linear Regression, Random Forest Regressor, Decision Tree Regressor, Support Vector Regressor (SVR), and K-Nearest Neighbors (KNN). To evaluate these models, I used performance metrics like Mean Squared Error (MSE) and Cross-Validation Scores. Additionally, I included data visualizations for EDA and feature analysis using Seaborn and Matplotlib.

## Technical Details

### Data Preprocessing

To ensure the dataset was ready for analysis, I handled missing values in columns such as `Lot Frontage`, `Garage Yr Blt`, and `Alley`. I encoded categorical variables using Label Encoding and standardized numerical features for models that required scaled inputs. These steps were critical in preparing the data for effective machine learning workflows.

### Exploratory Data Analysis (EDA)

I examined the distributions and relationships of various features using correlation matrices and scatter plots. During this process, I identified key predictors of housing prices, including `OverallQual`, `GrLivArea`, and `YearBuilt`. This analysis provided valuable insights into the dataset and informed the model-building process.

### Machine Learning Workflow

For the machine learning phase, I split the data into training and testing sets with an 80-20 ratio. I applied K-Fold Cross-Validation to ensure robust model evaluation and fine-tuned hyperparameters to optimize performance. By using these techniques, I was able to improve the accuracy and reliability of the predictive models.

### Tools and Libraries

- **Programming Language**: Python
- **Libraries**:
  - Data Manipulation: Pandas, Numpy
  - Statistical Analysis: Scipy, Statsmodels
  - Visualization: Seaborn, Matplotlib
  - Machine Learning: Scikit-learn

### Results

Among the models I developed, the Random Forest Regressor achieved the lowest MSE, demonstrating its effectiveness in handling large datasets with many features. I also created visualizations to interpret feature importance and evaluate model performance, which highlighted the key contributors to housing price predictions.

## Setup Instructions

### Prerequisites

To run this project, you will need Python 3.8 or later and Jupyter Notebook installed on your system.

### Installation

To set up the project, start by cloning the repository:

```bash
git clone https://github.com/nitvob/housing-price-predictor.git
cd housing-price-predictor
```

Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

### Dataset

The dataset is included in the project folder as `AmesHousing.csv`. If it is not present, you can download it from [the official source](https://jse.amstat.org/v19n3/decock/AmesHousing.txt) and save it to the project directory.

### Running the Project

To run the project, launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the `main.ipynb` file and run the cells sequentially. This will allow you to load and preprocess the data, perform EDA, train and evaluate machine learning models, and view the results and visualizations.

## Skills and Knowledge Gained

Through this project, I deepened my understanding of data preprocessing techniques, including handling missing values, encoding categorical variables, and standardizing numerical features. I strengthened my ability to conduct exploratory data analysis by identifying patterns, correlations, and key predictors of outcomes. Additionally, I honed my expertise in implementing and evaluating machine learning models, applying techniques such as K-Fold Cross-Validation and hyperparameter tuning to optimize performance.

I also gained valuable experience with Python's powerful data science ecosystem, including libraries like Pandas, Numpy, Scikit-learn, and Matplotlib. Working on this project enhanced my skills in communicating insights effectively through visualizations and interpreting model results to inform decisions. Overall, this project served as a platform to showcase my ability to approach complex problems, derive actionable insights, and build robust solutions using machine learning.
