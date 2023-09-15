# Data Poem Assignment

## Problem Statement

A time series-based regression dataset, that has 21 input features and 1 target variable. Each checkpoint represents the number cycles that have been rented from that particular checkpoint on the corresponding date and target is the sum of number of cycles that have been rented on that particular date. In addition to that you are also provided with weather data of the particular city. Also all the check points are in a single city for which weather data is provided. You need to analyse following mentioned points:

## Project Goals

The main objectives of this project are:

1. **Exploratory Data Analysis (EDA):** This involves handling anomalous values, gaining insights, and preparing the data for modeling.

2. **Implement Deep Learning Algorithms:** Apply deep learning algorithms to achieve a good Mean Absolute Percentage Error (MAPE) and overall error performance.

3. **Auto Hyperparameter Tuning:** Utilize auto hyperparameter tuning algorithms in combination with the deep learning models to optimize performance.

## Code Structure

The project code is divided into several sections, as outlined below:


### Section 1: Data Preprocessing

This section focuses on preparing the data for analysis. It involves the following steps:

1. **Library Imports:**
   - Importing necessary Python libraries such as pandas (for data manipulation), matplotlib (for plotting), and seaborn (for enhanced visualization).

2. **Data Loading:**
   - Reading the data from two CSV files (`Checkpoint_count.csv` and `weather.csv`) and storing them in separate DataFrames (`checkpoint_data` and `weather_data`).

3. **Basic EDA:**
   - Displaying the initial rows of the data to get a quick overview.
   - Obtaining summary statistics like mean, standard deviation, min, max, etc.
   - Checking the data types of each column (e.g., numeric, categorical).
   - Identifying missing values in the dataset.

### Section 2: Data Cleaning and Feature Engineering

This section focuses on preparing the data for analysis and modeling. It involves the following steps:

1. **Date Manipulation:**
   - Defining a function (`swap_dates`) to handle date formats where the day and month will be swapped. The function ensures the date is in the format `%d-%m-%Y`.

2. **Date Processing:**
   - Applying the `swap_dates` function to the 'Date' column in the 'checkpoint_data' DataFrame to ensure uniform date formatting.

3. **Column Operations:**
   - Dropping the original 'Date' column from 'checkpoint_data'.
   - Assigning the updated dates to the DataFrame.

4. **Data Merging:**
   - Renaming the 'Date/Time' column in the 'weather_data' DataFrame to 'Date'.
   - Performing an inner join between 'weather_data' and 'checkpoint_data' based on the 'Date' column, resulting in the 'merged_data' DataFrame.

5. **Column Cleanup:**
   - Handling irrelevant columns by dropping them for better clarity.
   - Renaming columns to improve readability and maintain consistency.

6. **Missing Value Handling:**
   - Using imputation techniques like K-Nearest Neighbors (KNN) imputation to fill missing values in specific columns.
   - Addressing special cases where values are represented as strings.

7. **Data Type Conversion:**
   - Converting the 'date' column to datetime format for ease of handling.

8. **Feature Engineering:**
   - Extracting the weekday information from the 'date' column and creating a new column 'weekend' to indicate whether the day is a weekend.

9. **Final Column Selection:**
   - Dropping unnecessary columns like 'day', 'spd_of_max_gust', 'max_temp', 'min_temp', and 'weekend'.

### Section 3: Exploratory Data Analysis (EDA)

This section is focused on gaining insights and visualizing relationships within the dataset:

1. **Pair Plots:**
   - Generating pair plots to visually explore relationships between multiple variables.

2. **Box Plots:**
   - Creating box plots to identify potential outliers and understand the distribution of the target variable across different categories.

### Section 4: Data Modeling

This section involves building and training machine learning models:

#### Deep Learning Model 1 (Feedforward Neural Network):

1. **Model Construction:**
   - Building a simple feedforward neural network with an input layer, hidden layers, and an output layer.

2. **Model Compilation:**
   - Compiling the model by specifying the optimizer and loss function.

3. **Model Training:**
   - Training the model on the preprocessed data.

4. **Model Evaluation:**
   - Assessing the model's performance using the Mean Absolute Percentage Error (MAPE).
   - MAPE: 0.3733

#### Deep Learning Model 2 (LSTM):

1. **Data Reshaping:**
   - Preparing the data for LSTM by reshaping it into the required format.

2. **Model Construction:**
   - Building an LSTM model with input and output layers.

3. **Model Compilation:**
   - Compiling the LSTM model.

4. **Model Training:**
   - Training the LSTM model on the preprocessed data.

5. **Model Evaluation:**
   - Assessing the LSTM model's performance using MAPE.
   - MAPE: 0.3765

#### Hyperparameter Tuning:

1. **Hyperparameter Search Space:**
   - Defining the range of hyperparameters to explore using Keras Tuner.

2. **Random Search:**
   - Implementing random search for hyperparameter tuning.

3. **Best Hyperparameters:**
   - Retrieving the best hyperparameters from the search.

4. **Model Rebuilding:**
   - Reconstructing the model with the optimized hyperparameters.

5. **Model Training and Evaluation:**
   - Training and evaluating the model with the tuned hyperparameters.
   - MAPE: 0.4281


## Usage

To run this project, follow these steps:

1. Ensure you have Python installed on your system.
2. Install the required libraries `pandas==1.3.3, matplotlib==3.4.3, seaborn==0.11.2, scikit-learn==0.24.2`
`tensorflow==2.6.0, keras-tuner==1.0.4`
3. Execute the code cells in your preferred environment, ensuring the necessary data files (`Checkpoint_count.csv` and `weather.csv`) are in the same directory.

## Model Outputs

1. **Feedforward Neural Network:**
   - MAPE: 0.3733

2. **LSTM Model:**
   - MAPE: 0.3765

3. **Hyperparameter Tuned Model:**
   - MAPE: 0.4281

### Conclusion

In this data science project, we conducted a thorough analysis of a time series-based regression dataset with 21 input features and 1 target variable. The dataset represents the number of cycles rented from various checkpoints on corresponding dates, and it also includes weather data for the specific city.

We followed a structured approach to preprocess the data, which included handling date formats, merging datasets, imputing missing values, and performing feature engineering. Exploratory Data Analysis (EDA) provided valuable insights into the relationships between variables.

We implemented two deep learning models: a Feedforward Neural Network and an LSTM model. The models were trained and evaluated, and the `Feedforward Neural Network achieved` a Mean Absolute Percentage Error (MAPE) of approximately `0.3733`, while the `LSTM model` achieved a MAPE of approximately `0.3765`.

Furthermore, we conducted hyperparameter tuning using the Keras Tuner library, optimizing the model performance. The `hyperparameter tuned model` achieved a MAPE of approximately `0.4281`.

Overall, this project showcases the application of deep learning models in analyzing time series data. The optimized models provide a strong foundation for predicting the number of cycles rented, which can be invaluable for planning and resource allocation.

For any questions or feedback, please contact `Saurabh Harak` at `jobsforsaurabhharak@gmail.com`.
