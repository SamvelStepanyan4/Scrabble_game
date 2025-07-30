# Scrabble Rating Prediction - Kaggle Competition Solution

## Overview
This repository contains a solution for a Kaggle competition focused on predicting player ratings in Scrabble games. The solution uses machine learning techniques, specifically an XGBoost model, to predict ratings based on game and turn data. The approach includes data preprocessing, feature engineering, and hyperparameter tuning to optimize the model's performance.

## Dataset
The dataset consists of four CSV files:
- **train.csv**: Contains training data with game IDs, player nicknames, scores, and ratings.
- **test.csv**: Contains test data with similar features but missing ratings, which the model predicts.
- **turns.csv**: Details individual turns in each game, including turn number, player nickname, rack, move, points, and turn type.
- **games.csv**: Contains game metadata such as time control, game end reason, winner, lexicon, and duration.

The goal is to predict the `rating` for players in the test set based on these features.

## Prerequisites
To run the code, you need the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`

You can install these dependencies using:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

## File Structure
- **Scrabble.ipynb**: The main Jupyter Notebook containing the solution, including data loading, preprocessing, feature engineering, model training, and prediction.
- **train.csv**: Training dataset.
- **test.csv**: Test dataset.
- **turns.csv**: Turn-level data.
- **games.csv**: Game-level metadata.
- **sample_submission.csv**: Template for submission file.
- **sub.csv**: Output file with predicted ratings.
- **README.md**: This file.

## Approach
1. **Data Loading and Exploration**:
   - Load the datasets (`train.csv`, `test.csv`, `turns.csv`, `games.csv`) using pandas.
   - Display initial rows to understand the data structure.

2. **Data Preprocessing**:
   - Handle missing values in `train`, `test`, and `turns` datasets by filling them with random choices based on the distribution of non-missing values.
   - Drop irrelevant or redundant columns (e.g., `move`, `rack`, `nickname`, `first`, `created_at`, `time_control_name`, `lexicon`, `game_end_reason`, `winner`, `initial_time_seconds`, `increment_seconds`, `game_duration_seconds`, `turn_number`, `location`, `turn_type`) to simplify the model.
   - Encode categorical variables using `LabelEncoder` to convert them into numerical format.

3. **Feature Engineering**:
   - Merge `train` and `test` datasets with `games` and `turns` to incorporate game and turn-level information.
   - Concatenate `train` and `test` datasets for consistent preprocessing.

4. **Model Training**:
   - Use an `XGBRegressor` model from the `xgboost` library for rating prediction.
   - Perform hyperparameter tuning with `GridSearchCV` to find the optimal number of estimators (`n_estimators` tested with values `[60, 65, 70]`).
   - Train the model on the preprocessed training data and predict ratings for the test set.

5. **Submission**:
   - Generate predictions and save them in `sub.csv` formatted according to `sample_submission.csv`.

## How to Run
1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```
2. Place the dataset files (`train.csv`, `test.csv`, `turns.csv`, `games.csv`, `sample_submission.csv`) in the same directory as the notebook.
3. Open `Scrabble.ipynb` in Jupyter Notebook or JupyterLab.
4. Run all cells in the notebook to preprocess the data, train the model, and generate predictions.
5. The output will be saved as `sub.csv` in the same directory.

## Results
The solution uses a tuned XGBoost model with the best parameter `n_estimators=70`. The predictions are saved in `sub.csv` for submission to the Kaggle competition.

## Notes
- The correlation analysis shows that `score_x` (final game score) and `points` (turn points) have a moderate positive correlation with `rating`, indicating their importance in the prediction.
- The preprocessing steps ensure that no missing values remain and categorical features are properly encoded.
- The notebook assumes that the test set labels are consistent with the training set after encoding.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Kaggle for providing the dataset and competition platform.
- The open-source community for libraries like `pandas`, `scikit-learn`, and `xgboost`.