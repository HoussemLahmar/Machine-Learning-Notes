# Exploratory Data Analysis (EDA) Steps

## 1. Data Loading

**Python Libraries:**
- `pandas`
- `numpy`

**Functions:**
- `pandas.read_csv()`
- `pandas.read_excel()`
- `pandas.read_sql()`

**Example:**
```python
import pandas as pd

# Load data from a CSV file
df = pd.read_csv('data.csv')
```

## 2. Data Inspection

**Python Libraries:**
- `pandas`

**Functions:**
- `df.head()`
- `df.tail()`
- `df.info()`
- `df.describe()`

**Example:**
```python
# Inspect the first few rows
print(df.head())

# Inspect data types and missing values
print(df.info())

# Get summary statistics
print(df.describe())
```

## 3. Data Cleaning

**Python Libraries:**
- `pandas`
- `numpy`

**Functions:**
- `df.dropna()`
- `df.fillna()`
- `df.drop_duplicates()`
- `df.replace()`

**Example:**
```python
# Drop missing values
df = df.dropna()

# Fill missing values with the mean
df.fillna(df.mean(), inplace=True)

# Drop duplicate rows
df = df.drop_duplicates()
```

## 4. Data Transformation

**Python Libraries:**
- `pandas`
- `numpy`

**Functions:**
- `df.apply()`
- `df.map()`
- `df.groupby()`
- `pd.get_dummies()`

**Example:**
```python
# Apply a transformation to a column
df['column'] = df['column'].apply(lambda x: x*2)

# Convert categorical variables to dummy variables
df = pd.get_dummies(df, columns=['categorical_column'])
```

## 5. Data Visualization

**Python Libraries:**
- `matplotlib`
- `seaborn`
- `pandas` (for built-in plotting)

**Functions:**
- `df.plot()`
- `plt.hist()`
- `sns.scatterplot()`
- `sns.heatmap()`
- `sns.pairplot()`

**Example:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram of a column
plt.hist(df['column'])
plt.show()

# Scatter plot of two columns
sns.scatterplot(x='column1', y='column2', data=df)
plt.show()

# Heatmap of correlation matrix
sns.heatmap(df.corr(), annot=True)
plt.show()
```

## 6. Correlation and Relationship Analysis

**Python Libraries:**
- `pandas`
- `seaborn`

**Functions:**
- `df.corr()`
- `sns.heatmap()`
- `sns.pairplot()`

**Example:**
```python
# Calculate correlations
correlation_matrix = df.corr()

# Heatmap of correlation matrix
sns.heatmap(correlation_matrix, annot=True)
plt.show()
```

## 7. Feature Engineering

**Python Libraries:**
- `pandas`
- `numpy`

**Functions:**
- `df.apply()`
- `df.groupby()`
- `df.rolling()`

**Example:**
```python
# Create new features
df['new_feature'] = df['feature1'] * df['feature2']

# Apply rolling window statistics
df['rolling_mean'] = df['feature'].rolling(window=3).mean()
```

## 8. Data Splitting

**Python Libraries:**
- `sklearn.model_selection`

**Functions:**
- `train_test_split()`

**Example:**
```python
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2, random_state=42)
```

## Preparing Data Before Building the Model

1. **Feature Scaling:**
   - Standardize or normalize features to ensure they are on a similar scale.
   - Libraries: `sklearn.preprocessing`

   **Example:**
   ```python
   from sklearn.preprocessing import StandardScaler

   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   ```

2. **Encoding Categorical Variables:**
   - Convert categorical variables into numerical form using techniques like one-hot encoding.
   - Libraries: `pandas`, `sklearn.preprocessing`

   **Example:**
   ```python
   from sklearn.preprocessing import OneHotEncoder

   encoder = OneHotEncoder(sparse=False)
   X_train_encoded = encoder.fit_transform(X_train[['categorical_feature']])
   ```

3. **Handling Missing Values:**
   - Impute missing values or remove rows/columns with missing data.
   - Libraries: `pandas`, `sklearn.impute`

   **Example:**
   ```python
   from sklearn.impute import SimpleImputer

   imputer = SimpleImputer(strategy='mean')
   X_train_imputed = imputer.fit_transform(X_train)
   ```

4. **Outlier Detection and Handling:**
   - Identify and handle outliers to avoid their impact on the model.
   - Libraries: `pandas`, `numpy`

   **Example:**
   ```python
   # Remove outliers using z-score
   from scipy import stats

   z_scores = stats.zscore(df)
   df = df[(z_scores < 3).all(axis=1)]
   ```

5. **Feature Selection:**
   - Select important features and discard irrelevant ones to improve model performance.
   - Libraries: `sklearn.feature_selection`

   **Example:**
   ```python
   from sklearn.feature_selection import SelectKBest, f_classif

   selector = SelectKBest(score_func=f_classif, k=10)
   X_train_selected = selector.fit_transform(X_train, y_train)
   ```

## Summary

Exploratory Data Analysis (EDA) involves loading, inspecting, cleaning, transforming, visualizing, and analyzing data to understand its characteristics and relationships. Python libraries such as Pandas, NumPy, Matplotlib, and Seaborn are commonly used for EDA. Preparing the data involves scaling features, encoding categorical variables, handling missing values, detecting outliers, and selecting important features before building and training machine learning models.

