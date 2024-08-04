import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

sns.set(style="whitegrid")

def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    data['employment_length'] = data['employment_length'].replace({'10+ years': '10', '< 1 year': '0.5', 'n/a': None})
    data['employment_length'] = data['employment_length'].str.extract('(\d+)', expand=False).astype(float)
    data['employment_length'].fillna(data['employment_length'].median(), inplace=True)

    data['funded_amount'].fillna(data['funded_amount'].median(), inplace=True)
    data['int_rate'].fillna(data['int_rate'].median(), inplace=True)
    data['term'].fillna(data['term'].mode()[0], inplace=True)

    data.dropna(subset=['loan_status'], inplace=True)
    
    return data

def handle_outliers(data: pd.DataFrame) -> pd.DataFrame:
    Q1 = data['loan_amount'].quantile(0.25)
    Q3 = data['loan_amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data['loan_amount'] >= lower_bound) & (data['loan_amount'] <= upper_bound)]
    return data

def normalize_data(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    scaler = StandardScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data

def create_new_features(data: pd.DataFrame) -> pd.DataFrame:
    data['loan_to_income_ratio'] = data['loan_amount'] / data['annual_inc']
    data['payment_to_income_ratio'] = data['instalment'] / (data['annual_inc'] / 12)
    return data

def univariate_analysis(data: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['loan_amount'], bins=30, kde=True)
    plt.title('Distribution of Loan Amounts')
    plt.xlabel('Loan Amount')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(data['int_rate'], bins=30, kde=True)
    plt.title('Distribution of Interest Rates')
    plt.xlabel('Interest Rate')
    plt.ylabel('Frequency')
    plt.show()

def bivariate_analysis(data: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='loan_amount', y='int_rate', data=data)
    plt.title('Loan Amount vs Interest Rate')
    plt.xlabel('Loan Amount')
    plt.ylabel('Interest Rate')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='employment_length', y='loan_amount', data=data)
    plt.title('Employment Length vs Loan Amount')
    plt.xlabel('Employment Length (years)')
    plt.ylabel('Loan Amount')
    plt.show()

def multivariate_analysis(data: pd.DataFrame):
    plt.figure(figsize=(12, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

def advanced_visualizations(data: pd.DataFrame):
    sns.pairplot(data[['loan_amount', 'int_rate', 'employment_length', 'annual_inc', 'loan_to_income_ratio']])
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='loan_status', y='loan_amount', data=data)
    plt.title('Loan Status vs Loan Amount')
    plt.xlabel('Loan Status')
    plt.ylabel('Loan Amount')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.violinplot(x='grade', y='int_rate', data=data)
    plt.title('Interest Rate Distribution Across Loan Grades')
    plt.xlabel('Loan Grade')
    plt.ylabel('Interest Rate')
    plt.show()

def summarize_findings(data: pd.DataFrame):
    print("Summary of Key Findings:")
    print("1. Distribution of loan amounts and interest rates show ...")
    print("2. Higher loan amounts tend to have ... interest rates.")
    print("3. Employment length has a significant impact on loan amounts ...")

    loan_status_counts = data['loan_status'].value_counts()
    print("\nNumber of Loans by Loan Status:")
    print(loan_status_counts)

def correlation_analysis(data: pd.DataFrame):
    plt.figure(figsize=(12, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

def pair_plot(data: pd.DataFrame):
    sns.pairplot(data[['loan_amount', 'int_rate', 'employment_length', 'annual_inc', 'loan_to_income_ratio', 'payment_to_income_ratio']])
    plt.show()

def feature_importance(data: pd.DataFrame):
    features = ['loan_amount', 'int_rate', 'employment_length', 'annual_inc', 'loan_to_income_ratio', 'payment_to_income_ratio']
    target = 'loan_status'
    
    X = data[features]
    y = data[target]
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importance')
    plt.show()

if __name__ == "__main__":
    data = load_data('loan_payments.csv')
    print(f"Data shape: {data.shape}")
    print(data.head())

    print("\nNull values in each column:")
    print(data.isnull().sum())

    data = handle_missing_values(data)
    print("\nNull values after handling missing values:")
    print(data.isnull().sum())

    data = handle_outliers(data)
    data = create_new_features(data)
    print("\nData with new features:")
    print(data.head())

    data = normalize_data(data, ['loan_amount', 'int_rate', 'annual_inc', 'loan_to_income_ratio', 'payment_to_income_ratio'])

    print("\nData types of each column:")
    print(data.dtypes)

    print("\nSummary statistics:")
    print(data.describe())

    univariate_analysis(data)
    bivariate_analysis(data)
    multivariate_analysis(data)
    advanced_visualizations(data)
    summarize_findings(data)
    correlation_analysis(data)
    pair_plot(data)
    feature_importance(data)
