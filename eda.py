import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_transform import DataTransform
from data_frame_info import DataFrameInfo
from data_frame_transform import DataFrameTransform
from plotter import Plotter

sns.set(style="whitegrid")

def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def main():
    data = load_data('loan_payments.csv')

    # Initialize transformer, info, and plotter classes
    transformer = DataTransform(data)
    data_info = DataFrameInfo(data)
    data_transformer = DataFrameTransform(data)
    plotter = Plotter(data)

    # Transform the data
    data = transformer.transform()

    # Task 1: Current state of the loans
    current_recovered = data['total_payment'] / data['loan_amount'] * 100
    six_month_payment = data['instalment'] * 6

    plt.figure(figsize=(10, 6))
    plt.hist(current_recovered, bins=30, alpha=0.7, label='Current Recovered')
    plt.axvline(current_recovered.mean(), color='r', linestyle='dashed', linewidth=2)
    plt.title('Current Recovered Percentage of Loans')
    plt.xlabel('Percentage Recovered')
    plt.ylabel('Number of Loans')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(six_month_payment, bins=30, alpha=0.7, label='Amount to be Paid in 6 Months')
    plt.title('Amount to be Paid in 6 Months')
    plt.xlabel('Amount')
    plt.ylabel('Number of Loans')
    plt.legend()
    plt.show()

    print(f"Current Recovered Percentage: {current_recovered.mean():.2f}%")
    print(f"Total Amount to be Paid in 6 Months: {six_month_payment.sum():.2f}")

    # Task 2: Calculating loss
    charged_off_loans = data[data['loan_status'] == 'Charged Off']
    charged_off_percentage = len(charged_off_loans) / len(data) * 100
    total_paid_towards_charged_off = charged_off_loans['total_payment'].sum()

    print(f"Percentage of Charged Off Loans: {charged_off_percentage:.2f}%")
    print(f"Total Amount Paid Towards Charged Off Loans: {total_paid_towards_charged_off:.2f}")

    # Task 3: Calculating projected loss
    charged_off_loans['projected_loss'] = charged_off_loans['loan_amount'] - charged_off_loans['total_payment']
    plt.figure(figsize=(10, 6))
    plt.hist(charged_off_loans['projected_loss'], bins=30, alpha=0.7, label='Projected Loss')
    plt.title('Projected Loss for Charged Off Loans')
    plt.xlabel('Projected Loss')
    plt.ylabel('Number of Loans')
    plt.legend()
    plt.show()
    print(f"Total Projected Loss: {charged_off_loans['projected_loss'].sum():.2f}")

    # Task 4: Possible loss
    behind_loans = data[data['loan_status'] == 'Late (31-120 days)']
    behind_loans_percentage = len(behind_loans) / len(data) * 100
    total_amount_behind = behind_loans['loan_amount'].sum()
    projected_loss_behind = behind_loans['loan_amount'] - behind_loans['total_payment']

    print(f"Percentage of Behind Loans: {behind_loans_percentage:.2f}%")
    print(f"Total Amount of Behind Loans: {total_amount_behind:.2f}")
    print(f"Projected Loss of Behind Loans: {projected_loss_behind.sum():.2f}")

    # Task 5: Indicators of loss
    grade_subset = data[['grade', 'loan_status']]
    purpose_subset = data[['purpose', 'loan_status']]
    home_ownership_subset = data[['home_ownership', 'loan_status']]

    plt.figure(figsize=(10, 6))
    sns.countplot(x='grade', hue='loan_status', data=grade_subset)
    plt.title('Loan Grade vs Loan Status')
    plt.xlabel('Loan Grade')
    plt.ylabel('Count')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.countplot(x='purpose', hue='loan_status', data=purpose_subset)
    plt.title('Loan Purpose vs Loan Status')
    plt.xlabel('Loan Purpose')
    plt.ylabel('Count')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.countplot(x='home_ownership', hue='loan_status', data=home_ownership_subset)
    plt.title('Home Ownership vs Loan Status')
    plt.xlabel('Home Ownership')
    plt.ylabel('Count')
    plt.show()

if __name__ == "__main__":
    main()
