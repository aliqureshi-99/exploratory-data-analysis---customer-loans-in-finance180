from db_utils import load_credentials, RDSDatabaseConnector

if __name__ == "__main__":
    credentials = load_credentials('credentials.yaml')
    connector = RDSDatabaseConnector(credentials)
    connector.init_engine()
    print("Engine initialized successfully!")
    df = connector.fetch_data('loan_payments')
    print(df.head())
    connector.save_data_to_csv(df, 'loan_payments.csv')
    print("Data saved to loan_payments.csv")
