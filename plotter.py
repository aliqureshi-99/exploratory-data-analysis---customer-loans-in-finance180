import matplotlib.pyplot as plt
import seaborn as sns

class Plotter:
    def __init__(self, data):
        self.data = data

    def plot_missing_values(self):
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.data.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values in the Dataset')
        plt.show()

    def plot_correlation_matrix(self):
        plt.figure(figsize=(12, 8))
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.show()
