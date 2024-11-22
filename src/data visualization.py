import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset (replace 'your_dataset.csv' with your file path)
file_path = 'titanic.csv'  # Update this to your dataset file path
data = pd.read_csv(file_path)

# Display the structure of the dataset
print("Dataset Overview:")
print(data.info())
print("\nFirst Few Rows:")
print(data.head())

# Ensure the dataset is sorted by PassengerId for sequential analysis
data.sort_values(by='PassengerId', inplace=True)

# Visualization 1: Bar Chart (Survival Counts)
plt.figure(figsize=(8, 5))
sns.countplot(data=data, x='Survived', palette=['lightcoral', 'skyblue'])
plt.title('Survival Counts')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Did Not Survive', 'Survived'], rotation=0)
plt.show()

# Visualization 2: Scatter Plot (Passenger ID vs Survival)
plt.figure(figsize=(10, 5))
sns.scatterplot(data=data, x='PassengerId', y='Survived', color='blue', alpha=0.7)
plt.title('Scatter Plot: Passenger ID vs Survival')
plt.xlabel('Passenger ID')
plt.ylabel('Survived (0 = No, 1 = Yes)')
plt.yticks([0, 1], ['Did Not Survive', 'Survived'])
plt.show()

# Visualization 3: Heatmap (Correlation Matrix)
plt.figure(figsize=(6, 4))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Visualization 4: Survival Distribution (Pie Chart)
plt.figure(figsize=(6, 6))
data['Survived'].value_counts().plot.pie(
    autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'skyblue'], labels=['Did Not Survive', 'Survived']
)
plt.title('Survival Distribution')
plt.ylabel('')
plt.show()

# Visualization 5: Cumulative Survival Plot
plt.figure(figsize=(10, 5))
data['Cumulative Survival'] = data['Survived'].cumsum() / len(data)
sns.lineplot(data=data, x='PassengerId', y='Cumulative Survival', color='green')
plt.title('Cumulative Survival Rate Over Passenger IDs')
plt.xlabel('Passenger ID')
plt.ylabel('Cumulative Survival Rate')
plt.show()

# Visualization 6: Grouped Bar Plot (Passenger Groups by Survival)
# Creating groups of Passenger IDs to show survival trends
data['Passenger Group'] = pd.cut(data['PassengerId'], bins=10, labels=[f'Group {i+1}' for i in range(10)])
grouped_data = data.groupby('Passenger Group')['Survived'].mean()

plt.figure(figsize=(10, 5))
sns.barplot(x=grouped_data.index, y=grouped_data.values, palette='viridis')
plt.title('Survival Rate by Passenger Group')
plt.xlabel('Passenger Group')
plt.ylabel('Survival Rate')
plt.xticks(rotation=45)
plt.show()

# Visualization 7: Density Plot (Passenger ID Distribution by Survival)
plt.figure(figsize=(10, 5))
sns.kdeplot(data=data, x='PassengerId', hue='Survived', fill=True, common_norm=False, palette='muted', alpha=0.5)
plt.title('Passenger ID Distribution by Survival')
plt.xlabel('Passenger ID')
plt.ylabel('Density')
plt.show()

# Visualization 8: Survival Proportion by Passenger ID (Stacked Area Plot)
plt.figure(figsize=(10, 5))
survival_proportion = data.groupby('PassengerId')['Survived'].mean()
survival_proportion.plot(kind='area', color='skyblue', alpha=0.5)
plt.title('Survival Proportion Across Passenger IDs')
plt.xlabel('Passenger ID')
plt.ylabel('Survival Proportion')
plt.show()
