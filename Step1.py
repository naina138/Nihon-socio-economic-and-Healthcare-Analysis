import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('Japan_life_expectancy.csv')

print(df.info())
print(df.describe())
print("\nMissing Values:\n", df.isnull().sum())
numeric_df = df.select_dtypes(include=['number'])  
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Life Expectancy Factors')
plt.show()
sns.pairplot(df, vars=['Life_expectancy', 'Physician', 'Income_per capita', 'Health_exp', 'Educ_exp', 'Welfare_exp'])
plt.show()
