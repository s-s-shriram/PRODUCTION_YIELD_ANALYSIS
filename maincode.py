import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from io import StringIO

# Sample CSV data (replace or expand as needed)
csv_data = StringIO("""
Date,Temperature,Humidity,Machine_Speed,Material_Input,Yield,Machine_Type,Material_Type
2025-01-01,72,45,150,120,88,A,X
2025-01-02,74,47,155,122,86,B,X
2025-01-03,70,43,148,118,85,A,Y
2025-01-04,69,42,147,119,87,B,Z
2025-01-05,73,46,152,121,89,C,Y
2025-01-06,75,49,158,124,91,C,Z
2025-01-07,68,40,145,117,84,A,X
2025-01-08,71,44,149,118,86,B,X
2025-01-09,72,45,151,120,90,C,Y
2025-01-10,76,50,160,125,93,B,Z
""")

# Read CSV from the string
df = pd.read_csv(csv_data)
df['Date'] = pd.to_datetime(df['Date'])

# Define features and target
features = ['Temperature', 'Humidity', 'Machine_Speed', 'Material_Input']
target = 'Yield'
X = df[features]
y = df[target]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# Visualization
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(df['Date'], df['Yield'], marker='o', linestyle='-', color='blue')
plt.title("Yield Over Time")
plt.xlabel("Date")
plt.ylabel("Yield")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
plt.hist(df['Yield'], bins=10, color='skyblue', edgecolor='black')
plt.title("Yield Distribution")
plt.xlabel("Yield")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(data=df, x='Material_Type', y='Yield', hue='Material_Type', palette='pastel', legend=False)
plt.title("Boxplot: Yield by Material Type")
plt.show()

plt.figure(figsize=(8,6))
sns.violinplot(x=pd.cut(df['Temperature'], bins=3), y='Yield', data=df)
plt.title("Violin Plot: Yield by Temperature Range")
plt.xticks(rotation=45)
plt.show()

machine_counts = df['Machine_Type'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(machine_counts, labels=machine_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Machine Type Distribution")
plt.axis('equal')
plt.show()

avg_yield = df.groupby('Machine_Type')['Yield'].mean().reset_index()
plt.figure(figsize=(8,6))
sns.barplot(x='Machine_Type', y='Yield', data=avg_yield, hue='Machine_Type', palette='Set2', legend=False)
plt.title("Average Yield by Machine Type")
plt.show()

sns.pairplot(df[['Temperature', 'Humidity', 'Machine_Speed', 'Yield']])
plt.suptitle("Pairwise Relationships", y=1.02)
plt.show()

plt.figure(figsize=(8,6))
sns.kdeplot(df['Yield'], fill=True, color='purple')
plt.title("KDE Plot: Yield Density")
plt.xlabel("Yield")
plt.ylabel("Density")
plt.show()


# For demonstration, let's assign alternating lines
df['Line'] = ['Line A', 'Line B', 'Line C'] * (len(df) // 3) + ['Line A'] * (len(df) % 3)

df['Date_str'] = df['Date'].dt.strftime('%Y-%m-%d')

heatmap_data = df.pivot(index='Date_str', columns='Line', values='Yield')

plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap='YlGnBu', cbar_kws={'label': 'Yield %'})
plt.title("Yield % Heatmap by Date and Line")
plt.xlabel("Line")
plt.ylabel("Date")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


fig = px.scatter(df, x='Temperature', y='Yield', color='Humidity',
                 title='Interactive: Temperature vs Yield (Colored by Humidity)',
                 size='Machine_Speed', hover_data=['Material_Input'])
fig.show()
