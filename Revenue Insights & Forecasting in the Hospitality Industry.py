# Revenue Insights in the Hospitality Industry

# 📦 Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from fbprophet import Prophet  # Install with: pip install prophet

# 📥 Load dataset
# Replace with your dataset path if needed
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/hotelBookings.csv")

# 🔍 Initial exploration
print("Shape:", df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())

# 🧹 Data cleaning
# Fill missing values

df['children'].fillna(0, inplace=True)
df.dropna(subset=['country'], inplace=True)  # Drop rows with missing country

# Add date column
df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' + 
                                     df['arrival_date_month'] + '-' +
                                     df['arrival_date_day_of_month'].astype(str), 
                                     format='%Y-%B-%d')

# Calculate total revenue = stays_in_weekend_nights + stays_in_week_nights * adr
df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['revenue'] = df['total_nights'] * df['adr']

# 🧠 Key Analysis

# Revenue trend over time
monthly_revenue = df.groupby(df['arrival_date'].dt.to_period('M')).agg({'revenue': 'sum'}).reset_index()
monthly_revenue['arrival_date'] = monthly_revenue['arrival_date'].astype(str)

plt.figure(figsize=(12,6))
sns.lineplot(data=monthly_revenue, x='arrival_date', y='revenue')
plt.xticks(rotation=45)
plt.title('Monthly Revenue Over Time')
plt.tight_layout()
plt.show()

# Revenue by hotel type
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='hotel', y='revenue')
plt.title('Revenue Distribution by Hotel Type')
plt.ylim(0, 1000)
plt.show()

# Revenue by country (Top 10)
country_rev = df.groupby('country').agg({'revenue': 'sum'}).sort_values(by='revenue', ascending=False).head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=country_rev.index, y=country_rev['revenue'])
plt.title('Top 10 Countries by Revenue')
plt.xticks(rotation=45)
plt.show()

# Forecasting with Prophet
# Aggregate daily revenue
daily_rev = df.groupby('arrival_date').agg({'revenue': 'sum'}).reset_index()
daily_rev.columns = ['ds', 'y']

model = Prophet()
model.fit(daily_rev)

future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

fig1 = model.plot(forecast)
plt.title('Revenue Forecast (Next 90 Days)')
plt.show()


