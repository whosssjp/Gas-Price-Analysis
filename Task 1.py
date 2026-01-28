import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.linear_model import LinearRegression

# 1️⃣ Load data
natgas_df = pd.read_csv('Nat_Gas.csv')
natgas_df

# 2️⃣ Plot prices
plt.plot(natgas_df['Dates'], natgas_df['Prices'])
plt.title("Natural Gas Prices")
plt.show()


# 3️⃣ January prices
natgas_df['Dates'] = pd.to_datetime(natgas_df['Dates'])
natgas_df['Year'] = natgas_df['Dates'].dt.year
natgas_df['Month'] = natgas_df['Dates'].dt.month

natgas_jan = natgas_df[natgas_df['Month'] == 1]
print("JANUARY DATA")
print(natgas_jan)

# 4️⃣ Linear Regression
from sklearn.linear_model import LinearRegression
x = np.array(natgas_df[natgas_df['Month'] == 1]['Year'] ).reshape(-1, 1)
y = np.array(natgas_df[natgas_df['Month'] == 1]['Prices'] ).reshape(-1, 1)
reg = LinearRegression().fit(x, y)
print(y.flatten())


# 5️⃣ Predict 2025 price
pred_2025 = reg.predict(np.array([[2025]]))
print(pred_2025)

# 6️⃣ Extrapolation for an addtional year
from sklearn.linear_model import LinearRegression
import numpy as np

def next_year_price(next_year):
    """Returns a list of predicted natural gas prices for each month of the following year."""

    price_list = []

    for i in range(12):
        x = np.array(
            natgas_df[natgas_df['Month'] == i + 1]['Year']
        ).reshape(-1, 1)

        y = np.array(
            natgas_df[natgas_df['Month'] == i + 1]['Prices']
        ).reshape(-1, 1)

        reg = LinearRegression().fit(x, y)
        price = reg.predict(np.array([[next_year]]))

        price_list.append(round(float(price[0][0]), 2))

    return price_list

gas_prices25 = next_year_price(2025)
np.array(gas_prices25)
print(gas_prices25)

# 7️⃣ Last of each month
import datetime as dt

def get_last_of_each_month(year):
    dates_array = []
    current_date = dt.datetime(year, 12, 31)  # Start from last day of year

    while current_date.year == year:
        dates_array.append(current_date.strftime('%Y-%m-%d'))

        month = current_date.month
        year_current = current_date.year

        # Move to first day of current month
        current_date = current_date.replace(year=year_current, month=month, day=1)

        # Move back one day → last day of previous month
        current_date -= dt.timedelta(days=1)

    return dates_array[::-1]  # Reverse AFTER loop finishes


# ---- Function call (OUTSIDE the function) ----
dates_2025 = get_last_of_each_month(2025)

for date in dates_2025:
    print(date)

# 8️⃣ Create DataFrame for 2025
projected_gas_prices25_df = pd.DataFrame({'Dates': dates_2025, 'Prices': gas_prices25})
projected_gas_prices25_df

projected_gas_prices25_df['Dates'] = pd.to_datetime(projected_gas_prices25_df['Dates'])

projected_gas_prices25_df['Year'] = projected_gas_prices25_df['Dates'].dt.year
projected_gas_prices25_df['Month'] = projected_gas_prices25_df['Dates'].dt.month
projected_gas_prices25_df
print(projected_gas_prices25_df)

# 9️⃣ Get historical and projected prices
gas_df = pd.concat([natgas_df, projected_gas_prices25_df], ignore_index=True)
gas_df
print(gas_df)

# 10️⃣ Final plot analysis
plt.plot(gas_df['Dates'], gas_df['Prices'], label = 'Predicted 2025')
plt.plot(natgas_df['Dates'], natgas_df['Prices'], label = 'Actuals 2021-24')
plt.ylabel('Gas Prices $')
plt.xlabel('Year')
plt.title('Gas Price Forecast', fontweight = 'bold')
plt.legend()
plt.show()




    









