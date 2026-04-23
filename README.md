# Flight-Price-Prediction

## Overview

This project performs **Feature Engineering and Exploratory Data Analysis (EDA)** on Indian domestic flight price data. The goal is to analyze factors affecting flight ticket prices — such as airline, number of stops, route, departure/arrival time, and duration — and prepare the data for machine learning model building.

---

## Dataset

| Detail | Info |
|---|---|
| **File** | `flight_price.xlsx` |
| **Records** | 10,683 flights |
| **Columns** | 11 (original) → expanded after feature engineering |
| **Target Variable** | Price (ticket price in INR) |

### Original Features

| Column | Type | Description |
|---|---|---|
| Airline | Categorical | Name of the airline (12 unique airlines) |
| Date_of_Journey | Object | Date of the flight journey (DD/MM/YYYY) |
| Source | Categorical | Departure city (5 unique cities) |
| Destination | Categorical | Arrival city (6 unique cities) |
| Route | Categorical | Flight route with stops |
| Dep_Time | Object | Departure time (HH:MM) |
| Arrival_Time | Object | Arrival time (HH:MM + date info) |
| Duration | Object | Total flight duration (e.g., 2h 50m) |
| Total_Stops | Categorical | Number of stops (non-stop, 1 stop, 2 stops, 3 stops, 4 stops) |
| Additional_Info | Categorical | Extra info (meal, no info, etc.) |
| Price | Numerical | **Target variable** — ticket price in INR |

### Airlines in Dataset
IndiGo, Air India, Jet Airways, SpiceJet, Multiple carriers, GoAir, Vistara, Air Asia, Vistara Premium economy, Jet Airways Business, Multiple carriers Premium economy, Trujet

### Source Cities
Bangalore, Kolkata, Delhi, Chennai, Mumbai

### Destination Cities
New Delhi, Bangalore, Cochin, Kolkata, Delhi, Hyderabad

---

## Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
```

---

## Steps Followed

### 1. Data Loading & Initial Exploration
- Loaded dataset from Excel using `pd.read_excel()`
- Checked shape, info, dtypes using `.info()`, `.describe()`

### 2. Feature Engineering

**Date_of_Journey column:**
- Split `Date_of_Journey` into 3 new columns: `Date`, `Month`, `Year` using `str.split('/')`
- Converted all 3 to `int` datatype
- Dropped original `Date_of_Journey` column

**Arrival_Time column:**
- Cleaned extra date info (e.g., "01:10 22 Mar") using `lambda x: x.split(' ')[0]`
- Split into `Arrival_Hour` and `Arrival_Minute`
- Converted to `int`, dropped original `Arrival_Time`

**Dep_Time column:**
- Split into `Dep_Hour` and `Dep_Minute`
- Converted to `int`, dropped original `Dep_Time`

**Total_Stops column:**
- Mapped ordinal text values to numerical:

```python
df['Total_Stops'] = df['Total_Stops'].map({
    'non-stop': 0,
    '1 stop': 1,
    '2 stops': 2,
    '3 stops': 3,
    '4 stops': 4,
    np.nan: 1
})
```

- Dropped `Route` column (redundant with stops info)

**Duration column:**
- Extracted `Hour` and `Minute` using regex:
```python
df['Hour'] = df['Duration'].str.extract(r'(\d+)h').fillna(0).astype(int)
df['Minute'] = df['Duration'].str.extract(r'(\d+)m').fillna(0).astype(int)
```
- Filled missing minutes with 0
- Dropped original `Duration` column

### 3. Encoding Categorical Features

**One-Hot Encoding** — Airline, Source, Destination:
```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
encoded_df = pd.DataFrame(
    encoder.fit_transform(df[['Airline','Source','Destination']]).toarray(),
    columns=encoder.get_feature_names_out()
)
df = pd.concat([df, encoded_df], axis=1)
df.drop(['Airline','Source','Destination'], axis=1, inplace=True)
```

**pd.get_dummies()** — Additional_Info:
```python
additional_encoded = pd.get_dummies(df['Additional_Info'], prefix='Additional_Info')
df = pd.concat([df, additional_encoded], axis=1)
df.drop('Additional_Info', axis=1, inplace=True)
```

---

## EDA — Key Questions Answered

### Q1. Which airline is most expensive?

<img width="589" height="685" alt="Image" src="https://github.com/user-attachments/assets/7d3c99cd-7cdb-4bf6-8171-6f6517517b66" />

**Insights:**
- **Jet Airways Business** is the most expensive airline with the highest ticket prices (median ~55,000 INR)
- **Jet Airways** also has higher average prices compared to most airlines
- Budget airlines like **IndiGo** and **SpiceJet** generally have the lowest prices
- **Trujet** has the lowest median price overall

---

### Q2. Which route is most expensive?

**Route vs Price Insights (from grouped analysis):**

| Route | Price |
|---|---|
| Bangalore → New Delhi | Most expensive |
| Delhi → Cochin | Second most expensive |
| Kolkata → Bangalore | Third most expensive |
| Chennai → Kolkata | Comparatively cheaper |

---

### Q3. Does number of stops affect price?

<img width="589" height="453" alt="Image" src="https://github.com/user-attachments/assets/811c6b28-a54a-483c-870d-e6be1262c68d" />

**Insights:**
- Flight price **increases as number of stops increases**
- **Non-stop flights** have the lowest average price (~5,024 INR)
- **1 stop** flights show a significant price jump (~11,000 INR median)
- **4 stops** flights have the highest average price (~17,686 INR)
- Number of stops is an **important factor** affecting ticket price

### Average Price by Number of Stops

| Stops | Avg Price (INR) |
|---|---|
| 0 (Non-stop) | ~5,024 |
| 1 stop | ~11,000 |
| 2 stops | ~13,000 |
| 3 stops | ~14,000 |
| 4 stops | ~17,686 |

---

## Key Insights Summary

- **More stops = Higher price** — Non-stop flights are cheapest on average
- **Jet Airways Business** is the premium airline with highest ticket prices
- **IndiGo and SpiceJet** are the most budget-friendly airlines
- **Bangalore → New Delhi** is the most expensive route
- Number of stops is the most important factor affecting flight ticket price
- Most flights in the dataset have **1 stop**

---

## Files in this Repository

| File | Description |
|---|---|
| `Flight_Price_Prediction_Clean.ipynb` | Jupyter Notebook with complete feature engineering and EDA |
| `flight_price.xlsx` | Original dataset |
| `README.md` | Project documentation |

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/nikitajadhav0022-afk/Flight-Price-Prediction.git
```

2. Install required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

3. Open the notebook:
```bash
jupyter notebook Flight_Price_Prediction_Clean.ipynb
```

---

## Tools & Technologies

- Python 3
- Pandas — data manipulation & feature engineering
- NumPy — numerical operations
- Matplotlib & Seaborn — data visualization
- Scikit-learn — OneHotEncoder for categorical encoding
- Jupyter Notebook
