# Singapore HDB Resale Price Analysis and Prediction

Data analysis on Singapore HDB resale data to visualize how different factors affect pricing, such as pricing trends over time, across towns and flat characteristics.\
A Streamlit web application that predicts resale prices of HDB flats in Singapore using machine learning.

## Data
[HDB Resale Flat Prices](https://data.gov.sg/datasets?page=1&query=hdb+resale&resultId=189) (from data.gov.sg)
- Resale transactions: Jan 2015 - June 2025, ~244,000 transactions with features such as `town`, `flat_type`, `floor_area_sqm`, `flat_model` and `remaining_lease`.

## Demo
[Live App Demo](https://singapore-hdb-resale-price-predictor-jefnhchbrdrrvczsxrqkyv.streamlit.app/)

<img width="1462" alt="streamlit web app" src="https://github.com/user-attachments/assets/cd4b6ddd-e656-4817-8e5c-1101da5e813f" />


## Installation
To run locally:
1. Clone repository
   ```bash
   git clone https://github.com/clarissafel/Singapore-HDB-Resale-Price-Predictor.git
   cd Singapore-HDB-Resale-Price-Predictor
2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
6. Run the app:
   ```bash
   streamlit run hdb-resale-price-predictor.py

## Top Features that affect resale price
1. `is_mature`
   - Mature estates typically command higher prices due to established amenities
2. `flat_type`
   - eg. `flat_type_4 ROOM` , `flat_type_EXECUTIVE`
   - Flat types like 4-room and executive that higher in demand tend to have higher prices.
3. `town`
   - eg. `town_PASIR RIS`, `town_BUKIT TIMAH`
   - Premium locations and high-end areas impacts price.
4. `floor_area_sqm`
   - Larger flats naturally cost more.
5. `flat_model`
   - eg. `flat_model_Terrace`, `flat_model_Model A`
   - Different types of flats like terrace and the common HDB model influences price.

## Prediction Model details

**Algorithm: Random Forest**\
Results:\
Mean Absolute Error: $69,664.93\
R-squared: 0.7815

**Algorithm: XGBoost**\
XGBoost Results:\
Mean Absolute Error: $68,176.52\
R-squared: 0.7913

**XGBoost** was selected as prediction model because:
1. Proven Performance - Consistently outperforms other algorithms in regression tasks with structured data like property prices
2. Handles Mixed Features - Effectively processes both numerical (floor area) and categorical (town, flat type) data
3. Feature Importance - Provides clear insights into what factors most influence HDB prices (as seen in our top features)
4. Robust to Outliers - Singapore's property market has extreme values that XGBoost handles well
5. Competition Success - Similar models have won real estate prediction competitions

