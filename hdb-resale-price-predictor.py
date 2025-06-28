import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('hdb_xgb_model.pkl')

@st.cache_resource
def load_columns():
    return joblib.load('hdb_columns.pkl')
training_cols = load_columns()

@st.cache_data
def load_data():
    return pd.read_csv("hdb_cleaned_data.csv")
hdb_data = load_data()

def to_k(x, pos):
    if x >= 1000:
        return f'{int(x // 1000)} k'
    return str(int(x))

# Main Streamlit app
def main():
    st.title('🏘️ Singapore HDB Resale Price Predictor')
    st.write("""
    Predict HDB resale prices based on flat features like location, size, and lease.
    """)

    # Load model
    model = load_model()
    if model is None:
        return
    
    #Input form
    st.sidebar.header('📝 Input Features')

    floor_area = st.sidebar.slider("Floor Area (sqm)", 30, 200, 90)
    flat_age = st.sidebar.slider("Flat Age (Years)", 0, 50, 10)
    remaining_lease = st.sidebar.slider("Remaining Lease (Years)", 20, 99, 70)

    all_towns = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
       'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
       'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
       'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
       'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
       'TOA PAYOH', 'WOODLANDS', 'YISHUN']
    town = st.sidebar.selectbox("Town", all_towns)

    # Preprocess inputs
    input_data = pd.DataFrame(columns=training_cols).fillna(0)
    input_data['floor_area_sqm'] = [floor_area]
    input_data['flat_age'] = [flat_age]
    input_data['remaining_lease'] = [remaining_lease]
    input_data['town_' + town] = [1]
    input_data['year'] = pd.Timestamp.now().year

    if st.sidebar.button("Predict Price"):
        prediction = model.predict(input_data)[0]
        st.success(f"### Predicted Resale Price: **${prediction:,.0f}**")
    
        # Show feature importance
        st.subheader("🔍 What's Driving This Price?")
        st.write("Top features influencing this prediction:")

        cat_cols = ['town', 'flat_type', 'flat_model']
        num_cols = ['floor_area_sqm', 'remaining_lease', 'year', 'month_num','flat_age', 'storey_mid', 'is_mature']
        try:
            xgb_model = model.named_steps['regressor']
            
            numeric_features = num_cols
            categorical_features = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_cols)
            feature_names = list(numeric_features) + list(categorical_features)
            
            # Create importance DataFrame
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': xgb_model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            st.markdown(feature_importance.style.hide(axis='index').to_html(), unsafe_allow_html=True)
            
                
        except Exception as e:
            st.warning(f"Could not display feature importance: {str(e)}")

    # Visualization: Price distribution by town
    st.subheader("📊 HDB Price Trends")
    if st.checkbox("Show Price Distribution by Town"):
        fig, ax = plt.subplots()
        ax = sns.boxplot(x='town', y='resale_price', data=hdb_data)
        ax.yaxis.set_major_formatter(FuncFormatter(to_k))
        plt.xticks(rotation=90)
        plt.ylabel('Resale Price')
        plt.xlabel('Town')
        st.pyplot(fig)

main()