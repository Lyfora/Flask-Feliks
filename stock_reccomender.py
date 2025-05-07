from flask import Flask, request, jsonify
import pandas as pd
from datetime import datetime
import joblib
import numpy as np
import os
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

app = Flask(__name__)

# File path for your CSV
CSV_FILE = 'data\data_bersih.csv'
#Load the use data!
DATA_FOLDER = 'data'
DATA_PATH = os.path.join(DATA_FOLDER,'df_reccomender_stock_use.csv')
df_reccomender = pd.read_csv(DATA_PATH)


#Load the model!
# Configuration
MODEL_FOLDER = 'models'
XGB_MODEL_PATH = os.path.join(MODEL_FOLDER, 'xgboost_20250505_184047.joblib')
LGBM_MODEL_PATH = os.path.join(MODEL_FOLDER, 'lgbm_20250505_184047.joblib')
xgb_model = joblib.load(XGB_MODEL_PATH)
lgbm_model = joblib.load(LGBM_MODEL_PATH)

# Function to load the DataFrame from CSV
def load_dataframe():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)

# Function to save the DataFrame to CSV
def save_dataframe(df):
    df.to_csv(CSV_FILE, index=False)

def load_reccomender():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)

def save_reccomender(df):
    df.to_csv(DATA_PATH)

@app.route('/products', methods=['GET'])
def get_products():
    df = load_dataframe()
    return jsonify(df.to_dict('records'))

# Function to get all existing product names
def get_products_name():
    df = load_dataframe()
    return df['product_name'].unique().tolist()

# Function for Grouping those data into Weekly per product/date
def grouping(df):
    df_daily = df.groupby(['product_name', 'created_at']).agg(
        quantity=('quantity', 'sum'),
    ).reset_index()
    return df_daily

product_dictionary = get_products_name()


#Make The Reccomender!
@app.route('/api/recommend/xgboost', methods=['GET'])
def recommend_with_xgboost():
    try:
        #Validating Request
        product_id = int(request.args.get('id'))
        if not product_id:
            return jsonify({'error': 'Missing product ID'}), 400
        
        #Validating Entry
        product_id = f'PROD{product_id}'

        if(product_id not in product_dictionary):
            return jsonify({'error': 'Wrong product ID'}), 400
        
        #Entry Valid!
        df_check = df_reccomender[df_reccomender['product_name']==product_id]
        data = np.array([df_check.iloc[-1,2:-1]])
        pred = int(xgb_model.predict(data)[0])
        print(pred)
        return jsonify({
            'pred' : pred
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

#Make The Reccomender!
@app.route('/api/recommend/lgbm', methods=['GET'])
def recommend_with_lgbm():
    try:
        #Validating Request
        product_id = int(request.args.get('id'))
        print(product_id)
        if not product_id:
            return jsonify({'error': 'Missing product ID'}), 400
        
        #Validating Entry
        product_id = f'PROD{product_id}'
        print(product_id)
        if(product_id not in product_dictionary):
            return jsonify({'error': 'Wrong product ID'}), 400
        
        #Entry Valid!
        df_check = df_reccomender[df_reccomender['product_name']==product_id]
        data = np.array([df_check.iloc[-1,2:-1]])
        
        pred = int(lgbm_model.predict(data)[0])
        return jsonify({
            'pred' : pred
        })
    except Exception as e:

        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)