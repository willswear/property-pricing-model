print("--- THE SCRIPT IS ALIVE! ---")
from flask import Flask, render_template_string, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# =========================================================================
# 1. DATA PIPELINE & MODEL TRAINING (Runs automatically when web app starts)
# =========================================================================

# Load the master dataset
df = pd.read_excel('Current_Past_Leases.xlsx')

# Clean out 'Not Found' square footage and $0 rates
df_clean = df[(df['Sqft'] != 'Not Found') & (df['Daily/Nightly Rate'] > 0)].copy()

# Calculate length of stay and filter out invalid leases
df_clean['Arrival_Date'] = pd.to_datetime(df_clean['Arrival'])
df_clean['Departure_Date'] = pd.to_datetime(df_clean['Departure'])
df_clean['Stay_Length_Days'] = (df_clean['Departure_Date'] - df_clean['Arrival_Date']).dt.days
df_clean = df_clean[df_clean['Stay_Length_Days'] > 0].copy()

# Extract the arrival month number
df_clean['Arrival_Month'] = df_clean['Arrival_Date'].dt.month

# Convert text specs to numbers safely
df_clean['Sqft'] = pd.to_numeric(df_clean['Sqft'], errors='coerce')
df_clean['Bathrooms'] = pd.to_numeric(df_clean['Bathrooms'], errors='coerce')

df_clean['Bedrooms'] = df_clean['Bedrooms'].replace({'Studio': 0, 'FivePlusBed': 5})
df_clean['Bedrooms'] = pd.to_numeric(df_clean['Bedrooms'], errors='coerce')

df_clean['Parking'] = df_clean['Parking'].replace({'Yes': 1, 'No': 0})
df_clean['Pets Allowed'] = df_clean['Pets Allowed'].replace({'Yes': 1, 'No': 0})

# Group by Property and Month to establish clean baseline nightly rates
df_grouped = df_clean.groupby(['Property', 'Arrival_Month'], as_index=False).agg({
    'Sqft': 'first',
    'Bedrooms': 'first',
    'Bathrooms': 'first',
    'Parking': 'first',
    'Pets Allowed': 'first',
    'Hood': 'first',
    'Daily/Nightly Rate': 'mean' 
})

# One-hot encode neighborhoods and arrival months
df_final_model_data = pd.get_dummies(df_grouped, columns=['Hood', 'Arrival_Month'], dtype=int)

# Drop rows with missing values in core features
df_final_model_data = df_final_model_data.dropna(subset=['Sqft', 'Bedrooms', 'Bathrooms', 'Parking', 'Pets Allowed'])

# Isolate features (X) and target (y)
dynamic_features = [col for col in df_final_model_data.columns if 'Hood_' in col or 'Arrival_Month_' in col]
features = ['Sqft', 'Bedrooms', 'Bathrooms', 'Parking', 'Pets Allowed'] + dynamic_features

X = df_final_model_data[features]
y = df_final_model_data['Daily/Nightly Rate']

# Train the production regression model
price_model = LinearRegression()
price_model.fit(X, y)

# Gather the list of unique neighborhoods for our website dropdown menu
all_hoods = sorted([col.replace("Hood_", "") for col in features if "Hood_" in col])

# =========================================================================
# 2. FLASK WEB APP ARCHITECTURE
# =========================================================================

app = Flask(__name__)

# Design a gorgeous, modern webpage dashboard interface using Tailwind CSS
HTML_LAYOUT = """
<!DOCTYPE html>
<html>
<head>
    <title>George</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-50 font-sans text-gray-900 min-h-screen flex flex-col justify-between">
    <div class="max-w-4xl mx-auto my-10 p-8 bg-white rounded-xl shadow-lg border border-gray-100">
        <h1 class="text-3xl font-bold text-indigo-900 mb-2">George</h1>
        <p class="text-gray-500 mb-8">Give George the characteristics of your property in DC. He will return with an estimate daily rate based on Attache's previously sold leases.</p>
        
        <form method="POST" class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
                <label class="block text-sm font-semibold text-gray-700 mb-1">Square Footage</label>
                <input type="number" name="sqft" value="{{ inputs.sqft or 900 }}" class="w-full p-2 border rounded-lg focus:ring-2 focus:ring-indigo-500" required>
            </div>
            <div>
                <label class="block text-sm font-semibold text-gray-700 mb-1">Bedrooms</label>
                <input type="number" name="bedrooms" value="{{ inputs.bedrooms or 2 }}" min="0" max="10" class="w-full p-2 border rounded-lg focus:ring-2 focus:ring-indigo-500" required>
            </div>
            <div>
                <label class="block text-sm font-semibold text-gray-700 mb-1">Bathrooms</label>
                <input type="number" name="bathrooms" value="{{ inputs.bathrooms or 1.5 }}" step="0.5" min="1" max="10" class="w-full p-2 border rounded-lg focus:ring-2 focus:ring-indigo-500" required>
            </div>
            <div>
                <label class="block text-sm font-semibold text-gray-700 mb-1">Dedicated Parking Spot?</label>
                <select name="parking" class="w-full p-2 border rounded-lg focus:ring-2 focus:ring-indigo-500">
                    <option value="Yes" {% if inputs.parking == 'Yes' %}selected{% endif %}>Yes</option>
                    <option value="No" {% if inputs.parking == 'No' %}selected{% endif %}>No</option>
                </select>
            </div>
            <div>
                <label class="block text-sm font-semibold text-gray-700 mb-1">Pets Allowed?</label>
                <select name="pets" class="w-full p-2 border rounded-lg focus:ring-2 focus:ring-indigo-500">
                    <option value="Yes" {% if inputs.pets == 'Yes' %}selected{% endif %}>Yes</option>
                    <option value="No" {% if inputs.pets == 'No' %}selected{% endif %}>No</option>
                </select>
            </div>
            <div>
                <label class="block text-sm font-semibold text-gray-700 mb-1">DC Neighborhood</label>
                <select name="neighborhood" class="w-full p-2 border rounded-lg focus:ring-2 focus:ring-indigo-500">
                    {% for hood in hoods %}
                    <option value="{{ hood }}" {% if inputs.neighborhood == hood %}selected{% endif %}>{{ hood }}</option>
                    {% endfor %}
                </select>
            </div>
            <div>
                <label class="block text-sm font-semibold text-gray-700 mb-1">Target Move-in Month (1-12)</label>
                <input type="number" name="month" value="{{ inputs.month or 6 }}" min="1" max="12" class="w-full p-2 border rounded-lg focus:ring-2 focus:ring-indigo-500" required>
            </div>
            <div class="md:col-span-2 mt-4">
                <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-4 rounded-lg shadow-md transition">
                    Generate Pricing Underwriting Report
                </button>
            </div>
        </form>

        {% if result %}
        <div class="mt-10 p-6 bg-indigo-50 rounded-xl border border-indigo-100">
            <h2 class="text-xl font-bold text-indigo-900 mb-4">📊 Recommended Underwriting Metrics</h2>
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div class="bg-white p-4 rounded-lg shadow-sm border">
                    <span class="text-xs font-bold text-gray-400 uppercase tracking-wider">Recommended List Price</span>
                    <p class="text-2xl font-extrabold text-indigo-600 mt-1">{{ result.price }} <span class="text-sm font-normal text-gray-500">/ night</span></p>
                </div>
                <div class="bg-white p-4 rounded-lg shadow-sm border">
                    <span class="text-xs font-bold text-gray-400 uppercase tracking-wider">Est. Gross Monthly Revenue</span>
                    <p class="text-2xl font-extrabold text-green-600 mt-1">{{ result.monthly }} <span class="text-sm font-normal text-gray-500">/ month</span></p>
                </div>
            </div>
        </div>
        {% endif %}
    </div>
    <footer class="text-center text-xs text-gray-400 py-4">Internal Pricing Tool v1.2</footer>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    inputs = {}
    
    if request.method == "POST":
        # Capture web form responses
        inputs['sqft'] = float(request.form["sqft"])
        inputs['bedrooms'] = float(request.form["bedrooms"])
        inputs['bathrooms'] = float(request.form["bathrooms"])
        inputs['parking'] = request.form["parking"]
        inputs['pets'] = request.form["pets"]
        inputs['neighborhood'] = request.form["neighborhood"]
        inputs['month'] = int(request.form["month"])
        
        # Format layout for the machine learning model inputs
        input_row = pd.DataFrame(0, index=[0], columns=features)
        input_row['Sqft'] = inputs['sqft']
        input_row['Bedrooms'] = inputs['bedrooms']
        input_row['Bathrooms'] = inputs['bathrooms']
        input_row['Parking'] = 1 if inputs['parking'] == "Yes" else 0
        input_row['Pets Allowed'] = 1 if inputs['pets'] == "Yes" else 0
        
        if f"Hood_{inputs['neighborhood']}" in input_row.columns:
            input_row[f"Hood_{inputs['neighborhood']}"] = 1
        if f"Arrival_Month_{inputs['month']}" in input_row.columns:
            input_row[f"Arrival_Month_{inputs['month']}"] = 1
            
        # Execute the prediction mathematical formula
        predicted_rate = price_model.predict(input_row)[0]
        est_monthly_rev = predicted_rate * 30
        
        result = {
            "price": f"${predicted_rate:.2f}",
            "monthly": f"${est_monthly_rev:,.2f}"
        }
        
    return render_template_string(HTML_LAYOUT, hoods=all_hoods, result=result, inputs=inputs)

if __name__ == "__main__":
    print("\nStarting up the underwriting data server...")
    app.run(debug=True)
