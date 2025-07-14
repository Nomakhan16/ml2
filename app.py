from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and features
model = joblib.load("models/model.pkl")
features = joblib.load("models/features.pkl")

def format_price(price):
    try:
        price = float(price)
    except:
        return "Invalid price"
    if price >= 1e7:
        return f"₹ {price / 1e7:.2f} Cr"
    elif price >= 1e5:
        return f"₹ {price / 1e5:.2f} Lakh"
    elif price >= 1e3:
        return f"₹ {price / 1e3:.1f}K"
    else:
        return f"₹ {price:,.0f}"

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            form = request.form
            location = form.get("location", "").strip()
            property_type = form.get("property_type", "").strip()
            size = float(form.get("size") or 0)
            bedrooms = int(form.get("bedrooms") or 0)

            # Replacing bathrooms with storey
            storey = form.get("storey", "1")
            storey = 5 if storey == "5+" else int(storey)

            year_built = int(form.get("year_built") or 0)
            lot_size = float(form.get("lot_size") or 0)
            garage = 1 if form.get("garage") else 0
            garden = 1 if form.get("garden") else 0
            swimming_pool = 1 if form.get("swimming_pool") else 0
            home_gym = 1 if form.get("home_gym") else 0

            input_dict = {
                "location": location,
                "property_type": property_type,
                "size": size,
                "bedrooms": bedrooms,
                "storey": storey,
                "year_built": year_built,
                "lot_size": lot_size,
                "garage": garage,
                "garden": garden,
                "swimming_pool": swimming_pool,
                "home_gym": home_gym,
            }

            df = pd.DataFrame([input_dict])
            df_encoded = pd.get_dummies(df).reindex(columns=features, fill_value=0)

            price = model.predict(df_encoded)[0]
            prediction = format_price(price)

        except Exception as e:
            import traceback
            traceback.print_exc()
            error = f"⚠️ Error: {str(e)}"

    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True)



