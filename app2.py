import pandas as pd
import joblib
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file
import io

# Load the pre-trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input data from the form
        air_pollution_index = float(request.form['air_pollution_index'])
        humidity = float(request.form['humidity'])
        wind_speed = float(request.form['wind_speed'])
        wind_direction = float(request.form['wind_direction'])
        visibility_in_miles = float(request.form['visibility_in_miles'])
        dew_point = float(request.form['dew_point'])
        temperature = float(request.form['temperature'])
        rain_p_h = float(request.form['rain_p_h'])
        snow_p_h = float(request.form['snow_p_h'])
        clouds_all = float(request.form['clouds_all'])
        day_of_week = int(request.form['day_of_week'])
        is_holiday = int(request.form['is_holiday'])
        
        # Create a DataFrame for the input data (excluding 'hour' since we will generate predictions for all 24 hours)
        input_data = pd.DataFrame({
            'air_pollution_index': [air_pollution_index] * 24,  # Repeat for 24 hours
            'humidity': [humidity] * 24,
            'wind_speed': [wind_speed] * 24,
            'wind_direction': [wind_direction] * 24,
            'visibility_in_miles': [visibility_in_miles] * 24,
            'dew_point': [dew_point] * 24,
            'temperature': [temperature] * 24,
            'rain_p_h': [rain_p_h] * 24,
            'snow_p_h': [snow_p_h] * 24,
            'clouds_all': [clouds_all] * 24,
            'day_of_week': [day_of_week] * 24,
            'is_holiday': [is_holiday] * 24,
            'hour': list(range(24))  # Automatically generate hours (0 to 23)
        })

        # Scale the input features
        scaled_input = scaler.transform(input_data)

        # Predict traffic volume for each hour (0 to 23)
        predicted_traffic = model.predict(scaled_input)

        # Create a time range for 24 hours
        hours = list(range(24))

        # Define traffic zones
        def traffic_zone(value):
            if value < 3000:
                return 'Low'
            elif value < 5000:
                return 'Medium'
            else:
                return 'High'

        # Classify each predicted value into Low, Medium, or High
        traffic_zones = [traffic_zone(t) for t in predicted_traffic]

        # Generate the plot with larger size
        fig, ax = plt.subplots(figsize=(14, 8))  # Increase the figure size (width, height)

        # Plot the traffic volume with color-coding based on the zones
        ax.plot(hours, predicted_traffic, marker='o', color='black', label='Predicted Traffic Volume')

        # Fill the areas for Low, Medium, and High zones
        ax.fill_between(hours, 0, 3000, color='green', alpha=0.2, label='Low')
        ax.fill_between(hours, 3000, 5000, color='yellow', alpha=0.2, label='Medium')
        ax.fill_between(hours, 5000, max(predicted_traffic), color='red', alpha=0.2, label='High')

        # Add labels and title
        ax.set_title('Traffic Volume Prediction for 24 Hours', fontsize=16)  # Title font size
        ax.set_xlabel('Time of Day (Hours)', fontsize=14)  # X-axis label font size
        ax.set_ylabel('Traffic Volume', fontsize=14)  # Y-axis label font size
        ax.set_xticks(hours)  # Set the x-axis to display 24 hours
        ax.set_xticklabels([f'{i}:00' for i in hours], fontsize=12, rotation=45)  # Rotate x-axis labels for better readability

        # Increase the spacing for the labels
        plt.tight_layout()  # This helps to make sure labels fit within the figure without overlapping
        ax.legend()

        # Save the plot to a BytesIO object
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)

        # Return the plot as an image
        return send_file(img, mimetype='image/png')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
