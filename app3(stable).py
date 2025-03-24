import os
from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# Load the pre-trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

# Folder to save the plots
app.config['UPLOAD_FOLDER'] = 'static/plots'

# Ensure the folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None  # Variable to store prediction results
    if request.method == 'POST':
        # Get input data from the form
        selected_date = request.form['date']  # Get the date input
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

        # Parse the selected date
        selected_date = datetime.strptime(selected_date, "%Y-%m-%d")

        # Create a DataFrame for the input data
        input_data = pd.DataFrame({
            'air_pollution_index': [air_pollution_index],
            'humidity': [humidity],
            'wind_speed': [wind_speed],
            'wind_direction': [wind_direction],
            'visibility_in_miles': [visibility_in_miles],
            'dew_point': [dew_point],
            'temperature': [temperature],
            'rain_p_h': [rain_p_h],
            'snow_p_h': [snow_p_h],
            'clouds_all': [clouds_all],
            'hour': [0],  # Placeholder for hour, we'll modify this later for the 24-hour prediction
            'day_of_week': [day_of_week],
            'is_holiday': [is_holiday]
        })

        # Scale the input features
        scaled_input = scaler.transform(input_data)

        # Generate traffic volume predictions for the entire day (24 hours)
        predicted_traffic = []
        for i in range(24):
            # Set the current hour in input data
            input_data['hour'] = [i]
            scaled_input = scaler.transform(input_data)
            prediction = model.predict(scaled_input)[0]
            predicted_traffic.append(prediction)

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

        # Generate the plot
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot the traffic volume with color-coding based on the zones
        ax.plot(hours, predicted_traffic, marker='o', color='black', label='Predicted Traffic Volume')

        # Fill the areas for Low, Medium, and High zones
        ax.fill_between(hours, 0, 3000, color='green', alpha=0.2, label='Low')
        ax.fill_between(hours, 3000, 5000, color='yellow', alpha=0.2, label='Medium')
        ax.fill_between(hours, 5000, max(predicted_traffic), color='red', alpha=0.2, label='High')

        # Add labels and title
        ax.set_title(f'Traffic Volume Prediction for {selected_date.strftime("%B %d, %Y")}', fontsize=16)
        ax.set_xlabel('Time of Day (Hours)')
        ax.set_ylabel('Traffic Volume')
        ax.set_xticks(hours)  # Set the x-axis to display 24 hours
        ax.set_xticklabels([f'{i}:00' for i in hours], fontsize=12, rotation=45)  # Rotate x-axis labels for better readability

        # Increase the spacing for the labels
        plt.tight_layout()  # This helps to make sure labels fit within the figure without overlapping
        ax.legend()

        # Save the plot as a PNG image
        plot_filename = f'predicted_traffic_{selected_date.strftime("%Y-%m-%d")}.png'
        plot_filepath = os.path.join(app.config['UPLOAD_FOLDER'], plot_filename)
        fig.savefig(plot_filepath)

        # Return the prediction results along with the plot
        return render_template('results.html', prediction=prediction, plot_url=plot_filename)

    return render_template('index1.html', prediction=prediction)  #index 1 stable results stable

if __name__ == '__main__':
    app.run(debug=True)
