import streamlit as st
import joblib
import numpy as np

# Load model and scaler once
model = joblib.load('linear_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

def preprocess_input(weight, resolution, ppi, cpu_core, cpu_freq, rear_cam, battery, sale, internal_mem, ram, front_cam, thickness):
	Sale_log = np.log1p(sale)
	internal_mem_log = np.log1p(internal_mem)
	ram_sqrt = np.sqrt(ram)
	front_cam_log = np.log1p(front_cam)
	thickness_sqrt = np.sqrt(thickness)
	return np.array([[weight, resolution, ppi, cpu_core, cpu_freq, rear_cam, battery, Sale_log, internal_mem_log, ram_sqrt, front_cam_log, thickness_sqrt]])

def main():
	st.title("📱 Mobile Price Prediction")
	st.write("Fill in the details below to predict the price of a mobile phone.")

	with st.form("input_form"):
		st.header("Enter Mobile Specifications")
		col1, col2 = st.columns(2)
		with col1:
			weight = st.number_input("Weight (g)", min_value=50.0, max_value=1000.0, value=100.0, step=25.0)
			resolution = st.number_input("Resolution (inches)", min_value=1.0, max_value=15.0, value=5.0, step=0.1)
			ppi = st.number_input("PPI", min_value=100.0, max_value=1000.0, value=300.0, step=50.0)
			cpu_core = st.selectbox("CPU Cores", [1.0, 2.0, 4.0, 6.0, 8.0], index=2)
			cpu_freq = st.number_input("CPU Frequency (GHz)", min_value=0.2, max_value=3.0, value=1.0, step=0.1)
			thickness = st.number_input("Thickness (mm)", min_value=5.0, max_value=20.0, value=9.0, step=0.5)
		with col2:
			sale = st.number_input("Sale Units", min_value=0, max_value=10000, value=100, step=1)
			internal_mem = st.selectbox("Internal Memory (GB)", [0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0], index=5)
			ram = st.selectbox("RAM (GB)", [0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.512, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0], index=8)
			rear_cam = st.number_input("Rear Camera (MP)", min_value=0.0, max_value=25.0, value=5.0, step=0.1)
			front_cam = st.number_input("Front Camera (MP)", min_value=0.0, max_value=25.0, value=5.0, step=0.1)
			battery = st.number_input("Battery (mAh)", min_value=800, max_value=10000, value=3000, step=50)
		submitted = st.form_submit_button("Predict Price")
	if submitted:
		input_data = preprocess_input(weight, resolution, ppi, cpu_core, cpu_freq, rear_cam, battery, sale, internal_mem, ram, front_cam, thickness)
		input_scaled = scaler.transform(input_data)
		predicted_price = model.predict(input_scaled)
		st.success(f"Predicted Price: ₹{predicted_price[0]:,.2f}")

if __name__ == "__main__":
    main()


