# import streamlit as st
# import requests

# st.set_page_config(page_title="Spam Detector")

# st.title("Spam Detector")

# st.write("This frontend communicates with a FastAPI backend to classify messages.")

# input_sms=st.text_area("Enter the message" ,height=150)

# if st.button('Predict'):
#     if input_sms:
#         backend_url = "http://127.0.0.1:8000/predict"
#         payload = {"text": input_sms}
#         try:
#             # 2. Sending the request
#             response = requests.post(backend_url, json=payload)
            
#             # 3. Check if the server actually replied successfully
#             if response.status_code == 200:
#                 data = response.json()
                
#                 # IMPORTANT: The key name must be "prediction" (match your main.py)
#                 prediction = data.get("prediction", "Error: Key not found")
                
#                 if prediction == "Spam":
#                     st.error(f"🚨 Result: {prediction}")
#                 else:
#                     st.success(f"🟢 Result: {prediction}")
#             else:
#                 st.error(f"Backend returned an error code: {response.status_code}")
                
#         except Exception as e:
#             st.error(f"Connection Error: {e}")
#     else:
#         st.warning("Please enter the text")




import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="AI Spam Detector", layout="wide")

st.title("🚫 Spam Detection System")
st.write("Full-stack AI application powered by FastAPI and MySQL.")

# Create Tabs for a cleaner UI
tab1, tab2 = st.tabs(["🔍 Predict", "📜 History"])

with tab1:
    st.header("Classify a Message")
    input_sms = st.text_area("Enter the message content here:", height=150)

    if st.button('Predict'):
        if input_sms:
            backend_url = "http://127.0.0.1:8000/predict"
            payload = {"text": input_sms}
            try:
                with st.spinner("Analyzing..."):
                    response = requests.post(backend_url, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    prediction = data.get("prediction")
                    status = data.get("status", "Processed")
                    
                    if prediction == "Spam":
                        st.error(f"### Result: {prediction}")
                    else:
                        st.success(f"### Result: {prediction}")
                    
                    st.info(f"System Note: {status}")
                else:
                    st.error(f"Backend Error: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Connection Error: {e}")
        else:
            st.warning("Please enter some text first.")

with tab2:
    st.header("Recent Database Entries")
    if st.button("Refresh History"):
        try:
            # Calling the new GET endpoint we will add to main.py
            history_url = "http://127.0.0.1:8000/history"
            response = requests.get(history_url)
            
            if response.status_code == 200:
                history_data = response.json()
                if history_data:
                    # Creating a DataFrame to show a nice table
                    df = pd.DataFrame(history_data)
                    # Renaming columns for better look
                    df.columns = ['ID', 'Original Message', 'Classification', 'Date & Time']
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No records found in the database.")
            else:
                st.error("Could not fetch history.")
        except Exception as e:
            st.error(f"Error: {e}")