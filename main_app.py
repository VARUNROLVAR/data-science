import numpy as np
import pickle
import streamlit as st

# Load the trained model and transformers (make sure to provide the correct path)
with open("final_model.pkl", "rb") as model_file:
    classifier = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("pca.pkl", "rb") as pca_file:
    pca = pickle.load(pca_file)

def predict_note_authentication(Education, Marital_Status, Income, Kidhome, Teenhome, Recency, NumDealsPurchases, NumWebPurchases, NumCatalogPurchases,
                                NumStorePurchases, NumWebVisitsMonth, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5, AcceptedCmp1, AcceptedCmp2, Complain, Response, total_amount_spent, Children):
    # Convert input to numerical values with default to 0
    try:
        Education = int(Education)
        Marital_Status = int(Marital_Status)
        Income = int(Income)
        Kidhome = int(Kidhome)
        Teenhome = int(Teenhome)
        Recency = int(Recency)
        NumDealsPurchases = int(NumDealsPurchases)
        NumWebPurchases = int(NumWebPurchases)
        NumCatalogPurchases = int(NumCatalogPurchases)
        NumStorePurchases = int(NumStorePurchases)
        NumWebVisitsMonth = int(NumWebVisitsMonth)
        AcceptedCmp3 = int(AcceptedCmp3)
        AcceptedCmp4 = int(AcceptedCmp4)
        AcceptedCmp5 = int(AcceptedCmp5)
        AcceptedCmp1 = int(AcceptedCmp1)
        AcceptedCmp2 = int(AcceptedCmp2)
        Complain = int(Complain)
        Response = int(Response)
        total_amount_spent = int(total_amount_spent)
        Children = int(Children)
    except ValueError as e:
        return f"Invalid value: {e}"

    # Prepare the feature array
    features = np.array([[Education, Marital_Status, Income, Kidhome, Teenhome, Recency, NumDealsPurchases, NumWebPurchases, NumCatalogPurchases,
                          NumStorePurchases, NumWebVisitsMonth, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5, AcceptedCmp1, AcceptedCmp2, Complain, Response, total_amount_spent, Children]])

    # Scale the features
    scaled_features = scaler.transform(features)

    # Apply PCA
    pca_features = pca.transform(scaled_features)

    # Predict the cluster
    prediction = classifier.predict(pca_features)
    
    return prediction

def main():
    st.title("Customer Segmentation")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Customer Segmentation ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
   
    # Create input fields with default values and validation
    Education = st.selectbox("Education", ["Graduation", "PhD", "Master", "Basic", "2n Cycle"], index=0)
    Marital_Status = st.selectbox("Marital Status", ["Single", "Together", "Married", "Divorced", "Widow", "Alone", "Absurd", "YOLO"], index=0)
    Income = st.text_input("Income", "0")
    Kidhome = st.text_input("Kidhome", "0")
    Teenhome = st.text_input("Teenhome", "0")
    Recency = st.text_input("Recency", "0")
    NumDealsPurchases = st.text_input("NumDealsPurchases", "0")
    NumWebPurchases = st.text_input("NumWebPurchases", "0")
    NumCatalogPurchases = st.text_input("NumCatalogPurchases", "0")
    NumStorePurchases = st.text_input("NumStorePurchases", "0")
    NumWebVisitsMonth = st.text_input("NumWebVisitsMonth", "0")
    AcceptedCmp3 = st.text_input("AcceptedCmp3", "0")
    AcceptedCmp4 = st.text_input("AcceptedCmp4", "0")
    AcceptedCmp5 = st.text_input("AcceptedCmp5", "0")
    AcceptedCmp1 = st.text_input("AcceptedCmp1", "0")
    AcceptedCmp2 = st.text_input("AcceptedCmp2", "0")
    Complain = st.text_input("Complain", "0")
    Response = st.text_input("Response", "0")
    total_amount_spent = st.text_input("total_amount_spent", "0")
    Children = st.text_input("Children", "0")

    # Validate and convert input
    def validate_input(input_value):
        try:
            return int(input_value)
        except ValueError:
            return None

    inputs = {
        "Education": validate_input(Education),
        "Marital_Status": validate_input(Marital_Status),
        "Income": validate_input(Income),
        "Kidhome": validate_input(Kidhome),
        "Teenhome": validate_input(Teenhome),
        "Recency": validate_input(Recency),
        "NumDealsPurchases": validate_input(NumDealsPurchases),
        "NumWebPurchases": validate_input(NumWebPurchases),
        "NumCatalogPurchases": validate_input(NumCatalogPurchases),
        "NumStorePurchases": validate_input(NumStorePurchases),
        "NumWebVisitsMonth": validate_input(NumWebVisitsMonth),
        "AcceptedCmp3": validate_input(AcceptedCmp3),
        "AcceptedCmp4": validate_input(AcceptedCmp4),
        "AcceptedCmp5": validate_input(AcceptedCmp5),
        "AcceptedCmp1": validate_input(AcceptedCmp1),
        "AcceptedCmp2": validate_input(AcceptedCmp2),
        "Complain": validate_input(Complain),
        "Response": validate_input(Response),
        "total_amount_spent": validate_input(total_amount_spent),
        "Children": validate_input(Children)
    }

    # Check if all inputs are valid
    if None in inputs.values():
        st.error("")
    else:
        result = ""
        if st.button("Predict"):
            result = predict_note_authentication(
                inputs["Education"], inputs["Marital_Status"], inputs["Income"], inputs["Kidhome"], inputs["Teenhome"],
                inputs["Recency"], inputs["NumDealsPurchases"], inputs["NumWebPurchases"], inputs["NumCatalogPurchases"],
                inputs["NumStorePurchases"], inputs["NumWebVisitsMonth"], inputs["AcceptedCmp3"], inputs["AcceptedCmp4"],
                inputs["AcceptedCmp5"], inputs["AcceptedCmp1"], inputs["AcceptedCmp2"], inputs["Complain"], inputs["Response"],
                inputs["total_amount_spent"], inputs["Children"]
            )
            st.success('The output is {}'.format(result))
    
    if st.button("About"):
        st.text("Built with Streamlit")

if __name__ == '__main__':
    main()

