import gradio as gr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# Load dataset
data = pd.read_csv('uber.csv')
data = data.drop('cars_available', axis=1)

# Features and target
X = data.drop('price_usd', axis=1)
Y = data['price_usd']

# Train-test split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)

# Train model
xgb = XGBRegressor()
xgb.fit(Xtrain, Ytrain)

# Evaluate model (just printing in logs, not in interface)
output = xgb.predict(Xtest)
score = r2_score(Ytest, output)
print("R² Score:", score)

# Gradio prediction function
def predict_price(rain, distance, time, traffic):
    user_data = [[rain, distance, time, traffic]]
    predicted_price = xgb.predict(user_data)[0]
    return f"Predicted Uber Price: ${predicted_price:,.2f}"

# Gradio Interface
interface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Slider(1, 10, step=1, label="Rain (1-10)", value=5),
        gr.Number(label="Distance (km)", value=10),
        gr.Slider(1, 24, step=1, label="Time (Hour of Day)", value=12),
        gr.Slider(1, 10, step=1, label="Traffic (1-10)", value=5),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Uber Price Prediction",
    description="Enter ride details to predict Uber price using XGBoost."
)

# Launch app
if __name__ == "__main__":
    interface.launch(share=True)
