import joblib
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import pandas as pd
from typing import Literal

model = joblib.load('mental_health_model.pkl')

app = FastAPI()

class studentData(BaseModel):
    Age: int = Field(..., ge = 10, le = 100)
    Gender: Literal['Male', 'Female']
    Country: str 
    Academic_Level: Literal['Undergraduate', 'Graduate', 'High School']
    Most_Used_Platform: Literal['Facebook', 'LinkedIn', 'Instagram', 'Snapchat', 'Twitter','YouTube', 'TikTok', 'LINE', 'KakaoTalk', 'VKontakte', 'WhatsApp','WeChat']
    Purpose_Of_Use: Literal['Networking', 'Education', 'Entertainment', 'News']
    Avg_Daily_Usage_Hours: float = Field(..., ge = 0, le = 24)
    Daily_Unlocks: int = Field(..., ge = 0)
    Study_Hours: float = Field(..., ge = 0)
    Physical_Activity_Hours: float = Field(..., ge = 0, le = 24)
    Sleep_Hours_Per_Night: float = Field(..., ge = 0, le = 24)
    Stress_Level: Literal['Medium', 'Low', 'Very High', 'High']


#Describe what we send back
class PredictionResponse(BaseModel):
    predected_mental_health_score: float


@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

app.mount("/static", StaticFiles(directory="."), name="static")

top_country = ['United States', 'India', 'United Kingdom', 'Canada', 'Australia', 'Germany', 'France']

@app.post("/predict", response_model = PredictionResponse)
def predict(data: studentData):
    if data.Country in top_country:
        country_group = data.Country
    else:
        country_group = 'Other'

    input_row = pd.DataFrame([{
        'Age'                       : data.Age,
        'Gender'                    : data.Gender,
        'Country'                   : data.Country,
        'Academic_Level'            : data.Academic_Level,
        'Most_Used_Platform'        : data.Most_Used_Platform,
        'Purpose_Of_Use'            : data.Purpose_Of_Use,
        'Avg_Daily_Usage_Hours'     : data.Avg_Daily_Usage_Hours,
        'Daily_Unlocks'             : data.Daily_Unlocks,
        'Study_Hours'               : data.Study_Hours,
        'Physical_Activity_Hours'   : data.Physical_Activity_Hours,
        'Sleep_Hours_Per_Night'     : data.Sleep_Hours_Per_Night,
        'Stress_Level'              : data.Stress_Level,
        "Grouped_country"           : country_group
    }])

    prediction = model.predict(input_row)[0]
    return PredictionResponse(predected_mental_health_score = (float(prediction)))