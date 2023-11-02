# # from fastapi import FastAPI
# # from pydantic import BaseModel
# # import joblib
# # import numpy as np
# # import pandas as pd
# # import pickle
# # from tensorflow import keras


# # # from tensorflow.keras.models import Sequential
# # # from tensorflow.keras.layers import Dense, Dropout
# # # from tensorflow.keras.metrics import AUC
# # # from tensorflow.keras.utils import to_categorical
# # # from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# # from datetime import datetime

# # app = FastAPI()

# # # pickle_in = open('model_pkl', 'rb')
# # # model = pickle.load(pickle_in)

# # # Load the trained model
# # # with open('model_pkl', 'rb') as f: 
# # #     model = pickle.load(f)

# # model = joblib.load('model.pkl')

# # class PredictionRequest(BaseModel):
# #     sensor_unit_id: int
# #     co2_level: float
# #     hummidity_level: float
# #     temperature: float
# #     light_intensity: float
# #     pir_reading: float

# # class PredictionResponse(BaseModel):
# #     sensor_id: int
# #     occupancy_rate: str
# #     room_status: str
# #     sent_time: datetime

# # @app.post("/predict", response_model=PredictionResponse)
# # async def predict(request: PredictionRequest):
    
# #     temp = request.temperature
# #     light  = request.light_intensity
# #     co2 = request.c02_level
# #     pir = request.pir_reading
    
# #     prediction  = model.Predict([[temp, light, co2, pir, co2]])
     
# #     print(prediction)
    
# #     room_occ = pd.series(prediction).idmax()
    
# #     occ_rate = lambda room_occ: "high" if room_occ == 0 else ("medium" if room_occ == 1 else ("low" if room_occ == 2 else "none"))
    
# #     return {
# #         "sensor_id": request.sensor_unit_id,
# #         "occupancy_rate" : occ_rate,
# #         "room_status" : "normal",
# #         "sent_time" : datetime.datetime.now()
# #     }   
    
# #     # x = request.x
# #     # prediction = model.predict(np.array([x]))[0]
# #     # return {"prediction": prediction}

# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import pandas as pd
# from datetime import datetime

# app = FastAPI()

# # Load the trained model using joblib
# model = joblib.load('joblibedmodel.joblib')

# class PredictionRequest(BaseModel):
#     sensor_unit_id: int
#     co2_level: float
#     humidity_level: float
#     temperature: float
#     light_intensity: float
#     pir_reading: int

# class PredictionResponse(BaseModel):
#     sensor_id: int
#     occupancy_rate: str
#     room_status: str
#     sent_time: datetime

# @app.post("/predict", response_model=PredictionResponse)
# async def predict(request: PredictionRequest):
    
#     print("hello world")
    
#     temp = request.temperature
#     light = request.light_intensity
#     co2 = request.co2_level
#     pir = request.pir_reading
    
#     print(temp, light, co2, pir)
    
#     prediction = model.predict([[temp, light, pir, co2]])
     
#     print(prediction)
    
#     room_occ = pd.Series(prediction).idxmax()
    
#     occ_rate = lambda room_occ: "high" if room_occ == 0 else ("medium" if room_occ == 1 else ("low" if room_occ == 2 else "none"))
    
#     return {
#         "sensor_id": request.sensor_unit_id,
#         "occupancy_rate": occ_rate(room_occ),
#         "room_status": "normal",
#         "sent_time": datetime.now()
#     }

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import random

app = FastAPI()

# Load the trained model using joblib
model = joblib.load('joblibedmodel.joblib')

class PredictionRequest(BaseModel):
    sensor_unit_id: int
    co2_level: float
    humidity_level: float
    temperature: float
    light_intensity: float
    pir_reading: int

class PredictionResponse(BaseModel):
    sensor_id: int
    occupancy_rate: str
    room_status: str
    sent_time: datetime

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    
    temp = request.temperature
    light = request.light_intensity
    co2 = request.co2_level
    pir = request.pir_reading
    
    input_data = np.array([[temp, light, pir, co2]])
    print(input_data)
    
    prediction = model.predict(input_data)
     
    room_occ = pd.Series(prediction[0]).idxmax()
    
    occ_rate = lambda room_occ: "high" if room_occ == 0 else ("medium" if room_occ == 1 else ("low" if room_occ == 2 else "none"))
    
    room_occ = random.random()
    
    occ_rate = lambda room_occ: "high" if room_occ <= 0.25 else ("medium" if room_occ <= 0.5 else ("low" if room_occ <= 0.75 else "none"))

    
    occupancy_rate = occ_rate(room_occ)
    
    print(occupancy_rate)
    
    # occupancy_rate = "none" if pir == 0 else ("low" if co2 < 200 else ("medium" if co2 < 500 else "high"))
    
    return {
        "sensor_id": request.sensor_unit_id,
        "occupancy_rate": occupancy_rate,
        "room_status": "normal",
        "sent_time": datetime.now()
    }