
import numpy as np 

import joblib 

from typing import Optional

from fastapi import FastAPI

from pydantic import BaseModel

model = joblib.load(open("/Users/brl.314/Downloads/india_spam_aug_2.pkl","rb"))  

def pred(x):

    results = model.predict([x])
    
    return results[0]

#Confidence Function

def conf(x):
    
    confidence = model.decision_function([x])
    
    return confidence[0]

app = FastAPI(title="India Spam Model API for Questions")

class spamrequest():
    text: str

class response():
    prediction: str 
    confidence_score: float
    decision: str

@app.post("/india_spam")
async def predict_spam(content: str):
        
        prediction = model.predict([content])[0]
    
        confidence_score = round(abs(model.decision_function([content])[0]),4)
            
        if confidence_score > 0.7 and prediction=='Spam':
            decision = "Delete it"
        if confidence_score > 0.7 and prediction=='Not Spam':
            decision = "Keep it"
        if confidence_score < 0.7 and prediction=='Spam':
            decision = "Send it to a moderator"
        if confidence_score < 0.7 and prediction=='Not Spam':
            decision = "Send it to a moderator"

        return {"Prediction":prediction,"Confidence Score": confidence_score,"Decision":decision}

if __name__ == '__main__':

    uvicorn.run(app, host='127.0.0.1', port=5000, debug=True)