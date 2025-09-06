from fastapi import FastAPI
from fastapi.responses import Response

app = FastAPI()

@app.post("/voice")
def testing_twilio():
    twiml_response = """
    <Response>
        <Say>Welcome to AI</Say>
    </Response>
    """
    return Response(content=twiml_response, media_type="application/xml")
