import uvicorn
from fastapi import FastAPI

app = FastAPI()


@app.get('/ping')
async def ping():
    return {"message": "Hello, I am alive"}


if __name__ == '__main__':  # Corrected line
    uvicorn.run(app, host='localhost', port=8000, reload=True)