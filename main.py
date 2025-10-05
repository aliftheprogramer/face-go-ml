from src.ml.api import app

# Optional: allow `python main.py` to run the server directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
