from fastapi import FastAPI
from service.api.api import main_router
app = FastAPI(project_name= "Facemask Detection")
app.include_router(main_router)

@app.get("/")
def read_values():
    return {"Hello": "Worldc"}