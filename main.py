from fastapi import FastAPI
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Request, Form
import utils
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# app object
app = FastAPI()

# mounting images and templates 
app.mount("/static", StaticFiles(directory= "static"), name="static")  
template = Jinja2Templates(directory= "templates")   

# @app.get("/")
# async def home(request: Request):
#     return template.TemplateResponse("index.html", {"request":request})

# @app.post("/")
# async def demo_home(request: Request, file:UploadFile= File(...)):
    
#     result = None
#     error = None

#     try:
#         result = utils.get_results(input_img=file)
#     except Exception as e:
#         error = e
#     return template.TemplateResponse("prediction.html", {"request":request, "result":result, "error":error})


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/heda")
def home(request: Request):
    return template.TemplateResponse("index.html", {"request":request})
