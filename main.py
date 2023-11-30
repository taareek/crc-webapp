from fastapi import FastAPI
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Request, Form
# import utils
# import utils_main
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# app object
app = FastAPI()

# mounting images and templates 
app.mount("/static", StaticFiles(directory= "static"), name="static")  
template = Jinja2Templates(directory= "templates")   


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/heda")
def home(request: Request):
    return template.TemplateResponse("index.html", {"request":request})
