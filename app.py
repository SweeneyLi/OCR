from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, Form, File, UploadFile
from predict_image import precict_image

app = FastAPI()
from typing import List


@app.post("/file/")
async def process_file(
        file: UploadFile = File(...)):
    print("file")
    contents = await file.read()
    cost_time, res = precict_image(contents, file.filename)
    return {
        "filename": file.filename,
        "time": cost_time,
        "res": res
    }

@app.post("/file2/")
async def process_file2(
        files: List[bytes] = File(...)):
    file = files[0]
    contents = await file.read()
    cost_time, res = precict_image(contents, file.filename)
    return {
        "filename": file.filename,
        "time": cost_time,
        "res": res
    }
# class People(BaseModel):
#     name: str
#     age: int
#
# @app.post('/insert')
# def insert(people: People):
#     age_after_10_years = people.age + 10
#     msg = f'此人名字叫做：{people.name}，十年后此人年龄：{age_after_10_years}'
#     return {'success': True, 'msg': msg}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app=app,
                host="0.0.0.0",
                port=8080,
                workers=1)
