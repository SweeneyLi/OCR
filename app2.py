from typing import List

from fastapi import FastAPI, File, UploadFile
from starlette.responses import HTMLResponse
from predict_image import precict_image

app = FastAPI()


@app.post("/uploadfile/")
async def uploadfile(file: UploadFile = File(...)):
    contents = await file.read()
    return precict_image(contents, file.filename, mult=False)



@app.post("/uploadfiles/")
async def uploadfiles(
        files: List[UploadFile] = File(...)
):
    # return {"filenames": [file.filename for file in files]}
    res_list = []
    for file in files:
        # file = files[0]
        contents = await file.read()
        res_info = precict_image(contents, file.filename)
        res_list.append(res_info)
    return res_list


@app.get("/")
async def main():
    content = """
<body>
<form action="/uploadfile/" enctype="multipart/form-data" method="post">
<input name="file" type="file" >
<input type="submit">
</form>
<form action="/uploadfiles/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
 """
    return HTMLResponse(content=content)


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app=app,
                host="0.0.0.0",
                port=8080,
                workers=1)
