from fastapi import FastAPI, Request, Response
import numpy as np
import tensorflow as tf
from PIL import Image
import base64
import io
from pydantic import BaseModel
from PIL import Image
from load_model import *

def load(image_file):
  image = tf.convert_to_tensor(image_file)
  w = tf.shape(image)[1]
  w = w // 2
  input_image = image[:, w:, :]
  input_image = tf.cast(input_image, tf.float32)

  return input_image

def resize(input_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return input_image

def normalize(input_image):
  input_image = (input_image / 127.5) - 1
  return input_image

def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1)

app = FastAPI()

class Item(BaseModel):
    image64: str

generator_Death_Valley = GenerationModels(name = 'Death_Valley', path=f'./models/generators/Death_Valley_generator.pth', device='cpu')
generator_Laytonville = GenerationModels(name = 'Laytonville', path=f'./models/generators/Laytonville_generator.pth', device='cpu')
generator_Post_Earthquake = GenerationModels(name = 'Post_Earthquake', path=f'./models/generators/Post_Earthquake_generator.pth', device='cpu')
generator_Mt_Rainier = GenerationModels(name = 'Mt_Rainier', path=f'./models/generators/Mt_Rainier_generator.pth', device='cpu')
generator_San_Gabriel = GenerationModels(name = 'San_Gabriel', path=f'./models/generators/San_Gabriel_generator.pth', device='cpu')

coloring_Death_Valley = tf.keras.models.load_model('models/coloring/Death_Valley_coloring.h5')
coloring_Laytonville = tf.keras.models.load_model('models/coloring/Laytonville_coloring.h5')
coloring_Mt_Rainier = tf.keras.models.load_model('models/coloring/Mt_Rainier_coloring.h5')
coloring_Post_Earthquake = tf.keras.models.load_model('models/coloring/Post_Earthquake_coloring.h5')
coloring_San_Gabriel = tf.keras.models.load_model('models/coloring/San_Gabriel_coloring.h5')

fill_model = tf.keras.models.load_model("models/model_fill.h5", custom_objects={"dice_coef":dice_coef}) 

@app.post("/coloring")
async def read_root(response: Response, request: Request):
    item = await request.json()
    heightmap = np.array(item['heightmap'].split(',')).astype(np.float32).reshape(256, 256)
    image64 = item['image64']
    dec_image = base64.b64decode(image64)
    image = Image.open(io.BytesIO(dec_image))
    data = np.asarray(image)
    if data.shape[2] == 4:
        data = data[:, :, :3]
    heightmap = np.stack((heightmap,)*3, axis=-1)
    data = heightmap
    data = np.expand_dims(data, axis=0)
    data_proc = load(data)
    data_proc = resize(data_proc, 256, 256)
    data_proc = normalize(data_proc)
    if item['dataset'] == 'death_valley':
        pred = coloring_Death_Valley(data_proc, training=True)
    elif item['dataset'] == 'laytonville':
        pred = coloring_Laytonville(data_proc, training=True)
    elif item['dataset'] == 'post_earthquake':
        pred = coloring_Post_Earthquake(data_proc, training=True)
    elif item['dataset'] == 'mt_rainier':
        pred = coloring_Mt_Rainier(data_proc, training=True)
    elif item['dataset'] == 'san_gabriel':
        pred = coloring_San_Gabriel(data_proc, training=True)
    
    pred_final = ((pred[0].numpy() * 0.5 + 0.5) * 255.0).astype(np.uint8)
    pred_img = Image.fromarray(pred_final)

    buffered = io.BytesIO()
    pred_img.save(buffered, format="PNG")
    enc_pred_img = base64.b64encode(buffered.getvalue())

    response.headers['Access-Control-Allow-Origin'] = '*'
    return {'image64': enc_pred_img}

@app.post("/generate")
async def generate_heightmap(response: Response, request: Request):
    item = await request.json()
    print(item['dataset'])
    if item['dataset'] == 'death_valley':
        generated_heightmaps, generated_heightmaps_scaled = generator_Death_Valley.create_heightmaps('Death_Valley', 1, save=False)
    elif item['dataset'] == 'laytonville':
        generated_heightmaps, generated_heightmaps_scaled = generator_Laytonville.create_heightmaps('Laytonville', 1, save=False)
    elif item['dataset'] == 'post_earthquake':
        generated_heightmaps, generated_heightmaps_scaled = generator_Post_Earthquake.create_heightmaps('Post_Earthquake', 1, save=False)
    elif item['dataset'] == 'mt_rainier':
        generated_heightmaps, generated_heightmaps_scaled = generator_Mt_Rainier.create_heightmaps('Mt_Rainier', 1, save=False)
    elif item['dataset'] == 'san_gabriel':
        generated_heightmaps, generated_heightmaps_scaled = generator_San_Gabriel.create_heightmaps('San_Gabriel', 1, save=False)

    generated_heightmaps = (generated_heightmaps[0] + 1)/2

    pred_final = ((generated_heightmaps - generated_heightmaps.min()) * 255.0).astype(np.uint8)
    pred_final = np.dstack((pred_final,pred_final,pred_final))
    pred_img = Image.fromarray(pred_final)
    buffered = io.BytesIO()
    pred_img.save(buffered, format="PNG")
    enc_pred_img = base64.b64encode(buffered.getvalue())

    response.headers['Access-Control-Allow-Origin'] = '*'
    return {'image64': enc_pred_img, 'heightmap': ','.join(generated_heightmaps.flatten().astype(str))}

@app.post("/inpaiting")
async def fill_satelite(response: Response, request: Request):
    image64 = (await request.json())['image64']
    dec_image = base64.b64decode(image64)
    image = Image.open(io.BytesIO(dec_image))
    data = np.asarray(image) / 255.0
    inputs = [data.reshape((1,)+data.shape), data.reshape((1,)+data.shape)]
    output = fill_model.predict(inputs)
    output.reshape(output.shape[1:]) * 255.0
    output = (output * 255.0).astype(np.uint8)[0]
    pred_img = Image.fromarray(output)
    buffered = io.BytesIO()
    pred_img.save(buffered, format="PNG")
    enc_pred_img = base64.b64encode(buffered.getvalue())
    response.headers['Access-Control-Allow-Origin'] = '*'
    return {'image64': enc_pred_img}

@app.options("/coloring")
@app.options("/generate")
@app.options("/inpaiting")
async def options(response: Response):
  response.headers['Access-Control-Allow-Origin'] = '*'
  response.headers['Access-Control-Allow-Headers'] = '*'
  return {'status': 'success'}