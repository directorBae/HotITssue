import matplotlib.pyplot as plt
from wordcloud import WordCloud

import os
from google.cloud import vision
import pandas as pd

import numpy as np

import io

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'YOUR-KEY-FILE'
 
client_options = {'api_endpoint': 'eu-vision.googleapis.com'}
client = vision.ImageAnnotatorClient(client_options=client_options)

data = pd.read_json("data.json").drop(['Unnamed: 0'], axis=1)
data.index = data['keyword'].astype('string').tolist()

df = data[['HOT']] * 100
fre = df[['HOT']].astype('int32')

word_freq = fre['HOT'].to_dict()

wordcloud = WordCloud(prefer_horizontal=True, margin=90, height=1000, width=1000).generate_from_frequencies(word_freq)

im = wordcloud.to_image()

buffer = io.BytesIO()
im.save(buffer, format='PNG')
content = buffer.getvalue()

image = vision.Image(content=content)
response = client.text_detection(image=image)
texts = response.text_annotations

resultlist = []

for text in texts:
    ocr_text = text.description
    x1 = text.bounding_poly.vertices[0].x
    y1 = text.bounding_poly.vertices[0].y
    x2 = text.bounding_poly.vertices[1].x
    y2 = text.bounding_poly.vertices[1].y
    resultlist.append([ocr_text, (int(x1), int(y1)), (int(x2), int(y2))])

print(resultlist)