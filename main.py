#PREPROCESS DATA, COLLECT DATA VIA AUTHENTICATE INSTAGRAM USER AND THEN THRU INSTA BASIC DISPLAY API GET PROFILE PICTURES OF ALL GIRLS 1000+ FOLLOWERS
import pandas as pd
import os

img_dir= "images/"
csv_file = "results.csv"

image_paths = [os.path.join(img_dir, image_file) for image_file in os.listdir(img_dir)]

#ATTRIBUTE EXTRACTION FROM PROFILE PICTURES
from deepface import DeepFace

#also use this: https://www.betaface.com/wpa/index.php/products ??
# try:
#   analysis = DeepFace.analyze(img_path = "images/liam.png", actions = ["age", "gender", "emotion", "race"])
#   print(analysis)
# except:
#   print("No face detected")

api_key = "d45fd466-51e2-4701-8da8-04351c872236"
api_url = "www.betafaceapi.com/api/v2/face"

results_list = []
for path in image_paths:
      try:
        deep_result = DeepFace.analyze(img_path = path, actions = ["age", "gender", "emotion", "race"])
        print(deep_result)
        payload = {"api_key": api_key, "file": open(path, "rb")}
        beta_result = requests.post(api_url, files=payload).json()
        print(beta_result)

        results_list.append({'deep_result':deep_result, 'beta_result':beta_result})
      except:
        print("No face detected")

df = pd.DataFrame(results_list)
df.to_csv(csv_file, index=False)


#MULTILINEAR REGRESSION MODEL OR CNN BASED OFF OF AGE, GENDER, EMOTION, RACE TO PIN POINT EXACT TYPE
from sklearn.linear_model import LinearRegression

data = pd.read_csv("results.csv")

#still have to preprocess data

# from sklearn.model_selection import train_test_split
# Data_train, Data_test = train_test_split(data, test_size = 0.2, random_state = 0)

#MODEL EVAL

#random forest or Dimensionality Reduction (PCA) + Clustering (K-Means) would be better

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(data)

#HYPERPARAMETER TUNING

#GET LINKS
import requests
from bs4 import BeautifulSoup

def get_video_links(search_keywords):
  url = "https://www.xvideos.com/?k=" + search_keywords
  response = requests.get(url)
  soup = BeautifulSoup(response.content, "html.parser")
  for div in soup.find_all("div", class_="thumb"):
    a_tag = div.find('a')
    if a_tag is not None:
        href = a_tag.get('href')
        print('https://www.xvideos.com' + href)
    else:
        print('a tag not found')

if __name__ == "__main__":
  search_keywords = input("Enter search keywords: ")
  print("Here are the video links for your search:")
  video_links = get_video_links(search_keywords)


# import pornhub

# search_keywords = ["evolvedfights"]
# client = pornhub.PornHub(search_keywords)

# for star in client.getStars(10):
#     print(star)
#     print(star["name"])
    
# for video in client.getVideos(10):
#     print(video["url"])

# for photo_url in client.getPhotos(5):
#     print(photo_url)

# video = client.getVideo("SOME VIDEO URL")
# print(video)
# print(video['accurate_views'])


#DEPLOYMENT TO porngram.ai

