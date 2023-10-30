#PREPROCESS DATA, COLLECT DATA VIA AUTHENTICATE INSTAGRAM USER AND THEN THRU INSTA BASIC DISPLAY API GET PROFILE PICTURES OF ALL GIRLS 1000+ FOLLOWERS
import pandas as pd
import os
import json, requests
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from deepface import DeepFace
import statistics
from bs4 import BeautifulSoup


img_dir= "images/"
csv_file = "results.csv"

image_paths = [os.path.join(img_dir, image_file) for image_file in os.listdir(img_dir)]

#ATTRIBUTE EXTRACTION FROM PROFILE PICTURES
api_key = "d45fd466-51e2-4701-8da8-04351c872236"
api_url = "www.betafaceapi.com/api/v2/face"

ages = []
genders = []
emotions = []
races = []

# #also use this: https://www.betaface.com/wpa/index.php/products ??
# try:
#   analysis = DeepFace.analyze(img_path = "images/ben.png", actions = ["age", "gender", "emotion", "race"])
#   print(analysis)
#   #print('Gender:' + analysis['dominant_gender'] + '  Emotion:' + analysis['dominant_emotion'] + '  Race:' + analysis['dominant_race'])

#   age = next((feature.get('age') for feature in analysis if 'age' in feature), None)
#   gender = next((feature.get('dominant_gender') for feature in analysis if 'dominant_gender' in feature), None)
#   emotion = next((feature.get('dominant_emotion') for feature in analysis if 'dominant_emotion' in feature), None)
#   race = next((feature.get('dominant_race') for feature in analysis if 'dominant_race' in feature), None)

#   results = [age, gender, emotion, race]
#   print(results)

#   ages.append(int(age))
#   genders.append(gender)
#   emotions.append(emotion)
#   races.append(race)
#   # payload = {"api_key": api_key, "file": open("images/ben.png", "rb")}
#   # beta_result = requests.post(api_url, files=payload).json()
#   # print(beta_result)
#   headers = {"Authorization":"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMWY2ZTkyMTktMTIwYi00MDI3LWI3Y2ItY2E2M2Y3NTNhOTg1IiwidHlwZSI6ImFwaV90b2tlbiJ9.2UHHLcaIY8Yw9rC9HxU7hcpZClAak5uDnUxbzeadjZk"}

#   url = "https://api.edenai.run/v2/image/face_detection"              	 
#   data={"show_original_response": False,"fallback_providers": "","providers": "google"}
#   files = {'file': open("images/ben.png",'rb')}

#   response = requests.post(url, data=data, files=files, headers=headers)
#   result = json.loads(response.text)
#   print(result['google']['items'])
# except:
#   print("No face detected")


def analyze_image(image_path):
    try:
        analysis = DeepFace.analyze(img_path=image_path, actions=["age", "gender", "emotion", "race"])

        age = next((feature.get('age') for feature in analysis if 'age' in feature), None)
        gender = next((feature.get('dominant_gender') for feature in analysis if 'dominant_gender' in feature), None)
        emotion = next((feature.get('dominant_emotion') for feature in analysis if 'dominant_emotion' in feature), None)
        race = next((feature.get('dominant_race') for feature in analysis if 'dominant_race' in feature), None)

        return int(age), gender, emotion, race

    except:
        print("No face detected")
        return None, None, None, None

for _ in range(50):
    age, gender, emotion, race = analyze_image('images/ben.png')

    if age is not None:
        ages.append(age)
        genders.append(gender)
        emotions.append(emotion)
        races.append(race)

# results_list = []
# for path in image_paths:
#       try:
#         deep_result = DeepFace.analyze(img_path = path, actions = ["age", "gender", "emotion", "race"])
#         print(deep_result)
#         payload = {"api_key": api_key, "file": open(path, "rb")}
#         beta_result = requests.post(api_url, files=payload).json()
#         print(beta_result)

#         results_list.append({'deep_result':deep_result, 'beta_result':beta_result})
#       except:
#         print("No face detected")


# data = results_list

gender_binary = [1 if gender == 'Male' else 0 for gender in genders]

emotion_label_encoder = LabelEncoder()
emotion_integer_encoded = emotion_label_encoder.fit_transform(emotions)
emotion_onehot_encoder = OneHotEncoder(sparse=False)
emotion_integer_encoded = emotion_integer_encoded.reshape(len(emotion_integer_encoded), 1)
emotion_onehot_encoded = emotion_onehot_encoder.fit_transform(emotion_integer_encoded)

race_label_encoder = LabelEncoder()
race_integer_encoded = race_label_encoder.fit_transform(races)
race_onehot_encoder = OneHotEncoder(sparse=False)
race_integer_encoded = race_integer_encoded.reshape(len(race_integer_encoded), 1)
race_onehot_encoded = race_onehot_encoder.fit_transform(race_integer_encoded)


#MODEL EVAL
feature_matrix = np.column_stack((ages, gender_binary, emotion_onehot_encoded, race_onehot_encoded))

X_train, X_test, y_train, y_test = train_test_split(feature_matrix, races, test_size=0.2, random_state=42)

#random forest 
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Random Forest Classifier Accuracy:", accuracy)

# conf_matrix = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:\n" + str(conf_matrix))

# unique_classes, counts = np.unique(y_pred, return_counts=True)
# most_common_predicted_type = unique_classes[np.argmax(counts)]
# print("Most Common Predicted Romantic Type:", most_common_predicted_type)



# #Dimensionality Reduction (PCA) + Clustering (K-Means)?
# pca = PCA(n_components=2)
# reduced_features = pca.fit_transform(feature_matrix)
# kmeans = KMeans(n_clusters=3)
# clusters = kmeans.fit_predict(reduced_features)

# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor.fit(data)

#algorithmic approach
mean_age = sum(ages) / len(ages)
standard_deviation_ages = (sum((age - mean_age) ** 2 for age in ages) / (len(ages) - 1)) ** 0.5
mode_gender = statistics.mode(genders)
mode_emotions = statistics.mode(emotions)
mode_race = statistics.mode(races)
print("mean age: " + str(mean_age) + "\nstandard_dev: " + str(standard_deviation_ages) + "\nGender: " + str(mode_gender) + "\nEmotions: " + str(mode_emotions) + "\nRace: " + str(mode_race))

#GET LINKS

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

# search_keywords = [""]
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