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

