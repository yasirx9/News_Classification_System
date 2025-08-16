import requests
from bs4 import BeautifulSoup
import pandas as pd

#Config
newsPerPage = 9
maxTries = 5
newsPerCategory = 5000

def news(url):
  p = url.rstrip('/').split('/')
  return p[-1].replace('-', ' ')

def visitCategory(category, numberOfNews):
  data = []
  headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
  }
  tries = 0
  pgno = 1
  while len(data) < numberOfNews and tries < maxTries:
    url = f"https://arynews.tv/category/{category}/page/{pgno}/"
    print(f"Visiting URL: {url}")
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
      print(f"Failed to retrieve page {pgno} of category {category}")
      tries += 1
      continue
    soup = BeautifulSoup(response.text, 'html.parser')
    div = soup.find(id="tdi_66")
    if div:
      tags = div.find_all(class_="td-module-meta-info")
      for tag in tags:
        a_tag = tag.find("a")
        if a_tag and a_tag.get('href'):
          link = a_tag.get('href')
          _category = "health" if category == "health-2" else category
          data.append([_category,news(link)])
    pgno += 1
  df = pd.DataFrame(data, columns=["Category", "News"])
  df.to_csv("dataset.csv", mode="a", header=not pd.io.common.file_exists("dataset.csv"), index=False)
  return len(data)

categories = {"sci-techno":0, "health-2":0, "sports":0, "pakistan":0}
for i in categories:
  categories[i] = visitCategory(i, newsPerCategory)
  print(f"{categories[i]} newses scraped for category: {i}")
