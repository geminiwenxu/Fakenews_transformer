
from bs4 import BeautifulSoup
import requests

url = 'https://perspektive-online.net/2022/11/ueber-30-lng-schiffe-warten-vor-der-europaeischen-kueste-auf-hoehere-preise/'
#website = requests.get(url)
#results = BeautifulSoup(website.content, 'html.parser')
#blogbeitraege = results.find(class_='seitenkopf__data columns twelve  m-ten  m-offset-one l-eight l-offset-two')

#print(blogbeitraege)

# another try will selenium

from selenium import webdriver

#url = "https://www.tagesschau.de/archiv/?datum=2019-11-01"  # some day
#driver = webdriver.Chrome(executable_path=r"/Users/darialinke/Downloads/chromedriver")
#driver.get(url)
#soup = BeautifulSoup(driver.page_source, 'html.parser')
"""

# get the urls
links = soup.find_all(class_='teaser-xs__link')
not_news_list = ['de/kommentar/','de/faktenfinder/','de/investigativ/','de/wissen/','de/multimedia/']
list_urls = []
for link in links:
    if not any(elem in link['href'] for elem in not_news_list):
        list_urls.append(link['href'])



#blogbeitraege = soup.find_all(class_='m-ten m-offset-one l-eight l-offset-two textabsatz columns twelve') # tagesschau
blogbeitraege = soup.find_all(class_='td-post-content tagdiv-type')  # hoax blauter bote

#title = soup.find(class_='article-breadcrumb__title--inside').text #Tagesschau
#title = soup.find(class_='current').text# hoax blauter bote
title = soup.find("meta", property="og:title")["content"]


#print(soup.find(class_='elementor-form').find('input', {'name':'referer_title'})['value'])
body = ''

for txt in blogbeitraege:
    body+= txt.text
    #print(txt.text)
print(title.replace("\n", "").strip())
print(body)
"""
import pandas as pd
hoax_sites_classes = pd.read_csv('/Users/darialinke/Downloads/GermanFakeNC/fake_news.txt', delimiter='¬')

print(hoax_sites_classes['article'])











