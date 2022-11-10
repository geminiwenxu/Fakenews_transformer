from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd


class Hoax_site():
    def __init__(self, url, class_title, class_article):
        self.url = url
        self.class_title = class_title
        self.class_article = class_article

    def get_text_from_url(self):
        try:
            driver = webdriver.Chrome(executable_path=r"/Users/darialinke/Downloads/chromedriver")
            driver.get(self.url)
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            if self.class_title == 'title':
                self.title = soup.find("meta", property="og:title")["content"]
            else:
                self.title = soup.find(class_=self.class_title).text

            self.blogbeitraege = soup.find_all(class_=self.class_article)
            self.body = ''
            for txt in self.blogbeitraege:
                self.body += txt.text
            return self.title.replace("\n", "").strip(), self.body.strip()

        except:
            return None, None


def get_hoax_sites():
    hoax_urls = pd.read_json('/Users/darialinke/Downloads/GermanFakeNC/GermanFakeNC.json')
    hoax_sites_classes = pd.read_csv('/Users/darialinke/Downloads/GermanFakeNC/Hoax Site, body and title class.txt', delimiter=';')

    # find respective class and title
    invalid_urls = []
    texts = []
    for ind in hoax_urls.index:
        element = []
        url = hoax_urls["URL"][ind]
        date = hoax_urls["Date"][ind]
        for ind in hoax_sites_classes.index:
            if url.find(hoax_sites_classes["Hoax_Site"][ind]) != -1:
                body_class = hoax_sites_classes["body"][ind]
                title_class = hoax_sites_classes["title"][ind]
                site = Hoax_site(url, title_class, body_class)
                title, body = site.get_text_from_url()
                if title == None:
                    invalid_urls.append(url)
                else:
                    element.append(url)
                    element.append(date)
                    element.append(title)
                    element.append(body)
        texts.append(element)

    df_results = pd.DataFrame(texts, columns =['url','date','title', 'article'])
    df_results.to_csv(path_or_buf= '/Users/darialinke/Downloads/GermanFakeNC/fake_news.txt', sep= '¬')

def get_news_urls(date):
    url = "https://www.tagesschau.de/archiv/?datum=" + date
    driver = webdriver.Chrome(executable_path=r"/Users/darialinke/Downloads/chromedriver")
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    links = soup.find_all(class_='teaser-xs__link')
    not_news_list = ['de/kommentar/', 'de/faktenfinder/', 'de/investigativ/', 'de/wissen/', 'de/multimedia/']
    list_urls = []
    for link in links:
        if not any(elem in link['href'] for elem in not_news_list):
            list_urls.append(link['href'])
    list_dates = [date] * len(list_urls)
    return list_urls, list_dates

def get_news_date():
    hoax_urls = pd.read_json('/Users/darialinke/Downloads/GermanFakeNC/GermanFakeNC.json')
    dates = [str(i.date()) for i in pd.to_datetime(hoax_urls['Date'].unique())]
    return dates


list_news_dates = get_news_date()
list_urls = []
list_dates = []
for date in list_news_dates:
    urls, dates = get_news_urls(date)
    for url in urls: list_urls.append(url)
    for d in dates: list_dates.append(d)


df_news_sites = pd.DataFrame(list(zip(list_urls,list_dates)), columns=['URL','Date'])
df_news_sites.to_csv(path_or_buf= '/Users/darialinke/Downloads/GermanFakeNC/news_urls.txt', sep= ';')




