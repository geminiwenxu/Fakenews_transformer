import locale

import dateparser
import pandas as pd
from bs4 import BeautifulSoup
from get_data_fakenc_correctiv import web_site
from selenium import webdriver

locale.setlocale(locale.LC_ALL, 'de_de')
import re
from selenium.webdriver.chrome.options import Options


def get_news_urls(url):
    options = Options()
    options.headless = True
    driver = webdriver.Chrome(r"/Users/darialinke/Downloads/chromedriver", options=options)
    driver.get(url)
    list_urls = []
    number_urls = 0
    news_list = ['/news/', '/politik/']
    try:
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        for a in soup.find_all('a', href=True):
            if any(elem in a['href'] for elem in news_list) and a['href'] not in list_urls \
                    and str(a['href']).find('home') == -1 and str(a['href']).find('startseite') == -1 \
                    and str(a['href']).find('uebersicht') == -1 \
                    and 'https://www.bild.de' + str(a['href']) not in list_urls:
                list_urls.append('https://www.bild.de' + str(a['href']))
                number_urls += 1
    except:
        pass

    df_news_sites = pd.DataFrame(list_urls, columns=['URL'])

    df_news_sites.to_csv(path_or_buf='urls_bild.csv', sep=';', mode='a', header=False, index=False)
    driver.close()
    driver.quit()


for month in range(1, 12):
    for day in range(1, 28):
        try:  # if there is nothing for this day, go to next day
            get_news_urls('https://www.bild.de/archive/2021/' + str(day) + '/' + str(month) + '/index.html')
        except:
            pass

# Bild; title;article-body;datetime datetime--article
# """
df_news_urls = pd.read_csv('urls_bild.csv', delimiter=';', header=None, names=['URL', 'Date'])
invalid_urls_ts = []
df_news_urls = df_news_urls.drop_duplicates().sample(frac=1).reset_index(drop=True)
texts = []
for ind in df_news_urls.iloc[:, :].index:
    element = []
    url = df_news_urls["URL"][ind]
    date = df_news_urls["Date"][ind]
    site = web_site(url, 'title', 'text')
    title, body, date = site.get_text_from_url()
    match_begin = re.search(r'\d{2}', date)
    match_end = re.search(r'\d{4}', date)
    try:
        date_processed = dateparser.parse(date[match_begin.start():match_end.end()], date_formats=['%d. %B %Y'],
                                          locales=['de'])
    except:
        date_processed = None

    if title is None:
        invalid_urls_ts.append(url)
    else:
        element.append(url)
        element.append(date_processed)
        element.append(title)
        element.append(body)
        element.append('true')
        element.append(1)
    texts.append(element)

    if len(texts) > 15000:
        break
df_results = pd.DataFrame(texts, columns=['url', 'date', 'title', 'text', 'label', 'label_id'])
df_results.to_json(path_or_buf='true_news_bild.json', force_ascii=False, orient='records')
