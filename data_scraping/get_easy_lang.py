import locale
import time

import dateparser
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By

from get_data_fakenc_correctiv import web_site

locale.setlocale(locale.LC_ALL, 'de_de')
import re
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains


def get_news_urls(url):
    options = Options()
    options.headless = True
    driver = webdriver.Chrome(executable_path=r"/Users/darialinke/Downloads/chromedriver")
    driver.get(url)

    list_urls = []
    number_urls = 0

    for i in range(0, 1000):
        print(i)
        time.sleep(3)
        element = driver.find_element(By.XPATH, '//*[@id="content"]/div[2]/section[5]/button')
        actions = ActionChains(driver)
        actions.move_to_element(element).perform()
        actions.click().perform()
        # weitere Beiträge Button drücken
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        for a in soup.find_all('a', href=True):
            if str(a['href']).find(url) != -1 \
                    and str(a['href']) not in list_urls:
                list_urls.append(str(a['href']))
                number_urls += 1

        df_news_sites = pd.DataFrame(list_urls, columns=['URL'])
        df_news_sites.to_csv(path_or_buf='easy_lang_urls_kz.csv', sep=';', mode='a', header=False, index=False)


get_news_urls('https://www.kleinezeitung.at/service/topeasy')

# kleine Zeitung; title;article-body prose;shrink-0 flex items-center
# df: title;article-content;article-header-author;date_formats=['%d.%m.%Y']
# ndr:title;modulepadding copytext;lastchanged;['%d.%m.%Y']
# sr:title;article__header__text;text--teaser-subhead;['%d.%m.%Y']


df_news_urls = pd.read_csv('easy_lang_urls_kz.csv', delimiter=';', header=None, names=['URL', 'Date'])
invalid_urls_ts = []
df_news_urls = df_news_urls.drop_duplicates().sample(frac=1).reset_index(drop=True)
texts = []
options = Options()
options.headless = True
driver = webdriver.Chrome(r"/Users/darialinke/Downloads/chromedriver", options=options)
count_text = 0
for ind in df_news_urls.iloc[:, :].index:
    element = []
    url = df_news_urls["URL"][ind]
    site = web_site(url, driver,
                    'title',
                    'txt clearfix',
                    'date'
                    )
    title, body, date = site.get_text_from_url()

    try:
        date_processed = dateparser.parse(date, date_formats=['%d.%m.%Y'], locales=['de'])
    except:
        date_processed = None

    if date_processed is None:
        try:
            match_begin = re.search(r'\d{2}', date)
            match_end = re.search(r'\d{4}', date)
            date_processed = dateparser.parse(date[match_begin.start():match_end.end()], date_formats=['%d. %B %Y'],
                                              locales=['de'])
        except:
            date_processed = None

    if body is None or date is None:
        invalid_urls_ts.append(url)
    else:
        element.append(url)
        element.append(date_processed)
        element.append(title)
        element.append(body)
        element.append('real')
        element.append('1')
        count_text += 1
        texts.append(element)

df_results = pd.DataFrame(texts, columns=['url', 'date', 'title', 'text', 'label', 'label_id'])
df_results.to_json(path_or_buf='true_news_kz.json', force_ascii=False, orient='records')
driver.close()
driver.quit()

# """
