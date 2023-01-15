import locale
import time

import dateparser
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By

from get_data_fakenc_correctiv import web_site

locale.setlocale(locale.LC_ALL, 'de_de')
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains


def get_news_urls(url):
    options = Options()
    options.headless = True
    driver = webdriver.Chrome(executable_path=r"/Users/darialinke/Downloads/chromedriver")
    driver.get(url)

    list_urls = []
    number_urls = 0
    not_news_list = ['https://www.volksverpetzer.de/unterstutzen/', 'https://www.volksverpetzer.de/author/',
                     'https://www.volksverpetzer.de/page/', 'https://www.volksverpetzer.de/datenschutzerklaerung/'
                                                            'https://volksverpetzer-shop.de/',
                     'https://www.volksverpetzer.de/category/video',
                     'https://www.volksverpetzer.de/author/gast/', 'https://www.volksverpetzer.de/ueber-uns/',
                     'https://www.volksverpetzer.de/spenden/',
                     'https://www.volksverpetzer.de/impressum-volksverpetzer/',
                     ]

    # scroll through pages
    for i in range(0, 100):
        try:
            time.sleep(5)
            element = driver.find_element(By.XPATH,
                                          '//*[@id="post-76"]/div/div/div/div[2]/div/div/div[2]/div/div/div/div[1]')
            actions = ActionChains(driver)
            actions.move_to_element(element).perform()
            actions.click().perform()

            # press button to see further articles
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            for a in soup.find_all('a', href=True):
                if str(a['href']).find('https://www.volksverpetzer.de/') != -1 \
                        and a['href'] != 'https://www.volksverpetzer.de/category/aktuelles/' \
                        and str(a['href']) not in list_urls \
                        and not any(elem in a['href'] for elem in not_news_list):
                    print("Found the URL:", a['href'])
                    list_urls.append(str(a['href']))
                    number_urls += 1
        except:
            pass

        df_news_sites = pd.DataFrame(list_urls, columns=['URL'])
        df_news_sites.to_csv(path_or_buf='urls_vp.csv', sep=';', mode='a', header=False, index=False)


get_news_urls('https://www.volksverpetzer.de')

# VP; title;text;et_pb_module et_pb_post_content et_pb_post_content_0_tb_body;published

df_news_urls = pd.read_csv('urls_vp.csv', delimiter=';', header=None, names=['URL', 'Date'])
invalid_urls_ts = []
df_news_urls = df_news_urls.drop_duplicates()
texts = []
for ind in df_news_urls.iloc[:, :].index:
    element = []
    url = df_news_urls["URL"][ind]
    site = web_site(url, 'title',
                    'et_pb_module et_pb_post_content et_pb_post_content_0_tb_body',
                    'published')
    title, body, date = site.get_text_from_url()
    try:
        date_processed = dateparser.parse(date, date_formats=['%b %d, %Y'], locales=['de'])
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

df_results = pd.DataFrame(texts, columns=['url', 'date', 'title', 'text', 'label', 'label_id'])
df_results.to_json(path_or_buf='true_news_vp.json', force_ascii=False, orient='records')
