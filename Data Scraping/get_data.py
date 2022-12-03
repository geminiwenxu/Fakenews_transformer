from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd
from sklearn.model_selection import train_test_split
import os

class web_site():
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
                self.body += txt.text.replace("\n", "")
            return self.title.replace("\n", "").strip(), self.body.strip()

        except:
            return None, None


def get_hoax_sites(hoax_urls):
    hoax_sites_classes = pd.read_csv('Hoax Site, body and title class.txt', delimiter=';')

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
                site = web_site(url, title_class, body_class)
                title, body = site.get_text_from_url()
                if title is None:
                    invalid_urls.append(url)
                else:
                    element.append(url)
                    element.append(date)
                    element.append(title)
                    element.append(body)
                    element.append('fake')
                    element.append('0')
                texts.append(element)

    df_results = pd.DataFrame(texts, columns =['url', 'date', 'title', 'article', 'label','labelID'])
    df_results = df_results[df_results['article'].isna() == False]
    df_results.to_json(path_or_buf='fake_news_correctiv.json', force_ascii=False, orient='records')

def get_news_urls(date):
    url = "https://www.tagesschau.de/archiv/?datum=" + date
    driver = webdriver.Chrome(executable_path=r"/Users/darialinke/Downloads/chromedriver")
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    links = soup.find_all(class_='teaser-xs__link')
    not_news_list = ['de/kommentar/', 'de/faktenfinder/', 'de/investigativ/', 'de/wissen/', 'de/multimedia/']
    list_urls = []
    number_urls = 0
    for link in links:
        if not any(elem in link['href'] for elem in not_news_list):
            list_urls.append(link['href'])
            number_urls+=1
            if number_urls>5:
                break
    list_dates = [date] * len(list_urls)
    df_news_sites = pd.DataFrame(list(zip(list_urls, list_dates)), columns=['URL', 'Date'])
    df_news_sites.to_csv(path_or_buf='news_urls.csv', sep=';', mode='a', header=False, index=False)


def get_dates_news_urls():
    #hoax_urls = pd.read_json('GermanFakeNC.json')
    hoax_urls = pd.read_csv('correctiv_fakes.csv', delimiter=';')
    dates = [str(i.date()) for i in pd.to_datetime(hoax_urls['Date'].unique())]
    for date in dates:
        get_news_urls(date)


def get_news_text():
    df_news_urls = pd.read_csv('news_urls.csv', delimiter=';', header=None, names=['URL','Date'])
    invalid_urls_ts = []
    df_news_urls = df_news_urls.drop_duplicates().sample(frac=1).reset_index(drop=True)
    texts = []
    for ind in df_news_urls.iloc[:,:].index:
        element = []
        url = df_news_urls["URL"][ind]
        date = df_news_urls["Date"][ind]
        site = web_site(url, 'article-breadcrumb__title--inside',
                        'm-ten m-offset-one l-eight l-offset-two textabsatz columns twelve')
        title, body = site.get_text_from_url()
        if title is None:
            invalid_urls_ts.append(url)
        else:
            element.append(url)
            element.append(date)
            element.append(title)
            element.append(body)
            element.append('true')
            element.append('1')
        texts.append(element)
    df_results = pd.DataFrame(texts, columns =['url', 'date', 'title', 'article', 'label','labelID'])
    df_results.to_json(path_or_buf='true_news_correctiv.json', force_ascii=False, orient='records')

if __name__ == "__main__":
    '''
    hoax_urls = pd.read_json('GermanFakeNC.json')
    hoax_urls = pd.read_csv('correctiv_fakes.csv', delimiter=';')
    get_hoax_sites(hoax_urls)
    get_news_text()
    pd_fake = pd.read_json('fake_news_1.json')
    pd_true = pd.read_json('true_news.json')
    df_all = pd.concat([pd_fake,pd_true.loc[:137,:]])
    df_all.to_json(path_or_buf='news_all.json', force_ascii=False, orient='records')
    '''
    news = pd.read_json('news_all.json')
    news = news.rename(columns={"labelID": "label_id", "article": "text"})

    # combine with corrective
    pd_fake = pd.read_json('fake_news_correctiv.json')
    pd_fake = pd_fake.rename(columns={"labelID": "label_id", "article": "text"})
    pd_fake['label_id'] = 0
    pd_true = pd.read_json('true_news_correctiv.json')
    pd_true = pd_true.rename(columns={"labelID": "label_id", "article": "text"})
    df_scrapped = pd.concat([pd_fake, pd_true, news])
    df_scrapped.dropna(inplace=True)

    #combine with fang
    df_fang = pd.read_json('fang_processed.json')
    df_all = pd.concat([df_scrapped, df_fang])
    #dublicates possible, find them
    df_all = df_all.drop_duplicates(subset='text', keep="first")

    news_train, news_test, = train_test_split(news, test_size = 0.2, random_state = 42)
    news_test, news_dev, = train_test_split(news_test, test_size=0.5, random_state=42)

    news_train.to_json(path_or_buf='news_train_new.json', force_ascii=False, orient='records')
    news_test.to_json(path_or_buf='news_test_new.json', force_ascii=False, orient='records')
    news_dev.to_json(path_or_buf='news_dev_new.json', force_ascii=False, orient='records')
    #

