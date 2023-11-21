from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from joblib import Parallel, delayed
import os
def scroll_to_end(driver):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
def append_article(driver, file_path):
    # Create a set to store unique URLs
    unique_urls = set()
    # Find and extract all the links on the page
    links = driver.find_elements(By.TAG_NAME, 'a')
    links = driver.find_elements(By.XPATH, '//*[@id="wrapper"]/section[2]/div[1]/div/div[3]/ul/li/div[2]/h3/a')
    article_links = [link.get_attribute('href') for link in links]
    print(len(links))
    unique_urls.update(article_links)
    # Create a folder named 'URLs' if it doesn't exist
    folder_path = 'URLs'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # Concatenate the folder path with the file name
    file_path_in_folder = os.path.join(folder_path, file_path)
    with open(file_path_in_folder, 'w', encoding='utf-8') as file:
        for url in unique_urls:
            file.write(url + '\n')
def scroll_to_bottom(driver, file_path):
    old_position = 0
    new_position = None
    while new_position != old_position:
        old_position = driver.execute_script(
            ("return (window.pageYOffset !== undefined) ?"
             " window.pageYOffset : (document.documentElement ||"
             " document.body.parentNode || document.body);"))
        time.sleep(3.5)
        driver.execute_script((
                "var scrollingElement = (document.scrollingElement ||"
                " document.body);scrollingElement.scrollTop ="
                " scrollingElement.scrollHeight;"))
        append_article(driver, file_path)
        new_position = driver.execute_script(
            ("return (window.pageYOffset !== undefined) ?"
             " window.pageYOffset : (document.documentElement ||"
             " document.body.parentNode || document.body);"))
def get_data(website_url, file_path):
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    driver.get(website_url)
    scroll_to_bottom(driver, file_path)
    unique_urls = set()
    try:
        last_height = driver.execute_script('return document.body.scrollHeight')
        flag = 1
        while flag == 1:
            driver.execute_script('window.scrollTo(0,document.body.scrollHeight)')
            try:
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height != last_height:
                    last_height = new_height
                    time.sleep(2)
                else:
                    print("End of page reached")
                    flag = 0
            except:
                flag = 0
            links = driver.find_elements(By.XPATH, '//*[@id="wrapper"]/section[2]/div[1]/div/div[3]/ul/li/div[2]/h3/a')
            article_links = [link.get_attribute('href') for link in links]
            print(len(links))
            unique_urls.update(article_links)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        driver.quit()
    return unique_urls
def main(website_url, file_path):
    unique_urls = get_data(website_url, file_path)
    print(unique_urls)
    # Create a folder named 'URLs' if it doesn't exist
    folder_path = 'URLs'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # Concatenate the folder path with the file name
    file_path_in_folder = os.path.join(folder_path, file_path)
    with open(file_path_in_folder, 'w', encoding='utf-8') as file:
        for url in unique_urls:
            file.write(url + '\n')
if __name__ == "__main__":
    # Define the URLs and file paths for both scripts
    urls_and_paths = [
        ('https://www.pmindia.gov.in/en/news-updates/', 'pmindia_unique_urls_english.txt'),
        ('https://www.pmindia.gov.in/hi/न्यूज-अपडेट्स/', 'pmindia_unique_urls_hindi.txt'),
        ('https://www.pmindia.gov.in/asm/বাতৰি-সংজোযন/', 'pmindia_unique_urls_assamese.txt'),
        ('https://www.pmindia.gov.in/bn/সর্বশেষ-প্রাপ্ত-সংবাদ/', 'pmindia_unique_urls_bengali.txt'),
        ('https://www.pmindia.gov.in/gu/ન્યુઝ-અપડેટ/', 'pmindia_unique_urls_gujrati.txt'),
        ('https://www.pmindia.gov.in/kn/ಇತ್ತೀಚಿನ-ಸುದ್ದಿಗಳು/', 'pmindia_unique_urls_kannada.txt'),
        ('https://www.pmindia.gov.in/ml/പുതിയ-വാർത്തകൾ/', 'pmindia_unique_urls_malayalam.txt'),
        ('https://www.pmindia.gov.in/mni/অনৌবা-পাউশিং/', 'pmindia_unique_urls_manipuri.txt'),
        ('https://www.pmindia.gov.in/mr/ताज्या-बातम्या/', 'pmindia_unique_urls_marathi.txt'),
        ('https://www.pmindia.gov.in/ory/ସଦ୍ୟତମ-ଖବର/', 'pmindia_unique_urls_oriya.txt'),
        ('https://www.pmindia.gov.in/pa/ਨਿਊਜ਼-ਅੱਪਡੇਟ/', 'pmindia_unique_urls_punjabi.txt'),
        ('https://www.pmindia.gov.in/ta/சமீபத்திய-செய்திகள்/', 'pmindia_unique_urls_tamil.txt'),
        ('https://www.pmindia.gov.in/te/తాజా-స‌మాచారం/', 'pmindia_unique_urls_telugu.txt'),
        ('https://www.pmindia.gov.in/ur/مختصر-تازہ-خبریں/', 'pmindia_unique_urls_urdu.txt')
    ]
    # Run scripts in parallel with prefer="threads"
    num_jobs = 14
    results = Parallel(n_jobs=num_jobs, prefer="threads")(
        delayed(main)(*url_path) for url_path in urls_and_paths
    )


