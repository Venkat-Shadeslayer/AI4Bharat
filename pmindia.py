from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.alert import Alert
import time

options = webdriver.ChromeOptions()
driver = webdriver.Chrome(options=options)

def scroll_to_end():
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

def append_article():
    # Create a set to store unique URLs
    unique_urls = set()
    # Find and extract all the links on the page
    links = [link.get_attribute('href') for link in driver.find_elements(By.TAG_NAME, 'a')]
    links = driver.find_elements(By.XPATH,'//*[@id="wrapper"]/section[2]/div[1]/div/div[3]/ul/li/div[2]/h3/a')
    article_links = [link.get_attribute('href') for link in links]
    print (len(links))
    unique_urls.update(article_links)
    file_path = 'pmindia_unique_urls.txt'
    with open(file_path, 'w', encoding='utf-8') as file:
        for url in unique_urls:
            file.write(url + '\n')





def scroll_to_bottom(driver):

    old_position = 0
    new_position = None

    while new_position != old_position:
        # Get old scroll position
        old_position = driver.execute_script(
                ("return (window.pageYOffset !== undefined) ?"
                 " window.pageYOffset : (document.documentElement ||"
                 " document.body.parentNode || document.body);"))
        # Sleep and Scroll
        time.sleep(2.5)
        driver.execute_script((
                "var scrollingElement = (document.scrollingElement ||"
                " document.body);scrollingElement.scrollTop ="
                " scrollingElement.scrollHeight;"))
        append_article()
        
        # Get new position
        new_position = driver.execute_script(
                ("return (window.pageYOffset !== undefined) ?"
                 " window.pageYOffset : (document.documentElement ||"
                 " document.body.parentNode || document.body);"))
        


driver.get("https://www.pmindia.gov.in/en/news-updates/")

scroll_to_bottom(driver)
'''
# Locate and click on the language dropdown
Languages = driver.find_element(By.XPATH, '//*[@id="lang_choice_polylang-2"]')
Languages.click()

# Accept any alerts if they are present
try:
    alert = Alert(driver)
    alert.accept()
except:
    pass

class InfiniteScroll(object):
    def __init__(self, last):
        self.last = last

    def __call__(self, driver):
        new = driver.execute_script('return document.body.scrollHeight')
        if new > self.last:
            return new
        else:
            return False'''

# Create a set to store unique URLs
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

        # Find and extract all the links on the page
        #links = [link.get_attribute('href') for link in driver.find_elements(By.TAG_NAME, 'a')]
        links = driver.find_elements(By.XPATH,'//*[@id="wrapper"]/section[2]/div[1]/div/div[3]/ul/li/div[2]/h3/a')
        article_links = [link.get_attribute('href') for link in links]
        print (len(links))

         # Filter out None values from the links list
        #links = [link for link in links if link is not None]


        # Add the links to the set (avoiding duplicates)
        unique_urls.update(article_links)

except Exception as e:
    print(f"An error occurred: {str(e)}")

# Close the Selenium driver
driver.quit()

# Write unique URLs to a text file
file_path = 'pmindia_unique_urls.txt'
with open(file_path, 'w', encoding='utf-8') as file:
    for url in unique_urls:
        file.write(url + '\n')



