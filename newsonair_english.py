from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import os


def scroll_to_bottom(driver, unique_urls):
    scroll_increment = 500  # Adjust the scroll increment as needed
    old_position = 0
    new_position = None

    while new_position != old_position:
        # Get old scroll position
        old_position = driver.execute_script(
            ("return (window.pageYOffset !== undefined) ?"
             " window.pageYOffset : (document.documentElement ||"
             " document.body.parentNode || document.body);"))

        # Scroll incrementally
        driver.execute_script(f"window.scrollBy(0, {scroll_increment});")
        time.sleep(.5)
        append_article(driver,unique_urls)

        # Append article links periodically
        if old_position != 0 and old_position == new_position:
            
            time.sleep(.5)

        # Get new position
        new_position = driver.execute_script(
            ("return (window.pageYOffset !== undefined) ?"
             " window.pageYOffset : (document.documentElement ||"
             " document.body.parentNode || document.body);"))

    

def append_article(driver, unique_urls):
    # Find the total number of articles on the page
    article_elements = driver.find_elements(By.XPATH, '/html/body/form/div[3]/section/div/div/div[1]/div/div[1]/div/div')
    num_articles = len(article_elements)

    # Loop through the range of the total number of articles
    for i in range(1, num_articles + 1):
        # Construct the XPath with the changing value of i
        article_xpath = f'/html/body/form/div[3]/section/div/div/div[1]/div/div[1]/div/div[{i}]/div/div[2]/div[1]/div/h6/a'
        
        # Find and extract all the links on the page based on the constructed XPath
        links = driver.find_elements(By.XPATH, article_xpath)
        article_links = [link.get_attribute('href') for link in links]

        # Update the set with the article links
        unique_urls.update(article_links)


options = webdriver.ChromeOptions()
#options.add_argument('--headless')
driver = webdriver.Chrome(options=options)


news = ["https://newsonair.gov.in/National-News.aspx", 
        "https://newsonair.gov.in/State-News.aspx",
        "https://newsonair.gov.in/International-News.aspx",
        "https://newsonair.gov.in/Sports-News.aspx",
        "https://newsonair.gov.in/Business-News.aspx",
        "https://newsonair.gov.in/Miscellaneous-News.aspx"]

# Create a folder for text files if it doesn't exist



for url in news:
    driver.get(url)

    # Initialize the set inside the loop for each new URL
    unique_urls = set()

    for page in range(6):
        # Perform incremental scroll down, scrape articles, and click on next
        scroll_to_bottom(driver, unique_urls)
        next_button = driver.find_element(By.XPATH, '//*[@id="ctl00_ContentPlaceHolder1_AddUserControl_lbNext"]')
        next_button.click()
        time.sleep(2)

        # Scroll to top, incremental scroll down, scrape articles
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(2)
        scroll_to_bottom(driver, unique_urls)
    folder_path = "URLs"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


    # Write unique URLs to a text file for each URL
    file_path = f'newsonair_english_{url.split("/")[-1]}.txt'

        #COncatenate folder path and the file path name
    file_path_in_folder = os.path.join(folder_path,file_path)

    with open(file_path_in_folder, 'w', encoding='utf-8') as file:
        for u_url in unique_urls:
            file.write(u_url + '\n')


# Close the webdriver
driver.quit()
