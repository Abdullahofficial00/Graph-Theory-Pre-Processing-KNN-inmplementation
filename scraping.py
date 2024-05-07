import requests
from bs4 import BeautifulSoup

def scrape(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = '\n'.join([p.text for p in paragraphs])
        return text
    else:
        print(f"Failed to fetch data for Status code: {response.status_code}")
        return None

def scrape_and_save_data():
    urls = []
    num_urls = int(input("Enter the number of URLs you want to scrape: "))
    for i in range(num_urls):
        url = input(f"Enter URL {i+1}: ")
        urls.append(url)

    for i, url in enumerate(urls):
        raw_text = scrape(url)
        if raw_text:
            word_count = len(raw_text.split())
            file_name = f"data_{i}.txt"
            with open(file_name, 'w', encoding='utf-8') as file:
                file.write(raw_text)
                print(f"Scraped data from {url} and saved to {file_name}. Word count: {word_count}")

# Example usage:
scrape_and_save_data()
print("Data saved successfully.")
