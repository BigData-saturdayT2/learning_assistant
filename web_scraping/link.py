import requests
from bs4 import BeautifulSoup
import re
import time

# Keywords to filter data-related links
TECH_KEYWORDS = [
    'data', 'data-science', 'machine-learning', 'artificial-intelligence', 'deep-learning',
    'python', 'statistics', 'data-analytics', 'data-engineering', 'data-pipelines',
    'data-processing', 'data-visualization', 'data-wrangling', 'data-mining',
    'data-management', 'data-governance', 'data-security', 'data-modeling',
    'data-quality', 'data-transformation', 'data-lakes', 'data-warehousing',
    'predictive-modeling', 'classification', 'regression', 'clustering',
    'dimensionality-reduction', 'time-series-analysis', 'database', 'sql', 'etl'
]

BASE_URL = "https://www.geeksforgeeks.org/"

# Function to collect links related to data topics from GeeksforGeeks
def scrape_tech_links(base_url, keywords, max_links=1000):
    links = set()
    pages_to_scrape = [base_url]
    visited_pages = set()
    try:
        while pages_to_scrape and len(links) < max_links:
            current_url = pages_to_scrape.pop(0)
            if current_url in visited_pages:
                continue
            visited_pages.add(current_url)

            response = requests.get(current_url)
            if response.status_code != 200:
                print(f"Unable to fetch page: {current_url}, Status code: {response.status_code}")
                continue
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                title = a_tag.get_text(strip=True).lower()
                # Add links that match the keywords either in the URL or title
                if any(keyword in href.lower() for keyword in keywords) or any(keyword in title for keyword in keywords):
                    full_link = href if href.startswith('http') else base_url + href
                    if full_link not in links:
                        links.add(full_link)
                        # Explore internal links further
                        if base_url in full_link and full_link not in visited_pages and full_link not in pages_to_scrape:
                            pages_to_scrape.append(full_link)
                
                if len(links) >= max_links:
                    break
            time.sleep(1)
    
    except requests.RequestException as e:
        print(f"Error occurred: {e}")
    
    return links

if __name__ == "__main__":
    tech_links = scrape_tech_links(BASE_URL, TECH_KEYWORDS)
    
    # Save the collected links to a file
    with open("links.txt", "w") as file:
        file.write(f"Total Links Collected: {len(tech_links)}\n")
        for link in tech_links:
            file.write(f"{link}\n")
    
    print(f"Total Links Collected: {len(tech_links)}")
    for link in tech_links:
        print(link)
