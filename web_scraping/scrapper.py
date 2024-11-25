import requests
from bs4 import BeautifulSoup

page_url = "https://www.geeksforgeeks.org/introduction-deep-learning/"

resp = requests.get(page_url)

if resp.status_code == 200:
    # Prase the HTML content of the page
    soup = BeautifulSoup(resp.text, 'html.parser')

    # Extract the article tittle
    articel_title = soup.find('div', class_='article-title').get_text(strip=True)

    # Extract the main cotent
    main_div = soup.find('div', class_='text')
    
    # Extract text from paragraphs, spnas, and blockquotes
    pragraphs = [p.get_text(strip=True) for p in main_div.find_all('p')]
    spnas = [span.get_text(strip=True) for span in main_div.find_all('span')]
    blckquotes = [blockquote.get_text(strip=True) for blockquote in main_div.find_all('blockquote')]

    ordered_lst_items = []
    unordered_lst_items = []

    ordered_lsts = main_div.find_all('ol')  
    unordered_lsts = main_div.find_all('ul')  

    # Extract ordered list itmes
    for ol in ordered_lsts:
        for li in ol.find_all('li'):
            ordered_lst_items.append(li.get_text(strip=True))
    
    # Extract unordered list itmes
    for ul in unordered_lsts:
        for li in ul.find_all('li'):
            unordered_lst_items.append(li.get_text(strip=True))

    # Extract subhedings
    subhedings = [h2.get_text(strip=True) for h2 in main_div.find_all('h2')]

    # Write scraped data to a txt file
    with open("scraped_data.txt", "w", encoding="utf-8") as f:
        f.write(f"Tittle: {articel_title}\n\n")

        if subhedings:
            f.write("Subhedings:\n")
            for subheading in subhedings:
                f.write(f"- {subheading}\n")
            f.write("\n")

        if pragraphs:
            f.write("Pragraphs:\n")
            for paragraph in pragraphs:
                f.write(paragraph + "\n")
            f.write("\n")

        if spnas:
            f.write("Spnas:\n")
            for span in spnas:
                f.write(span + "\n")
            f.write("\n")

        if blckquotes:
            f.write("Blockquotes:\n")
            for blockquote in blckquotes:
                f.write(blockquote + "\n")
            f.write("\n")

        if ordered_lst_items:
            f.write("Ordered List Itmes:\n")
            for i, item in enumerate(ordered_lst_items, 1):
                f.write(f"{i}. {item}\n")
            f.write("\n")

        if unordered_lst_items:
            f.write("Unordered List Itmes:\n")
            for item in unordered_lst_items:
                f.write(f"- {item}\n")
            f.write("\n")

    print("Scarped data saved to 'scraped_data.txt'")
else:
    print(f"Failed to retrive the page. Status code: {resp.status_code}")