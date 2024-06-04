import time
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup

import argparse


id2month = {
  1: "January",
  2: "February",
  3: "March",
  4: "April",
  5: "May",
  6: "June",
  7: "July",
  8: "August",
  9: "September",
  10: "October",
  11: "November",
  12: "December"
}


def get_year_links(base_url):
    """
    Get all year links from the base URL.
    """
    year_links = []
    page = requests.get(base_url)
    soup = BeautifulSoup(page.content, 'html.parser')

    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('/browse/supremecourt/') and len(href.strip('/').split('/')) == 3:
            year_links.append(base_url + href.strip().split('/')[-2])
    return year_links


def get_month_links(year_url):
    """
    Get all month links from the given year URL.
    """
    month_links = []
    page = requests.get(year_url)
    soup = BeautifulSoup(page.content, 'html.parser')

    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('/search/?formInput=doctypes:supremecourt') and 'fromdate' in href:
            month_links.append('https://indiankanoon.org' + href)
    return month_links


def get_judgement_links(month_url):
    """
    Get all judgement links from the given month URL.
    """
    judgement_links = []
    
    page = requests.get(month_url)
    soup = BeautifulSoup(page.content, 'html.parser')
    
    total_judgements = get_total_judgements(soup)
    total_pages = 1 if total_judgements<=10 else ((total_judgements // 10) + 1)

    if total_judgements==0:
        return judgement_links
    
    for page_num in range(total_pages):
        curr_page_url = month_url + f"&pagenum={page_num}"
        curr_page = requests.get(curr_page_url)
        curr_soup = BeautifulSoup(curr_page.content, 'html.parser')

        links_on_page = curr_soup.find_all('a', href=True)
        
        if not links_on_page:  
            break

        for link in links_on_page:
            href = link['href']
            if href.startswith('/doc/'):
                judgement_link = 'https://indiankanoon.org' + href
                judgement_links.append(judgement_link) 
    return judgement_links


def get_total_judgements(soup):
    """
    Extract the total number of judgements from the HTML soup.
    """
    div = soup.find('div', class_='results_middle')
    if div:
        b_tag = div.find('b')
        if b_tag:
            text = b_tag.text
            parts = text.split(' ')
            if len(parts) > 2 and parts[-1].isdigit():
                return int(parts[-1])
    return 0


def get_clean_text(text):
    """
    Basic preprocessing to remove `\t`, `\n` and extra white-spaces. 
    """
    text = text.strip()
    text = text.replace("\t", " ")
    text = re.sub(r'\s+', ' ', text)
    text = text.replace("\n", " ")
    return text

def get_judgement_text(judgement_url):
    """
    Get the full text of a judgement from the given URL.
    """

    judgement_details = {}

    print_view_url = judgement_url + "?type=print"
    page = requests.get(print_view_url)
    soup = BeautifulSoup(page.content, 'html.parser')
    
    # Find Issue element (can be multiple)
    div = soup.find('div', class_='judgments')

    if div:
        # Doc Title:
        doc_title = div.find('h2', class_='doc_title')
        if doc_title:
            doc_title_text = doc_title.text
            judgement_details['Title'] = doc_title_text
        
        # Bench:
        doc_bench = div.find('h3', class_='doc_bench')
        if doc_bench:
            doc_bench_a_tags = doc_bench.find_all('a')
            doc_bench_names = ", ".join([doc_bench_a_tag.get_text(strip=True) for doc_bench_a_tag in doc_bench_a_tags])
            judgement_details['Bench'] = doc_bench_names

        # Issue:
        issue_element = div.find_all('p', {'data-structure': 'Issue'})
        if issue_element:
            issue_text = "\n".join([element.get_text(strip=True) for element in issue_element])
            judgement_details['Issue'] = get_clean_text(issue_text)

        # Facts:
        facts_elements = div.find_all('p', {'data-structure': 'Facts'})
        if facts_elements:
            facts_text = "\n".join([element.get_text() for element in facts_elements])
            judgement_details['Facts'] = get_clean_text(facts_text)  

        # Precendent:
        precedent_elements = div.find_all('p', {'data-structure': 'Precedent'})
        if precedent_elements:                                          
            precedent_text = "\n".join([element.get_text() for element in precedent_elements])
            judgement_details['Precedent'] = get_clean_text(precedent_text)


    return judgement_details


def main(base_url, num_rows):
  
    year_links = get_year_links(base_url)

    all_judgements=[]

    counter = 0 
    
    for i, year_link in enumerate(year_links):
        print(f"Year {i+1}/{len(year_links)}: {year_link}")
        month_links = get_month_links(year_link)[1:]

        
        for j, month_link in enumerate(month_links):
            print(f"Month {j+1}/{len(month_links)}: {month_link}")
            judgement_links = get_judgement_links(month_link)


            for k, judgement_link in enumerate(judgement_links):
                print(f"Crawling {k+1}/{len(judgement_links)}: {judgement_link}")
                judgement_details = get_judgement_text(judgement_link)

                # Make a dict for each judgment
                judgement = dict()
                judgement['Year']   =   f"{1950+i}"
                judgement['Month']  =   id2month[j+1]
                judgement['ID']     =   judgement_link.split("/")[-2]
                judgement.update(judgement_details)

                all_judgements.append(judgement)

                counter +=1
                time.sleep(1)

                if counter > num_rows:
                    break
                
            if counter > num_rows:
                break
        
        if counter > num_rows:
            print("[INFO] Reached maximum rows. Exiting.")
            break
        
        
    
    df = pd.DataFrame.from_dict(all_judgements)
    print(df.head(10))
    df.to_excel("sc_judgement_data.xlsx")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process base URL input")
    parser.add_argument("base_url", type=str, help="The base URL")
    parser.add_argument("num_rows", type=int, help="Number of rows/judgements to scrap")        # Quick Testing
    args = parser.parse_args()
    main(args.base_url, args.num_rows)


