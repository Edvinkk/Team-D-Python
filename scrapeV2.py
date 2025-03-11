from bs4 import BeautifulSoup #for parsing html content 
import requests #to make http requests to fetch web pages
import os

#list of urls to scrape
urls = [
    'https://en.wikipedia.org/wiki/MDMA',
    'https://en.wikipedia.org/wiki/Cocaine',
    'https://en.wikipedia.org/wiki/Ketamine',
    'https://en.wikipedia.org/wiki/Cannabis_(drug)',
    'https://en.wikipedia.org/wiki/Nitrous_oxide',
    'https://en.wikipedia.org/wiki/Heroin',
    'https://en.wikipedia.org/wiki/Methamphetamine',
    'https://en.wikipedia.org/wiki/Fentanyl',
    'https://en.wikipedia.org/wiki/LSD',
    'https://en.wikipedia.org/wiki/Benzodiazepine',
    'https://en.wikipedia.org/wiki/Psilocybin_mushroom',
    'https://en.wikipedia.org/wiki/Phencyclidine',
    'https://en.wikipedia.org/wiki/Opioid',
    'https://en.wikipedia.org/wiki/Lean_(drug)' 
      
    ]

# get current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# set the file to ssave in current directory
outputFile = os.path.join(current_directory, "scrapedData.txt")


#opens the file in write mode overwriting any existing content and using utf-8 encoding to handle special characters
with open(outputFile, 'w', encoding='utf-8') as file:
    
    #loops through the urls in the urls list   
    for url in urls:
        
        #writes the url to the file reference
        file.write(f"Scraped {url}\n\n")
        
        #make a http request to fetch the page content
        response = requests.get(url)
        
        #checks if the http request was succesful 
        if response.status_code == 200:
            
            #parses the html content using BeautifulSoup with html.parser
            soup = BeautifulSoup(response.text, 'html.parser')
            
            #extracts all paragraph tags from the html content and write to the file being seperated by a new line
            for paragraph in soup.find_all('p'):
                file.write(paragraph.get_text())
                file.write('\n')
            #writes a line of dashes (80) to seperate the content from different urls
            file.write("\n" + "-"*80 + "\n")
        else:
            #if the page couldnt be fetched print an error message
            errorMsg = f"Failed to fetch {url}"
            print(errorMsg)
          
#after scraping is completed it prints a msg to the console            
print(f"scraped data saved to {outputFile}")


