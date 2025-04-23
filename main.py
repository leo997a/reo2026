from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import json

app = FastAPI()

# للسماح بالوصول من Streamlit Cloud
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/extract")
def extract_match_data(url: str = Query(...)):
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
    try:
        driver.get(url)
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        script_tag = soup.find(lambda tag: tag.name == 'script' and 'matchCentreData' in tag.text)
        if not script_tag:
            return {"error": "matchCentreData not found"}
        raw = script_tag.text
        match_json = raw.split("matchCentreData: ")[1].split(",\n")[0]
        return json.loads(match_json)
    except Exception as e:
        return {"error": str(e)}
    finally:
        driver.quit()
