"""
Generates chunks for QA pair generation 

Downloads papers from semantic scholar, saves papers and metadata to disk,
extract chunks from each relevant sections per paper for QA pair generation
"""

import json
import time
import requests
import os
import re
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from semanticscholar import SemanticScholar
from requests.exceptions import HTTPError, ConnectionError, Timeout, SSLError
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# Schematic scholar api key
load_dotenv()
S2_API_KEY = os.getenv('S2_API_KEY')

def is_relevant_paper(title, abstract):
    """ Checks if paper is relevant for download."""
    title_text = title.lower()
    abstract_text = (abstract or "").lower()
    full_text = title_text + abstract_text

    # Policy / Survey / Review papers
    policy_signals = [
        "delphi", "questionnaire", "expert survey", "likert scale",
        "semi-structured interview", "focus group", "policy",
        "sustainability assessment", "life cycle assessment",
        "review", "literature review", "state of the art", "meta-analysis"
    ]
    if any(signal in full_text for signal in policy_signals):
        return False

    # 2. REJECT: Asphalt / Bitumen (but NOT binder)
    if any(x in full_text for x in ["asphalt", "asphaltic", "bitumen", "bituminous"]):
        return False

    # 3. KEEP: Must have at least one experimental / replacement keyword
    experimental_keywords = [
        "compressive strength", "tensile strength", "flexural strength",
        "mechanical properties",  "mix design", "mix proportion",
        "mix ratio", "workability", "slump",
        "water-cement","curing", "durability",
        "microstructure", "SEM", "XRD", "TGA",
        "specimen", "testing", "experimental",  "trial mix",
        "sample preparation", "test results",
        "replacement", "replace", "substitution", "substituted",
        "supplementary cementitious",  "cement replacement",
        "aggregate replacement", "alternative material"
    ]
    if not any(kw in full_text for kw in experimental_keywords):
        return False

    # 4. KEEP: Must be about cementitious materials (check full text, broader list)
    core_subject_terms = [
        "concrete", "mortar", "cement paste",
        "cement", "cementitious", "cementitious material",
        "geopolymer", "alkali-activated", "alkali activated",
    ]
    if not any(term in full_text for term in core_subject_terms):
        return False

    return True

def download_s2_papers(api_key, materials_list, paper_count_per_material, save_dir):
    """ Downloads papers from Semantic scholar to disk"""
    # metadata
    metadata = []

    # headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
    }

    # sch object
    sch = SemanticScholar(api_key=api_key)

    # store paper_id to avoid duplicates
    paper_ids = set()

    # loop over papers
    for material in materials_list:

        search_query = f'"{material}" concrete'

        results = sch.search_paper(
            query = search_query,
            publication_types= ['JournalArticle'],
            open_access_pdf=True,
            year="2020-",
            #limit=100
        )

        saved_downloads_count = 0
        result_iter = iter(results)

        while True:
            try:
                result = next(result_iter)
            except StopIteration:
                  break
            except Exception as e:
                print(f" API error mid-pagination for '{search_query}': {e} — retrying in 5s")
                time.sleep(5)
                continue

            try:
                paper_id = result["paperId"]
                oa = result["openAccessPdf"]

                if not oa:
                    continue

                pdf_url = oa["url"]
                license= oa ["license"]
            except Exception:
                continue

            if paper_id not in paper_ids:

                if license == "CCBY" and (is_relevant_paper(result["title"], result["abstract"])):

                  pdf_path = save_dir / f"{paper_id}.pdf"


                  if not pdf_url:
                      continue

                  try:
                      response = requests.get(pdf_url, headers=headers, timeout=60)
                      response.raise_for_status()

                  except (SSLError, ConnectionError):
                      try:
                          response = requests.get(pdf_url, headers=headers, timeout=60, verify=False)
                          response.raise_for_status()
                      except Exception as e:
                          print(f"Skip {paper_id}: retry failed - {e}")
                          continue

                  except (HTTPError, Timeout) as e:
                      print(f"Skip {paper_id}: {e}")
                      continue


                  # obtained respomse, check its pdf
                  try:
                      content_type = response.headers.get("content-type", "").lower()
                      if "pdf" in content_type or response.content[:4] == b"%PDF":
                          with open(pdf_path, "wb") as f:
                              f.write(response.content)
                          print(f"{paper_id}: saved to disk")

                          saved_downloads_count += 1
                          paper_ids.add(paper_id)

                          # Update metadata for every successful downloads
                          metadata.append({

                          # core identifiers
                          "paper_id": paper_id,
                          "doi": result["externalIds"].get("DOI"),
                          "title": result["title"],
                          "authors": [a["name"] for a in result["authors"]],
                          "year": result["year"],
                          "venue": result["venue"],
                          "journalName": result["journal"].get("name"),
                          "publicationDate": result["publicationDate"],
                          "isOpenAccess": result["isOpenAccess"],
                          "url": result["url"],

                          # Impact metrics
                          "citationCount": result["citationCount"],

                          # Abstract text
                          "abstract": result["abstract"],

                          "retrieved_for": material,
                          "time_downloaded": datetime.now().strftime("%B %d, %Y at %I:%M %p")
                          })

                      else:
                        print(f"Skip {paper_id}: not a PDF")

                  except Exception as e:
                      print(f"Skip {paper_id}: unexpected error - {e}")
                      continue

            if saved_downloads_count == paper_count_per_material:
              break

    # save paper metadata
    with open (METADATA_PATH, "w") as f:
        for paper in metadata:
          f.write(json.dumps(paper, ensure_ascii=False) + "\n")

    # Check num of downloaded papers
    downloaded = len(list(PAPER_DIR.glob("*.pdf")))
    print(f"\nDownloaded {downloaded} papers.")
    print(f"Downloaded papers metadata saved to {METADATA_PATH}")

MATERIALS = [

    # Industrial by-products / supplementary cementitious materials
    "fly ash",
    "ground granulated blast furnace slag",
    "silica fume",
    "steel slag",
    "copper slag",
    "metakaolin",
    "calcined clay",
    "volcanic ash",
    "glass powder",

    # Agricultural / biomass ashes
    "rice husk ash",
    "rice straw ash",
    "palm oil fuel ash",
    "oil palm fibre ash",
    "sugarcane bagasse ash",
    "coconut shell ash",
    "palm kernel shell ash",
    "cassava peel ash",
    "corn cob ash",
    "corn stalk ash",
    "groundnut shell ash",
    "bamboo leaf ash",
    "bamboo ash",
    "sawdust ash",
    "wood ash",
    "eggshell powder",
    "plantain peel ash",
    "banana leaf ash",
    "bean pod ash",
    "cocoa pod ash",
    "cocoa shell ash",
    "cotton stalk ash",
    "wheat straw ash",

    # Alternative / waste aggregates
    "palm kernel shell",
    "coconut shell",
    "recycled concrete aggregate",
    "recycled brick aggregate",
    "recycled ceramic aggregate",
    "recycled glass aggregate",
    "waste tyre aggregate",
    "rubber aggregate",
    "recycled plastic aggregate",

    # Alternative natural / waste fibres
    "coconut fibre",
    "coir fibre",
    "palm fibre",
    "sisal fibre",
    "jute fibre",
    "hemp fibre",
    "kenaf fibre",
    "bamboo fibre",
    "sugarcane bagasse fibre",
    "waste textile fibre",

    # Other recycled / waste-derived materials
    "ceramic waste powder",
    "waste ceramic powder",
    "marble dust",
    "granite powder",
    "quarry dust",
    "red mud",
    "paper sludge ash",
    "waste paper ash",
]

# Create necessary file paths
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

PAPER_DIR =  DATA_DIR / "papers"
PAPER_DIR.mkdir(parents=True, exist_ok=True)

METADATA_PATH = PAPER_DIR / "downloaded_paper_metadata.jsonl"

# Download papers
download_s2_papers(api_key = S2_API_KEY, materials_list= MATERIALS,
                   paper_count_per_material =1, save_dir=PAPER_DIR)


def clean_markdown(text):
    """Simple cleaning for docling extracted markdown"""
    text = re.sub(r'<!-- image -->', '', text)
    text = re.sub(r'<!-- formula-not-decoded -->', '', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    text = re.sub(r' +', ' ', text)
    return text