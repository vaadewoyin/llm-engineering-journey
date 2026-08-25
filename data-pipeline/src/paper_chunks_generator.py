"""
Generates chunks for QA pair generation 

Downloads papers from semantic scholar, saves papers and metadata to disk,
extract chunks from each relevant sections per paper for QA pair generation
"""

from datetime import datetime
import json
import os
from pathlib import Path
import re
import time
import requests
from dotenv import load_dotenv
from requests.exceptions import ConnectionError, HTTPError, SSLError, Timeout
from semanticscholar import SemanticScholar
import html
import tiktoken
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

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

def create_file_name(result):
    """Create short pdf file"""
    id = result["paperId"]
    title = result["title"]
    year = result["year"]
    title_mod= '_'.join(title.split()[:8])
    return f"{id[:5]}_{title_mod}_{year}"

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

def docling_paper_converter():
    # Configure PDF pipeline
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.do_ocr = False
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = False

    # Create converter
    converter = DocumentConverter(format_options={InputFormat.PDF:
                                                  PdfFormatOption(pipeline_options=pipeline_options)})
    
    return converter

def convert_paper_to_markdown(converter, paper_path):
    # Convert PDF
    result = converter.convert(paper_path)
    md_text = result.document.export_to_markdown()
    return md_text


# clean markdown file
def clean_markdown(text):
    """Clean docling extracted markdown"""
    text = re.sub(r'<!-- image -->', '', text)
    text = re.sub(r'<!-- formula-not-decoded -->', '', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    text = re.sub(r' +', ' ', text)
    text = html.unescape(text)
    return text

# find headers
def get_headers(text):
    """
    Find all Markdown headers in the text.
    Returns a list of (title, position, level).
    """
    pattern = r'^(#{1,6})\s+(.+?)$'
    headers = []

    # Find all matches
    for match in re.finditer(pattern, text, re.MULTILINE):
        level = len(match.group(1))  # Number of # symbols
        title = match.group(2).strip()  # The title text
        position = match.start()  # Where it starts in the text

        headers.append((title, position, level))

    return headers


def find_sections(headers, section):
    """Find sections using Introduction as an anchor."""
    section = section.lower()
    
    SECTION_PATTERNS = {
        "methodology": ["method", "methodology", "materials and methods", "materials & methods",
                       "experimental methods", "experimental procedure", "materials"],
        "results_discussion": ["results", "discussion", "result and discussion",
                              "results and discussion", "results & discussion", 
                              "results and analysis", "results & analysis"],
    }
    
    # Handle conclusion
    if section == "conclusion":
        for text, pos, level in headers:
            if "conclusion" in text.lower():
                return text, pos, level
        return None
    
    # Find Introduction position
    intro_pos = None
    for idx, (text, pos, level) in enumerate(headers):
        if "introduction" in text.lower():
            intro_pos = idx
            break
    
    # If Introduction found, start searching after it
    if intro_pos is not None:
        start_idx = intro_pos + 1
    else:
        start_idx = 0  # Fallback: start from beginning
    
    # Search only the relevant headers
    if section == "methodology":
        patterns = SECTION_PATTERNS["methodology"]
    elif section == "results":
        patterns = SECTION_PATTERNS["results_discussion"]
    else:
        return None
    
    # Search from start_idx to end
    for text, pos, level in headers[start_idx:]:
        text_lower = text.lower()
        for keyword in patterns:
            if keyword in text_lower:
                return text, pos, level
    
    return None

def align_to_paragraph(text, start_pos, end_pos=None):
    """Align start and end positions to paragraph boundaries."""
    # Align Start 
    if start_pos <= 0:
        aligned_start = 0
    else:
        # Search backwards for double newline
        search_start = max(0, start_pos - 5000)
        last_break = text.rfind('\n\n', search_start, start_pos)
        
        if last_break != -1:
            aligned_start = last_break + 2  # Skip the \n\n
        else:
            # Fallback: search for single newline
            last_break = text.rfind('\n', search_start, start_pos)
            if last_break != -1:
                aligned_start = last_break + 1
            else:
                aligned_start = 0
    
    # Align End
    if end_pos is None or end_pos >= len(text):
        aligned_end = len(text)
    else:
        # Search forward for double newline
        next_break = text.find('\n\n', end_pos)
        
        if next_break != -1:
            aligned_end = next_break
        else:
            # Fallback: search for single newline
            next_break = text.find('\n', end_pos)
            if next_break != -1:
                aligned_end = next_break
            else:
                aligned_end = len(text)
    
    return aligned_start, aligned_end

def extract_fallback(md_text, headers, conclusion_pos):
    """
    Fallback for extraction:
    - If conclusion exists: take from halfway through headers before Conclusion.
    - If conclusion doesn't exist: take ~15,000 chars from middle of paper.
    
    Always aligns to paragraph boundaries.
    """
    if not conclusion_pos:
        # No conclusion: take the middle 15,000 characters
        mid = len(md_text) // 2
        start, end = align_to_paragraph(md_text, mid, min(mid + 15000, len(md_text)))
        return md_text[start:end]

    conclusion_position = conclusion_pos[1]

    # Get headers before conclusion
    headers_before = []
    for title, pos, level in headers:
        if pos < conclusion_position:
            headers_before.append((title, pos, level))
        else:
            break

    num_headers = len(headers_before)

    if num_headers == 0:
        # No headers before conclusion: take last 10,000 chars before conclusion
        start = max(0, conclusion_position - 10000)
        start, end = align_to_paragraph(md_text, start, conclusion_position)
        return md_text[start:end]

    # Take from halfway point (header index)
    half_index = num_headers // 2
    start_pos = headers_before[half_index][1]
    
    # Align to paragraph boundaries
    start, end = align_to_paragraph(md_text, start_pos, conclusion_position)
    
    return md_text[start:end]

def extract_section(md_text, headers, section_to_extract):
    """
    Extract a specific section from the Markdown text.
    Returns the section text, or None if not found.
    """
    # Find section positions
    methodology_result = find_sections(headers, "methodology")
    results_discussion_result = find_sections(headers, "results_discussion")
    conclusion_result = find_sections(headers, "conclusion")
    
    # Unpack results 
    methodology_pos = methodology_result[1] if methodology_result else None
    results_discussion_pos = results_discussion_result[1] if results_discussion_result else None
    conclusion_pos = conclusion_result[1] if conclusion_result else None
    
    if section_to_extract == "methodology" and methodology_pos is not None:
        end_pos = results_discussion_pos if results_discussion_pos else len(md_text)
        return md_text[methodology_pos:end_pos]
    
    elif section_to_extract == "results_discussion" and results_discussion_pos is not None:
        end_pos = conclusion_pos if conclusion_pos else len(md_text)
        return md_text[results_discussion_pos:end_pos]
    
    elif section_to_extract == "conclusion" and conclusion_pos is not None:
        return md_text[conclusion_pos:]
    
    # --- Fallback: If section not found, try the "halfway to conclusion" approach ---
    elif section_to_extract in ["methodology", "results_discussion", "conclusion"]:
        return extract_fallback(md_text, headers, conclusion_result)
    
    else:
        return None




# pipeline
tokenizer = tiktoken.encoding_for_model("gpt-4")