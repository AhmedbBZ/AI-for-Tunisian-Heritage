"""
Tunisian Heritage Data Collection Script -- IMPROVED VERSION
============================================================
This script downloads and processes Tunisian heritage texts from open sources.

IMPORTANT: Install required libraries first:
pip install requests beautifulsoup4 PyPDF2 lxml
"""

import requests
from bs4 import BeautifulSoup
import os
import json
import time
from pathlib import Path
import PyPDF2
from urllib.parse import quote
import re

# ==========================================
# CONFIGURATION
# ==========================================

# Create data directory
DATA_DIR = Path("tunisian_heritage_data")
DATA_DIR.mkdir(exist_ok=True)

# Subdirectories
(DATA_DIR / "pdfs").mkdir(exist_ok=True)
(DATA_DIR / "texts").mkdir(exist_ok=True)
(DATA_DIR / "metadata").mkdir(exist_ok=True)
(DATA_DIR / "raw_html").mkdir(exist_ok=True)

# Headers to avoid being blocked
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9'
}

# Project Gutenberg - Public domain books
GUTENBERG_BOOKS = [
    {
        "id": "gutenberg_barbary_states",
        "title": "The Barbary States - North Africa",
        "url": "https://www.gutenberg.org/cache/epub/37703/pg37703.txt",
        "metadata": {"type": "historical", "era": "colonial", "language": "EN"}
    },
    {
        "id": "gutenberg_mediterranean",
        "title": "The Mediterranean - Its Role in History",
        "url": "https://www.gutenberg.org/files/63849/63849-0.txt",
        "metadata": {"type": "historical", "era": "ancient_modern", "language": "EN"}
    }
]

# Internet Archive items
ARCHIVE_ORG_TEXTS = [
    "behindcloseddoor0000heja",
    "historyofmodernt0000perk", 
    "tunisiacrossroad0000perk",
    "tunis1920mcgi",
    "tunisianationali0000ande",
]

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def download_text_from_url(url, use_headers=True):
    """Download text content from URL"""
    try:
        headers = HEADERS if use_headers else {}
        response = requests.get(url, timeout=30, headers=headers, allow_redirects=True)
        response.raise_for_status()
        
        # Try to detect encoding
        if response.encoding:
            return response.text
        else:
            return response.content.decode('utf-8', errors='ignore')
    
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:100]}")
        return None


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    print(f"Extracting text from: {pdf_path.name if hasattr(pdf_path, 'name') else pdf_path}")
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            for page_num in range(min(num_pages, 50)):  # First 50 pages only
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
                
                if (page_num + 1) % 10 == 0:
                    print(f"  Processed {page_num + 1}/{num_pages} pages")
        
        print(f"  ✓ Extracted {len(text)} characters")
        return text
    
    except Exception as e:
        print(f"  ✗ Error extracting text: {str(e)}")
        return ""


def save_metadata(book_id, metadata):
    """Save metadata as JSON"""
    metadata_file = DATA_DIR / "metadata" / f"{book_id}.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved metadata: {metadata_file.name}")


def clean_text(text):
    """Basic text cleaning"""
    # Remove extra whitespace
    text = re.sub(r'\n\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def clean_gutenberg_text(text):
    """Remove Gutenberg header and footer"""
    # Remove Gutenberg header
    start_markers = ["*** START OF", "***START OF"]
    for marker in start_markers:
        if marker in text:
            text = text.split(marker, 1)[1]
            break
    
    # Remove Gutenberg footer
    end_markers = ["*** END OF", "***END OF"]
    for marker in end_markers:
        if marker in text:
            text = text.split(marker, 1)[0]
            break
    
    return text.strip()


# ==========================================
# DATA COLLECTION FUNCTIONS
# ==========================================

def download_from_gutenberg():
    """Download books from Project Gutenberg"""
    print("\n" + "="*50)
    print("DOWNLOADING FROM PROJECT GUTENBERG")
    print("="*50 + "\n")
    
    for book in GUTENBERG_BOOKS:
        print(f"--- {book['title']} ---")
        
        text_path = DATA_DIR / "texts" / f"{book['id']}.txt"
        
        if text_path.exists():
            print(f"  ✓ Already exists: {text_path.name}")
        else:
            text = download_text_from_url(book['url'], use_headers=True)
            if text:
                # Clean Gutenberg header/footer
                text = clean_gutenberg_text(text)
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"  ✓ Saved: {text_path.name} ({len(text)} chars)")
                
                # Save metadata
                save_metadata(book['id'], book['metadata'])
            
            time.sleep(1)
        print()


def download_from_archive_org_api():
    """Download texts from Internet Archive using their API"""
    print("\n" + "="*50)
    print("DOWNLOADING FROM INTERNET ARCHIVE (API)")
    print("="*50 + "\n")
    
    for item_id in ARCHIVE_ORG_TEXTS:
        print(f"--- {item_id} ---")
        
        text_path = DATA_DIR / "texts" / f"archive_{item_id}.txt"
        
        if text_path.exists():
            print(f"  ✓ Already exists")
            continue
        
        try:
            # Get item metadata
            metadata_url = f"https://archive.org/metadata/{item_id}"
            response = requests.get(metadata_url, headers=HEADERS, timeout=20)
            
            if response.status_code == 200:
                data = response.json()
                
                # Find text file
                text_file = None
                if 'files' in data:
                    for f in data['files']:
                        name = f.get('name', '')
                        if name.endswith('_djvu.txt') or name.endswith('.txt'):
                            text_file = name
                            break
                
                if text_file:
                    text_url = f"https://archive.org/download/{item_id}/{text_file}"
                    text = download_text_from_url(text_url)
                    
                    if text and len(text) > 100:
                        with open(text_path, 'w', encoding='utf-8') as f:
                            f.write(clean_text(text))
                        print(f"  ✓ Downloaded text file ({len(text)} chars)")
                        
                        save_metadata(f"archive_{item_id}", {
                            "source": "Internet Archive",
                            "item_id": item_id,
                            "type": "historical",
                            "language": "EN"
                        })
                    else:
                        print(f"  ⚠ Text file too short or empty")
                else:
                    print(f"  ⚠ No text file found in item")
            else:
                print(f"  ✗ API returned {response.status_code}")
        
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:80]}")
        
        time.sleep(2)
    
    print()


def scrape_wikipedia_tunisia():
    """Scrape Wikipedia pages using API"""
    print("\n" + "="*50)
    print("DOWNLOADING FROM WIKIPEDIA")
    print("="*50 + "\n")
    
    api_url = "https://en.wikipedia.org/w/api.php"
    
    wikipedia_pages = [
        "Tunisian_independence",
        "History_of_Tunisia",
        "Culture_of_Tunisia",
        "French_protectorate_of_Tunisia",
        "Tunisian_revolution",
        "Habib_Bourguiba",
        "Carthage",
        "Berbers"
    ]
    
    for page_name in wikipedia_pages:
        print(f"Fetching: {page_name}")
        
        text_file = DATA_DIR / "texts" / f"wikipedia_{page_name}.txt"
        
        if text_file.exists():
            print(f"  ✓ Already exists")
            continue
        
        try:
            params = {
                'action': 'query',
                'prop': 'extracts',
                'explaintext': True,
                'titles': page_name.replace('_', ' '),
                'format': 'json',
                'formatversion': '2'
            }
            
            response = requests.get(api_url, params=params, headers=HEADERS, timeout=20)
            data = response.json()
            
            if 'query' in data and 'pages' in data['query']:
                page = data['query']['pages'][0]
                text = page.get('extract', '')
                
                if text and len(text) > 500:
                    with open(text_file, 'w', encoding='utf-8') as f:
                        f.write(f"Title: {page.get('title', page_name)}\n\n{text}")
                    
                    print(f"  ✓ Saved ({len(text)} chars)")
                    
                    metadata = {
                        "source": "Wikipedia",
                        "page": page_name,
                        "type": "encyclopedia",
                        "language": "EN",
                        "location": "Tunisia"
                    }
                    save_metadata(f"wikipedia_{page_name}", metadata)
                else:
                    print(f"  ⚠ Text too short or empty")
            else:
                print(f"  ✗ No pages found in response")
        
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:80]}")
        
        time.sleep(1)
    
    print()


def create_sample_arabic_stories():
    """Create sample Arabic & French stories"""
    print("\n" + "="*50)
    print("CREATING SAMPLE ARABIC & FRENCH STORIES")
    print("="*50 + "\n")
    
    sample_stories = [
        {
            "id": "sample_resistance_sousse",
            "text": """قصة المقاومة في سوسة - ١٩٥٢

في عام ١٩٥٢، قاد محمد الزواري، الشهيد البطل، مجموعة من المقاومين في مدينة سوسة ضد الاستعمار الفرنسي. كان الزواري رجلاً شجاعاً من أبناء المدينة، عمل في البداية كتاجر بسيط لكنه رفض الخضوع للظلم.

نظم الزواري ورفاقه هجمات على المراكز الفرنسية، واستخدموا معرفتهم بأزقة المدينة القديمة للهروب من القوات الاستعمارية. كانت النساء يساعدن المقاومين بإخفاء الأسلحة ونقل الرسائل.

في إحدى الليالي، حاصرت القوات الفرنسية منزل الزواري. قاتل ببسالة حتى آخر رمق، ورفض الاستسلام. استشهد في تلك الليلة، لكن قصته بقيت حية في ذاكرة أهل سوسة.

يروي كبار السن أن روح الزواري ظلت تلهم الشباب حتى نالت تونس استقلالها في ١٩٥٦.
""",
            "metadata": {
                "type": "resistance_story",
                "era": "colonial_1952",
                "language": "AR",
                "location": "Sousse",
                "story_type": "martyr_legend",
                "source": "oral_tradition"
            }
        },
        {
            "id": "sample_french_resistance_1943",
            "text": """La Résistance Tunisienne - 1943

Récit de résistance à Bizerte

Pendant la Seconde Guerre mondiale, quand les forces allemandes occupèrent la Tunisie en 1942-1943, de nombreux Tunisiens rejoignirent la résistance contre l'occupation.

Ali Ben Salem, un pêcheur de Bizerte, utilisait son bateau pour aider les résistants français et tunisiens. Il transportait des messages, des armes et parfois des personnes recherchées par les Allemands.

Une nuit de février 1943, Ali reçut une mission dangereuse : aider trois pilotes britanniques dont l'avion s'était écrasé près de la côte. Malgré les patrouilles allemandes, Ali navigua dans l'obscurité et sauva les trois hommes.

Les Allemands soupçonnèrent Ali et fouillèrent son bateau plusieurs fois, mais ne trouvèrent jamais de preuves. Après la libération de la Tunisie en mai 1943, Ali continua sa vie de pêcheur, rarement parlant de ses actes héroïques.

Ses petits-enfants racontent aujourd'hui comment leur grand-père aidait tout le monde, sans distinction de nationalité ou de religion.
""",
            "metadata": {
                "type": "resistance_story",
                "era": "wwii_1943",
                "language": "FR",
                "location": "Bizerte",
                "story_type": "war_hero",
                "source": "oral_tradition"
            }
        }
    ]
    
    for story in sample_stories:
        text_file = DATA_DIR / "texts" / f"{story['id']}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(story['text'])
        
        save_metadata(story['id'], story['metadata'])
        print(f"  ✓ Created: {story['id']}")
    
    print()


def process_local_pdfs():
    """Process any PDFs manually placed in the pdfs folder"""
    print("\n" + "="*50)
    print("PROCESSING LOCAL PDFs")
    print("="*50 + "\n")
    
    pdf_files = list((DATA_DIR / "pdfs").glob("*.pdf"))
    
    if not pdf_files:
        print("  ℹ No PDF files found in pdfs folder")
        print()
        return
    
    for pdf_path in pdf_files:
        text_path = DATA_DIR / "texts" / f"{pdf_path.stem}.txt"
        
        if text_path.exists():
            print(f"  ✓ Already processed: {pdf_path.name}")
            continue
        
        print(f"Processing: {pdf_path.name}")
        text = extract_text_from_pdf(pdf_path)
        
        if text and len(text) > 100:
            cleaned_text = clean_text(text)
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            print(f"  ✓ Extracted text ({len(cleaned_text)} chars)")
            
            save_metadata(pdf_path.stem, {
                "source": "local_pdf",
                "filename": pdf_path.name,
                "type": "document",
                "language": "unknown"
            })
        else:
            print(f"  ⚠ Could not extract text or file too short")
    
    print()


def generate_dataset_summary():
    """Generate a summary of collected data"""
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50 + "\n")
    
    texts = list((DATA_DIR / "texts").glob("*.txt"))
    num_texts = len(texts)
    
    if num_texts > 0:
        total_chars = sum(len(open(f, 'r', encoding='utf-8').read()) for f in texts)
        print(f"📄 Total texts: {num_texts}")
        print(f"💾 Total characters: {total_chars:,}")
        print(f"\n📚 Downloaded texts:")
        for text_file in sorted(texts):
            size_kb = text_file.stat().st_size / 1024
            print(f"  • {text_file.name} ({size_kb:.1f} KB)")
    else:
        print("⚠ No texts downloaded yet")
    
    print()


def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("  TUNISIAN HERITAGE DATA COLLECTION SCRIPT v2.0")
    print("="*60)
    
    print("\nThis script will download from:")
    print("  • ✅ Project Gutenberg (public domain books)")
    print("  • ✅ Wikipedia API (encyclopedia articles)")
    print("  • ✅ Internet Archive API (digital library)")
    print("  • ✅ Sample stories (Arabic & French)")
    print("\nData will be saved to:", DATA_DIR.absolute())
    print()
    
    input("Press Enter to start downloading...")
    
    # Run all collection functions
    download_from_gutenberg()
    scrape_wikipedia_tunisia()
    download_from_archive_org_api()
    create_sample_arabic_stories()
    process_local_pdfs()
    
    # Generate summary
    generate_dataset_summary()
    
    print("\n" + "="*60)
    print("  ✓ DATA COLLECTION COMPLETE!")
    print("="*60)
    print(f"\nAll data saved to: {DATA_DIR.absolute()}")
    print("\nNext steps:")
    print("  1. Review downloaded texts in 'texts' folder")
    print("  2. Check metadata in 'metadata' folder") 
    print("  3. Use these texts for your RAG system ingestion")
    print()
    
    # Show statistics
    texts = list((DATA_DIR / "texts").glob("*.txt"))
    if texts:
        total_chars = sum(len(open(f, 'r', encoding='utf-8').read()) for f in texts)
        print(f"📊 Downloaded {len(texts)} texts with {total_chars:,} total characters")
    print()


if __name__ == "__main__":
    main()
