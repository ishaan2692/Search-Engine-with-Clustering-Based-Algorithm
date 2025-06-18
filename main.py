import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import sqlite3
import hashlib
import time
from io import BytesIO
from PIL import Image
import random
import concurrent.futures
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import umap
from datetime import datetime

# Initialize SQLite database with storage management
conn = sqlite3.connect('electronics_products.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS products
             (id TEXT PRIMARY KEY, title TEXT, description TEXT, 
              price REAL, url TEXT, image BLOB, category TEXT, 
              brand TEXT, specs TEXT, relevance_score REAL, 
              last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
conn.commit()

# Electronics sites with high success rates
ELECTRONICS_SITES = [
    "https://www.bestbuy.com/site/laptop-computers/all-laptops/pcmcat138500050001.c",
    "https://www.newegg.com/Desktop-Computers/SubCategory/ID-10",
    "https://www.amazon.com/s?i=computers-intl-ship&bbn=16225007011",
    "https://www.bhphotovideo.com/c/browse/computers/ci/958"
]

# Load pre-trained BERT model for embeddings
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Advanced product schema with multiple fallbacks
PRODUCT_SCHEMA = {
    'title': [
        'h1.product-title', 'h1.product-name', 'h1.title', 
        '[data-test="product-title"]', 'h1', 'span#productTitle',
        'h1.product-name', 'h1.product-title', 'h1.page-title'
    ],
    'description': [
        '.product-description', '.description-content', 
        '#product-overview', '[data-feature-name="productDescription"]',
        '#feature-bullets', '.product-information',
        'div#productDescription', 'div.description', 'div.details'
    ],
    'price': [
        '.price', '.priceView-hero-price', '.price-current', 
        '[data-test="product-price"]', 'span.a-price-whole',
        '.priceView-customer-price', 'span.price-characteristic',
        'span.price-item', 'div.price', 'div.pricing'
    ],
    'image': [
        'img.product-image', 'img.primary-image', 
        '[data-test="product-gallery-image"]', 'img#landingImage',
        '.primary-image', 'img.main-image', 'img#main-image'
    ],
    'specs': [
        '.specifications', '.specs-table', '#product-details',
        '.spec-container', 'div#technicalSpecifications_section',
        'table.specs', 'div.spec-list', 'div.tech-specs'
    ]
}

# Browser-mimicking headers to avoid blocks 
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
]

# Streamlit app
st.set_page_config(page_title="TechSearch Pro", layout="wide")
st.title("🔍 Advanced Electronics Search with Smart Storage")
st.markdown("Sophisticated scraping with intelligent storage management (max 100 products)")

# Functions
def safe_get(soup, selectors):
    for selector in selectors:
        try:
            element = soup.select_one(selector)
            if element:
                if 'img' in selector:
                    return element.get('src', '')
                return element.get_text(strip=True)
        except:
            continue
    return ''

def get_random_headers():
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Referer': 'https://www.google.com/',
        'DNT': '1',
        'Accept-Encoding': 'gzip, deflate, br'
    }

def extract_with_retry(soup, selectors, retries=3):
    """Robust extraction with multiple attempts and selectors"""
    for _ in range(retries):
        result = safe_get(soup, selectors)
        if result:
            return result
        time.sleep(0.5)
    return ''

def crawl_site(url, max_products=100):
    """Advanced crawling with pagination and comprehensive product discovery"""
    base_domain = url.split('//')[-1].split('/')[0]
    product_links = set()
    visited_pages = set()
    to_visit = [url]
    
    while to_visit and len(product_links) < max_products:
        current_url = to_visit.pop(0)
        if current_url in visited_pages:
            continue
        
        try:
            headers = get_random_headers()
            response = requests.get(current_url, headers=headers, timeout=25)
            if response.status_code != 200:
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            visited_pages.add(current_url)
            
            # Find product links using multiple strategies
            link_candidates = set()
            
            # Strategy 1: Direct product links
            for link in soup.find_all('a', href=True):
                href = link['href']
                if not href or href.startswith(('javascript:', 'mailto:', 'tel:')):
                    continue
                    
                full_url = requests.compat.urljoin(current_url, href)
                
                # Match electronics product URL patterns
                if any(pat in full_url for pat in ['/p/', '/product/', '/prodid/', '/item/', '/dp/', '-product-']):
                    if base_domain in full_url and not any(x in full_url for x in ['cart', 'checkout', 'account']):
                        link_candidates.add(full_url)
            
            # Strategy 2: JSON-LD product data
            try:
                json_ld = soup.find('script', type='application/ld+json')
                if json_ld:
                    import json
                    data = json.loads(json_ld.string)
                    if isinstance(data, list):
                        data = data[0]
                    if data.get('@type') == 'Product' and data.get('url'):
                        product_url = data['url']
                        if base_domain in product_url:
                            link_candidates.add(product_url)
            except:
                pass
            
            # Strategy 3: Meta tags
            og_url = soup.find('meta', property='og:url')
            if og_url and og_url.get('content') and base_domain in og_url['content']:
                link_candidates.add(og_url['content'])
            
            # Add valid candidates to product links
            for link in link_candidates:
                if len(product_links) < max_products:
                    product_links.add(link)
            
            # Find next page for pagination
            next_page = None
            # Method 1: Link with rel="next"
            next_link = soup.find('link', rel='next')
            if next_link and next_link.get('href'):
                next_page = requests.compat.urljoin(current_url, next_link['href'])
            
            # Method 2: Common pagination patterns
            if not next_page:
                pagination_selectors = [
                    'a.next', 'a.pagination-next', 'li.next a', 
                    'a:contains("Next")', 'a:contains("›")', 'a:contains(">")'
                ]
                for selector in pagination_selectors:
                    next_btn = soup.select_one(selector)
                    if next_btn and next_btn.get('href'):
                        next_page = requests.compat.urljoin(current_url, next_btn['href'])
                        break
            
            # Add next page to visit
            if next_page and next_page not in visited_pages:
                to_visit.append(next_page)
            
            # Random delay to avoid detection
            time.sleep(random.uniform(1.0, 3.0))
            
        except Exception as e:
            st.warning(f"Error crawling {current_url}: {str(e)}")
    
    return list(product_links)[:max_products]

def extract_specs(soup):
    """Advanced technical specification extraction"""
    specs = {}
    try:
        # Try specification tables
        tables = soup.select('table.specs-table, table.spec-table, table.specs')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['th', 'td'])
                if len(cells) == 2:
                    key = cells[0].get_text(strip=True).replace(':', '')
                    value = cells[1].get_text(strip=True)
                    if key and value:
                        specs[key] = value
        
        # Try specification lists
        spec_lists = soup.select('div.spec-container, div.spec-list, ul.specs')
        for spec_list in spec_lists:
            items = spec_list.find_all(['li', 'div.spec-item'])
            for item in items:
                text = item.get_text(strip=True)
                if ':' in text:
                    key, value = text.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    if key and value:
                        specs[key] = value
        
        # Try JSON-LD specifications
        json_ld = soup.find('script', type='application/ld+json')
        if json_ld:
            try:
                import json
                data = json.loads(json_ld.string)
                if isinstance(data, list):
                    data = data[0]
                if data.get('@type') == 'Product':
                    # Extract from product schema
                    for prop in ['processor', 'ram', 'storage', 'display', 'graphics']:
                        if prop in data:
                            specs[prop.capitalize()] = data[prop]
                    if 'additionalProperty' in data:
                        for prop in data['additionalProperty']:
                            if 'name' in prop and 'value' in prop:
                                specs[prop['name']] = prop['value']
            except:
                pass
    
    except Exception as e:
        st.warning(f"Error extracting specs: {str(e)}")
    
    return str(specs)

def calculate_relevance_score(product):
    """Calculate relevance score based on completeness of data"""
    score = 0
    if product.get('title'): score += 25
    if product.get('description'): score += 20
    if product.get('price') and product['price'] > 0: score += 20
    if product.get('image'): score += 15
    if product.get('specs') and len(product['specs']) > 20: score += 10
    if product.get('category') != "Other": score += 10
    return score

def scrape_product(url):
    try:
        headers = get_random_headers()
        response = requests.get(url, headers=headers, timeout=30)  # Extended timeout
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        product = {'url': url}
        for key, selectors in PRODUCT_SCHEMA.items():
            product[key] = extract_with_retry(soup, selectors)
        
        # Price extraction
        if product['price']:
            price_text = re.sub(r'[^\d.,]', '', product['price'])
            match = re.search(r'(\d{1,3}(?:,\d{3})*\.\d+)|\d+', price_text)
            if match:
                try:
                    product['price'] = float(match.group().replace(',', ''))
                except:
                    product['price'] = 0.0
        
        # Electronics category detection
        category = "Other"
        electronics_categories = {
            "Laptop": ['laptop', 'notebook', 'ultrabook', 'chromebook'],
            "Smartphone": ['phone', 'iphone', 'android', 'smartphone', 'mobile'],
            "Tablet": ['tablet', 'ipad', 'android tablet'],
            "Camera": ['camera', 'dslr', 'mirrorless', 'camcorder'],
            "TV": ['tv', 'television', 'oled', 'qled', 'smart tv'],
            "Audio": ['headphone', 'earbud', 'speaker', 'soundbar', 'audio'],
            "Component": ['cpu', 'gpu', 'ram', 'ssd', 'motherboard', 'hard drive']
        }
        for cat, terms in electronics_categories.items():
            if any(t in url.lower() or (product.get('title') and t in product['title'].lower()) for t in terms):
                category = cat
                break
        product['category'] = category
        
        # Brand detection
        brand = "Unknown"
        common_brands = ['apple', 'samsung', 'dell', 'hp', 'lenovo', 'sony', 
                         'lg', 'asus', 'acer', 'msi', 'canon', 'nvidia', 'intel']
        for b in common_brands:
            if b in url.lower() or (product.get('title') and b in product['title'].lower()):
                brand = b.capitalize()
                break
        product['brand'] = brand
        
        # Extract technical specifications
        product['specs'] = extract_specs(soup)
        
        # Calculate relevance score
        product['relevance_score'] = calculate_relevance_score(product)
        
        # Image handling
        img_data = b''
        if product['image'] and product['image'].startswith('http'):
            try:
                img_response = requests.get(product['image'], headers=headers, timeout=20)
                if img_response.status_code == 200:
                    img_data = img_response.content
            except:
                pass
        product['image'] = img_data
        
        return product
    
    except Exception as e:
        st.warning(f"Error scraping {url}: {str(e)}")
        return None

def manage_storage(conn, new_product_ids):
    """Advanced storage management to maintain 100 highest-quality products"""
    c = conn.cursor()
    
    # Get current count
    c.execute("SELECT COUNT(*) FROM products")
    current_count = c.fetchone()[0]
    
    # If under limit, just add new products
    if current_count + len(new_product_ids) <= 100:
        return
    
    # Get all products with relevance scores
    c.execute("SELECT id, relevance_score, last_updated FROM products")
    all_products = c.fetchall()
    
    # Create dataframe for sorting
    df = pd.DataFrame(all_products, columns=['id', 'relevance_score', 'last_updated'])
    
    # Calculate retention score (70% relevance + 30% recency)
    df['recency_score'] = (pd.to_datetime(df['last_updated']) - pd.Timestamp('1970-01-01')).dt.total_seconds()
    max_recency = df['recency_score'].max()
    min_recency = df['recency_score'].min()
    
    if max_recency > min_recency:
        df['recency_normalized'] = (df['recency_score'] - min_recency) / (max_recency - min_recency)
    else:
        df['recency_normalized'] = 1.0
    
    df['retention_score'] = 0.7 * df['relevance_score'] + 0.3 * df['recency_normalized'] * 100
    
    # Sort by retention score (descending)
    df = df.sort_values('retention_score', ascending=False)
    
    # Determine which products to keep
    keep_ids = df.head(100)['id'].tolist()
    
    # Delete products not in the keep list
    placeholders = ','.join(['?'] * len(keep_ids))
    c.execute(f"DELETE FROM products WHERE id NOT IN ({placeholders})", keep_ids)
    conn.commit()

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = np.array([])
if 'cluster_centers' not in st.session_state:
    st.session_state.cluster_centers = []

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    
    if st.button("🔄 Refresh Database", key="refresh_db"):
        new_product_ids = []
        all_products = []
        for site in ELECTRONICS_SITES:
            with st.spinner(f"Crawling {site.split('//')[-1].split('/')[0]}..."):
                try:
                    product_links = crawl_site(site, max_products=25)
                    st.info(f"Found {len(product_links)} product links")
                    
                    if not product_links:
                        continue
                    
                    progress_bar = st.progress(0)
                    scraped_count = 0
                    
                    # Use threading for faster scraping
                    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                        futures = {executor.submit(scrape_product, url): url for url in product_links}
                        
                        for i, future in enumerate(concurrent.futures.as_completed(futures)):
                            try:
                                product = future.result()
                                if product:
                                    # Create unique ID
                                    url_hash = hashlib.sha256(product['url'].encode()).hexdigest()
                                    
                                    # Save to DB
                                    c.execute('''INSERT OR REPLACE INTO products 
                                              (id, title, description, price, url, image, category, brand, specs, relevance_score) 
                                              VALUES (?,?,?,?,?,?,?,?,?,?)''',
                                              (url_hash, product.get('title', ''), product.get('description', ''),
                                              product.get('price', 0), product['url'], product.get('image', b''), 
                                              product.get('category', 'Other'), product.get('brand', 'Unknown'),
                                              product.get('specs', ''), product.get('relevance_score', 0)))
                                    conn.commit()
                                    
                                    new_product_ids.append(url_hash)
                                    all_products.append(product)
                                    scraped_count += 1
                            except Exception as e:
                                st.warning(f"Error processing product: {str(e)}")
                            
                            progress_bar.progress((i + 1) / len(product_links))
                    
                    st.success(f"Added {scraped_count} products")
                except Exception as e:
                    st.error(f"Error processing site: {str(e)}")
        
        if all_products:
            # Manage storage to keep only 100 best products
            manage_storage(conn, new_product_ids)
            
            # Load data for clustering
            with st.spinner("Preparing data for clustering..."):
                c.execute("SELECT * FROM products")
                products = c.fetchall()
                if products:
                    st.session_state.df = pd.DataFrame(products, columns=[
                        'id','title','description','price','url','image',
                        'category','brand','specs','relevance_score','last_updated'
                    ])
                    
                    # Generate embeddings
                    st.session_state.df['text_data'] = st.session_state.df['title'] + " " + st.session_state.df['description'] + " " + st.session_state.df['specs']
                    st.session_state.embeddings = model.encode(st.session_state.df['text_data'].tolist(), show_progress_bar=False)
                    
                    # Generate clusters
                    reducer = umap.UMAP(n_components=5, random_state=42)
                    reduced_embeddings = reducer.fit_transform(st.session_state.embeddings)
                    scaler = StandardScaler()
                    scaled_embeddings = scaler.fit_transform(reduced_embeddings)
                    dbscan = DBSCAN(eps=0.5, min_samples=3)
                    clusters = dbscan.fit_predict(scaled_embeddings)
                    st.session_state.df['cluster'] = clusters
                    
                    # Calculate cluster centers
                    st.session_state.cluster_centers = []
                    for cluster_id in np.unique(clusters):
                        if cluster_id == -1:
                            continue
                        cluster_points = scaled_embeddings[clusters == cluster_id]
                        if len(cluster_points) > 0:
                            cluster_center = np.mean(cluster_points, axis=0)
                            st.session_state.cluster_centers.append(cluster_center)
                    
                    st.success("Clustering completed!")
            st.balloons()
            st.success(f"✅ Database updated with {len(st.session_state.df)} products")
        else:
            st.warning("No products were added to the database")
    
    if st.button("🧹 Clear Database", key="clear_db"):
        c.execute("DELETE FROM products")
        conn.commit()
        st.session_state.df = pd.DataFrame()
        st.session_state.embeddings = np.array([])
        st.session_state.cluster_centers = []
        st.success("Database cleared!")
    
    st.divider()
    
    if not st.session_state.df.empty:
        st.info("📊 Database Statistics")
        st.write(f"**Total Products:** {len(st.session_state.df)}")
        st.write(f"**Average Relevance Score:** {st.session_state.df['relevance_score'].mean():.1f}/100")
        
        # Show top categories
        st.write("**Top Categories:**")
        category_counts = st.session_state.df['category'].value_counts().head(5)
        for cat, count in category_counts.items():
            st.write(f"- {cat}: {count} products")
        
        # Show cluster info
        st.write("**Product Clusters:**")
        cluster_counts = st.session_state.df['cluster'].value_counts()
        for cluster_id, count in cluster_counts.items():
            if cluster_id != -1:
                st.write(f"- Cluster {cluster_id}: {count} products")
    
    st.divider()
    st.caption("ℹ️ Note: Stores maximum 100 highest-quality products")

# Main interface
st.header("🔍 Intelligent Electronics Search")

# Display product statistics
if not st.session_state.df.empty:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Products", len(st.session_state.df))
    col2.metric("Categories", len(st.session_state.df['category'].unique()))
    col3.metric("Avg. Relevance", f"{st.session_state.df['relevance_score'].mean():.1f}/100")

# Search section
search_query = st.text_input("Search electronics products:", 
                             placeholder="Gaming laptops, wireless headphones, 4K cameras...",
                             key="search_input")

if st.button("🔍 Search", key="search_btn") or search_query:
    if not st.session_state.df.empty and st.session_state.embeddings.size > 0:
        # Generate query embedding
        query_embedding = model.encode([search_query])[0]
        
        # Find closest clusters
        cluster_similarities = []
        for center in st.session_state.cluster_centers:
            similarity = np.dot(query_embedding, center) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(center))
            cluster_similarities.append(similarity)
        
        # Get top clusters
        top_cluster_ids = np.argsort(cluster_similarities)[-3:][::-1]
        
        # Get products from top clusters
        cluster_products = st.session_state.df[st.session_state.df['cluster'].isin(top_cluster_ids)]
        
        # Calculate similarities within these clusters
        product_indices = cluster_products.index
        product_embeddings = st.session_state.embeddings[product_indices]
        similarities = np.dot(product_embeddings, query_embedding) / (
            np.linalg.norm(product_embeddings, axis=1) * np.linalg.norm(query_embedding))
        
        # Add similarities to dataframe
        cluster_products = cluster_products.copy()
        cluster_products['similarity'] = similarities
        
        # Sort by similarity
        results = cluster_products.sort_values('similarity', ascending=False).head(10)
        
        # Display results
        st.subheader(f"🔎 Top {len(results)} Results for '{search_query}'")
        
        # Show cluster insights
        cluster_ids = results['cluster'].unique()
        if len(cluster_ids) > 0:
            st.markdown("### 🧠 AI Cluster Insights")
            for cluster_id in cluster_ids:
                cluster_products = st.session_state.df[st.session_state.df['cluster'] == cluster_id]
                common_category = cluster_products['category'].mode()[0]
                common_brand = cluster_products['brand'].mode()[0]
                st.markdown(f"- **Cluster {cluster_id}**: {len(cluster_products)} products ({common_category}, {common_brand})")
        
        # Display results grid
        st.subheader("💻 Recommended Products")
        cols = st.columns(3)
        for idx, (_, row) in enumerate(results.iterrows()):
            with cols[idx % 3]:
                with st.container():
                    st.markdown(f"#### {row['title'] if row['title'] else 'Untitled Product'}")
                    
                    # Display image
                    if row['image']:
                        try:
                            img = Image.open(BytesIO(row['image']))
                            st.image(img, use_column_width=True)
                        except:
                            st.image("https://via.placeholder.com/300x200?text=No+Image", use_column_width=True)
                    else:
                        st.image("https://via.placeholder.com/300x200?text=No+Image", use_column_width=True)
                    
                    # Price and metadata
                    if row['price'] and row['price'] > 0:
                        st.markdown(f"**Price:** ${row['price']:,.2f}")
                    else:
                        st.markdown("**Price:** Not available")
                    
                    st.markdown(f"**Brand:** {row['brand']}")
                    st.markdown(f"**Category:** {row['category']}")
                    st.markdown(f"**Relevance Score:** {row['relevance_score']}/100")
                    st.markdown(f"**Similarity:** {row['similarity']*100:.1f}%")
                    
                    # Specs expander
                    if row['specs'] and row['specs'] != '{}':
                        with st.expander("Technical Specifications"):
                            try:
                                specs = eval(row['specs'])
                                for key, value in specs.items():
                                    st.markdown(f"**{key}:** {value}")
                            except:
                                st.write(row['specs'])
                    
                    # Link to product
                    st.link_button("View Product", row['url'])
                    st.divider()
    else:
        st.warning("Database is not ready. Please refresh the database first.")

# Show database table
st.divider()
st.subheader("📦 Product Database (Max 100 Products)")
c.execute("SELECT title, price, category, brand, relevance_score, last_updated FROM products ORDER BY relevance_score DESC")
db_data = c.fetchall()

if db_data:
    db_df = pd.DataFrame(db_data, columns=['Title', 'Price', 'Category', 'Brand', 'Relevance', 'Last Updated'])
    st.dataframe(db_df, hide_index=True, height=400)
    
    # Show storage info
    c.execute("SELECT COUNT(*) FROM products")
    count = c.fetchone()[0]
    st.info(f"Storing {count} of maximum 100 products. Lowest relevance score: {db_df['Relevance'].min()}/100")
else:
    st.info("Database is empty. Click 'Refresh Database' to populate.")

# Close connection
conn.close()