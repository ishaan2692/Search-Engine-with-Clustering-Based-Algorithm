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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import umap

# Initialize SQLite database
conn = sqlite3.connect('electronics_products.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS products
             (id TEXT PRIMARY KEY, title TEXT, description TEXT, 
              price REAL, url TEXT, image BLOB, category TEXT, 
              brand TEXT, specs TEXT)''')
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

# Electronics product schema
PRODUCT_SCHEMA = {
    'title': [
        'h1.product-title', 'h1.product-name', 'h1.title', 
        '[data-test="product-title"]', 'h1', 'span#productTitle'
    ],
    'description': [
        '.product-description', '.description-content', 
        '#product-overview', '[data-feature-name="productDescription"]',
        '#feature-bullets', '.product-information'
    ],
    'price': [
        '.price', '.priceView-hero-price', '.price-current', 
        '[data-test="product-price"]', 'span.a-price-whole',
        '.priceView-customer-price'
    ],
    'image': [
        'img.product-image', 'img.primary-image', 
        '[data-test="product-gallery-image"]', 'img#landingImage',
        '.primary-image'
    ],
    'specs': [
        '.specifications', '.specs-table', '#product-details',
        '.spec-container', 'div#technicalSpecifications_section'
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
st.set_page_config(page_title="TechCluster", layout="wide")
st.title("🧠 Smart Electronics Search with ML Clustering")
st.markdown("Discover electronics products grouped by AI-powered similarity clusters")

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
        'DNT': '1'
    }

def crawl_site(url, depth=1):
    visited = set()
    to_visit = [url]
    product_links = []
    
    while to_visit and depth > 0:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue
            
        try:
            headers = get_random_headers()
            response = requests.get(current_url, headers=headers, timeout=20)
            if response.status_code != 200:
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            visited.add(current_url)
            
            # Find product links
            for link in soup.find_all('a', href=True):
                href = link['href']
                if not href or href.startswith(('javascript:', 'mailto:', 'tel:')):
                    continue
                    
                full_url = requests.compat.urljoin(current_url, href)
                
                # Match electronics product URL patterns
                if any(pat in full_url for pat in ['/p/', '/product/', '/prodid/', '/item/', '/dp/']):
                    if full_url not in product_links and not any(x in full_url for x in ['cart', 'checkout', 'account']):
                        product_links.append(full_url)
                elif depth > 1 and full_url.startswith(url) and '#' not in full_url:
                    to_visit.append(full_url)
            
            depth -= 1
            time.sleep(random.uniform(0.8, 1.8))
            
        except Exception as e:
            st.warning(f"Error crawling {current_url}: {str(e)}")
    
    return list(set(product_links))

def extract_specs(soup):
    """Extract technical specifications as key-value pairs"""
    specs = {}
    try:
        # Try to find specification tables
        tables = soup.select('table.specs-table, table.spec-table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['th', 'td'])
                if len(cells) == 2:
                    key = cells[0].get_text(strip=True)
                    value = cells[1].get_text(strip=True)
                    specs[key] = value
        
        # Try to find specification lists
        spec_lists = soup.select('div.spec-container, div.spec-list')
        for spec_list in spec_lists:
            items = spec_list.find_all(['li', 'div.spec-item'])
            for item in items:
                if ':' in item.text:
                    key, value = item.text.split(':', 1)
                    specs[key.strip()] = value.strip()
    
    except Exception as e:
        st.warning(f"Error extracting specs: {str(e)}")
    
    return str(specs)  # Convert to string for storage

def scrape_product(url):
    try:
        headers = get_random_headers()
        response = requests.get(url, headers=headers, timeout=25)
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        product = {'url': url}
        for key, selectors in PRODUCT_SCHEMA.items():
            product[key] = safe_get(soup, selectors)
        
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
        
        # Image handling
        img_data = b''
        if product['image'] and product['image'].startswith('http'):
            try:
                img_response = requests.get(product['image'], headers=headers, timeout=15)
                if img_response.status_code == 200:
                    img_data = img_response.content
            except:
                pass
        
        # Create unique ID
        url_hash = hashlib.sha256(url.encode()).hexdigest()
        
        # Save to DB
        c.execute('''INSERT OR IGNORE INTO products VALUES (?,?,?,?,?,?,?,?,?)''',
                  (url_hash, product.get('title', ''), product.get('description', ''),
                   product.get('price', 0), url, img_data, category, brand, product.get('specs', '')))
        conn.commit()
        
        return product
    
    except Exception as e:
        st.warning(f"Error scraping {url}: {str(e)}")
        return None

def scrape_with_retry(url, retries=3):
    for attempt in range(retries):
        try:
            return scrape_product(url)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            wait_time = (attempt + 1) * 6
            time.sleep(wait_time)
            continue
        except Exception as e:
            st.warning(f"Error on attempt {attempt+1} for {url}: {str(e)}")
            time.sleep(3)
    
    st.warning(f"Failed to scrape {url} after {retries} retries")
    return None

def generate_embeddings(df):
    """Generate BERT embeddings for product text data"""
    texts = df['title'] + " " + df['description'] + " " + df['specs']
    embeddings = model.encode(texts.tolist(), show_progress_bar=False)
    return embeddings

def cluster_products(df, embeddings):
    """Cluster products using DBSCAN algorithm"""
    # Reduce dimensionality with UMAP
    reducer = umap.UMAP(n_components=5, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Standardize features
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(reduced_embeddings)
    
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(scaled_embeddings)
    
    # Add clusters to dataframe
    df['cluster'] = clusters
    
    # Calculate cluster centers
    cluster_centers = []
    for cluster_id in np.unique(clusters):
        if cluster_id == -1:  # Skip noise points
            continue
        cluster_points = scaled_embeddings[clusters == cluster_id]
        cluster_center = np.mean(cluster_points, axis=0)
        cluster_centers.append(cluster_center)
    
    return df, cluster_centers, scaled_embeddings

def visualize_clusters(df, embeddings):
    """Create visualization of product clusters"""
    # Reduce to 2D for visualization
    tsne = TSNE(n_components=2, random_state=42)
    vis_embeddings = tsne.fit_transform(embeddings)
    
    df['x'] = vis_embeddings[:, 0]
    df['y'] = vis_embeddings[:, 1]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    scatter = sns.scatterplot(
        data=df, x='x', y='y', hue='cluster', 
        palette='viridis', style='category', s=100
    )
    
    plt.title("Electronics Product Clusters")
    plt.xlabel("TSNE Dimension 1")
    plt.ylabel("TSNE Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    return plt

def search_products(query, df, embeddings, top_k=10):
    """Find most relevant products using cluster-based search"""
    # Generate query embedding
    query_embedding = model.encode([query])[0]
    
    # Find most relevant clusters
    cluster_similarities = []
    for center in st.session_state.cluster_centers:
        similarity = np.dot(query_embedding, center) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(center))
        cluster_similarities.append(similarity)
    
    # Get top clusters
    top_cluster_ids = np.argsort(cluster_similarities)[-3:][::-1]
    
    # Get products from top clusters
    cluster_products = df[df['cluster'].isin(top_cluster_ids)]
    
    # Calculate similarities within these clusters
    product_embeddings = embeddings[cluster_products.index]
    similarities = np.dot(product_embeddings, query_embedding) / (
        np.linalg.norm(product_embeddings, axis=1) * np.linalg.norm(query_embedding))
    
    # Add similarities to dataframe
    cluster_products['similarity'] = similarities
    
    # Get top products
    results = cluster_products.sort_values('similarity', ascending=False).head(top_k)
    
    return results

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
        all_products = []
        for site in ELECTRONICS_SITES:
            with st.spinner(f"Crawling {site.split('//')[-1].split('/')[0]}..."):
                try:
                    product_links = crawl_site(site, depth=1)
                    st.info(f"Found {len(product_links)} product links")
                    
                    if not product_links:
                        continue
                    
                    progress_bar = st.progress(0)
                    scraped_count = 0
                    
                    # Use threading for faster scraping
                    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                        futures = {executor.submit(scrape_with_retry, url): url for url in product_links}
                        
                        for i, future in enumerate(concurrent.futures.as_completed(futures)):
                            try:
                                product = future.result()
                                if product:
                                    all_products.append(product)
                                    scraped_count += 1
                            except Exception as e:
                                pass
                            
                            progress_bar.progress((i + 1) / len(product_links))
                    
                    st.success(f"Added {scraped_count} products")
                except Exception as e:
                    st.error(f"Error processing site: {str(e)}")
        
        if all_products:
            st.balloons()
            st.success(f"✅ Total added: {len(all_products)} products")
            
            # Generate embeddings and clusters after refresh
            with st.spinner("Generating embeddings and clusters..."):
                c.execute("SELECT * FROM products")
                products = c.fetchall()
                st.session_state.df = pd.DataFrame(products, columns=[
                    'id','title','description','price','url','image',
                    'category','brand','specs'
                ])
                st.session_state.embeddings = generate_embeddings(st.session_state.df)
                st.session_state.df, st.session_state.cluster_centers, scaled_embeddings = cluster_products(
                    st.session_state.df, st.session_state.embeddings
                )
                st.success("Clusters generated!")
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
        st.info("📊 Cluster Analysis")
        cluster_counts = st.session_state.df['cluster'].value_counts()
        st.write(f"**Total Clusters:** {len(cluster_counts)}")
        st.write(f"**Products in Clusters:** {len(st.session_state.df[st.session_state.df['cluster'] != -1])}")
        st.write(f"**Noise Points:** {len(st.session_state.df[st.session_state.df['cluster'] == -1])}")
        
        st.divider()
        
        st.info("🔍 Top Clusters by Size")
        for cluster_id, count in cluster_counts.head(5).items():
            if cluster_id != -1:
                cluster_category = st.session_state.df[st.session_state.df['cluster'] == cluster_id]['category'].mode()[0]
                st.write(f"**Cluster {cluster_id}**: {count} products ({cluster_category})")
    
    st.divider()
    st.info("📦 Database Stats:")
    c.execute("SELECT COUNT(*) FROM products")
    count = c.fetchone()[0]
    st.write(f"**Products:** {count}")
    
    if count > 0:
        c.execute("SELECT category, COUNT(*) FROM products GROUP BY category")
        for row in c.fetchall():
            st.write(f"- **{row[0]}**: {row[1]}")
    
    st.divider()
    st.caption("ℹ️ Note: Scraping real e-commerce sites. Use responsibly.")

# Main interface
st.header("🔍 Intelligent Electronics Search")

# Display cluster visualization
if not st.session_state.df.empty and len(st.session_state.df) > 10:
    with st.expander("📊 Product Cluster Visualization", expanded=True):
        st.write("This visualization shows how our AI has grouped similar electronics products:")
        fig = visualize_clusters(st.session_state.df.copy(), st.session_state.embeddings)
        st.pyplot(fig)
        st.caption("Each point represents a product, colored by its cluster. Products in the same cluster have similar features.")

# Search section
search_query = st.text_input("Search electronics products:", 
                             placeholder="Gaming laptops, wireless headphones, 4K cameras...",
                             key="search_input")

# Search options
if not st.session_state.df.empty:
    col1, col2 = st.columns(2)
    with col1:
        categories = st.session_state.df['category'].unique().tolist()
        selected_category = st.selectbox("Filter by category:", ["All"] + categories)
    with col2:
        brands = st.session_state.df['brand'].unique().tolist()
        selected_brand = st.selectbox("Filter by brand:", ["All"] + brands)

# Price range filter
if not st.session_state.df.empty:
    min_price = st.session_state.df['price'].min()
    max_price = st.session_state.df['price'].max()
    price_range = st.slider("Price range:", min_value=min_price, max_value=max_price, 
                            value=(min_price, max_price))

# Search button
if st.button("Search", key="search_btn") or search_query:
    if not st.session_state.df.empty and st.session_state.embeddings.size > 0:
        with st.spinner("Finding the best electronics products using AI clustering..."):
            results = search_products(search_query, st.session_state.df, st.session_state.embeddings, top_k=12)
            
            if results.empty:
                st.warning("No matching products found. Try a different search term.")
                st.stop()
            
            # Apply filters
            if selected_category != "All":
                results = results[results['category'] == selected_category]
         
