import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
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
import json

# Load pre-trained BERT model for embeddings
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# In-memory product storage with 100-product limit
if 'products' not in st.session_state:
    st.session_state.products = []
    st.session_state.embeddings = None
    st.session_state.cluster_centers = []
    st.session_state.last_updated = None

# Indian fashion e-commerce sites
INDIAN_FASHION_SITES = [
    "https://www.myntra.com/men-tshirts",
    "https://www.myntra.com/women-dresses",
    "https://www.ajio.com/men-tshirts",
    "https://www.ajio.com/westernwear-dresses",
    "https://www.flipkart.com/clothing-and-accessories/men/clothing/tshirts",
    "https://www.flipkart.com/clothing-and-accessories/women/clothing/western-wear/dresses"
]

# Fashion product schema
FASHION_SCHEMA = {
    'title': [
        'h1.pdp-title', 'h1.pdp-name', 'h1.pdp-product-title', 
        'h1.product-title', 'h1.title', 'span.product-title',
        'h1.product-name', 'h1.pdp-title', 'h1.page-title'
    ],
    'description': [
        '.pdp-product-description', '.pdp-description', 
        '.product-details', '.product-description', 
        '.description-content', '.pdp-product-details',
        '.product-detail', '.pdp-description-content'
    ],
    'price': [
        '.pdp-price', '.pdp-product-price', '.price', 
        '.product-price', '.pdp-final-price', 
        '.final-price', '.selling-price', '.price-value'
    ],
    'image': [
        '.pdp-image', '.image-grid-image', '.image-wrapper img',
        '.product-gallery-image', '.product-image', '.main-image',
        '.pdp-main-image', '.gallery-image'
    ],
    'specs': [
        '.pdp-sizeFitDesc', '.pdp-product-size', '.size-information',
        '.product-sizes', '.size-chart', '.pdp-size-description',
        '.product-details-container', '.product-attributes'
    ],
    'category': [
        '.pdp-product-category', '.breadcrumb-item.active',
        '.breadcrumb-item:last-child', '.breadcrumb-item:last-child a',
        '.product-category', '.pdp-category'
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
st.set_page_config(page_title="DesiStyle Search", layout="wide")
st.title("👗 DesiStyle - Indian Fashion Search Engine")
st.markdown("Discover fashion from top Indian e-commerce sites")

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

def crawl_site(url, max_products=25):
    """Fashion-focused crawling with Indian site optimization"""
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
                
                # Match fashion product URL patterns for Indian sites
                if any(pat in full_url for pat in ['/p/', '/product/', '/prd/', '/item/', '/dp/', '-product-', '/dress', '/tshirt']):
                    if base_domain in full_url and not any(x in full_url for x in ['cart', 'checkout', 'account']):
                        link_candidates.add(full_url)
            
            # Strategy 2: JSON-LD product data
            try:
                json_ld = soup.find('script', type='application/ld+json')
                if json_ld:
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
            
            # Method 2: Common pagination patterns for Indian sites
            if not next_page:
                pagination_selectors = [
                    'a.next', 'a.pagination-next', 'li.next a', 
                    'a:contains("Next")', 'a:contains("›")', 'a:contains(">")',
                    'button.next', 'button:contains("Next")'
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
    """Extract fashion specifications"""
    specs = {}
    try:
        # Size information
        size_elems = soup.select('.size-buttons, .size-options, .size-variant')
        if size_elems:
            sizes = [size.get_text(strip=True) for size in size_elems]
            specs['Sizes'] = ", ".join(sizes)
        
        # Color information
        color_elems = soup.select('.color-options, .color-variant, .color-swatch')
        if color_elems:
            colors = [color.get('title') or color.get_text(strip=True) for color in color_elems]
            specs['Colors'] = ", ".join([c for c in colors if c])
        
        # Fabric information
        fabric_elems = soup.select('.fabric, .material, .compositions')
        if fabric_elems:
            fabrics = [fabric.get_text(strip=True) for fabric in fabric_elems]
            specs['Fabric'] = ", ".join(fabrics)
        
        # Care instructions
        care_elems = soup.select('.care-instructions, .wash-care')
        if care_elems:
            care = [care.get_text(strip=True) for care in care_elems]
            specs['Care'] = ", ".join(care)
        
    except Exception as e:
        st.warning(f"Error extracting specs: {str(e)}")
    
    return str(specs)

def calculate_relevance_score(product):
    """Calculate relevance score for fashion products"""
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
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        product = {'url': url}
        for key, selectors in FASHION_SCHEMA.items():
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
        
        # Fashion category detection
        category = "Other"
        fashion_categories = {
            "Men T-Shirts": ['tshirt', 't-shirt', 'tee'],
            "Women Dresses": ['dress', 'gown', 'frock'],
            "Men Shirts": ['shirt', 'formal shirt'],
            "Women Tops": ['top', 'blouse', 'kurti'],
            "Jeans": ['jeans', 'denim'],
            "Footwear": ['shoe', 'sandal', 'footwear', 'slipper'],
            "Accessories": ['bag', 'watch', 'sunglass', 'accessory']
        }
        for cat, terms in fashion_categories.items():
            if any(t in url.lower() or (product.get('title') and t in product['title'].lower()) for t in terms):
                category = cat
                break
        product['category'] = category
        
        # Brand detection for Indian fashion
        brand = "Unknown"
        indian_brands = ['h&m', 'zara', 'levis', 'wrogn', 'roadster', 'only', 'vero moda', 
                         'biba', 'w', 'flying machine', 'max', 'pantaloons']
        for b in indian_brands:
            if b in url.lower() or (product.get('title') and b in product['title'].lower()):
                brand = b.capitalize()
                break
        product['brand'] = brand
        
        # Extract fashion specifications
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
        product['image_data'] = img_data
        
        # Add timestamp
        product['timestamp'] = datetime.now().isoformat()
        
        return product
    
    except Exception as e:
        st.warning(f"Error scraping {url}: {str(e)}")
        return None

def manage_storage(new_products):
    """Manage in-memory storage to maintain 100 highest-quality products"""
    all_products = st.session_state.products + new_products
    
    # Create dataframe for sorting
    df = pd.DataFrame(all_products)
    
    if len(df) == 0:
        return []
    
    # Calculate retention score (70% relevance + 30% recency)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['recency_score'] = (df['timestamp'] - pd.Timestamp('1970-01-01')).dt.total_seconds()
    max_recency = df['recency_score'].max()
    min_recency = df['recency_score'].min()
    
    if max_recency > min_recency:
        df['recency_normalized'] = (df['recency_score'] - min_recency) / (max_recency - min_recency)
    else:
        df['recency_normalized'] = 1.0
    
    df['retention_score'] = 0.7 * df['relevance_score'] + 0.3 * df['recency_normalized'] * 100
    
    # Sort by retention score (descending)
    df = df.sort_values('retention_score', ascending=False)
    
    # Keep only top 100 products
    top_products = df.head(100).to_dict('records')
    
    return top_products

def generate_embeddings(products):
    """Generate BERT embeddings for fashion products"""
    texts = [f"{p['title']} {p['description']} {p['specs']}" for p in products]
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings

def cluster_products(products, embeddings):
    """Cluster fashion products using DBSCAN"""
    if len(products) < 5:  # Need at least 5 products for clustering
        return products, [], embeddings
    
    # Reduce dimensionality with UMAP
    reducer = umap.UMAP(n_components=5, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Standardize features
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(reduced_embeddings)
    
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=3)
    clusters = dbscan.fit_predict(scaled_embeddings)
    
    # Add clusters to products
    for i, product in enumerate(products):
        product['cluster'] = clusters[i]
    
    # Calculate cluster centers
    cluster_centers = []
    for cluster_id in np.unique(clusters):
        if cluster_id == -1:  # Skip noise points
            continue
        cluster_points = scaled_embeddings[clusters == cluster_id]
        if len(cluster_points) > 0:
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_centers.append(cluster_center)
    
    return products, cluster_centers, embeddings

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    
    if st.button("🔄 Refresh Products", key="refresh_db"):
        new_products = []
        for site in INDIAN_FASHION_SITES:
            with st.spinner(f"Crawling {site.split('//')[-1].split('/')[0]}..."):
                try:
                    product_links = crawl_site(site, max_products=10)
                    st.info(f"Found {len(product_links)} fashion items")
                    
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
                                    new_products.append(product)
                                    scraped_count += 1
                            except Exception as e:
                                st.warning(f"Error processing product: {str(e)}")
                            
                            progress_bar.progress((i + 1) / len(product_links))
                    
                    st.success(f"Added {scraped_count} fashion items")
                except Exception as e:
                    st.error(f"Error processing site: {str(e)}")
        
        if new_products:
            # Manage storage to keep only 100 best products
            st.session_state.products = manage_storage(new_products)
            
            # Generate embeddings and clusters
            if st.session_state.products:
                with st.spinner("Analyzing fashion products..."):
                    embeddings = generate_embeddings(st.session_state.products)
                    st.session_state.products, st.session_state.cluster_centers, st.session_state.embeddings = cluster_products(
                        st.session_state.products, embeddings
                    )
                st.success("Product analysis completed!")
                st.session_state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.balloons()
            st.success(f"✅ Total products: {len(st.session_state.products)}")
        else:
            st.warning("No products were added")
    
    if st.button("🧹 Clear Products", key="clear_db"):
        st.session_state.products = []
        st.session_state.embeddings = None
        st.session_state.cluster_centers = []
        st.session_state.last_updated = None
        st.success("Products cleared!")
    
    st.divider()
    
    if st.session_state.products:
        st.info("📊 Product Statistics")
        st.write(f"**Total Products:** {len(st.session_state.products)}")
        
        # Category distribution
        categories = [p['category'] for p in st.session_state.products]
        category_counts = pd.Series(categories).value_counts()
        st.write("**Top Categories:**")
        for cat, count in category_counts.head(5).items():
            st.write(f"- {cat}: {count}")
        
        # Brands
        brands = [p['brand'] for p in st.session_state.products]
        brand_counts = pd.Series(brands).value_counts()
        st.write("**Top Brands:**")
        for brand, count in brand_counts.head(5).items():
            st.write(f"- {brand}: {count}")
        
        # Average relevance
        avg_relevance = sum(p['relevance_score'] for p in st.session_state.products) / len(st.session_state.products)
        st.write(f"**Avg. Relevance:** {avg_relevance:.1f}/100")
        
        if st.session_state.last_updated:
            st.write(f"**Last Updated:** {st.session_state.last_updated}")
    
    st.divider()
    st.caption("ℹ️ Note: Stores maximum 100 highest-quality fashion products")

# Main interface
st.header("👗 Discover Indian Fashion")

# Display product stats
if st.session_state.products:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Products", len(st.session_state.products))
    
    categories = len(set(p['category'] for p in st.session_state.products))
    col2.metric("Categories", categories)
    
    avg_relevance = sum(p['relevance_score'] for p in st.session_state.products) / len(st.session_state.products)
    col3.metric("Avg. Relevance", f"{avg_relevance:.1f}/100")
    
    if st.session_state.last_updated:
        st.caption(f"Last updated: {st.session_state.last_updated}")

# Search section
search_query = st.text_input("Search fashion items:", 
                             placeholder="Men's shirts, women dresses, accessories...",
                             key="search_input")

# Search button
if st.button("🔍 Search", key="search_btn") or search_query:
    if not st.session_state.products or st.session_state.embeddings is None:
        st.warning("No products available. Please refresh products first.")
        st.stop()
    
    if not search_query.strip():
        st.warning("Please enter a search query")
        st.stop()
        
    # Generate query embedding
    query_embedding = model.encode([search_query])[0]
    
    # Find closest clusters
    cluster_similarities = []
    for center in st.session_state.cluster_centers:
        similarity = np.dot(query_embedding, center) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(center))
        cluster_similarities.append(similarity)
    
    # Get top clusters
    if cluster_similarities:
        top_cluster_ids = np.argsort(cluster_similarities)[-3:][::-1]
        
        # Get products from top clusters
        cluster_products = [p for p in st.session_state.products if p.get('cluster') in top_cluster_ids]
    else:
        cluster_products = st.session_state.products
    
    # Calculate product similarities
    product_embeddings = st.session_state.embeddings
    similarities = []
    for i, product in enumerate(cluster_products):
        similarity = np.dot(product_embeddings[i], query_embedding) / (
            np.linalg.norm(product_embeddings[i]) * np.linalg.norm(query_embedding))
        similarities.append(similarity)
    
    # Add similarities to products
    for i, product in enumerate(cluster_products):
        product['similarity'] = similarities[i]
    
    # Sort by similarity
    results = sorted(cluster_products, key=lambda x: x['similarity'], reverse=True)[:12]
    
    # Display results
    st.subheader(f"👗 Top {len(results)} Results for '{search_query}'")
    
    # Show cluster insights
    if cluster_similarities and results:
        cluster_ids = set(p['cluster'] for p in results)
        st.markdown("### 🧠 Fashion Cluster Insights")
        for cluster_id in cluster_ids:
            cluster_items = [p for p in st.session_state.products if p.get('cluster') == cluster_id]
            common_category = pd.Series([p['category'] for p in cluster_items]).mode()[0]
            common_brand = pd.Series([p['brand'] for p in cluster_items]).mode()[0]
            st.markdown(f"- **Cluster {cluster_id}**: {len(cluster_items)} items ({common_category}, {common_brand})")
    
    # Display results grid
    st.subheader("✨ Recommended Fashion Items")
    cols = st.columns(3)
    for idx, product in enumerate(results):
        with cols[idx % 3]:
            with st.container():
                st.markdown(f"#### {product['title'] if product['title'] else 'Untitled Product'}")
                
                # Display image
                if product.get('image_data'):
                    try:
                        img = Image.open(BytesIO(product['image_data']))
                        st.image(img, use_column_width=True, caption=product['brand'])
                    except:
                        st.image("https://via.placeholder.com/300x400?text=Fashion+Image", 
                                 use_column_width=True)
                else:
                    st.image("https://via.placeholder.com/300x400?text=No+Image", 
                             use_column_width=True)
                
                # Price and metadata
                if product['price'] and product['price'] > 0:
                    st.markdown(f"**Price:** ₹{product['price']:,.2f}")
                else:
                    st.markdown("**Price:** Not available")
                
                st.markdown(f"**Brand:** {product['brand']}")
                st.markdown(f"**Category:** {product['category']}")
                st.markdown(f"**Relevance Score:** {product['relevance_score']}/100")
                st.markdown(f"**Match Score:** {product['similarity']*100:.1f}%")
                
                # Specs expander
                if product['specs'] and product['specs'] != '{}':
                    with st.expander("Product Details"):
                        try:
                            specs = eval(product['specs'])
                            for key, value in specs.items():
                                st.markdown(f"**{key}:** {value}")
                        except:
                            st.write(product['specs'])
                
                # Link to product
                st.link_button("View Product", product['url'])
                st.divider()

# Show product list
st.divider()
st.subheader("📋 Product Catalog (Max 100 Items)")

if st.session_state.products:
    # Create display dataframe
    display_data = []
    for product in st.session_state.products:
        display_data.append({
            'Title': product['title'][:50] + '...' if product['title'] and len(product['title']) > 50 else product['title'],
            'Price': f"₹{product['price']:,.2f}" if product['price'] > 0 else "N/A",
            'Brand': product['brand'],
            'Category': product['category'],
            'Relevance': product['relevance_score']
        })
    
    df = pd.DataFrame(display_data)
    st.dataframe(df, hide_index=True, height=400)
    
    # Show storage info
    min_relevance = min(p['relevance_score'] for p in st.session_state.products)
    st.info(f"Storing {len(st.session_state.products)} fashion items. Minimum relevance score: {min_relevance}/100")
else:
    st.info("Catalog is empty. Click 'Refresh Products' to populate.")