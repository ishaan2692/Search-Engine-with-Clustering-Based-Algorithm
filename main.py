import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import json
import time
import random
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import concurrent.futures
from urllib.parse import urljoin, urlparse
import hashlib

# Test e-commerce sites that allow scraping
ECOMMERCE_SITES = [
    "https://webscraper.io/test-sites/e-commerce/allinone",
    "https://webscraper.io/test-sites/e-commerce/static",
    "https://scrapeme.live/shop/"
]

# Streamlit UI
st.set_page_config(page_title="GlobalShop Search", layout="wide")
st.title("🌍 Global E-commerce Search Engine")
st.markdown("Discover products from global online stores with enhanced scraping")

# Initialize session state
if 'products' not in st.session_state:
    st.session_state.products = []
    st.session_state.vectorizer = None
    st.session_state.search_index = {}
    st.session_state.last_refresh = None

# Browser headers for scraping
def get_random_headers():
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
    ]
    return {
        'User-Agent': random.choice(user_agents),
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Referer': 'https://www.google.com/',
        'DNT': '1'
    }

# Microservice 1: Site Discovery
def discover_ecommerce_sites():
    """Discover global e-commerce sites"""
    return ECOMMERCE_SITES

# Microservice 2: Product Discovery
def discover_products(site_url, max_products=15):
    """Discover product pages on a specific e-commerce site"""
    try:
        headers = get_random_headers()
        response = requests.get(site_url, headers=headers, timeout=15)
        if response.status_code != 200:
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        product_links = set()
        
        # Find all product links
        for link in soup.find_all('a', href=True):
            href = link['href'].strip()
            
            # Skip non-product links
            if not href or href.startswith(('#', 'javascript:', 'mailto:')):
                continue
                
            # Make absolute URL
            full_url = urljoin(site_url, href)
            
            # Skip external links
            if urlparse(full_url).netloc != urlparse(site_url).netloc:
                continue
                
            # Check if it's a product page
            if '/product/' in full_url or '/shop/' in full_url or '/item/' in full_url:
                product_links.add(full_url)
                
            # Check for pagination
            if 'page=' in href or 'p=' in href:
                # Add more pages to scrape
                product_links.update(discover_products(full_url, max_products))
        
        return list(product_links)[:max_products]
    
    except Exception as e:
        st.warning(f"Error discovering products on {site_url}: {str(e)}")
        return []

# Microservice 3: Product Scraping
def scrape_product(product_url):
    """Scrape product details from a product page"""
    try:
        headers = get_random_headers()
        response = requests.get(product_url, headers=headers, timeout=20)
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Create unique product ID
        product_id = hashlib.sha256(product_url.encode()).hexdigest()
        
        # Extract product details with robust selectors
        title = None
        price = None
        description = None
        image_url = None
        category = None
        rating = None
        
        # Try different strategies for each field
        
        # Title
        title_selectors = [
            'h1.product-title', 'h1.product-name', 'h1.title', 
            'h1.product_title', 'h1.page-title', 'h1.product-name',
            'h1.product-title', 'h1.product_title'
        ]
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text(strip=True)
                break
        
        # Price
        price_selectors = [
            '.price', '.product-price', '.amount', 
            '.product_price', '.price-value', '.current-price',
            'p.price', 'span.price', 'div.price'
        ]
        for selector in price_selectors:
            element = soup.select_one(selector)
            if element:
                price_text = element.get_text(strip=True)
                # Extract numeric price
                matches = re.findall(r'[\d,.]+', price_text)
                if matches:
                    try:
                        price = float(matches[0].replace(',', ''))
                        break
                    except:
                        continue
        
        # Description
        description_selectors = [
            '.product-description', '.description', 
            '.product-details', '.product_info', 
            '.product-short-description', '.product_excerpt',
            'div.description', 'div.product-description'
        ]
        for selector in description_selectors:
            element = soup.select_one(selector)
            if element:
                description = element.get_text(strip=True)
                break
        
        # Image
        image_selectors = [
            'img.product-image', 'img.wp-post-image', 
            'img.attachment-shop_single', 'img.main-image',
            'img.primary-image', 'img.product-img',
            'div.product-image img', 'figure.product-image img'
        ]
        for selector in image_selectors:
            element = soup.select_one(selector)
            if element and element.get('src'):
                image_url = element['src']
                if not image_url.startswith('http'):
                    image_url = urljoin(product_url, image_url)
                break
        
        # Download image content
        image_data = None
        if image_url:
            try:
                img_response = requests.get(image_url, headers=headers, timeout=15)
                if img_response.status_code == 200:
                    image_data = img_response.content
            except:
                pass
        
        # Category
        category_selectors = [
            '.product-category', '.breadcrumb', 
            '.product_meta', '.category', 
            '.posted_in', 'div.product-category'
        ]
        for selector in category_selectors:
            element = soup.select_one(selector)
            if element:
                category = element.get_text(strip=True)
                # Clean category text
                if ':' in category:
                    category = category.split(':')[-1].strip()
                break
        
        # Rating
        rating_selectors = [
            '.rating', '.star-rating', 
            '.product-rating', '.review-count',
            '.woocommerce-product-rating'
        ]
        for selector in rating_selectors:
            element = soup.select_one(selector)
            if element:
                rating_text = element.get_text(strip=True)
                # Extract numeric rating
                matches = re.findall(r'[\d.]+', rating_text)
                if matches:
                    try:
                        rating = float(matches[0])
                        if rating > 5:  # Normalize if out of range
                            rating = rating / 2
                        break
                    except:
                        continue
        
        # Create product dictionary
        product = {
            'id': product_id,
            'url': product_url,
            'title': title or "Untitled Product",
            'price': price or 0.0,
            'description': description or "No description available",
            'image_data': image_data,
            'category': category or "Uncategorized",
            'rating': rating or 0.0,
            'features': ""
        }
        
        # Add features from description
        if description:
            product['features'] = " ".join(description.split()[:20])
        
        return product
    
    except Exception as e:
        st.warning(f"Error scraping {product_url}: {str(e)}")
        return None

# Microservice 4: Vectorization and Indexing
def vectorize_and_index(products):
    """Create search index for products"""
    if not products:
        return None, {}
    
    # Create text data for vectorization
    text_data = [
        f"{p['title']} {p['category']} {p['features']}" 
        for p in products
    ]
    
    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    vectors = vectorizer.fit_transform(text_data)
    
    # Create search index
    search_index = {}
    for i, product in enumerate(products):
        product['vector'] = vectors[i]
        # Index by category
        if product['category'] not in search_index:
            search_index[product['category']] = []
        search_index[product['category']].append(i)
    
    return vectorizer, search_index

# Microservice 5: Enhanced Search Service
def search_products(query, products, vectorizer, search_index, top_k=10):
    """Enhanced product search with category filtering"""
    if not products or not vectorizer:
        return []
    
    # Vectorize query
    query_vec = vectorizer.transform([query])
    
    # Find relevant categories
    category_scores = {}
    for category, indices in search_index.items():
        category_vec = np.mean([products[i]['vector'].toarray() for i in indices], axis=0)
        score = cosine_similarity(query_vec, category_vec)[0][0]
        category_scores[category] = score
    
    # Get top categories
    top_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Get products from top categories
    candidate_indices = []
    for category, score in top_categories:
        candidate_indices.extend(search_index[category])
    
    # Remove duplicates
    candidate_indices = list(set(candidate_indices))
    
    # Calculate similarities for candidate products
    similarities = []
    for idx in candidate_indices:
        similarity = cosine_similarity(query_vec, products[idx]['vector'])[0][0]
        similarities.append((idx, similarity))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Get top products
    top_indices = [idx for idx, _ in similarities[:top_k]]
    results = [products[idx] for idx in top_indices]
    
    return results

# Main Orchestration Service
def run_scraping_pipeline(max_products=100):
    """Run the scraping pipeline"""
    st.info("Starting discovery of e-commerce sites...")
    sites = discover_ecommerce_sites()
    st.success(f"Discovered {len(sites)} e-commerce sites")
    
    all_products = []
    product_urls = set()
    
    # Discover product URLs
    with st.spinner("Discovering products across sites..."):
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_site = {executor.submit(discover_products, site, 15): site for site in sites}
            
            for future in concurrent.futures.as_completed(future_to_site):
                urls = future.result()
                for url in urls:
                    if len(product_urls) < max_products:
                        product_urls.add(url)
    
    st.success(f"Discovered {len(product_urls)} product pages")
    
    # Scrape products
    with st.spinner(f"Scraping {len(product_urls)} products..."):
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(scrape_product, url): url for url in product_urls}
            
            for future in concurrent.futures.as_completed(future_to_url):
                product = future.result()
                if product:
                    all_products.append(product)
    
    st.success(f"Successfully scraped {len(all_products)} products")
    
    # Vectorize and index
    with st.spinner("Indexing products for search..."):
        vectorizer, search_index = vectorize_and_index(all_products)
    
    return all_products, vectorizer, search_index

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    
    max_products = st.slider("Max Products to Scrape", 10, 100, 50)
    
    if st.button("🔄 Refresh Products", key="refresh_products"):
        with st.spinner("Running scraping pipeline..."):
            products, vectorizer, search_index = run_scraping_pipeline(max_products)
            
            if products:
                st.session_state.products = products
                st.session_state.vectorizer = vectorizer
                st.session_state.search_index = search_index
                st.session_state.last_refresh = time.strftime("%Y-%m-%d %H:%M:%S")
                st.balloons()
                st.success(f"✅ {len(products)} products ready for search!")
    
    st.divider()
    
    if st.session_state.products:
        st.info("📊 Product Statistics")
        st.write(f"**Total Products:** {len(st.session_state.products)}")
        
        # Category distribution
        categories = [p['category'] for p in st.session_state.products]
        category_counts = pd.Series(categories).value_counts()
        st.write("**Categories:**")
        for cat, count in category_counts.items():
            st.write(f"- {cat}: {count}")
        
        # Price stats
        prices = [p['price'] for p in st.session_state.products if p['price'] > 0]
        if prices:
            st.write(f"**Avg. Price:** ${sum(prices)/len(prices):.2f}")
            st.write(f"**Price Range:** ${min(prices):.2f} - ${max(prices):.2f}")
        
        if st.session_state.last_refresh:
            st.write(f"**Last Refresh:** {st.session_state.last_refresh}")
    
    st.divider()
    st.caption("Note: This scrapes test e-commerce sites that allow scraping")

# Main interface
st.header("🔍 Product Search")

# Search form
col1, col2 = st.columns([3, 1])
with col1:
    search_query = st.text_input("Search for products:", placeholder="Wireless headphones, running shoes, smart watches...")
with col2:
    st.write("")
    st.write("")
    search_clicked = st.button("🔍 Search", key="search_btn")

# Add filters
if st.session_state.products:
    categories = sorted(set(p['category'] for p in st.session_state.products))
    selected_category = st.selectbox("Filter by Category", ["All"] + categories)
    
    min_price = min(p['price'] for p in st.session_state.products if p['price'] > 0)
    max_price = max(p['price'] for p in st.session_state.products)
    price_range = st.slider("Price Range", min_price, max_price, (min_price, max_price))
    
    min_rating = min(p['rating'] for p in st.session_state.products)
    max_rating = max(p['rating'] for p in st.session_state.products)
    min_rating_filter = st.slider("Minimum Rating", min_rating, max_rating, 3.0)

# Show results
if search_clicked or search_query:
    if not st.session_state.products:
        st.warning("No products available. Please refresh products first.")
        st.stop()
    
    if not search_query.strip():
        st.warning("Please enter a search query")
        st.stop()
    
    with st.spinner("Searching products..."):
        results = search_products(
            search_query, 
            st.session_state.products, 
            st.session_state.vectorizer,
            st.session_state.search_index,
            top_k=12
        )
        
        # Apply filters
        if selected_category != "All":
            results = [p for p in results if p['category'] == selected_category]
        
        results = [p for p in results if price_range[0] <= p['price'] <= price_range[1]]
        results = [p for p in results if p['rating'] >= min_rating_filter]
        
        if not results:
            st.warning("No products match your search criteria")
            st.stop()
        
        st.subheader(f"🔎 Found {len(results)} matching products")
        
        # Display results in a grid
        cols = st.columns(3)
        for idx, product in enumerate(results):
            with cols[idx % 3]:
                with st.container():
                    # Product header
                    st.markdown(f"#### {product['title']}")
                    
                    # Display image
                    if product.get('image_data'):
                        try:
                            img = Image.open(BytesIO(product['image_data']))
                            st.image(img, use_column_width=True, caption=product['category'])
                        except:
                            st.image("https://via.placeholder.com/300x300?text=Product+Image", 
                                     use_column_width=True)
                    else:
                        st.image("https://via.placeholder.com/300x300?text=No+Image", 
                                 use_column_width=True)
                    
                    # Product details
                    col1, col2 = st.columns(2)
                    col1.metric("Price", f"${product['price']:.2f}")
                    col2.metric("Rating", f"{product['rating']:.1f} ⭐")
                    
                    st.caption(f"**Category:** {product['category']}")
                    
                    # Description expander
                    with st.expander("Product Details"):
                        st.write(product['description'])
                        if product['features']:
                            st.markdown("**Key Features:**")
                            st.write(product['features'])
                    
                    st.link_button("View Product", product['url'])
                    st.divider()

# Product visualization
if st.session_state.products:
    st.divider()
    st.subheader("📊 Product Distribution by Category")
    
    # Create category distribution chart
    categories = [p['category'] for p in st.session_state.products]
    category_counts = pd.Series(categories).value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 4))
    category_counts.plot(kind='bar', color='skyblue', ax=ax)
    plt.title("Products by Category")
    plt.xlabel("Category")
    plt.ylabel("Number of Products")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

# Show all products
st.divider()
st.subheader("📦 All Products")

if st.session_state.products:
    # Create display data
    display_data = []
    for product in st.session_state.products:
        display_data.append({
            'Title': product['title'][:50] + '...' if len(product['title']) > 50 else product['title'],
            'Price': f"${product['price']:.2f}",
            'Category': product['category'],
            'Rating': product['rating']
        })
    
    df = pd.DataFrame(display_data)
    st.dataframe(df, height=400, hide_index=True)
    
    st.info(f"Showing {len(st.session_state.products)} products. Use search to find specific items.")
else:
    st.info("No products available. Click 'Refresh Products' to start.")