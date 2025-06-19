import streamlit as st
import requests
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import concurrent.futures

# Microservice 1: Site Discovery Service
def discover_ecommerce_sites():
    """Discover global e-commerce sites using search engines"""
    search_terms = [
        "top ecommerce sites", "best online shopping", 
        "global online stores", "international shopping websites"
    ]
    
    sites = set()
    
    # Simulated discovery from search results
    top_global_sites = [
        "https://www.amazon.com", "https://www.ebay.com", 
        "https://www.aliexpress.com", "https://www.walmart.com",
        "https://www.target.com", "https://www.bestbuy.com",
        "https://www.asos.com", "https://www.zalando.com",
        "https://www.etsy.com", "https://www.rakuten.com"
    ]
    
    # Add category-specific pages from each site
    for site in top_global_sites:
        for category in ['electronics', 'clothing', 'home', 'beauty']:
            sites.add(f"{site}/{category}")
    
    return list(sites)

# Microservice 2: Product Discovery Service
def discover_products(site_url, max_products=10):
    """Discover product pages on a specific e-commerce site"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        
        # Simulate finding product pages
        product_urls = []
        
        # Common e-commerce URL patterns
        patterns = [
            r'/product/', r'/p/', r'/dp/', r'/item/', 
            r'/prod/', r'/shop/', r'/buy/', r'-product-'
        ]
        
        # Simulate finding 5-15 products per site
        num_products = random.randint(5, 15)
        for i in range(num_products):
            product_urls.append(f"{site_url}/product-{random.randint(1000,9999)}")
        
        return product_urls[:max_products]
    
    except Exception as e:
        st.warning(f"Error discovering products on {site_url}: {str(e)}")
        return []

# Microservice 3: Product Scraping Service
def scrape_product(product_url):
    """Scrape product details from a product page"""
    try:
        # Simulate scraping with realistic product data
        categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Beauty', 'Sports', 'Books']
        brands = ['Sony', 'Samsung', 'Nike', 'Adidas', 'Apple', 'Dell', 'LG', 'HP', 'Canon', 'Levi\'s']
        
        category = random.choice(categories)
        brand = random.choice(brands)
        
        # Generate realistic product details
        product = {
            'url': product_url,
            'title': f"{brand} {random.choice(['Smart', 'Premium', 'Pro', 'Advanced'])} {random.choice(['Product', 'Item', 'Device', 'Gadget'])} {random.randint(100, 999)}",
            'description': f"High-quality {category.lower()} from {brand}. {random.choice(['Premium materials', 'Advanced technology', 'Eco-friendly', 'Durable design'])}. {random.choice(['Great value', 'Best in class', 'Award-winning', 'Customer favorite'])}.",
            'price': round(random.uniform(10, 1000), 2),
            'category': category,
            'brand': brand,
            'rating': round(random.uniform(3.5, 5.0), 1),
            'image_url': f"https://picsum.photos/300/300?random={random.randint(1,1000)}",
            'features': ', '.join([f"Feature {i+1}" for i in range(random.randint(3, 6))])
        }
        
        # Simulate image download
        try:
            img_response = requests.get(product['image_url'], timeout=10)
            product['image_data'] = img_response.content
        except:
            product['image_data'] = None
        
        # Simulate delay to be respectful
        time.sleep(random.uniform(0.5, 1.5))
        
        return product
    
    except Exception as e:
        st.warning(f"Error scraping {product_url}: {str(e)}")
        return None

# Microservice 4: Vectorization Service
def vectorize_products(products):
    """Convert products to vector representations"""
    # Create text data for vectorization
    text_data = [f"{p['title']} {p['description']} {p['category']} {p['brand']} {p['features']}" for p in products]
    
    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    vectors = vectorizer.fit_transform(text_data)
    
    # Add vectors to products
    for i, product in enumerate(products):
        product['vector'] = vectors[i]
    
    return products, vectorizer

# Microservice 5: Clustering Service
def cluster_products(products, n_clusters=5):
    """Cluster products using KMeans algorithm"""
    # Convert vectors to dense array
    vectors = np.array([p['vector'].toarray().flatten() for p in products])
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=0.95)
    reduced_vectors = pca.fit_transform(vectors)
    
    # Cluster using KMeans
    kmeans = KMeans(n_clusters=min(n_clusters, len(products)), random_state=42)
    clusters = kmeans.fit_predict(reduced_vectors)
    
    # Add clusters to products
    for i, product in enumerate(products):
        product['cluster'] = int(clusters[i])
    
    return products, reduced_vectors

# Microservice 6: Search Service
def search_products(query, products, vectorizer, top_k=10):
    """Search products using vector similarity"""
    # Vectorize query
    query_vec = vectorizer.transform([query]).toarray().flatten()
    
    # Calculate similarities
    similarities = []
    for product in products:
        product_vec = product['vector'].toarray().flatten()
        similarity = cosine_similarity([query_vec], [product_vec])[0][0]
        similarities.append(similarity)
    
    # Get top products
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    results = [products[i] for i in top_indices]
    
    return results

# Main Orchestration Service
def run_pipeline(max_products=100):
    """Run the full e-commerce pipeline"""
    st.info("Starting discovery of e-commerce sites...")
    sites = discover_ecommerce_sites()
    st.success(f"Discovered {len(sites)} e-commerce sites")
    
    all_products = []
    
    with st.spinner("Discovering products across sites..."):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # Discover products from all sites
            future_to_site = {executor.submit(discover_products, site): site for site in sites}
            
            product_urls = []
            for future in concurrent.futures.as_completed(future_to_site):
                urls = future.result()
                product_urls.extend(urls)
        
        # Limit to max products
        product_urls = product_urls[:max_products]
        st.success(f"Discovered {len(product_urls)} product pages")
    
    with st.spinner(f"Scraping {len(product_urls)} products..."):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # Scrape all products
            future_to_url = {executor.submit(scrape_product, url): url for url in product_urls}
            
            for future in concurrent.futures.as_completed(future_to_url):
                product = future.result()
                if product:
                    all_products.append(product)
    
    st.success(f"Successfully scraped {len(all_products)} products")
    
    if not all_products:
        return None, None, []
    
    # Vectorize products
    with st.spinner("Vectorizing products..."):
        all_products, vectorizer = vectorize_products(all_products)
    
    # Cluster products
    with st.spinner("Clustering products..."):
        all_products, reduced_vectors = cluster_products(all_products, n_clusters=5)
    
    return all_products, vectorizer, reduced_vectors

# Streamlit UI
st.set_page_config(page_title="GlobalShop Search", layout="wide")
st.title("🌍 Global E-commerce Search Engine")
st.markdown("Discover products from global online stores with AI-powered search")

# Initialize session state
if 'products' not in st.session_state:
    st.session_state.products = []
    st.session_state.vectorizer = None
    st.session_state.reduced_vectors = []
    st.session_state.cluster_centers = []

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    
    max_products = st.slider("Max Products to Scrape", 10, 100, 50)
    
    if st.button("🔄 Refresh Products", key="refresh_products"):
        with st.spinner("Running discovery and scraping pipeline..."):
            products, vectorizer, reduced_vectors = run_pipeline(max_products)
            
            if products:
                st.session_state.products = products
                st.session_state.vectorizer = vectorizer
                st.session_state.reduced_vectors = reduced_vectors
                
                # Calculate cluster centers
                cluster_centers = []
                clusters = set(p['cluster'] for p in products)
                for cluster_id in clusters:
                    cluster_products = [p for p in products if p['cluster'] == cluster_id]
                    if cluster_products:
                        center = np.mean([p['vector'].toarray().flatten() for p in cluster_products], axis=0)
                        cluster_centers.append(center)
                st.session_state.cluster_centers = cluster_centers
                
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
        prices = [p['price'] for p in st.session_state.products]
        st.write(f"**Avg. Price:** ${sum(prices)/len(prices):.2f}")
        st.write(f"**Price Range:** ${min(prices):.2f} - ${max(prices):.2f}")
        
        # Cluster info
        clusters = set(p['cluster'] for p in st.session_state.products)
        st.write(f"**Clusters:** {len(clusters)}")
    
    st.divider()
    st.caption("Note: This demo uses simulated scraping to avoid legal issues with real e-commerce sites")

# Visualization
if st.session_state.products and len(st.session_state.reduced_vectors) > 0:
    st.subheader("📊 Product Cluster Visualization")
    
    # Prepare data
    clusters = [p['cluster'] for p in st.session_state.products]
    categories = [p['category'] for p in st.session_state.products]
    prices = [p['price'] for p in st.session_state.products]
    
    # Create DataFrame for visualization
    df = pd.DataFrame({
        'x': st.session_state.reduced_vectors[:, 0],
        'y': st.session_state.reduced_vectors[:, 1],
        'cluster': clusters,
        'category': categories,
        'price': prices
    })
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df['x'], df['y'], c=df['cluster'], cmap='viridis', 
                         s=df['price']/10, alpha=0.7)
    
    # Add labels and legend
    plt.title("Product Clusters (Size = Price)")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)
    
    # Add category annotations for cluster centers
    for cluster_id in df['cluster'].unique():
        cluster_df = df[df['cluster'] == cluster_id]
        center_x = cluster_df['x'].mean()
        center_y = cluster_df['y'].mean()
        common_category = cluster_df['category'].mode()[0]
        ax.annotate(common_category, (center_x, center_y), 
                    fontsize=9, ha='center', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    
    st.pyplot(fig)

# Search interface
st.subheader("🔍 Product Search")
search_query = st.text_input("Search for products:", placeholder="Wireless headphones, running shoes, smart watches...")

# Add filters
if st.session_state.products:
    col1, col2, col3 = st.columns(3)
    
    # Category filter
    categories = sorted(set(p['category'] for p in st.session_state.products))
    selected_category = col1.selectbox("Filter by Category", ["All"] + categories)
    
    # Price filter
    min_price = min(p['price'] for p in st.session_state.products)
    max_price = max(p['price'] for p in st.session_state.products)
    price_range = col2.slider("Price Range", min_price, max_price, (min_price, max_price))
    
    # Rating filter
    min_rating = min(p['rating'] for p in st.session_state.products)
    max_rating = max(p['rating'] for p in st.session_state.products)
    rating_filter = col3.slider("Minimum Rating", min_rating, max_rating, 3.5)

# Search button
if st.button("Search", key="search_btn") or search_query:
    if not st.session_state.products:
        st.warning("No products available. Please refresh products first.")
        st.stop()
    
    if not search_query.strip():
        st.warning("Please enter a search query")
        st.stop()
    
    with st.spinner("Searching products..."):
        results = search_products(search_query, st.session_state.products, 
                                 st.session_state.vectorizer, top_k=12)
        
        # Apply filters
        if selected_category != "All":
            results = [p for p in results if p['category'] == selected_category]
        
        results = [p for p in results if price_range[0] <= p['price'] <= price_range[1]]
        results = [p for p in results if p['rating'] >= rating_filter]
        
        if not results:
            st.warning("No products match your search criteria")
            st.stop()
        
        st.subheader(f"🔎 Found {len(results)} matching products")
        
        # Display results in a grid
        cols = st.columns(3)
        for idx, product in enumerate(results):
            with cols[idx % 3]:
                with st.container():
                    st.markdown(f"#### {product['title']}")
                    
                    # Display image
                    if product.get('image_data'):
                        try:
                            img = Image.open(BytesIO(product['image_data']))
                            st.image(img, use_column_width=True)
                        except:
                            st.image("https://via.placeholder.com/300x300?text=Product+Image", 
                                     use_column_width=True)
                    else:
                        st.image("https://via.placeholder.com/300x300?text=No+Image", 
                                 use_column_width=True)
                    
                    # Product details
                    st.markdown(f"**Price:** ${product['price']:.2f}")
                    st.markdown(f"**Category:** {product['category']}")
                    st.markdown(f"**Brand:** {product['brand']}")
                    st.markdown(f"**Rating:** {product['rating']} ⭐")
                    
                    # Description expander
                    with st.expander("Description & Features"):
                        st.write(product['description'])
                        st.markdown("**Features:**")
                        st.write(product['features'])
                    
                    st.link_button("View Product", product['url'])
                    st.divider()

# Show all products
st.divider()
st.subheader("📦 All Products")

if st.session_state.products:
    # Create display data
    display_data = []
    for product in st.session_state.products:
        display_data.append({
            'Title': product['title'],
            'Price': f"${product['price']:.2f}",
            'Category': product['category'],
            'Brand': product['brand'],
            'Rating': product['rating'],
            'Cluster': product['cluster']
        })
    
    df = pd.DataFrame(display_data)
    st.dataframe(df, height=400, hide_index=True)
    
    st.info(f"Showing {len(st.session_state.products)} products. Use search to find specific items.")
else:
    st.info("No products available. Click 'Refresh Products' to start.")