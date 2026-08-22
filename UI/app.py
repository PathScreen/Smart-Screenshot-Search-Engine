import streamlit as st 
import pandas as pd 
import numpy as np 
import torch 
import re 
import os
 
from sentence_transformers import SentenceTransformer 
from sklearn.metrics.pairwise import cosine_similarity 
from transformers import CLIPProcessor, CLIPModel 
from PIL import Image 


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# ============================================================
# PATHS 
# ============================================================ 
 
SEMANTIC_METADATA_PATH = os.path.join(
    PROJECT_ROOT,
    "Data",
    "metadata_ocr_3150.csv"
)

SEMANTIC_EMBEDDINGS_PATH = os.path.join(
    PROJECT_ROOT,
    "Data",
    "screenshot_embeddings_3150.npy"
)

VISUAL_METADATA_PATH = os.path.join(
    PROJECT_ROOT,
    "Data",
    "metadata_4000.csv"
)

VISUAL_EMBEDDINGS_PATH = os.path.join(
    PROJECT_ROOT,
    "Data",
    "visual_embeddings_4000.npy"
) 
SEMANTIC_METADATA_PATH = os.path.join(
    PROJECT_ROOT,
    "Data",
    "metadata_ocr_3150.csv"
)

SEMANTIC_EMBEDDINGS_PATH = os.path.join(
    PROJECT_ROOT,
    "Data",
    "screenshot_embeddings_3150.npy"
)

VISUAL_METADATA_PATH = os.path.join(
    PROJECT_ROOT,
    "Data",
    "metadata_4000.csv"
)

VISUAL_EMBEDDINGS_PATH = os.path.join(
    PROJECT_ROOT,
    "Data",
    "visual_embeddings_4000.npy"
) 
 
# ============================================================
# PAGE CONFIG 
# ============================================================ 
 
st.set_page_config( 
    page_title="Smart Screenshot Search Engine", 
    page_icon="🔍", 
    layout="wide", 
    initial_sidebar_state="expanded" 
) 
 
 
# ============================================================
# CUSTOM CSS 
# ============================================================ 
 
st.markdown( 
    """ 
    <style> 
 
    /* ======================================================== 
       GLOBAL 
       ======================================================== */ 
 
    .stApp { 
        background: #f6f8fc; 
    } 
 
    .block-container { 
        max-width: 1450px; 
        padding-top: 2rem; 
        padding-bottom: 3rem; 
    } 
 
    h1, h2, h3, h4, h5, h6 { 
        color: #172033 !important; 
        font-family: Arial, sans-serif; 
    } 
 
    p, label { 
        color: #475467; 
        font-family: Arial, sans-serif; 
    } 
 
 
    /* ======================================================== 
       HEADER 
       ======================================================== */ 
 
    .main-title { 
        text-align: center; 
        font-size: 42px; 
        font-weight: 800; 
        color: #172033; 
        margin-top: 5px; 
        margin-bottom: 5px; 
        letter-spacing: -1px; 
    } 
 
    .subtitle { 
        text-align: center; 
        color: #667085; 
        font-size: 17px; 
        margin-bottom: 30px; 
    } 
 
 
    /* ======================================================== 
       METRICS 
       ======================================================== */ 
 
    div[data-testid="stMetric"] { 
        background: #ffffff; 
        border: 1px solid #e4e7ec; 
        border-radius: 16px; 
        padding: 17px; 
        box-shadow: 0 4px 14px rgba(16, 24, 40, 0.05); 
    } 
 
    div[data-testid="stMetricLabel"] { 
        color: #667085 !important; 
    } 
 
    div[data-testid="stMetricValue"] { 
        color: #172033 !important; 
        font-weight: 800; 
    } 
 
 
    /* ======================================================== 
       SIDEBAR 
       ======================================================== */ 
 
    section[data-testid="stSidebar"] { 
        background: #ffffff; 
        border-right: 1px solid #e4e7ec; 
    } 
 
    section[data-testid="stSidebar"] * { 
        font-family: Arial, sans-serif; 
    } 
 
    .sidebar-title { 
        font-size: 21px; 
        font-weight: 800; 
        color: #172033; 
        margin-bottom: 18px; 
    } 
 
    .sidebar-info { 
        background: #f6f8fc; 
        border: 1px solid #e4e7ec; 
        border-radius: 12px; 
        padding: 13px; 
        margin-top: 12px; 
        color: #667085; 
        font-size: 13px; 
        line-height: 1.5; 
    } 
 
 
    /* ======================================================== 
       SEARCH AREA 
       ======================================================== */ 
 
    .search-section-title { 
        font-size: 26px; 
        font-weight: 800; 
        color: #172033; 
        margin-top: 10px; 
        margin-bottom: 4px; 
    } 
 
    .search-section-subtitle { 
        color: #667085; 
        font-size: 14px; 
        margin-bottom: 15px; 
    } 
 
    div[data-testid="stTextInput"] input { 
        height: 58px; 
        border-radius: 15px; 
        border: 1px solid #d0d5dd; 
        background: #ffffff; 
        color: #172033 !important; 
        font-size: 16px; 
        padding-left: 17px; 
        box-shadow: 0 3px 12px rgba(16, 24, 40, 0.05); 
    } 
 
    div[data-testid="stTextInput"] input::placeholder { 
        color: #98a2b3 !important; 
    } 
 
    div[data-testid="stTextInput"] input:focus { 
        border-color: #6366f1; 
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.12); 
    } 
 
 
    /* ======================================================== 
       BUTTONS 
       ======================================================== */ 
 
    div.stButton > button { 
        min-height: 56px; 
        border-radius: 15px; 
        border: none; 
        background: #172033; 
        color: #ffffff !important; 
        font-size: 15px; 
        font-weight: 700; 
        box-shadow: 0 4px 12px rgba(16, 24, 40, 0.14); 
        transition: 0.2s ease; 
    } 
 
    div.stButton > button:hover { 
        background: #303b55; 
        color: #ffffff !important; 
    } 
 
    div.stButton > button p, 
    div.stButton > button span { 
        color: #ffffff !important; 
    } 
 
 
    /* ======================================================== 
       RESULTS 
       ======================================================== */ 
 
    .result-title { 
        font-size: 26px; 
        font-weight: 800; 
        color: #172033; 
        margin-top: 8px; 
        margin-bottom: 5px; 
    } 
 
    .result-count { 
        color: #667085; 
        font-size: 14px; 
        margin-bottom: 18px; 
    } 
 
    .result-category { 
        font-size: 14px; 
        font-weight: 700; 
        color: #344054; 
        margin-top: 8px; 
        margin-bottom: 5px; 
    } 
 
    .result-match { 
        font-size: 13px; 
        color: #667085; 
        margin-top: 6px; 
        margin-bottom: 5px; 
    } 
 
    .result-filename { 
        font-size: 12px; 
        color: #667085; 
        overflow-wrap: anywhere; 
        margin-top: 6px; 
        margin-bottom: 10px; 
    } 
 
 
    /* ======================================================== 
       RESULT CARDS 
       ======================================================== */ 
 
    div[data-testid="stImage"] { 
        border-radius: 12px; 
        overflow: hidden; 
    } 
 
 
    /* ======================================================== 
       EXPANDERS 
       ======================================================== */ 
 
    div[data-testid="stExpander"] { 
        background: #ffffff; 
        border: 1px solid #e4e7ec; 
        border-radius: 11px; 
        margin-top: 6px; 
    } 
 
 
    /* ======================================================== 
       FOOTER 
       ======================================================== */ 
 
    .footer { 
        text-align: center; 
        color: #98a2b3; 
        font-size: 13px; 
        padding-top: 15px; 
        padding-bottom: 10px; 
    } 
 
    hr { 
        margin-top: 25px; 
        margin-bottom: 25px; 
        border: none; 
        border-top: 1px solid #e4e7ec; 
    } 
 
    </style> 
    """, 
    unsafe_allow_html=True 
) 
 
 
# ============================================================
# CATEGORY NAMES 
# ============================================================ 
 
category_display = { 
    "Addresses_Locations": "📍 Addresses & Locations", 
    "Interior_Design": "🏠 Interior Design", 
    "Products_Shopping": "🛍️ Products & Shopping", 
    "Receipts_Bills": "🧾 Receipts & Bills", 
    "Recipes": "🍳 Recipes", 
    "Tickets_Bookings": "🎟️ Tickets & Bookings" 
} 
 
 
# ============================================================
# VISUAL SEARCH WORDS 
# ============================================================ 
 
visual_keywords = { 
    "car", "cars", 
    "bike", "bicycle", "motorcycle", 
    "laptop", "computer", 
    "phone", "mobile", 
    "sofa", "chair", "table", "bed", 
    "pizza", "burger", "food", 
    "dog", "cat", "tree", "flower", 
    "shirt", "shoe", "shoes", 
    "bag", "watch", 
    "bottle", "tv", "television", 
    "cake", "sandwich", "coffee", 
    "dress", "jacket", "jeans", 
    "plant", "house", "building", 
    "carpet", "lamp", "desk" 
} 
 
 
# ============================================================
# SEARCH VOCABULARY 
# ============================================================ 
 
known_search_words = { 
    "address", "addresses", "location", "locations", 
    "map", "maps", "place", "places", 
 
    "interior", "design", "home", "house", 
    "room", "bedroom", "living", "decor", 
 
    "product", "products", "shopping", 
    "shop", "buy", "price", "item", 
 
    "receipt", "receipts", "bill", "bills", 
    "invoice", "payment", "amount", "total", 
 
    "recipe", "recipes", "cooking", 
    "dish", "ingredient", "ingredients", 
    "food", "meal", 
 
    "ticket", "tickets", "booking", "bookings", 
    "flight", "train", "bus", "hotel", 
    "reservation", "travel", 
 
    "name", "date", "time", "number", 
    "contact", "email", "phone", 
    "order", "delivery", "confirmation", 
    "discount", "offer", 
    "restaurant", "airport", "station", 
    "menu", 
 
    *visual_keywords 
} 
 
 
# ============================================================
# LOAD DATA 
# ============================================================ 
 
@st.cache_data 
def load_semantic_metadata(): 
    return pd.read_csv( 
        SEMANTIC_METADATA_PATH 
    ) 
 
 
@st.cache_resource 
def load_semantic_embeddings(): 
    return np.load( 
        SEMANTIC_EMBEDDINGS_PATH 
    ) 
 
 
@st.cache_data 
def load_visual_metadata(): 
    return pd.read_csv( 
        VISUAL_METADATA_PATH 
    ) 
 
 
@st.cache_resource 
def load_visual_embeddings(): 
    return np.load( 
        VISUAL_EMBEDDINGS_PATH 
    ) 
 
 
# ============================================================
# LOAD SEMANTIC MODEL 
# ============================================================ 
 
@st.cache_resource 
def load_semantic_model(): 
 
    device = ( 
        "cuda" 
        if torch.cuda.is_available() 
        else "cpu" 
    ) 
 
    return SentenceTransformer( 
        "all-MiniLM-L6-v2", 
        device=device 
    ) 
 
 
# ============================================================
# LOAD CLIP 
# ============================================================ 
 
@st.cache_resource 
def load_clip_model(): 
 
    device = ( 
        "cuda" 
        if torch.cuda.is_available() 
        else "cpu" 
    ) 
 
    model = CLIPModel.from_pretrained( 
        "openai/clip-vit-base-patch32" 
    ).to(device) 
 
    processor = CLIPProcessor.from_pretrained( 
        "openai/clip-vit-base-patch32" 
    ) 
 
    model.eval() 
 
    return model, processor, device 
   
 
# ============================================================
# LOAD EVERYTHING 
# ============================================================ 
 
semantic_df = load_semantic_metadata() 
semantic_embeddings = load_semantic_embeddings() 
 
visual_df = load_visual_metadata() 
visual_embeddings = load_visual_embeddings() 
 
semantic_model = load_semantic_model() 
 
clip_model, clip_processor, clip_device = ( 
    load_clip_model() 
) 
 
       
# ============================================================
# MERGE OCR TEXT INTO VISUAL DATA 
# ============================================================ 
 
# Create OCR column first 
visual_df["ocr_text"] = "" 
 
 
# Normalize filename helper 
def get_filename_key(path): 
 
    return ( 
        str(path) 
        .replace("\\", "/") 
        .strip() 
        .lower() 
        .split("/")[-1] 
    ) 
 
 
# Create matching keys 
semantic_df["_match_key"] = ( 
    semantic_df["filename"] 
    .apply(get_filename_key) 
) 
 
visual_df["_match_key"] = ( 
    visual_df["filename"] 
    .apply(get_filename_key) 
) 
 
 
# Create OCR lookup dictionary 
ocr_lookup = dict( 
    zip( 
        semantic_df["_match_key"], 
        semantic_df["ocr_text"] 
    ) 
) 
 
 
# Add OCR text to visual dataframe 
visual_df["ocr_text"] = ( 
    visual_df["_match_key"] 
    .map(ocr_lookup) 
) 
 
 
# Replace missing OCR values 
visual_df["ocr_text"] = ( 
    visual_df["ocr_text"] 
    .fillna("") 
) 
 
 
# Remove temporary matching columns 
semantic_df.drop( 
    columns=["_match_key"], 
    inplace=True, 
    errors="ignore" 
) 
 
visual_df.drop( 
    columns=["_match_key"], 
    inplace=True, 
    errors="ignore" 
) 
 
# ============================================================
# VALIDATION 
# ============================================================ 
 
if len(semantic_df) != len(semantic_embeddings): 
 
    st.error( 
        f"Semantic data mismatch: " 
        f"{len(semantic_df)} metadata rows vs " 
        f"{len(semantic_embeddings)} embeddings." 
    ) 
 
    st.stop() 
 
 
if len(visual_df) != len(visual_embeddings): 
 
    st.error( 
        f"Visual data mismatch: " 
        f"{len(visual_df)} metadata rows vs " 
        f"{len(visual_embeddings)} embeddings." 
    ) 
 
    st.stop() 
 
 
# ============================================================
# HEADER 
# ============================================================ 
 
st.markdown( 
    '<div class="main-title">' 
    '🔍 Smart Screenshot Search Engine' 
    '</div>', 
    unsafe_allow_html=True 
) 
 
st.markdown( 
    '<div class="subtitle">' 
    'Find what you are looking for across your screenshots.' 
    '</div>', 
    unsafe_allow_html=True 
) 
 
 
# ============================================================
# STATISTICS 
# ============================================================ 
 
total_images = len(visual_df) 
 
total_categories = ( 
    visual_df["category"] 
    .nunique() 
) 
 
embedding_dimension = ( 
    semantic_embeddings.shape[1] 
) 
 
 
col1, col2, col3, col4 = st.columns(4) 
 
 
with col1: 
 
    st.metric( 
        "📸 Screenshots Indexed", 
        total_images 
    ) 
 
with col2: 
 
    st.metric( 
        "📂 Categories", 
        total_categories 
    ) 
 
with col3: 
 
    st.metric( 
        "🧠 Embedding Size", 
        embedding_dimension 
    ) 
 
with col4: 
 
    st.metric( 
        "⚡ Processing", 
        "GPU" 
        if torch.cuda.is_available() 
        else "CPU" 
    ) 
 
 
st.divider() 
 
 
# ============================================================
# SEARCH HISTORY 
# ============================================================ 
 
if "search_history" not in st.session_state: 
 
    st.session_state.search_history = [] 
 
 
# ============================================================
# SIDEBAR 
# ============================================================ 
 
with st.sidebar: 
 
    st.markdown( 
        '<div class="sidebar-title">' 
        '⚙️ Search Settings' 
        '</div>', 
        unsafe_allow_html=True 
    ) 
 
 
    # ========================================================
    # CATEGORY 
    # ======================================================== 
 
    category_options = [ 
        "🗂️ All Categories" 
    ] 
 
    for category in sorted( 
        visual_df["category"] 
        .dropna() 
        .unique() 
    ): 
 
        category_options.append( 
            category_display.get( 
                category, 
                category 
            ) 
        ) 
 
 
    selected_display_category = st.selectbox( 
        "📂 Category", 
        category_options 
    ) 
 
 
    if ( 
        selected_display_category 
        == "🗂️ All Categories" 
    ): 
 
        selected_category = "All Categories" 
 
    else: 
 
        selected_category = next( 
            ( 
                key 
                for key, value in category_display.items() 
                if value 
                == selected_display_category 
            ), 
            selected_display_category 
        ) 
 
 
    # ========================================================
    # NUMBER OF RESULTS 
    # ======================================================== 
 
    number_of_results = st.slider( 
        "🎚️ Number of results", 
        min_value=1, 
        max_value=10, 
        value=6 
    ) 
 
 
    st.divider() 
 
 
    # ========================================================
    # DATASET OVERVIEW 
    # ======================================================== 
 
    st.markdown( 
        '<div class="sidebar-title">' 
        '📊 Dataset Overview' 
        '</div>', 
        unsafe_allow_html=True 
    ) 
 
 
    category_counts = ( 
        visual_df["category"] 
        .value_counts() 
    ) 
 
 
    for category, count in category_counts.items():

        display_name = category_display.get(
            category,
            category
        )

        st.html(
            f"""
            <div style="
                background:#f8f9fc;
                border:1px solid #e4e7ec;
                border-radius:10px;
                padding:10px 12px;
                margin-bottom:8px;
                font-family:Arial,sans-serif;
            ">

                <div style="
                    font-size:13px;
                    color:#475467;
                    font-weight:600;
                ">
                    {display_name}
                </div>

                <div style="
                    font-size:20px;
                    color:#172033;
                    font-weight:800;
                    margin-top:3px;
                ">
                    {count}
                    <span style="
                        font-size:12px;
                        color:#98a2b3;
                        font-weight:500;
                    ">
                        screenshots
                    </span>
                </div>

            </div>
            """
        )


    # ========================================================
    # CATEGORY DISTRIBUTION BAR GRAPH 
    # ======================================================== 
 
    st.markdown( 
        '<div class="sidebar-title" ' 
        'style="margin-top:20px;">' 
        '📈 Category Distribution' 
        '</div>', 
        unsafe_allow_html=True 
    ) 
 
    chart_data = category_counts.rename( 
        index={ 
            category: category_display.get( 
                category, 
                category 
            ) 
            for category in category_counts.index 
        } 
    ) 
 
    st.bar_chart( 
        chart_data, 
        use_container_width=True 
    ) 
 
 
# ============================================================
# SEARCH SECTION 
# ============================================================ 
 
st.markdown( 
    '<div class="search-section-title">' 
    'Explore by meaning' 
    '</div>', 
    unsafe_allow_html=True 
) 
 
st.markdown( 
    '<div class="search-section-subtitle">' 
    'Search using text, meaning, objects, places, products, ' 
    'recipes, receipts, tickets, and more.' 
    '</div>', 
    unsafe_allow_html=True 
) 
 
 
search_col, button_col = st.columns( 
    [6, 1] 
) 
 
 
with search_col: 
 
    query = st.text_input( 
        "Search", 
        placeholder=( 
            "🔍 Try: shopping bill, hotel booking, " 
            "pizza, address, laptop..." 
        ), 
        label_visibility="collapsed" 
    ) 
 
 
with button_col: 
 
    search_button = st.button( 
        "🔍 Search", 
        use_container_width=True 
    ) 
 
 
# ============================================================
# QUERY CLEANING 
# ============================================================ 
 
def clean_query(text): 
 
    text = str(text).lower().strip() 
 
    text = re.sub( 
        r"[^a-zA-Z0-9\s]", 
        " ", 
        text 
    ) 
 
    text = re.sub( 
        r"\s+", 
        " ", 
        text 
    ) 
 
    return text.strip() 
 
 
# ============================================================
# GIBBERISH DETECTION 
# ============================================================ 
 
def looks_like_gibberish(text): 
 
    cleaned = clean_query(text) 
 
    if not cleaned: 
        return True 
 
    words = cleaned.split() 
 
    meaningful_count = 0 
 
 
    for word in words: 
 
        if word in known_search_words: 
 
            meaningful_count += 1 
            continue 
 
        if len(word) <= 2: 
 
            meaningful_count += 1 
            continue 
 
        if re.search( 
            r"(.)\1{3,}", 
            word 
        ): 
 
            continue 
 
        vowel_count = sum( 
            char in "aeiou" 
            for char in word 
        ) 
 
        if vowel_count >= 1: 
 
            meaningful_count += 1 
 
 
    if meaningful_count == 0: 
 
        return True 
 
 
    if ( 
        len(words) == 1 
        and len(words[0]) >= 8 
    ): 
 
        word = words[0] 
 
 
        vowel_count = sum( 
            char in "aeiou" 
            for char in word 
        ) 
 
 
        if vowel_count == 0: 
 
            return True 
 
 
        if re.search( 
            r"[bcdfghjklmnpqrstvwxyz]{6,}", 
            word 
        ): 
 
            return True 
 
 
    return False 
 
 
# ============================================================
# VISUAL QUERY DETECTION 
# ============================================================ 
 
def is_visual_query(text): 
 
    words = set( 
        re.findall( 
            r"[a-zA-Z]+", 
            text.lower() 
        ) 
    ) 
 
 
    visual_found = ( 
        words.intersection( 
            visual_keywords 
        ) 
    ) 
 
 
    semantic_words = { 
        "receipt", "bill", "invoice", 
        "recipe", "ingredient", 
        "booking", "ticket", 
        "address", "location", 
        "payment", "order", 
        "confirmation", "hotel", 
        "flight", "train", "bus", 
        "shopping", "product" 
    } 
 
 
    semantic_found = ( 
        words.intersection( 
            semantic_words 
        ) 
    ) 
 
 
    if ( 
        visual_found 
        and not semantic_found 
    ): 
 
        return True 
 
 
    return False 
 
 
# ============================================================
# SEMANTIC SEARCH 
# ============================================================ 
 
def semantic_search( 
    query, 
    search_df, 
    search_embeddings 
): 
 
    query_embedding = ( 
        semantic_model.encode( 
            [query], 
            convert_to_numpy=True 
        ) 
    ) 
 
 
    similarities = ( 
        cosine_similarity( 
            query_embedding, 
            search_embeddings 
        )[0] 
    ) 
 
 
    results = search_df.copy() 
 
 
    results["similarity"] = ( 
        similarities 
    ) 
 
 
    return results.sort_values( 
        "similarity", 
        ascending=False 
    ) 
 
 
# ============================================================
# CLIP SEARCH 
# ============================================================ 
 
def clip_search( 
    query, 
    search_df, 
    search_embeddings 
): 
 
    inputs = clip_processor( 
        text=[query], 
        return_tensors="pt", 
        padding=True 
    ) 
 
 
    inputs = { 
        key: value.to(clip_device) 
        for key, value in inputs.items() 
    } 
 
 
    with torch.inference_mode(): 
 
        output = ( 
            clip_model.get_text_features( 
                **inputs 
            ) 
        ) 
 
 
    if hasattr( 
        output, 
        "pooler_output" 
    ): 
 
        query_embedding = ( 
            output.pooler_output 
        ) 
 
    else: 
 
        query_embedding = output 
 
 
    query_embedding = ( 
        query_embedding 
        / query_embedding.norm( 
            p=2, 
            dim=-1, 
            keepdim=True 
        ) 
    ) 
 
 
    image_embeddings = torch.tensor( 
        search_embeddings, 
        dtype=torch.float32, 
        device=clip_device 
    ) 
 
 
    image_embeddings = ( 
        image_embeddings 
        / image_embeddings.norm( 
            p=2, 
            dim=-1, 
            keepdim=True 
        ) 
    ) 
 
 
    similarities = ( 
        image_embeddings 
        @ query_embedding.T 
    ) 
 
 
    similarities = ( 
        similarities 
        .squeeze() 
        .detach() 
        .cpu() 
        .numpy() 
    ) 
 
 
    results = search_df.copy() 
 
 
    results["similarity"] = ( 
        similarities 
    ) 
 
 
    return results.sort_values( 
        "similarity", 
        ascending=False 
    ) 
 
 
# ============================================================
# SEARCH EXECUTION 
# ============================================================ 
 
if search_button: 
 
    cleaned_query = clean_query( 
        query 
    ) 
 
 
    # ========================================================
    # EMPTY SEARCH 
    # ======================================================== 
 
    if not cleaned_query: 
 
        st.warning( 
            "Please enter something to search." 
        ) 
 
        st.stop() 
 
 
    # ========================================================
    # GIBBERISH 
    # ======================================================== 
 
    if looks_like_gibberish( 
        cleaned_query 
    ): 
 
        st.divider() 
 
        st.markdown( 
            '<div class="result-title">' 
            '📌 Relevant Results' 
            '</div>', 
            unsafe_allow_html=True 
        ) 
 
        st.info( 
            "No meaningful search detected. " 
            "Try something like shopping bill, " 
            "recipe, address, hotel booking, " 
            "pizza, laptop, or ticket." 
        ) 
 
        st.stop() 
 
 
    # ========================================================
    # SEARCH HISTORY 
    # ======================================================== 
 
    if ( 
        cleaned_query 
        not in st.session_state.search_history 
    ): 
 
        st.session_state.search_history.insert( 
            0, 
            cleaned_query 
        ) 
 
 
    st.session_state.search_history = ( 
        st.session_state.search_history[:10] 
    ) 
 
 
    # ========================================================
    # SELECT DATA 
    # ======================================================== 
 
    if selected_category == "All Categories": 
 
        semantic_search_df = ( 
            semantic_df.copy() 
        ) 
 
        semantic_search_embeddings = ( 
            semantic_embeddings 
        ) 
 
        visual_search_df = ( 
            visual_df.copy() 
        ) 
 
        visual_search_embeddings = ( 
            visual_embeddings 
        ) 
 
    else: 
 
        # ----------------------------------------------------
        # SEMANTIC DATA 
        # ---------------------------------------------------- 
 
        semantic_mask = ( 
            semantic_df["category"] 
            == selected_category 
        ) 
 
 
        semantic_search_df = ( 
            semantic_df[ 
                semantic_mask 
            ].copy() 
        ) 
 
 
        semantic_search_embeddings = ( 
            semantic_embeddings[ 
                semantic_mask.values 
            ] 
        ) 
 
 
        # ----------------------------------------------------
        # VISUAL DATA 
        # ---------------------------------------------------- 
 
        visual_mask = ( 
            visual_df["category"] 
            == selected_category 
        ) 
 
 
        visual_search_df = ( 
            visual_df[ 
                visual_mask 
            ].copy() 
        ) 
 
 
        visual_search_embeddings = ( 
            visual_embeddings[ 
                visual_mask.values 
            ] 
        ) 
 
 
    # ========================================================
    # DATA CHECK 
    # ======================================================== 
 
    if ( 
        len(semantic_search_df) == 0 
        and len(visual_search_df) == 0 
    ): 
 
        st.warning( 
            "No screenshots found in this category." 
        ) 
 
        st.stop() 
 
 
    # ========================================================
    # SEARCH TYPE 
    # ======================================================== 
 
    visual_query = is_visual_query( 
        cleaned_query 
    ) 
 
 
    # ========================================================
    # SEARCH 
    # ======================================================== 
 
    with st.spinner( 
        "Understanding your search..." 
    ): 
 
 
        if visual_query: 
 
            if len( 
                visual_search_df 
            ) == 0: 
 
                st.info( 
                    "No visual screenshots were found " 
                    "in this category." 
                ) 
 
                st.stop() 
 
 
            results = clip_search( 
                cleaned_query, 
                visual_search_df, 
                visual_search_embeddings 
            ) 
 
            search_method = ( 
                "🖼️ CLIP Visual Search" 
            ) 
 
 
        else: 
 
            if len( 
                semantic_search_df 
            ) == 0: 
 
                st.info( 
                    "No text-searchable screenshots " 
                    "were found in the selected category." 
                ) 
 
                st.stop() 
 
 
            results = semantic_search( 
                cleaned_query, 
                semantic_search_df, 
                semantic_search_embeddings 
            ) 
 
            search_method = ( 
                "🧠 Semantic Search" 
            ) 
 
 
    # ========================================================
    # IMPORTANT:
    # DO NOT USE A HARD CLIP THRESHOLD
    # ======================================================== 
 
    # CLIP similarity values can be lower than expected.
    # Therefore, we always show the best matches.
 
    relevant_results = ( 
        results 
        .head(number_of_results) 
        .copy() 
    ) 
 
 
    result_count = len( 
        relevant_results 
    ) 
 
 
    # ========================================================
    # RESULTS HEADER 
    # ======================================================== 
 
    st.divider() 
 
 
    st.markdown( 
        '<div class="result-title">' 
        '📌 Relevant Results' 
        '</div>', 
        unsafe_allow_html=True 
    ) 
 
 
    st.markdown( 
        f""" 
        <div class="result-count"> 
 
            <b>{result_count}</b> 
            result{"s" if result_count != 1 else ""} 
 
            &nbsp; • &nbsp; 
 
            {search_method} 
 
            &nbsp; • &nbsp; 
 
            Search: "{cleaned_query}" 
 
        </div> 
        """, 
        unsafe_allow_html=True 
    ) 
 
 
    # ========================================================
    # NO RESULTS 
    # ======================================================== 
 
    if result_count == 0: 
 
        st.info( 
            "No screenshots were found." 
        ) 
 
 
    # ========================================================
    # DISPLAY RESULTS 
    # ======================================================== 
 
    else: 
 
        result_columns = st.columns( 
            3, 
            gap="large" 
        ) 
 
 
        for index, (_, row) in enumerate( 
            relevant_results.iterrows() 
        ): 
 
 
            with result_columns[ 
                index % 3 
            ]: 
 
 
                try: 
 
                    # ================================================= 
                    # IMAGE PATH 
                    # ================================================= 
 
                    image_path = str( 
                        row["filepath"] 
                    ) 
 
 
                    image = Image.open( 
                        image_path 
                    ).convert("RGB") 
 
 
                    # ================================================= 
                    # IMAGE 
                    # ================================================= 
 
                    st.image( 
                        image, 
                        use_container_width=True 
                    ) 
 
 
                    # ================================================= 
                    # CATEGORY 
                    # ================================================= 
 
                    category_text = ( 
                        category_display.get( 
                            row["category"], 
                            row["category"] 
                        ) 
                    ) 
 
 
                    st.markdown( 
                        f""" 
                        <div class="result-category"> 
                            {category_text} 
                        </div> 
                        """, 
                        unsafe_allow_html=True 
                    ) 
 
 
                    # ================================================= 
                    # SIMILARITY 
                    # ================================================= 
 
                    similarity = float( 
                        row["similarity"] 
                    ) 
 
 
                    # Convert cosine score to a 
                    # safe display value. 
 
                    display_similarity = ( 
                        max( 
                            0.0, 
                            min( 
                                similarity, 
                                1.0 
                            ) 
                        ) 
                    ) 
 
 
                    st.progress( 
                        display_similarity 
                    ) 
 
 
                    st.markdown( 
                        f""" 
                        <div class="result-match"> 
                            🎯 Match: 
                            <b>{similarity * 100:.1f}%</b> 
                        </div> 
                        """, 
                        unsafe_allow_html=True 
                    ) 
 
 
                    # ================================================= 
                    # FILENAME 
                    # ================================================= 
 
                    filename = str( 
                        row.get( 
                            "filename", 
                            "Unknown" 
                        ) 
                    ) 
 
 
                    st.markdown( 
                        f""" 
                        <div class="result-filename"> 
                            📄 {filename} 
                        </div> 
                        """, 
                        unsafe_allow_html=True 
                    ) 
 
 
                    # ================================================= 
                    # OCR TEXT 
                    # ================================================= 
 
                    with st.expander( 
                        "📝 View OCR Text" 
                    ): 
 
                        ocr_text = row.get( 
                            "ocr_text", 
                            "" 
                        ) 
 
 
                        if ( 
                            pd.notna(ocr_text) 
                            and 
                            str( 
                                ocr_text 
                            ).strip() 
                            and 
                            str( 
                                ocr_text 
                            ).strip().lower() 
                            != "nan" 
                        ): 
 
                            st.text_area( 
                                "Extracted Text", 
                                value=str( 
                                    ocr_text 
                                )[:5000], 
                                height=250, 
                                disabled=True, 
                                label_visibility="collapsed" 
                            ) 
 
                        else: 
 
                            st.info( 
                                "No OCR text available " 
                                "for this screenshot." 
                            ) 
 
 
                    # ================================================= 
                    # FULL IMAGE 
                    # ================================================= 
 
                    with st.expander( 
                        "🖼️ View Full Image" 
                    ): 
 
                        st.image( 
                            image, 
                            use_container_width=True 
                        ) 
 
 
                except Exception as e: 
 
                    st.error( 
                        f"Could not open image: {e}" 
                    ) 
 
 
# ============================================================
# RECENT SEARCHES 
# ============================================================ 
 
st.divider() 
 
 
st.subheader( 
    "🕘 Recent Searches" 
) 
 
 
if len( 
    st.session_state.search_history 
) == 0: 
 
    st.caption( 
        "No searches yet." 
    ) 
 
 
else: 
 
    for previous_search in ( 
        st.session_state.search_history 
    ): 
 
        st.write( 
            f"🔎 {previous_search}" 
        ) 
 
 
    if st.button( 
        "🧹 Clear Search History" 
    ): 
 
        st.session_state.search_history = [] 
 
        st.rerun() 
 
 
# ============================================================
# FOOTER 
# ============================================================ 
 
st.divider() 
 
 
st.markdown( 
    """ 
    <div class="footer"> 
        Smart Screenshot Search Engine 
        • OCR + Semantic Search + CLIP Visual Search 
    </div> 
    """, 
    unsafe_allow_html=True 
)