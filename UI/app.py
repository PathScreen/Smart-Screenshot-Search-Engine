import streamlit as st 
import pandas as pd 
import numpy as np 
import torch 
import re 
import os
import gc
 
from sentence_transformers import SentenceTransformer 
from sklearn.metrics.pairwise import cosine_similarity 
from transformers import CLIPProcessor, CLIPModel 
from PIL import Image
import easyocr
DEPLOYED_MODE = False

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
 
 
# ============================================================
# PAGE CONFIG 
# ============================================================ 
 
st.set_page_config( 
    page_title="VisionX", 
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
# LOAD SEMANTIC MODEL LAZILY
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
# LOAD CLIP LAZILY
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
# RELEASE AI MODELS FROM MEMORY
# ============================================================

def release_ai_models():

    try:
        load_semantic_model.clear()
    except Exception:
        pass

    try:
        load_clip_model.clear()
    except Exception:
        pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
   
# ============================================================
# LOAD OCR READER
# ============================================================

def load_ocr_reader():

    model_dir = os.path.join(
        PROJECT_ROOT,
        "easyocr_models"
    )

    return easyocr.Reader(
        ["en"],
        gpu=False,
        model_storage_directory=model_dir,
        download_enabled=False
    )
# ============================================================
# OCR FUNCTION FOR UPLOADED IMAGE
# ============================================================

def extract_ocr_text(image_path):

    reader = load_ocr_reader()

    try:

        results = reader.readtext(
            image_path,
            detail=0,
            paragraph=True
        )

        return " ".join(results).strip()

    finally:

        del reader
        gc.collect()

# ============================================================
# GENERATE SEMANTIC EMBEDDING
# ============================================================

def generate_semantic_embedding(text):

    semantic_model = load_semantic_model()

    try:

        embedding = semantic_model.encode(
            text,
            convert_to_numpy=True
        )

        return embedding.astype(np.float32)

    finally:

        del semantic_model
        release_ai_models()

# ============================================================
# EMBEDDING FUNCTIONS FOR UPLOADED IMAGE
# ============================================================

def generate_visual_embedding(image):

    clip_model, clip_processor, clip_device = (
        load_clip_model()
    )

    try:

        inputs = clip_processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(clip_device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            vision_outputs = clip_model.vision_model(
                pixel_values=inputs["pixel_values"]
            )

            pooled_output = (
                vision_outputs.pooler_output
            )

            image_features = (
                clip_model.visual_projection(
                    pooled_output
                )
            )

        embedding = (
            image_features
            .cpu()
            .numpy()[0]
        )

        embedding = (
            embedding
            / np.linalg.norm(embedding)
        )

        return embedding.astype(np.float32)

    finally:

        del clip_model
        del clip_processor
        release_ai_models()

# ============================================================
# PROCESS AND SAVE UPLOADED IMAGE
# ============================================================

def process_uploaded_image(uploaded_file, category):

    # --------------------------------------------------------
    # DATA PATHS
    # --------------------------------------------------------

    category_folder = os.path.join(
        PROJECT_ROOT,
        "Data",
        category
    )

    # --------------------------------------------------------
    # FILE NAME VALIDATION
    # --------------------------------------------------------

    original_name = os.path.basename(
        uploaded_file.name
    )

    name, extension = os.path.splitext(
        original_name
    )

    extension = extension.lower()

    if extension not in [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    ]:

        raise ValueError(
            "Only JPG, JPEG, PNG and WEBP images are supported."
        )

    # --------------------------------------------------------
    # LOAD CURRENT DATA FIRST
    # --------------------------------------------------------

    current_semantic_df = pd.read_csv(
        SEMANTIC_METADATA_PATH
    )

    current_visual_df = pd.read_csv(
        VISUAL_METADATA_PATH
    )

    current_semantic_embeddings = np.load(
        SEMANTIC_EMBEDDINGS_PATH
    )

    current_visual_embeddings = np.load(
        VISUAL_EMBEDDINGS_PATH
    )

    # --------------------------------------------------------
    # SAFETY CHECK:
    # CSV AND NPY MUST ALREADY MATCH
    # --------------------------------------------------------

    if len(current_semantic_df) != len(
        current_semantic_embeddings
    ):

        raise ValueError(
            "Semantic metadata and embeddings are already "
            "mismatched. Upload cancelled."
        )

    if len(current_visual_df) != len(
        current_visual_embeddings
    ):

        raise ValueError(
            "Visual metadata and embeddings are already "
            "mismatched. Upload cancelled."
        )

    # --------------------------------------------------------
    # CHECK EMBEDDING DIMENSIONS
    # --------------------------------------------------------

    if current_semantic_embeddings.ndim != 2:

        raise ValueError(
            "Semantic embeddings must be a 2-dimensional array."
        )

    if current_visual_embeddings.ndim != 2:

        raise ValueError(
            "Visual embeddings must be a 2-dimensional array."
        )

    if current_semantic_embeddings.shape[1] != 384:

        raise ValueError(
            "Semantic embedding dataset must contain 384-dimensional embeddings."
        )

    if current_visual_embeddings.shape[1] != 512:

        raise ValueError(
            "Visual embedding dataset must contain 512-dimensional embeddings."
        )

    # --------------------------------------------------------
    # CHECK METADATA COLUMNS
    # --------------------------------------------------------

    required_semantic_columns = {
        "filename",
        "filepath",
        "category",
        "ocr_text"
    }

    required_visual_columns = {
        "filename",
        "filepath",
        "category"
    }

    if not required_semantic_columns.issubset(
        current_semantic_df.columns
    ):

        raise ValueError(
            "Semantic metadata CSV does not contain "
            "the expected columns."
        )

    if not required_visual_columns.issubset(
        current_visual_df.columns
    ):

        raise ValueError(
            "Visual metadata CSV does not contain "
            "the expected columns."
        )

    # --------------------------------------------------------
    # CREATE CATEGORY FOLDER
    # --------------------------------------------------------

    os.makedirs(
        category_folder,
        exist_ok=True
    )

    # --------------------------------------------------------
    # AVOID FILE NAME COLLISION
    # --------------------------------------------------------

    filename = original_name

    image_path = os.path.join(
        category_folder,
        filename
    )

    counter = 1
    filename_was_changed = False
 
    while os.path.exists(image_path):

        filename = (
            f"{name}_{counter}{extension}"
        )

        image_path = os.path.join(
            category_folder,
            filename
        )
 
        counter += 1
        filename_was_changed = True

    # --------------------------------------------------------
    # SAVE IMAGE
    # --------------------------------------------------------

    with open(
        image_path,
        "wb"
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )

    try:

        # ----------------------------------------------------
        # VERIFY IMAGE
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        ocr_text = extract_ocr_text(
            image_path
        )

        if not ocr_text:

            ocr_text = ""

        # ----------------------------------------------------
        # SEMANTIC EMBEDDING
        # ----------------------------------------------------

        semantic_embedding = (
            generate_semantic_embedding(
                ocr_text
            )
        )

        # ----------------------------------------------------
        # VISUAL EMBEDDING
        # ----------------------------------------------------

        visual_embedding = (
            generate_visual_embedding(
                image
            )
        )
        
        del image
        gc.collect()

        # ----------------------------------------------------
        # VERIFY NEW EMBEDDING DIMENSIONS
        # ----------------------------------------------------

        if semantic_embedding.shape[0] != 384:

            raise ValueError(
                "Generated semantic embedding does not have 384 dimensions."
            )

        if visual_embedding.shape[0] != 512:

            raise ValueError(
                "Generated visual embedding does not have 512 dimensions."
            )

        # ----------------------------------------------------
        # RELATIVE FILE PATH
        # ----------------------------------------------------

        relative_path = os.path.join(
            "Data",
            category,
            filename
        ).replace("\\", "/")

        # ----------------------------------------------------
        # CREATE NEW METADATA ROWS
        # ----------------------------------------------------

        semantic_row = {
            "filename": filename,
            "filepath": relative_path,
            "category": category,
            "ocr_text": ocr_text
        }

        visual_row = {
            "filename": filename,
            "filepath": relative_path,
            "category": category
        }

        # ----------------------------------------------------
        # APPEND METADATA
        # ----------------------------------------------------

        semantic_new_row = pd.DataFrame(
            [semantic_row],
            columns=current_semantic_df.columns
        )

        visual_new_row = pd.DataFrame(
            [visual_row],
            columns=current_visual_df.columns
        )

        updated_semantic_df = pd.concat(
            [
                current_semantic_df,
                semantic_new_row
            ],
            ignore_index=True
        )

        updated_visual_df = pd.concat(
            [
                current_visual_df,
                visual_new_row
            ],
            ignore_index=True
        )

        # ----------------------------------------------------
        # APPEND EMBEDDINGS
        # ----------------------------------------------------

        updated_semantic_embeddings = np.vstack(
            [
                current_semantic_embeddings,
                semantic_embedding
            ]
        )

        updated_visual_embeddings = np.vstack(
            [
                current_visual_embeddings,
                visual_embedding
            ]
        )

        # ----------------------------------------------------
        # FINAL SYNCHRONIZATION CHECK
        # ----------------------------------------------------

        if len(updated_semantic_df) != len(
            updated_semantic_embeddings
        ):

            raise ValueError(
                "Semantic data could not be synchronized."
            )

        if len(updated_visual_df) != len(
            updated_visual_embeddings
        ):

            raise ValueError(
                "Visual data could not be synchronized."
            )
        gc.collect()

        # ----------------------------------------------------
        # CREATE BACKUPS
        # ----------------------------------------------------

        backup_paths = {

            "semantic_csv":
                SEMANTIC_METADATA_PATH + ".backup",

            "semantic_npy":
                SEMANTIC_EMBEDDINGS_PATH + ".backup",

            "visual_csv":
                VISUAL_METADATA_PATH + ".backup",

            "visual_npy":
                VISUAL_EMBEDDINGS_PATH + ".backup"
        }

        # ----------------------------------------------------
        # COPY CURRENT FILES TO BACKUP
        # ----------------------------------------------------

        import shutil

        shutil.copy2(
            SEMANTIC_METADATA_PATH,
            backup_paths["semantic_csv"]
        )

        shutil.copy2(
            SEMANTIC_EMBEDDINGS_PATH,
            backup_paths["semantic_npy"]
        )

        shutil.copy2(
            VISUAL_METADATA_PATH,
            backup_paths["visual_csv"]
        )

        shutil.copy2(
            VISUAL_EMBEDDINGS_PATH,
            backup_paths["visual_npy"]
        )

        try:

            # ------------------------------------------------
            # SAVE UPDATED SEMANTIC DATA
            # ------------------------------------------------

            updated_semantic_df.to_csv(
                SEMANTIC_METADATA_PATH,
                index=False
            )

            np.save(
                SEMANTIC_EMBEDDINGS_PATH,
                updated_semantic_embeddings
            )

            # ------------------------------------------------
            # SAVE UPDATED VISUAL DATA
            # ------------------------------------------------

            updated_visual_df.to_csv(
                VISUAL_METADATA_PATH,
                index=False
            )

            np.save(
                VISUAL_EMBEDDINGS_PATH,
                updated_visual_embeddings
            )

        except Exception:

            # ------------------------------------------------
            # RESTORE ORIGINAL FILES
            # ------------------------------------------------

            shutil.copy2(
                backup_paths["semantic_csv"],
                SEMANTIC_METADATA_PATH
            )

            shutil.copy2(
                backup_paths["semantic_npy"],
                SEMANTIC_EMBEDDINGS_PATH
            )

            shutil.copy2(
                backup_paths["visual_csv"],
                VISUAL_METADATA_PATH
            )

            shutil.copy2(
                backup_paths["visual_npy"],
                VISUAL_EMBEDDINGS_PATH
            )

            raise

        finally:

            # ------------------------------------------------
            # REMOVE TEMPORARY BACKUPS
            # ------------------------------------------------

            for backup_file in backup_paths.values():

                if os.path.exists(backup_file):

                    os.remove(backup_file)

        # ----------------------------------------------------
        # CLEAR CACHED DATA
        # ----------------------------------------------------

        load_semantic_metadata.clear()
        load_semantic_embeddings.clear()

        load_visual_metadata.clear()
        load_visual_embeddings.clear()

        return {
            "filename": filename,
            "original_filename": original_name,
            "category": category,
            "ocr_text": ocr_text,
            "filename_was_changed": filename_was_changed
        }


    except Exception:

        # ----------------------------------------------------
        # REMOVE UPLOADED IMAGE IF PROCESSING FAILED
        # ----------------------------------------------------

        if os.path.exists(image_path):

            os.remove(image_path)

        raise

# ============================================================
# LOAD DATA INDEXES
# ============================================================

semantic_df = load_semantic_metadata()
semantic_embeddings = load_semantic_embeddings()

visual_df = load_visual_metadata()
visual_embeddings = load_visual_embeddings() 
       
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


# Create matching key using category + filename
def get_match_key(row):

    return (
        str(row["category"]).strip().lower()
        + "|"
        + get_filename_key(row["filename"])
    )


# Create matching keys
semantic_df["_match_key"] = semantic_df.apply(
    get_match_key,
    axis=1
)

visual_df["_match_key"] = visual_df.apply(
    get_match_key,
    axis=1
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
    '🔍VisionX' 
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
    "🔎 Search Modes",
    "2"
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

    if not DEPLOYED_MODE:

        # ========================================================
        # UPLOAD NEW IMAGE
        # ========================================================

        st.markdown(
            '<div class="sidebar-title">'
            '📤 Add New Screenshot'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <style>

            /* Hide the Material icon glyph */
            div[data-testid="stFileUploader"] [data-testid="stIconMaterial"] {
                display: none !important;
            }

            /* Style the upload button */
            div[data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"] {
                width: 100% !important;
                box-sizing: border-box !important;
                white-space: nowrap !important;
                padding: 8px 14px !important;
                border-radius: 8px !important;
                background: #172033 !important;
                border: none !important;
            }

            div[data-testid="stFileUploader"] button:hover {
                background: #303b55 !important;
            }

            /* Label text */
            div[data-testid="stFileUploader"] button p {
                margin: 0 !important;
                color: #ffffff !important;
                font-size: 13px !important;
                font-weight: 600 !important;
            }

            </style>
            """,
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )

        upload_category_display = st.selectbox(
            "📂 Select Category",
            [
                category_display[category]
                for category in category_display
            ],
            key="upload_category"
        )

        upload_category = next(
            (
                key
                for key, value in category_display.items()
                if value == upload_category_display
            ),
            None
        )

        if st.button(
            "📤 Add to Search Engine",
            width="stretch"
        ):

            if uploaded_file is None:

                st.warning(
                    "Please select an image first."
                )

            else:

                try:

                    with st.spinner(
                        "Processing image... OCR + AI embeddings"
                    ):

                        upload_result = process_uploaded_image(
                            uploaded_file,
                            upload_category
                        )

                    if upload_result["filename_was_changed"]:

                        st.warning(
                            f"⚠️ {upload_result['original_filename']} already existed.\n\n"
                            f"Saved as {upload_result['filename']} instead."
                        )

                    else:

                        st.success(
                            f"'{upload_result['filename']}' "
                            "was added successfully!"
                        )

                    if upload_result["ocr_text"]:

                        st.info(
                            "OCR text extracted successfully."
                        )

                        st.caption(
                            "Extracted OCR: "
                            + upload_result["ocr_text"]
                        )

                    else:

                        st.info(
                            "Image added. No readable text was detected."
                        )

                    st.rerun()

                except Exception as e:

                    st.error(
                        f"Upload failed: {e}"
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
        width="stretch"
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
        width="stretch" 
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

    semantic_model = load_semantic_model()

    try:

        query_embedding = (
            semantic_model.encode(
                [query],
                convert_to_numpy=True
            )
        )

    finally:

        del semantic_model
    similarities = (
        cosine_similarity(
            query_embedding,
            search_embeddings
        )[0]
    )

    results = search_df.copy()

    results["similarity"] = similarities

    # --------------------------------------------------------
    # OCR EXACT MATCH BOOST
    # --------------------------------------------------------

    query_words = set(
        clean_query(query).split()
    )

    def calculate_ocr_boost(text):

        text = clean_query(text)

        if not text:
            return 0.0

        ocr_words = set(text.split())

        # Exact word match
        exact_matches = (
            query_words.intersection(ocr_words)
        )

        if exact_matches:
            return 1.0

        # Query appears as a complete phrase
        if clean_query(query) in text:
            return 1.0

        return 0.0

    results["ocr_boost"] = (
        results["ocr_text"]
        .fillna("")
        .apply(calculate_ocr_boost)
    )

    # --------------------------------------------------------
    # FINAL HYBRID TEXT SCORE
    # --------------------------------------------------------

    results["final_score"] = (
        results["similarity"] * 0.7
        +
        results["ocr_boost"] * 0.3
    )

    return results.sort_values(
        ["ocr_boost", "final_score"],
        ascending=[False, False]
    )
 
 
# ============================================================
# CLIP SEARCH 
# ============================================================ 
 
def clip_search(
    query,
    search_df,
    search_embeddings
):

    clip_model, clip_processor, clip_device = (
        load_clip_model()
    )

    try:

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

    finally:

        del clip_model
        del clip_processor

        if "inputs" in locals():
            del inputs

        if "image_embeddings" in locals():
            del image_embeddings

        if "query_embedding" in locals():
            del query_embedding

        if "output" in locals():
            del output

        if "similarities" in locals():
            del similarities
 
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
        f"**{result_count}** "
        f"result{'s' if result_count != 1 else ''} "
        f"• {search_method} "
        f"• Search: \"{cleaned_query}\""
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
 
                    image_path = os.path.join(
                        PROJECT_ROOT,
                        str(row["filepath"]).replace("\\", "/")
                    )

                    image = Image.open(
                       image_path
                    ).convert("RGB")

                    # ================================================= 
                    # IMAGE 
                    # ================================================= 
 
                    st.image( 
                        image, 
                        width="stretch" 
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
                                label_visibility="collapsed",
                                key=f"ocr_text_{index}_{row['filename']}"
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
                            width="stretch" 
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