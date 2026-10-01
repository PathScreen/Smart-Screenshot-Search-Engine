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
    
    .favorite-button-container div.stButton > button {
        min-height: 32px !important;
        height: 32px !important;
        padding: 4px 10px !important;
        border-radius: 8px !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    } 
 
    div.stButton > button p, 
    div.stButton > button span { 
        color: #ffffff !important; 
    } 
    
    div[data-testid="stFormSubmitButton"] > button {
        min-height: 56px;
        border-radius: 15px;
        border: none;
        background: #172033;
        color: #ffffff !important;
        font-size: 15px;
        font-weight: 700;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #303b55;
    }

    div[data-testid="stFormSubmitButton"] > button p {
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
    "Tickets_Bookings": "🎟️ Tickets & Bookings",
    "Other_Uncategorized": "📁 Other / Uncategorized"
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

    embedding = semantic_model.encode(
        text,
        convert_to_numpy=True
    )

    return embedding.astype(np.float32)

# ============================================================
# EMBEDDING FUNCTIONS FOR UPLOADED IMAGE
# ============================================================

def generate_visual_embedding(image):
    clip_model, clip_processor, clip_device = load_clip_model()

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

        pooled_output = vision_outputs.pooler_output

        image_features = clip_model.visual_projection(
            pooled_output
        )

    embedding = image_features.cpu().numpy()[0]

    embedding = embedding / np.linalg.norm(embedding)

    return embedding.astype(np.float32)

# ============================================================
# ATOMIC SAVE FUNCTIONS
# ============================================================

def atomic_save_csv(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_save_npy(arr, path):
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


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

        visual_embedding = generate_visual_embedding(image)

        release_ai_models()

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

            atomic_save_csv(
                updated_semantic_df,
                SEMANTIC_METADATA_PATH
            )

            atomic_save_npy(
                updated_semantic_embeddings,
                SEMANTIC_EMBEDDINGS_PATH
            )

            # ------------------------------------------------
            # SAVE UPDATED VISUAL DATA
            # ------------------------------------------------

            atomic_save_csv(
                updated_visual_df,
                VISUAL_METADATA_PATH
            )

            atomic_save_npy(
                updated_visual_embeddings,
                VISUAL_EMBEDDINGS_PATH
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
    'Search your screenshots using text, meaning, and visual content.'
    '</div>',
    unsafe_allow_html=True
) 
 
# ============================================================
# STATISTICS 
# ============================================================ 
 
total_images = len(visual_df) 
 
total_categories = 7
 
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
 
if "last_search" not in st.session_state:
    st.session_state.last_search = None

if "favorites" not in st.session_state:
    st.session_state.favorites = []

if "show_favorites" not in st.session_state:
    st.session_state.show_favorites = False

if "search_history" not in st.session_state:
    st.session_state.search_history = []

if "detect_duplicates" not in st.session_state:
    st.session_state.detect_duplicates = False

if "run_duplicate_scan" not in st.session_state:
    st.session_state.run_duplicate_scan = False

if "duplicate_groups" not in st.session_state:
    st.session_state.duplicate_groups = []

if "duplicate_threshold_used" not in st.session_state:
    st.session_state.duplicate_threshold_used = 95

if "duplicate_category_used" not in st.session_state:
    st.session_state.duplicate_category_used = "🗂️ All Categories"

if "duplicate_max_groups_used" not in st.session_state:
    st.session_state.duplicate_max_groups_used = 10

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
        if "upload_notice" in st.session_state:
            st.success(
                st.session_state.pop("upload_notice")
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
                        msg = (
                            f"⚠️ {upload_result['original_filename']} "
                            f"already existed. "
                            f"Saved as {upload_result['filename']}."
                        )
                    else:
                        msg = (
                            f"✅ '{upload_result['filename']}' "
                            f"was added successfully!"
                        )

                    if upload_result["ocr_text"]:
                        msg += (
                            "\n\nOCR text extracted: "
                            + upload_result["ocr_text"][:300]
                        )
                    else:
                        msg += (
                            "\n\nNo readable text was detected."
                        )

                    st.session_state.upload_notice = msg
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

    for category in category_display:
        category_options.append(
            category_display[category]
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

    sort_option = st.selectbox(
        "↕️ Sort by",
        [
            "Highest Match",
            "Lowest Match",
            "Filename A–Z",
            "Filename Z–A"
        ]
    )

    st.divider()


    # ========================================================
    # FAVORITES
    # ========================================================

    st.markdown(
        '<div class="sidebar-title">'
        '⭐ Favorites'
        '</div>',
        unsafe_allow_html=True
    )

    favorite_count = len(
        st.session_state.favorites
    )

    st.caption(
        f"{favorite_count} favorite"
        f"{'s' if favorite_count != 1 else ''} saved."
    )

    if st.button(
        "⭐ View Favorites",
        width="stretch"
    ):

        st.session_state.show_favorites = True


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

    st.divider()

    # ========================================================
    # DUPLICATE DETECTION SETTINGS
    # ========================================================

    st.markdown(
        '<div class="sidebar-title">'
        '🔍 Duplicate Detection'
        '</div>',
        unsafe_allow_html=True
    )

    duplicate_threshold = st.slider(
        "Similarity threshold",
        min_value=90,
        max_value=99,
        value=95,
        step=1
    )

    duplicate_category_options = [
        "🗂️ All Categories"
    ]

    for category in category_display:

        duplicate_category_options.append(
            category_display[category]
        )

    duplicate_category_display = st.selectbox(
        "📂 Category",
        duplicate_category_options,
        key="duplicate_category"
    )

    duplicate_max_groups = st.selectbox(
        "🔢 Maximum groups",
        [
            5,
            10,
            20,
            50,
            "All"
        ],
        index=1
    )

    if st.button(
        "🔍 Detect Duplicates",
        width="stretch"
    ):

        st.session_state.detect_duplicates = True
        st.session_state.run_duplicate_scan = True

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
 
 
with st.form("search_form", clear_on_submit=False):

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

        search_button = st.form_submit_button(
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
            char in "aeiouy" 
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
            char in "aeiouy" 
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
 
STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for",
    "and", "or", "is", "are", "my", "me", "with", "from",
    "by", "show", "find", "all"
}


def semantic_search(query, search_df, search_embeddings):
    semantic_model = load_semantic_model()

    query_embedding = semantic_model.encode(
        [query],
        convert_to_numpy=True
    )

    similarities = cosine_similarity(
        query_embedding,
        search_embeddings
    )[0]

    results = search_df.copy()
    results["similarity"] = similarities

    cleaned = clean_query(query)

    query_words = {
        word
        for word in cleaned.split()
        if word not in STOPWORDS
    }

    def calculate_ocr_boost(text):
        text = clean_query(text)

        if not text or not query_words:
            return 0.0

        ocr_words = set(text.split())

        fraction = (
            len(query_words & ocr_words)
            / len(query_words)
        )

        if cleaned in text:
            fraction = 1.0

        return fraction

    results["ocr_boost"] = (
        results["ocr_text"]
        .fillna("")
        .apply(calculate_ocr_boost)
    )

    results["final_score"] = (
        results["similarity"] * 0.7
        +
        results["ocr_boost"] * 0.3
    )

    results["display_score"] = (
        results["final_score"]
        .clip(0, 1)
    )

    return results.sort_values(
        "final_score",
        ascending=False
    )

 
# ============================================================
# CLIP SEARCH 
# ============================================================ 
 
def clip_search(query, search_df, search_embeddings):
    clip_model, clip_processor, clip_device = load_clip_model()

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
        output = clip_model.get_text_features(**inputs)

    text_emb = (
        output.pooler_output
        if hasattr(output, "pooler_output")
        else output
    )

    text_emb = text_emb.detach().cpu().numpy()[0]
    text_emb = text_emb / np.linalg.norm(text_emb)

    img_emb = (
        search_embeddings
        / np.linalg.norm(
            search_embeddings,
            axis=1,
            keepdims=True
        )
    )

    similarities = img_emb @ text_emb

    results = search_df.copy()
    results["similarity"] = similarities

    LOW, HIGH = 0.15, 0.35

    results["display_score"] = (
        (results["similarity"] - LOW)
        / (HIGH - LOW)
    ).clip(0, 1)

    return results.sort_values(
        "similarity",
        ascending=False
    )

# ============================================================
# FIND SIMILAR SCREENSHOTS
# ============================================================

def find_similar_screenshots(row, top_n=5):

    target_mask = (
        (visual_df["filename"].astype(str) == str(row["filename"]))
        &
        (visual_df["category"].astype(str) == str(row["category"]))
    )

    matching_indices = visual_df.index[target_mask]

    if len(matching_indices) == 0:

        return pd.DataFrame()

    target_index = matching_indices[0]

    target_position = visual_df.index.get_loc(
        target_index
    )

    target_embedding = visual_embeddings[
        target_position
    ]

    embeddings = visual_embeddings.copy()

    embeddings = (
        embeddings
        /
        np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )
    )

    target_embedding = (
        target_embedding
        /
        np.linalg.norm(target_embedding)
    )

    similarities = embeddings @ target_embedding

    similarities[target_position] = -1

    similar_results = visual_df.copy()

    similar_results["similarity"] = similarities

    similar_results = (
        similar_results
        .sort_values(
            "similarity",
            ascending=False
        )
        .head(top_n)
        .copy()
    )

    return similar_results

# ============================================================
# DUPLICATE / NEAR-DUPLICATE DETECTION
# ============================================================

def detect_duplicate_groups(
    search_df,
    search_embeddings,
    similarity_threshold=0.95,
    max_groups=10,
    batch_size=256
):

    embeddings = search_embeddings.astype(
        np.float32,
        copy=True
    )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1.0

    embeddings = (
        embeddings
        / norms
    )

    total = len(embeddings)

    parent = np.arange(
        total
    )

    def find(x):

        while parent[x] != x:

            parent[x] = parent[
                parent[x]
            ]

            x = parent[x]

        return x

    def union(a, b):

        root_a = find(a)
        root_b = find(b)

        if root_a != root_b:

            parent[root_b] = root_a

    # --------------------------------------------------------
    # COMPARE EMBEDDINGS IN BATCHES
    # --------------------------------------------------------

    for start in range(
        0,
        total,
        batch_size
    ):

        end = min(
            start + batch_size,
            total
        )

        batch = embeddings[
            start:end
        ]

        similarities = (
            batch @ embeddings.T
        )

        for local_index in range(
            end - start
        ):

            global_index = (
                start
                + local_index
            )

            later_indices = np.arange(
                global_index + 1,
                total
            )

            if len(later_indices) == 0:
                continue

            row_similarities = (
                similarities[
                    local_index,
                    later_indices
                ]
            )

            matching_indices = (
                later_indices[
                    row_similarities
                    >= similarity_threshold
                ]
            )

            for match_index in matching_indices:

                union(
                    global_index,
                    int(match_index)
                )

    # --------------------------------------------------------
    # CREATE GROUPS
    # --------------------------------------------------------

    groups = {}

    for index in range(total):

        root = find(index)

        if root not in groups:

            groups[root] = []

        groups[root].append(
            index
        )

    duplicate_groups = []

    for indices in groups.values():

        if len(indices) >= 2:

            group_df = (
                search_df
                .iloc[indices]
                .copy()
            )

            duplicate_groups.append(
                group_df
            )

    # Largest groups first
    duplicate_groups.sort(
        key=len,
        reverse=True
    )

    # Limit number of groups
    duplicate_groups = (
        duplicate_groups[:max_groups]
    )

    return duplicate_groups

# ============================================================
# SEARCH EXECUTION
# ============================================================

MAX_STORED_RESULTS = 10


def run_search(cleaned_query):

    if not cleaned_query:

        return {
            "notice": (
                "warning",
                "Please enter something to search."
            )
        }


    if looks_like_gibberish(
        cleaned_query
    ):

        return {
            "notice": (
                "info",
                "No meaningful search detected. "
                "Try something like shopping bill, "
                "recipe, address, hotel booking, "
                "pizza, laptop, or ticket."
            )
        }


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

        return {
            "notice": (
                "warning",
                "No screenshots found in this category."
            )
        }


    # ========================================================
    # SEARCH TYPE
    # ========================================================

    visual_query = is_visual_query(
        cleaned_query
    )


    # ========================================================
    # SEARCH
    # ========================================================

    if visual_query:

        if len(visual_search_df) == 0:

            return {
                "notice": (
                    "info",
                    "No visual screenshots were found "
                    "in this category."
                )
            }


        results = clip_search(
            cleaned_query,
            visual_search_df,
            visual_search_embeddings
        )

        search_method = (
            "🖼️ CLIP Visual Search"
        )

    else:

        if len(semantic_search_df) == 0:

            return {
                "notice": (
                    "info",
                    "No text-searchable screenshots "
                    "were found in the selected category."
                )
            }


        results = semantic_search(
            cleaned_query,
            semantic_search_df,
            semantic_search_embeddings
        )

        search_method = (
            "🧠 Semantic Search"
        )


    return {
        "results": (
            results
            .head(MAX_STORED_RESULTS)
            .copy()
        ),
        "method": search_method,
        "query": cleaned_query
    }


# ============================================================
# RUN SEARCH
# ============================================================

if search_button:

    cleaned_query = clean_query(
        query
    )


    # --------------------------------------------------------
    # SEARCH HISTORY
    # --------------------------------------------------------

    if (
        cleaned_query
        and not looks_like_gibberish(
            cleaned_query
        )
    ):

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


    with st.spinner(
        "Understanding your search..."
    ):

        st.session_state.last_search = (
            run_search(
                cleaned_query
            )
        )


# ============================================================
# DISPLAY STORED RESULTS
# ============================================================

import html as html_lib


last_search = st.session_state.get(
    "last_search"
)


if last_search:

    st.divider()


    st.markdown(
        '<div class="result-title">'
        '📌 Relevant Results'
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # NOTICE
    # ========================================================

    if "notice" in last_search:

        level, message = (
            last_search["notice"]
        )

        getattr(
            st,
            level
        )(message)


    # ========================================================
    # RESULTS
    # ========================================================

    else:

        relevant_results = last_search["results"].copy()

        if sort_option == "Highest Match":
            relevant_results = relevant_results.sort_values(
                "display_score",
                ascending=False
            )

        elif sort_option == "Lowest Match":
            relevant_results = relevant_results.sort_values(
                "display_score",
                ascending=True
            )

        elif sort_option == "Filename A–Z":
            relevant_results = relevant_results.sort_values(
                "filename",
                ascending=True,
                key=lambda column: column.astype(str).str.lower()
            )

        elif sort_option == "Filename Z–A":
            relevant_results = relevant_results.sort_values(
                "filename",
                ascending=False,
                key=lambda column: column.astype(str).str.lower()
            )

        relevant_results = relevant_results.head(
            number_of_results
        )

        result_count = len(
            relevant_results
        )

        st.markdown(
            f"**{result_count}** "
            f"result{'s' if result_count != 1 else ''} "
            f"• {last_search['method']} "
            f"• Search: "
            f"\"{last_search['query']}\""
        )


        # ====================================================
        # NO RESULTS
        # ====================================================

        if result_count == 0:

            st.info(
                "No screenshots were found."
            )


        # ====================================================
        # DISPLAY RESULTS
        # ====================================================

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

                        # =====================================
                        # IMAGE PATH
                        # =====================================

                        image_path = os.path.join(
                            PROJECT_ROOT,
                            str(
                                row["filepath"]
                            ).replace(
                                "\\",
                                "/"
                            )
                        )


                        image = Image.open(
                            image_path
                        ).convert("RGB")


                        # =====================================
                        # IMAGE
                        # =====================================

                        st.image(
                            image,
                            width="stretch"
                        )


                        # =====================================
                        # FAVORITE
                        # =====================================

                        favorite_key = (
                            f"{row['category']}|"
                            f"{row['filename']}"
                        )

                        is_favorite = (
                            favorite_key
                            in st.session_state.favorites
                        )

                        with st.container():

                            st.markdown(
                                '<div class="favorite-button-container">',
                                unsafe_allow_html=True
                            )

                            if st.button(
                                "★ Favorited"
                                if is_favorite
                                else "☆ Favorite",
                                key=(
                                    f"favorite_{index}_"
                                    f"{row['filename']}"
                                )
                            ):

                                if is_favorite:

                                    st.session_state.favorites.remove(
                                        favorite_key
                                    )

                                else:

                                    st.session_state.favorites.append(
                                        favorite_key
                                    )

                                st.rerun()

                            st.markdown(
                                '</div>',
                                unsafe_allow_html=True
                            )


                        # =====================================
                        # CATEGORY
                        # =====================================

                        category_text = (
                            html_lib.escape(
                                category_display.get(
                                    row["category"],
                                    str(
                                        row["category"]
                                    )
                                )
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


                        # =====================================
                        # SCORE
                        # =====================================

                        score = float(
                            row.get(
                                "display_score",
                                row["similarity"]
                            )
                        )


                        score = max(
                            0.0,
                            min(
                                score,
                                1.0
                            )
                        )


                        st.progress(
                            score
                        )


                        st.markdown(
                            f"""
                            <div class="result-match">
                                🎯 Match:
                                <b>{score * 100:.1f}%</b>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                        # =====================================
                        # FILENAME
                        # =====================================

                        filename = (
                            html_lib.escape(
                                str(
                                    row.get(
                                        "filename",
                                        "Unknown"
                                    )
                                )
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


                        # =====================================
                        # OCR TEXT
                        # =====================================

                        with st.expander(
                            "📝 View OCR Text"
                        ):

                            ocr_text = row.get(
                                "ocr_text",
                                ""
                            )


                            if (
                                pd.notna(
                                    ocr_text
                                )
                                and str(
                                    ocr_text
                                ).strip()
                                and str(
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
                                    key=(
                                        f"ocr_text_{index}_"
                                        f"{row['filename']}"
                                    )
                                )

                            else:

                                st.info(
                                    "No OCR text available "
                                    "for this screenshot."
                                )


                        # =====================================
                        # FULL IMAGE
                        # =====================================

                        with st.expander(
                            "🖼️ View Full Image"
                        ):

                            st.image(
                                image,
                                width="stretch"
                            )


                        # =====================================
                        # FIND SIMILAR
                        # =====================================

                        if st.button(
                            "🔗 Find Similar",
                            key=(
                                f"find_similar_{index}_"
                                f"{row['filename']}"
                            ),
                            width="stretch"
                        ):

                            similar_results = (
                                find_similar_screenshots(
                                    row,
                                    top_n=5
                                )
                            )

                            if similar_results.empty:

                                st.info(
                                    "No similar screenshots found."
                                )

                            else:

                                st.markdown(
                                    "**🔗 Similar Screenshots**"
                                )

                                similar_columns = st.columns(
                                    5,
                                    gap="small"
                                )

                                for similar_index, (
                                    _,
                                    similar_row
                                ) in enumerate(
                                    similar_results.iterrows()
                                ):

                                    with similar_columns[
                                        similar_index % 5
                                    ]:

                                        similar_image_path = (
                                            os.path.join(
                                                PROJECT_ROOT,
                                                str(
                                                    similar_row[
                                                        "filepath"
                                                    ]
                                                ).replace(
                                                    "\\",
                                                    "/"
                                                )
                                            )
                                        )

                                        try:

                                            similar_image = (
                                                Image.open(
                                                    similar_image_path
                                                ).convert(
                                                    "RGB"
                                                )
                                            )

                                            st.image(
                                                similar_image,
                                                width="stretch"
                                            )

                                            similar_score = float(
                                                similar_row[
                                                    "similarity"
                                                ]
                                            )

                                            st.caption(
                                                f"{similar_score * 100:.1f}%"
                                            )

                                        except Exception:

                                            st.caption(
                                                "Image unavailable"
                                            )


                        # =====================================
                        # WHY THIS RESULT
                        # =====================================

                        with st.expander(
                            "🧠 Why this result?"
                        ):

                            st.markdown(
                                f"**Search method:** {last_search['method']}"
                            )

                            st.markdown(
                                f"**Your query:** `{last_search['query']}`"
                            )

                            st.markdown(
                                f"**Category:** {row.get('category', 'Unknown')}"
                            )

                            st.markdown(
                                f"**Match:** {score * 100:.1f}%"
                            )

                            if (
                                "Semantic" in last_search["method"]
                                and pd.notna(
                                    row.get(
                                        "ocr_text",
                                        ""
                                    )
                                )
                                and str(
                                    row.get(
                                        "ocr_text",
                                        ""
                                    )
                                ).strip()
                            ):

                                ocr_preview = str(
                                    row.get(
                                        "ocr_text",
                                        ""
                                    )
                                ).strip()

                                if len(ocr_preview) > 300:
                                    ocr_preview = (
                                        ocr_preview[:300]
                                        + "..."
                                    )

                                st.markdown(
                                    "**OCR content:**"
                                )

                                st.caption(
                                    ocr_preview
                                )

                            elif "Visual" in last_search["method"]:

                                st.caption(
                                    "This result was retrieved "
                                    "because its visual content "
                                    "is similar to your query."
                                )


                    except Exception:

                        st.error(
                            "Unable to display this result."
                        )
if (
    st.session_state.get(
        "run_duplicate_scan",
        False
    )
    and st.session_state.detect_duplicates
):

    if duplicate_category_display == "🗂️ All Categories":

        duplicate_search_df = visual_df.copy()

        duplicate_search_embeddings = (
            visual_embeddings
        )

    else:

        duplicate_category = next(
            (
                key
                for key, value
                in category_display.items()
                if value == duplicate_category_display
            ),
            None
        )

        duplicate_mask = (
            visual_df["category"]
            == duplicate_category
        )

        duplicate_search_df = visual_df[
            duplicate_mask
        ].copy()

        duplicate_search_embeddings = (
            visual_embeddings[
                duplicate_mask.values
            ]
        )

    if len(duplicate_search_df) >= 2:

        if duplicate_max_groups == "All":
            max_groups = len(
                duplicate_search_df
            )
        else:
            max_groups = int(
                duplicate_max_groups
            )

        with st.spinner(
            "Scanning screenshots for visual duplicates..."
        ):

            st.session_state.duplicate_groups = (
                detect_duplicate_groups(
                    duplicate_search_df,
                    duplicate_search_embeddings,
                    similarity_threshold=(
                        duplicate_threshold / 100
                    ),
                    max_groups=max_groups
                )
            )

    else:

        st.session_state.duplicate_groups = []

    st.session_state.duplicate_threshold_used = (
        duplicate_threshold
    )

    st.session_state.duplicate_category_used = (
        duplicate_category_display
    )

    st.session_state.duplicate_max_groups_used = (
        duplicate_max_groups
    )

    st.session_state.run_duplicate_scan = False


# ============================================================
# DUPLICATE / NEAR-DUPLICATE RESULTS
# ============================================================

if st.session_state.detect_duplicates:

    st.divider()

    st.markdown(
        '<div class="result-title">'
        '🔍 Duplicate / Near-Duplicate Screenshots'
        '</div>',
        unsafe_allow_html=True
    )

    st.caption(
        f"Visual similarity threshold: "
        f"{st.session_state.get('duplicate_threshold_used', duplicate_threshold)}%"
    )

    duplicate_groups = st.session_state.get(
        "duplicate_groups",
        []
    )

    if not duplicate_groups:

        st.success(
            "No duplicate or near-duplicate "
            "screenshots found."
        )

    else:

        st.markdown(
            f"**{len(duplicate_groups)}** "
            "duplicate group"
            f"{'s' if len(duplicate_groups) != 1 else ''} "
            "found."
        )

        for group_number, group_df in enumerate(
            duplicate_groups,
            start=1
        ):

            st.markdown(
                f"### Duplicate Group {group_number}"
            )

            st.caption(
                f"{len(group_df)} "
                "screenshots in this group"
            )

            duplicate_columns = st.columns(
                min(
                    len(group_df),
                    5
                ),
                gap="small"
            )

            for image_index, (
                _,
                duplicate_row
            ) in enumerate(
                group_df.iterrows()
            ):

                with duplicate_columns[
                    image_index % 5
                ]:

                    duplicate_image_path = os.path.join(
                        PROJECT_ROOT,
                        str(
                            duplicate_row[
                                "filepath"
                            ]
                        ).replace("\\", "/")
                    )

                    try:

                        duplicate_image = Image.open(
                            duplicate_image_path
                        ).convert("RGB")

                        st.image(
                            duplicate_image,
                            width="stretch"
                        )

                        st.caption(
                            str(
                                duplicate_row[
                                    "filename"
                                ]
                            )
                        )

                    except Exception:

                        st.caption(
                            "Image unavailable"
                        )

            st.divider()

    if st.button(
        "✖ Close Duplicate Detection",
        width="stretch"
    ):

        st.session_state.detect_duplicates = False
        st.session_state.run_duplicate_scan = False
        st.session_state.duplicate_groups = []

        st.rerun()
# ============================================================
# FAVORITES SECTION
# ============================================================

if st.session_state.show_favorites:

    st.divider()

    st.markdown(
        '<div class="result-title">'
        '⭐ Favorite Screenshots'
        '</div>',
        unsafe_allow_html=True
    )

    favorites = st.session_state.favorites

    if not favorites:

        st.info(
            "You have not added any screenshots to Favorites yet."
        )

    else:

        favorite_rows = []

        for favorite_key in favorites:

            try:

                favorite_category, favorite_filename = (
                    favorite_key.split(
                        "|",
                        1
                    )
                )

            except ValueError:

                continue

            matching_rows = visual_df[
                (
                    visual_df["category"]
                    .astype(str)
                    .str.lower()
                    == favorite_category.lower()
                )
                &
                (
                    visual_df["filename"]
                    .astype(str)
                    .str.lower()
                    == favorite_filename.lower()
                )
            ]

            if not matching_rows.empty:

                favorite_rows.append(
                    matching_rows.iloc[0]
                )

        if not favorite_rows:

            st.info(
                "No favorite screenshots are available."
            )

        else:

            favorite_columns = st.columns(
                3,
                gap="large"
            )

            for favorite_index, favorite_row in enumerate(
                favorite_rows
            ):

                with favorite_columns[
                    favorite_index % 3
                ]:

                    favorite_image_path = os.path.join(
                        PROJECT_ROOT,
                        str(
                            favorite_row["filepath"]
                        ).replace(
                            "\\",
                            "/"
                        )
                    )

                    try:

                        favorite_image = Image.open(
                            favorite_image_path
                        ).convert("RGB")

                        st.image(
                            favorite_image,
                            width="stretch"
                        )

                        favorite_category_text = (
                            category_display.get(
                                favorite_row["category"],
                                str(
                                    favorite_row["category"]
                                )
                            )
                        )

                        st.caption(
                            favorite_category_text
                        )

                        st.caption(
                            str(
                                favorite_row["filename"]
                            )
                        )

                        favorite_key = (
                            f"{favorite_row['category']}|"
                            f"{favorite_row['filename']}"
                        )

                        if st.button(
                            "★ Remove from Favorites",
                            key=(
                                f"remove_favorite_"
                                f"{favorite_index}_"
                                f"{favorite_row['filename']}"
                            ),
                            width="stretch"
                        ):

                            if favorite_key in (
                                st.session_state.favorites
                            ):

                                st.session_state.favorites.remove(
                                    favorite_key
                                )

                            st.rerun()

                    except Exception:

                        st.caption(
                            "Image unavailable"
                        )

    if st.button(
        "✖ Close Favorites",
        width="stretch"
    ):

        st.session_state.show_favorites = False

        st.rerun()

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
        VisionX 
        • OCR + Semantic Search + CLIP Visual Search 
    </div> 
    """, 
    unsafe_allow_html=True 
)