import streamlit as st
import google.generativeai as genai
import os
import json
import pandas as pd
import fitz  # PyMuPDF
import PyPDF2
import time
import warnings
from contextlib import redirect_stderr, redirect_stdout
import io
from io import BytesIO
import subprocess
import tempfile

# Suppress non-critical warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub")
warnings.filterwarnings("ignore", message="CropBox missing from /Page")
warnings.filterwarnings("ignore", message=".*CropBox.*")
warnings.filterwarnings("ignore", message=".*MediaBox.*")
warnings.filterwarnings("ignore", message=".*defaulting to MediaBox.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="streamlit")
warnings.filterwarnings("ignore", category=UserWarning, module="fitz")
warnings.filterwarnings("ignore", category=UserWarning, message=".*CropBox.*")

# Additional PyMuPDF specific filters
import sys
import logging
import os

# Suppress all PyMuPDF console output
if 'fitz' in sys.modules:
    logging.getLogger('fitz').setLevel(logging.CRITICAL)

# Environment variable to suppress PyMuPDF messages
os.environ['PYMUPDF_WARNINGS'] = '0'

# Additional fitz-specific suppression
try:
    import fitz
    # Suppress fitz internal warnings
    fitz.TOOLS.mupdf_warnings(False)
except:
    pass

def safe_get(dictionary, key, default_value):
    """Safely get value from dictionary with error protection"""
    try:
        if not isinstance(dictionary, dict):
            return default_value
        value = dictionary.get(key, default_value)
        return value if value is not None else default_value
    except Exception:
        return default_value

def clean_learned_library():
    """Clean learned library from corrupted entries"""
    if 'learned_library' not in st.session_state:
        st.session_state.learned_library = []
        return
    
    original_count = len(st.session_state.learned_library)
    cleaned_library = []
    
    for i, entry in enumerate(st.session_state.learned_library):
        try:
            # Validate entry structure
            if not isinstance(entry, dict):
                st.warning(f"Removing non-dict entry at index {i}")
                continue
            
            # Ensure required keys exist
            if 'filename' not in entry:
                st.warning(f"Adding missing filename to entry at index {i}")
                entry['filename'] = f'Unknown_{i}'
            
            # Validate other critical keys
            required_keys = ['id', 'timestamp', 'filename']
            for key in required_keys:
                if key not in entry:
                    entry[key] = f'Unknown_{key}_{i}'
            
            cleaned_library.append(entry)
            
        except Exception as e:
            st.warning(f"Removing corrupted entry at index {i}: {e}")
            continue
    
    st.session_state.learned_library = cleaned_library
    
    if original_count != len(cleaned_library):
        st.info(f"🧹 Cleaned library: {original_count} → {len(cleaned_library)} entries")

# Enhanced context manager để suppress ALL PyMuPDF console output
class SuppressPyMuPDFOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._original_warnings = warnings.showwarning
        
        # Redirect all output to null
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        # Suppress warnings completely
        warnings.showwarning = lambda *args: None
        
        # Additional suppression for fitz
        try:
            import fitz
            self._original_fitz_warnings = getattr(fitz, '_original_warning_func', None)
            if hasattr(fitz.TOOLS, 'mupdf_warnings'):
                fitz.TOOLS.mupdf_warnings(False)
        except:
            pass
            
        return self
    
    def __exit__(self, type, value, traceback):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        warnings.showwarning = self._original_warnings
        
        # Restore fitz warnings if needed
        try:
            import fitz
            if hasattr(fitz.TOOLS, 'mupdf_warnings'):
                fitz.TOOLS.mupdf_warnings(True)
        except:
            pass

# Safe import MarkItDown với fallback
MARKITDOWN_AVAILABLE = False
try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
    # Remove: st.success("🚀 MarkItDown loaded successfully!")
except ImportError as e:
    # Only show error if needed for debugging, otherwise silent
    # st.warning(f"⚠️ MarkItDown import failed: {e}")
    # st.info("📄 Fallback to PDF-only mode with PyMuPDF + PyPDF2")
    MarkItDown = None

# Cấu hình trang
st.set_page_config(
    page_title="📚 Systematic Literature Review Tool",
    page_icon="📊",
    layout="wide"
)

# Learned Library Management
LEARNED_LIBRARY_FILE = "learned_library.json"

def load_learned_library():
    """Load learned library from JSON file"""
    try:
        if os.path.exists(LEARNED_LIBRARY_FILE):
            with open(LEARNED_LIBRARY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Lỗi khi load learned library: {e}")
    return []

def save_learned_library(learned_data):
    """Save learned library to JSON file"""
    try:
        with open(LEARNED_LIBRARY_FILE, 'w', encoding='utf-8') as f:
            json.dump(learned_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"Lỗi khi save learned library: {e}")
        return False

def add_to_learned_library(analysis_result, metadata=None):
    """Add analysis result to learned library with metadata"""
    try:
        # Clean library first to remove any corrupted entries
        clean_learned_library()
        
        # --- BẮT ĐẦU ĐOẠN CODE CẦN THỤT VÀO ---
        if not analysis_result:
            st.warning("Empty analysis result passed to library")
            return None
        
        # Debug: Show analysis_result structure
        st.write("🔍 Debug - Analysis result type:", type(analysis_result))
        if isinstance(analysis_result, dict):
            st.write("🔍 Debug - Analysis result keys:", list(analysis_result.keys()))
            st.write("🔍 Debug - Has filename key:", 'filename' in analysis_result)
        else:
            st.write("🔍 Debug - Analysis result value:", str(analysis_result)[:100])
            return None
        
        # Validate analysis_result type
        if not isinstance(analysis_result, dict):
            st.error(f"Expected dict, got {type(analysis_result).__name__}")
            return None
        
        # Safe get filename with multiple fallbacks
        filename = None
        try:
            if 'filename' in analysis_result:
                filename = analysis_result['filename']
            else:
                filename = safe_get(analysis_result, 'filename', 'Unknown')
        except Exception as e:
            st.warning(f"Error getting filename: {e}")
            filename = 'Unknown'
        
        # Final filename validation
        if not filename or not isinstance(filename, str):
            filename = str(filename) if filename is not None else 'Unknown'
        
        st.write(f"🔍 Debug - Final filename: '{filename}'")
    
        # Create learned entry
        learned_entry = {
            "id": f"learned_{int(time.time())}_{len(st.session_state.learned_library)}",
            "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            "filename": filename,
            "title": safe_get(analysis_result, 'tieu_de', 'No title'),
            "authors": safe_get(analysis_result, 'tac_gia', 'Unknown authors'),
            "year": safe_get(analysis_result, 'nam_xuat_ban', 'Unknown year'),
            "field": safe_get(analysis_result, 'linh_vuc_nc', 'Unknown field'),
            "theories": safe_get(analysis_result, 'ly_thuyet_nen', 'No theories'),
            "variables_independent": safe_get(analysis_result, 'bien_doc_lap', 'No variables'),
            "variables_dependent": safe_get(analysis_result, 'bien_phu_thuoc', 'No variables'),
            "methodology": safe_get(analysis_result, 'pp_phan_tich', 'No method'),
            "findings": safe_get(analysis_result, 'ket_qua_nc', 'No findings'),
            "limitations": safe_get(analysis_result, 'han_che_huong_tiep', 'No limitations'),
            "scales": safe_get(analysis_result, 'thang_do', 'No scales'),
            "doi": safe_get(analysis_result, 'doi', 'No DOI'),
            "full_data": analysis_result,  # Store complete analysis for reference
            "metadata": metadata or {}
        }
    
        # Check if already exists (by filename) - with safety checks for old entries
        existing_index = None
        for i, entry in enumerate(st.session_state.learned_library):
            try:
                # Safe access to filename in existing entries
                entry_filename = entry.get('filename', 'Unknown') if isinstance(entry, dict) else 'Unknown'
                if entry_filename == learned_entry['filename']:
                    existing_index = i
                    break
            except Exception as entry_error:
                # Skip corrupted entries
                st.warning(f"Skipping corrupted entry at index {i}: {entry_error}")
                continue
    
        if existing_index is not None:
            # Update existing entry
            st.session_state.learned_library[existing_index] = learned_entry
        else:
            # Add new entry
            st.session_state.learned_library.append(learned_entry)
    
        return learned_entry
        # --- KẾT THÚC ĐOẠN CODE CẦN THỤT VÀO ---
    
    except Exception as e:
        st.error(f"Critical error in add_to_learned_library: {e}")
        st.error(f"Error type: {type(e).__name__}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

def search_learned_library(query, search_fields=None):
    """Search learned library by query"""
    if not query:
        return st.session_state.learned_library
    
    query = query.lower()
    if search_fields is None:
        search_fields = ['title', 'authors', 'field', 'theories', 'variables_independent', 
                        'variables_dependent', 'methodology', 'findings', 'filename']
    
    results = []
    for entry in st.session_state.learned_library:
        for field in search_fields:
            field_value = str(entry.get(field, '')).lower()
            if query in field_value:
                results.append(entry)
                break
    
    return results

def get_learned_stats():
    """Get statistics from learned library"""
    if not st.session_state.learned_library:
        return {}
    
    stats = {
        'total_documents': len(st.session_state.learned_library),
        'unique_fields': len(set([entry.get('field', '').strip() for entry in st.session_state.learned_library if entry.get('field', '').strip()])),
        'years_range': [],
        'top_theories': {},
        'top_methods': {},
        'recent_additions': []
    }
    
    # Years analysis
    years = []
    for entry in st.session_state.learned_library:
        year_str = str(entry.get('year', ''))
        if year_str.isdigit():
            years.append(int(year_str))
    
    if years:
        stats['years_range'] = [min(years), max(years)]
    
    # Top theories
    theories_count = {}
    for entry in st.session_state.learned_library:
        theories = str(entry.get('theories', '')).strip()
        if theories and theories.lower() not in ['no information', 'không có thông tin', 'no theories']:
            # Split multiple theories
            theory_list = [t.strip() for t in theories.replace(',', ';').split(';')]
            for theory in theory_list:
                if theory:
                    theories_count[theory] = theories_count.get(theory, 0) + 1
    
    stats['top_theories'] = dict(sorted(theories_count.items(), key=lambda x: x[1], reverse=True)[:10])
    
    # Top methods
    methods_count = {}
    for entry in st.session_state.learned_library:
        method = str(entry.get('methodology', '')).strip()
        if method and method.lower() not in ['no information', 'không có thông tin', 'no method']:
            methods_count[method] = methods_count.get(method, 0) + 1
    
    stats['top_methods'] = dict(sorted(methods_count.items(), key=lambda x: x[1], reverse=True)[:10])
    
    # Recent additions (last 10)
    stats['recent_additions'] = sorted(st.session_state.learned_library, 
                                     key=lambda x: x.get('timestamp', ''), reverse=True)[:10]
    
    return stats

@st.cache_resource
def init_gemini(model_name='gemini-2.5-flash', max_retries=2):
    """Khởi tạo Gemini AI với model được chọn và cơ chế retry"""
    # Danh sách API keys (sẽ luân chuyển khi gặp lỗi quota)
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = [
            "AIzaSyBNU2IteZpqb93aISVU38Z0fN9r_Wc3_qs",  # Original key
            "AIzaSyC9B1zWj45gYuHpvuOy2EcAtxURi91-Gb4",  # New key 1
            "AIzaSyDx8MdE88VtSA1_ha0HsUY84V4LFfbpwp0"   # New key 2
        ]
        st.session_state.current_key_index = 0
    
    # Lấy API key hiện tại từ danh sách
    api_key = st.session_state.api_keys[st.session_state.current_key_index]
    
    # Thử khởi tạo model với retry
    for attempt in range(max_retries + 1):
        try:
            genai.configure(api_key=api_key)
            
            # Cấu hình safety settings để giảm thiểu lỗi
            safety_settings = {
                "HARASSMENT": "block_none",
                "HATE": "block_none",
                "SEXUAL": "block_none",
                "DANGEROUS": "block_none"
            }
            
            # Tạo model với cấu hình đơn giản và an toàn
            generation_config = {
                "temperature": 0.2,  # Giảm temperature để tăng tính nhất quán
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,  # Tăng token để đảm bảo đủ dữ liệu
            }
            
            # Tạo model - thử với và không với cấu hình tùy chỉnh
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                return model
            except Exception as config_error:
                # Nếu lỗi với cấu hình tùy chỉnh, thử lại với cấu hình mặc định
                st.warning(f"Thử lại với cấu hình mặc định: {str(config_error)}")
                model = genai.GenerativeModel(model_name=model_name)
                return model
                
        except Exception as e:
            error_msg = str(e).lower()
            
            # Kiểm tra lỗi quota hoặc rate limit
            if "quota" in error_msg or "rate limit" in error_msg or "429" in error_msg:
                # Thử chuyển sang API key tiếp theo
                num_keys = len(st.session_state.api_keys)
                old_key_index = st.session_state.current_key_index
                st.session_state.current_key_index = (old_key_index + 1) % num_keys
                new_key_index = st.session_state.current_key_index
                
                # Nếu đã thử hết tất cả các key
                if new_key_index == old_key_index:
                    if attempt < max_retries:
                        # Đợi thêm thời gian trước khi thử lại
                        wait_time = (attempt + 1) * 2  # 2s, 4s, ...
                        st.warning(f"Tất cả API keys đều bị giới hạn, đợi {wait_time}s và thử lại...")
                        time.sleep(wait_time)
                        continue
                else:
                    # Thông báo đang chuyển sang key mới
                    new_key = st.session_state.api_keys[new_key_index]
                    masked_key = new_key[:6] + "..." + new_key[-4:]
                    st.info(f"Đang chuyển sang API key khác: {masked_key}")
                    
                    # Cập nhật API key và thử lại
                    api_key = new_key
                    continue
            
            # Các lỗi khác
            st.error(f"Lỗi khi khởi tạo Gemini AI với model {model_name}: {str(e)}")
            return None
            
    # Nếu đã thử hết tất cả các lần và không thành công
    return None

@st.cache_resource
def init_markitdown():
    """Khởi tạo MarkItDown converter với fallback handling"""
    if not MARKITDOWN_AVAILABLE:
        return None
    try:
        return MarkItDown()
    except Exception as e:
        st.error(f"Lỗi khi khởi tạo MarkItDown: {str(e)}")
        return None

def extract_text_from_document(uploaded_file):
    """Trích xuất text từ multiple file formats using MarkItDown hoặc fallback"""
    file_type = uploaded_file.type
    file_name = uploaded_file.name.lower()
    
    # Nếu MarkItDown available, thử sử dụng
    if MARKITDOWN_AVAILABLE:
        try:
            md = init_markitdown()
            if md:
                uploaded_file.seek(0)
                result = md.convert_stream(uploaded_file)
                if result and result.text_content and result.text_content.strip():
                    # Determine extraction method based on file type - but don't show technical details
                    # if file_type == "application/pdf":
                    #     st.success("✅ MarkItDown: PDF extracted with structure preservation")
                    # elif "word" in file_type or file_name.endswith(('.docx', '.doc')):
                    #     st.success("✅ MarkItDown: Word document converted to Markdown")
                    # elif "powerpoint" in file_type or file_name.endswith(('.pptx', '.ppt')):
                    #     st.success("✅ MarkItDown: PowerPoint extracted with slides structure")
                    # elif "excel" in file_type or file_name.endswith(('.xlsx', '.xls')):
                    #     st.success("✅ MarkItDown: Excel data extracted with table structure")
                    # else:
                    #     st.success("✅ MarkItDown: Document extracted successfully")
                    
                    return result.text_content
            
        except Exception as e:
            # st.warning(f"MarkItDown failed: {e}. Using fallback methods...")
            pass
    else:
        # st.info("📄 Using fallback extraction methods")
        pass
    
    # Fallback cho PDF files
    if file_type == "application/pdf" or file_name.endswith('.pdf'):
        return extract_text_from_pdf_fallback(uploaded_file)
    else:
        if MARKITDOWN_AVAILABLE:
            st.error(f"Unsupported file type: {file_type}. Please upload PDF, Word, PowerPoint, or Excel files.")
        else:
            st.error(f"PDF-only mode active. Please upload PDF files only.")
        return None

def extract_text_from_pdf_fallback(uploaded_file):
    """Fallback methods for PDF extraction với suppressed output and image extraction"""
    try:
        # Fallback 1: PyMuPDF with suppressed output
        pdf_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        
        with SuppressPyMuPDFOutput():
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            
            # Extract text and images from each page
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                
                # Extract text
                page_text = page.get_text()
                text += page_text
                
                # Extract images and convert to text descriptions
                image_list = page.get_images()
                if image_list:
                    text += f"\n\n[PAGE {page_num + 1} IMAGES: {len(image_list)} images found - may contain conceptual models, tables, or diagrams]\n"
                    
                    # Try to extract image descriptions
                    for img_index, img in enumerate(image_list):
                        try:
                            # Get image data
                            xref = img[0]
                            pix = fitz.Pixmap(pdf_document, xref)
                            
                            # Check if image is valid
                            if pix.n - pix.alpha < 4:  # GRAY or RGB
                                # For now, just note that images exist
                                text += f"[IMAGE {img_index + 1} on page {page_num + 1} - may contain research model or variable relationships]\n"
                            
                            pix = None  # Free memory
                        except Exception as e:
                            text += f"[IMAGE {img_index + 1} on page {page_num + 1} - extraction failed]\n"
                
                # Extract tables if any
                tables = page.find_tables()
                if tables:
                    text += f"\n[PAGE {page_num + 1} TABLES: {len(tables)} tables found - may contain variable definitions or measurement scales]\n"
                    
                    # Try to extract table content
                    for table_idx, table in enumerate(tables):
                        try:
                            # Extract table data
                            table_data = table.extract()
                            if table_data and len(table_data) > 0:
                                text += f"\n[TABLE {table_idx + 1} CONTENT on page {page_num + 1}]:\n"
                                # Add first few rows to give context
                                for row_idx, row in enumerate(table_data[:5]):  # First 5 rows
                                    if row:
                                        text += f"Row {row_idx + 1}: {' | '.join([str(cell) if cell else '' for cell in row])}\n"
                                if len(table_data) > 5:
                                    text += f"... and {len(table_data) - 5} more rows\n"
                                text += "\n"
                        except Exception as e:
                            text += f"[TABLE {table_idx + 1} extraction failed on page {page_num + 1}]\n"
                    
            pdf_document.close()
        
        if text.strip():
            # st.info("📄 PyMuPDF: PDF extracted with text, images, and tables detection")
            return text
            
    except Exception as e:
        # st.warning(f"PyMuPDF failed: {e}. Trying PyPDF2...")
        pass
        
    try:
        # Fallback 2: PyPDF2
        uploaded_file.seek(0)
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        if text.strip():
            # st.info("📋 PyPDF2: PDF extracted successfully")
            return text
    except Exception as e:
        st.error(f"Cannot process this file: {e}")
        return None

def extract_text_from_pdf_file(filepath):
    """Trích xuất text từ file PDF trong hệ thống using MarkItDown với suppressed output"""
    try:
        # Sử dụng MarkItDown trước
        md = init_markitdown()
        if md:
            result = md.convert(filepath)
            if result and result.text_content and result.text_content.strip():
                return result.text_content
        
    except Exception as e:
        st.warning(f"MarkItDown failed for {filepath}: {e}. Falling back to PyMuPDF...")
        
    try:
        # Fallback 1: PyMuPDF with suppressed output
        with SuppressPyMuPDFOutput():
            pdf_document = fitz.open(filepath)
            text = ""
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text += page.get_text()
            pdf_document.close()
        
        if text.strip():
            return text
            
    except Exception as e:
        st.warning(f"PyMuPDF failed for {filepath}: {e}. Trying PyPDF2...")
        
    try:
        # Fallback 2: PyPDF2
        with open(filepath, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"All extraction methods failed for {filepath}: {e}")
        return None

def detect_important_sections(text):
    """Detect and extract important sections for analysis"""
    sections = {}
    text_lower = text.lower()
    
    # Define section patterns
    section_patterns = {
        'abstract': ['abstract', 'tóm tắt'],
        'introduction': ['introduction', 'giới thiệu', 'mở đầu'],
        'literature': ['literature review', 'theoretical framework', 'conceptual framework', 
                      'tổng quan tài liệu', 'cơ sở lý thuyết', 'khung lý thuyết'],
        'methodology': ['methodology', 'method', 'research method', 'phương pháp nghiên cứu'],
        'results': ['results', 'findings', 'kết quả'],
        'discussion': ['discussion', 'thảo luận'],
        'limitations': ['limitations', 'limitation', 'hạn chế', 'future research', 'nghiên cứu tiếp theo'],
        'references': ['references', 'bibliography', 'tài liệu tham khảo']
    }
    
    # Find sections
    for section_name, keywords in section_patterns.items():
        for keyword in keywords:
            start_pos = text_lower.find(keyword)
            if start_pos != -1:
                sections[section_name] = start_pos
                break
    
    return sections

def extract_priority_content(text, max_chars=100000):
    """Extract priority content from long documents"""
    if len(text) <= max_chars:
        return text, 100.0
    
    sections = detect_important_sections(text)
    priority_content = []
    
    # Always include beginning (Abstract + Introduction)
    priority_content.append(text[:15000])
    remaining_chars = max_chars - 15000
    
    # Extract key sections
    section_priorities = ['literature', 'methodology', 'limitations']
    
    for section in section_priorities:
        if section in sections and remaining_chars > 5000:
            start_pos = sections[section]
            # Extract section (up to 20k chars or until next major section)
            end_pos = min(start_pos + 20000, len(text))
            
            # Try to find natural ending (next section)
            for other_section, other_pos in sections.items():
                if other_pos > start_pos and other_pos < end_pos:
                    end_pos = other_pos
                    break
            
            section_text = text[start_pos:end_pos]
            if len(section_text) <= remaining_chars:
                priority_content.append(f"\n\n=== {section.upper()} SECTION ===\n")
                priority_content.append(section_text)
                remaining_chars -= len(section_text) + 50
    
    final_text = ''.join(priority_content)
    coverage_percent = (len(final_text) / len(text)) * 100
    return final_text, coverage_percent

def analyze_document_with_gemini(model, text, filename=""):
    """Phân tích tài liệu và trích xuất thông tin theo format yêu cầu với smart content extraction"""
    if not model:
        return None
    
    # Validate and clean filename
    if not filename or not isinstance(filename, str):
        filename = "unknown_file"
    
    # Clean filename for safe display
    safe_filename = str(filename).replace('\x00', '').strip()
    
    # Smart content extraction for long documents
    analyzed_text, coverage_percent = extract_priority_content(text, 100000)
    
    # Show coverage info to user if document was truncated
    if coverage_percent < 95:
        st.info(f"📊 Document coverage: {coverage_percent:.1f}% (Smart section extraction applied)")
    
    text = analyzed_text  # Use the smart-extracted content
    
    # Debug: Show detected sections if available
    if st.session_state.get('debug_mode', False):
        sections = detect_important_sections(text)
        if sections:
            st.info(f"🔍 Detected sections: {list(sections.keys())}")
    
    # Determine input format for AI prompt
    input_format = "Markdown format để preserve structure" if MARKITDOWN_AVAILABLE else "plain text"
    structure_note = """
    TỐI ƯU CHO MARKDOWN INPUT:
    - Document đã được convert sang Markdown để preserve structure (headings, tables, lists)
    - Ưu tiên extract thông tin từ structured elements (## Abstract, ## Introduction, ## Methodology, ## Results, tables, etc.)
    - Sử dụng table structure trong Markdown để extract data chính xác hơn
    - Pay attention to heading hierarchy và list formatting
    """ if MARKITDOWN_AVAILABLE else """
    EXTRACT FROM PLAIN TEXT:
    - Document là plain text format
    - Tìm kiếm patterns như "Abstract:", "Introduction:", "Methodology:", "Results:", etc.
    - Identify table data và list items từ text formatting
    - Extract thông tin dựa trên context clues
    """
    
    prompt = f"""
    Hãy phân tích CHI TIẾT tài liệu nghiên cứu sau và trích xuất thông tin theo format JSON.

    ⚠️ QUAN TRỌNG - YÊU CẦU ĐẶC BIỆT:
    
    1️⃣ LÝ THUYẾT NỀN - TÌM CHÍNH XÁC:
    - ĐỌC KỸ: Literature Review, Theoretical Framework, Conceptual Framework, Introduction
    - TÌM TẤT CẢ THEORIES: Bất kỳ lý thuyết nào được mention, VÍ DỤ:
      * Technology theories: TAM, UTAUT, TPB, DOI, TTF
      * Management: RBV, Dynamic Capabilities, Stakeholder Theory, Agency Theory
      * Psychology: Social Cognitive Theory, Flow Theory, Self-Determination Theory
      * Quality: SERVQUAL, Service-Profit Chain, Gaps Model
      * Economics: Transaction Cost Theory, Network Theory
      * Marketing: Consumer Behavior Theory, Relationship Marketing Theory
      * VÀ BẤT KỲ LÝ THUYẾT NÀO KHÁC được tác giả đề cập
    - TÌM AUTHOR CITATIONS: Bất kỳ tác giả nào (Davis 1989, Rogers 2003, Porter 1985, etc.)
    - KHÔNG GENERIC: Tránh "lý thuyết về X" → phải tìm tên cụ thể của theory
    - MULTIPLE THEORIES: Liệt kê TẤT CẢ, phân cách bằng dấu phẩy
    - CHECK REFERENCES: Tìm trong bibliography/reference list
    - PATTERN SEARCH: Tìm "theory", "model", "framework", "approach" + tên riêng
    
    ⚠️ LƯU Ý QUAN TRỌNG: Danh sách trên chỉ là VÍ DỤ! Hãy tìm TẤT CẢ lý thuyết từ mọi lĩnh vực:
    Tâm lý học, Kinh tế, Quản lý, Xã hội học, Tài chính, Marketing, Kỹ thuật, Y học, Giáo dục, etc.
    KHÔNG BỎ SÓT bất kỳ theory nào được tác giả đề cập!
    
    2️⃣ THANG ĐO - TÌM ĐẦY ĐỦ:
    - Đọc kỹ phần Methodology, Measurement, Instrument, Questionnaire
    - TÌM BẢNG THANG ĐO (Table/Bảng có chứa mã hóa như CLTT1, CLHT1, CLDV1, ST1, SHL1, etc.)
    - Trích xuất TẤT CẢ items trong bảng thang đo (VD: "CLTT1: Nội dung thông tin được hiển thị hữu ích")
    - Ghi rõ loại thang đo (VD: Likert 5-point, Likert 7-point)
    - Tìm tên cụ thể của scale (VD: SERVQUAL, TAM Scale, Customer Satisfaction Scale)
    - Tìm tác giả của thang đo (VD: "adapted from Davis (1989)")
    - Nếu có nhiều thang đo, liệt kê đầy đủ
    
    3️⃣ HYPOTHESES & RESEARCH MODEL:
    - Đọc kỹ phần Hypothesis, Research Model, Conceptual Framework
    - TÌM TẤT CẢ HYPOTHESES: H1, H2, H3, H4... với nội dung đầy đủ
    - TRÍCH XUẤT MỐI QUAN HỆ: Arrows, paths, direct/indirect effects
    - PHÂN TÍCH MODEL: Các thành phần và vai trò của chúng
    - TÌM TRONG HÌNH ẢNH: Conceptual model diagrams, research framework figures
    
    4️⃣ BIẾN NGHIÊN CỨU - PHÂN LOẠI CHÍNH XÁC:
    - Đọc kỹ phần Hypothesis, Conceptual Model, Variables, Research Model
    - **ĐẶC BIỆT CHÚ Ý**: Tìm trong HÌNH ẢNH, SƠ ĐỒ, BẢNG BIỂU để xác định vai trò biến
    - **KIỂM TRA**: Conceptual Framework diagrams, Research Model figures, Variable relationship charts
    - **TÌM KIẾM**: Tables về variable definitions, measurement scales, hypothesis testing
    - PHÂN BIỆT RÕ VAI TRÒ CỦA TỪNG BIẾN:
    
    📌 **INDEPENDENT VARIABLES (Biến độc lập):**
    - Biến được tác giả MANIPULATE hoặc MEASURE để kiểm tra tác động
    - Biến INPUT vào mô hình nghiên cứu
    - VD: Technology adoption, Service quality, Customer satisfaction
    
    📌 **DEPENDENT VARIABLES (Biến phụ thuộc):**
    - Biến OUTCOME được đo lường để đánh giá kết quả
    - Biến được dự đoán bởi independent variables
    - VD: Performance, Profit, Success, Financial results
    
    📌 **MEDIATING VARIABLES (Biến trung gian):**
    - Biến GIẢI THÍCH cơ chế tác động từ independent → dependent
    - Biến "trung chuyển" trong mối quan hệ
    - VD: Trust (giữa Service Quality → Performance), Efficiency (giữa Technology → Results)
    
    📌 **MODERATING VARIABLES (Biến điều tiết):**
    - Biến ẢNH HƯỞNG đến cường độ mối quan hệ independent → dependent
    - Biến "điều chỉnh" tác động
    - VD: Experience, Age, Industry type
    
    📌 **CONTROL VARIABLES (Biến kiểm soát):**
    - Biến được KIỂM SOÁT để loại bỏ ảnh hưởng nhiễu
    - Biến không phải focus chính của nghiên cứu
    - VD: Company size, Industry, Geographic location
    
    ⚠️ QUAN TRỌNG - RULES FOR VARIABLE CLASSIFICATION: 
    - Đọc kỹ HYPOTHESIS để hiểu mối quan hệ
    - Kiểm tra CONCEPTUAL MODEL để xác định vai trò
    - **MỘT BIẾN CHỈ CÓ THỂ CÓ MỘT VAI TRÒ DUY NHẤT** trong mỗi nghiên cứu
    - **KHÔNG ĐƯỢC LIỆT KÊ CÙNG MỘT BIẾN Ở NHIỀU LOẠI**
    - **NẾU BIẾN CÓ VAI TRÒ TRUNG GIAN → CHỈ ĐẶT VÀO "mediating", KHÔNG đặt vào "dependent"**
    - **NẾU BIẾN LÀ KẾT QUẢ CUỐI CÙNG → CHỈ ĐẶT VÀO "dependent", KHÔNG đặt vào "mediating"**
    - VD: "Operational efficiency" nếu là mediator → CHỈ để trong "mediating", KHÔNG để trong "dependent"
    - VD: "Financial performance" nếu là outcome cuối cùng → CHỈ để trong "dependent", KHÔNG để trong "mediating"

    Hãy phân tích tài liệu nghiên cứu sau (được extract dưới dạng {input_format}) và trích xuất thông tin theo format JSON với các trường sau:
    
    {{
        "nam_xuat_ban": "năm xuất bản (số)",
        "tac_gia": "tên tác giả đầy đủ",
        "tieu_de": "tiêu đề bài nghiên cứu",
        "muc_tieu_nc": "mục tiêu nghiên cứu chính",
        "linh_vuc_nc": "lĩnh vực nghiên cứu",
        "dia_diem_nc": "địa điểm nghiên cứu/khu vực",
        "ly_thuyet_nen": "TÊN CỤ THỂ và ĐẦY ĐỦ TẤT CẢ các lý thuyết nền được sử dụng trong nghiên cứu này từ BẤT KỲ LĨNH VỰC NÀO. VD: 'Technology Acceptance Model (TAM)', 'Resource-Based View (RBV)', 'Social Exchange Theory', 'Institutional Theory', 'Game Theory', 'Expectancy Theory', v.v. Không giới hạn loại lý thuyết - tìm TẤT CẢ theories được mention. Nếu không có lý thuyết rõ ràng thì ghi 'No specific theory mentioned'.",
        "mau": "mô tả mẫu nghiên cứu (số lượng, đặc điểm)",
        "pp_phan_tich": "phương pháp phân tích dữ liệu/thống kê",
        "bien_doc_lap": "INDEPENDENT VARIABLES - Biến INPUT được tác giả kiểm tra tác động (VD: Technology adoption, Service quality, Innovation). CHỈ liệt kê các biến có vai trò CAUSE/PREDICTOR trong nghiên cứu.",
        "bien_phu_thuoc": "DEPENDENT VARIABLES - Biến OUTCOME/RESULT được đo lường để đánh giá kết quả (VD: Performance, Profit, Success, Financial results). CHỈ liệt kê các biến có vai trò EFFECT/OUTCOME trong nghiên cứu.", 
        "bien_trung_gian": "MEDIATING VARIABLES - Biến GIẢI THÍCH cơ chế tác động từ independent → dependent (VD: Trust, Efficiency, Satisfaction). CHỈ liệt kê các biến có vai trò MEDIATOR trong mối quan hệ.",
        "bien_dieu_tiet": "MODERATING VARIABLES - Biến ẢNH HƯỞNG đến cường độ mối quan hệ independent → dependent (VD: Experience, Age, Industry). CHỈ liệt kê các biến có vai trò MODERATOR trong nghiên cứu.",
        "bien_kiem_soat": "CONTROL VARIABLES - Biến được KIỂM SOÁT để loại bỏ ảnh hưởng nhiễu (VD: Company size, Industry, Location). CHỈ liệt kê các biến có vai trò CONTROL trong nghiên cứu.",
        "bien_khac": "OTHER VARIABLES - Các biến khác không thuộc các loại trên (nếu có)",
        "ket_qua_nc": "findings/kết quả nghiên cứu chính",
        "han_che_huong_tiep": "hạn chế và hướng nghiên cứu tiếp theo CHI TIẾT",
        "thang_do": "THANG ĐO CHI TIẾT (VD: 'Likert 5-point scale, CLTT1: Nội dung thông tin hữu ích, CLTT2: Thông tin đầy đủ, CLHT1: Website dễ sử dụng, etc.')",
        "co_mo_hinh": "có mô hình hay không (Yes/No)",
        "thanh_phan_mo_hinh": "CÁC THÀNH PHẦN MÔ HÌNH VÀ HYPOTHESES CHI TIẾT - Trích xuất đầy đủ các thành phần của mô hình nghiên cứu bao gồm: (1) Tên các biến/thành phần chính, (2) CÁC HYPOTHESES CHI TIẾT (H1, H2, H3...) với nội dung đầy đủ, (3) Mối quan hệ giữa các biến (arrows, paths), (4) Loại mối quan hệ (direct, indirect, moderating, mediating). VD: 'H1: Supply Chain Risk Management has a positive effect on Operational Efficiency, H2: Operational Efficiency mediates the relationship between Supply Chain Risk Management and Financial Performance, H3: Supplier Integration moderates the relationship between Supply Chain Risk Management and Operational Efficiency, Model includes: Supply Chain Risk Management → Operational Efficiency → Financial Performance, with Supplier Integration as moderator'",
        "doi": "số DOI (nếu có)"
    }}
    
    5️⃣ HẠN CHẾ & HƯỚNG NGHIÊN CỨU:
    - Đọc kỹ phần Limitations, Future Research, Conclusion
    - Trích xuất đầy đủ limitation và future research direction
    
    6️⃣ VÍ DỤ TRÍCH XUẤT THANG ĐO:
    Nếu thấy bảng như:
    "CLTT1: Information quality is useful
     CLTT2: Information is complete  
     CLHT1: System is easy to use"
    
    → Ghi vào field "thang_do": "Likert scale, CLTT1: Information quality is useful, CLTT2: Information is complete, CLHT1: System is easy to use"

    QUAN TRỌNG - QUY TẮC VỀ NGÔN NGỮ:
    - GIỮ NGUYÊN NGÔN NGỮ GỐC của bài báo (English, Vietnamese, etc.)
    - KHÔNG DỊCH nội dung sang tiếng Việt
    - Nếu bài báo bằng tiếng Anh → giữ nguyên tiếng Anh
    - Nếu bài báo bằng tiếng Việt → giữ nguyên tiếng Việt
    - Chỉ trích xuất thông tin y như trong bài gốc
    {structure_note}
    
    HƯỚNG DẪN ĐỌC:
    - Ưu tiên: Abstract, Introduction, Literature Review, Methodology, Results, Discussion, Limitations
    - 🔍 ĐẶC BIỆT CHÚ Ý BẢNG BIỂU: Tìm "Bảng", "Table", "Appendix" có chứa thang đo
    - 📋 PATTERN NHẬN DIỆN TABLE: Tìm mã hóa dạng "XXX1", "XXX2" (VD: CLTT1, CLHT1, TAM1, SERV1)
    - 📊 EXTRACT TABLE CONTENT: Copy nguyên văn nội dung bảng thang đo vào field "thang_do"
    - 🖼️ **QUAN TRỌNG**: Tìm kiếm trong HÌNH ẢNH, SƠ ĐỒ để xác định vai trò biến
    - 📈 **KIỂM TRA**: Conceptual Model figures, Research Framework diagrams, Variable relationship charts
    - 🔍 **TÌM KIẾM**: Keywords trong hình: "Independent", "Dependent", "Mediating", "Moderating", arrows, relationships
    - 📋 **HYPOTHESES EXTRACTION**: Tìm kiếm tất cả hypotheses với format H1:, H2:, H3: hoặc Hypothesis 1, Hypothesis 2, etc.
    - 🔗 **RELATIONSHIPS**: Tìm mối quan hệ giữa các biến (→, arrows, paths, direct/indirect effects)
    - 📊 **MODEL COMPONENTS**: Trích xuất đầy đủ tên biến và vai trò của chúng trong mô hình
    - Tìm kiếm keywords: "scale", "measurement", "instrument", "questionnaire", "theory", "model", "framework", "bảng", "table", "hypothesis", "H1", "H2", "H3"
    - Nếu không tìm thấy thông tin nào, ghi "No information" hoặc "Không có thông tin" (tùy ngôn ngữ bài gốc)
    - Trả về JSON hợp lệ
    - Trích xuất chính xác từ nội dung văn bản VÀ HÌNH ẢNH
    - Giữ nguyên thuật ngữ chuyên môn và tên riêng
    
    Tên file: {safe_filename}
    
    Nội dung tài liệu ({input_format}):
    {text[:100000]}
    """
    
    prompt_parts = [prompt]
    
    # Đo thời gian bắt đầu
    start_time = time.time()
    
    try:
        response = model.generate_content(prompt_parts)
        # Đo thời gian kết thúc
        end_time = time.time()
        st.info(f"Thời gian phân tích Gemini cho {safe_filename}: {end_time - start_time:.2f} giây")

        # Robustly extract text from response (support multipart responses)
        response_text = extract_response_text(response)

        # Tìm JSON trong response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1

        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            try:
                parsed_json = json.loads(json_str)
                st.write(f"Kết quả phân tích cho {safe_filename}:")
                st.json(parsed_json)
                return parsed_json
            except json.JSONDecodeError as parse_error:
                st.error(f"Lỗi parse JSON cho {safe_filename}: {parse_error}")
                st.error(f"JSON string: {json_str[:500]}...")
                return None
        else:
            st.error(f"Không tìm thấy JSON hợp lệ trong response cho {safe_filename}")
            st.error(f"Response preview: {response_text[:200]}...")
            return None

    except json.JSONDecodeError as e:
        st.error(f"Lỗi parse JSON cho {safe_filename}: {e}")
        return None
    except Exception as e:
        st.error(f"Lỗi khi gọi Gemini API cho {safe_filename}: {e}")
        st.error(f"Chi tiết lỗi: {type(e).__name__}: {str(e)}")
        return None

def create_analysis_dataframe(results):
    """Tạo DataFrame từ kết quả phân tích"""
    columns = [
        "STT", "Năm xuất bản", "Tên tác giả", "Tiêu đề bài nghiên cứu (Title)",
        "Mục tiêu nghiên cứu", "Lĩnh vực NC", "Địa điểm NC (Area)", "Lý Thuyết nền",
        "Mẫu (sample)", "PP phân tích dữ liệu (Statistical method)", 
        "Biến độc lập", "Biến phụ thuộc", "Biến trung gian", "Biến điều tiết",
        "Biến kiểm soát", "Biến khác nếu có", "Findings (Kết quả nghiên cứu)",
        "Hạn chế và hướng NC tiếp theo (Limitations)", "Thang đo (Measurement)",
        "Mô hình có hay không", "Mô hình gồm các thành phần gì (nếu có)", "Số DOI"
    ]
    
    data = []
    for i, result in enumerate(results, 1):
        if result is None:
            continue
            
        row = [
            str(i),  # STT
            safe_get_value(result, 'nam_xuat_ban'),
            safe_get_value(result, 'tac_gia'),
            safe_get_value(result, 'tieu_de'),
            safe_get_value(result, 'muc_tieu_nc'),
            safe_get_value(result, 'linh_vuc_nc'),
            safe_get_value(result, 'dia_diem_nc'),
            safe_get_value(result, 'ly_thuyet_nen'),
            safe_get_value(result, 'mau'),
            safe_get_value(result, 'pp_phan_tich'),
            safe_get_value(result, 'bien_doc_lap'),
            safe_get_value(result, 'bien_phu_thuoc'),
            safe_get_value(result, 'bien_trung_gian'),
            safe_get_value(result, 'bien_dieu_tiet'),
            safe_get_value(result, 'bien_kiem_soat'),
            safe_get_value(result, 'bien_khac'),
            safe_get_value(result, 'ket_qua_nc'),
            safe_get_value(result, 'han_che_huong_tiep'),
            format_complex_string_for_display(safe_get_value(result, 'thang_do'), "thang_do"),
            safe_get_value(result, 'co_mo_hinh'),
            format_complex_string_for_display(safe_get_value(result, 'thanh_phan_mo_hinh'), "thanh_phan_mo_hinh"),
            safe_get_value(result, 'doi')
        ]
        
        data.append(row)  # FIXED: Actually append the row to data list
    
    normalized_data = []
    row_counter = 1
    
    for i, result in enumerate(results, 1):
        if result is None:
            continue
            
        # Original data for the row (before normalization of variables)
        original_row_values = {
            "STT_Original": str(i),
            "Năm xuất bản": safe_get_value(result, 'nam_xuat_ban'),
            "Tên tác giả": safe_get_value(result, 'tac_gia'),
            "Tiêu đề bài nghiên cứu (Title)": safe_get_value(result, 'tieu_de'),
            "Mục tiêu nghiên cứu": safe_get_value(result, 'muc_tieu_nc'),
            "Lĩnh vực NC": safe_get_value(result, 'linh_vuc_nc'),
            "Địa điểm NC (Area)": safe_get_value(result, 'dia_diem_nc'),
            "Lý Thuyết nền": safe_get_value(result, 'ly_thuyet_nen'),
            "Mẫu (sample)": safe_get_value(result, 'mau'),
            "PP phân tích dữ liệu (Statistical method)": safe_get_value(result, 'pp_phan_tich'),
            "Findings (Kết quả nghiên cứu)": safe_get_value(result, 'ket_qua_nc'),
            "Hạn chế và hướng NC tiếp theo (Limitations)": safe_get_value(result, 'han_che_huong_tiep'),
            "Thang đo (Measurement)": format_complex_string_for_display(safe_get_value(result, 'thang_do'), "thang_do"),
            "Mô hình có hay không": safe_get_value(result, 'co_mo_hinh'),
            "Mô hình gồm các thành phần gì (nếu có)": format_complex_string_for_display(safe_get_value(result, 'thanh_phan_mo_hinh'), "thanh_phan_mo_hinh"),
            "Số DOI": safe_get_value(result, 'doi')
        }

        # Initialize all variable columns to empty string
        for var_col in ["Biến phụ thuộc", "Biến độc lập", "Biến trung gian", 
                       "Biến điều tiết", "Biến kiểm soát", "Biến khác nếu có"]:
            original_row_values[var_col] = ""

        # Variables to normalize
        variable_fields = {
            "Biến độc lập": parse_variables(safe_get_value(result, 'bien_doc_lap')),
            "Biến phụ thuộc": parse_variables(safe_get_value(result, 'bien_phu_thuoc')),
            "Biến trung gian": parse_variables(safe_get_value(result, 'bien_trung_gian')),
            "Biến điều tiết": parse_variables(safe_get_value(result, 'bien_dieu_tiet')),
            "Biến kiểm soát": parse_variables(safe_get_value(result, 'bien_kiem_soat')),
            "Biến khác nếu có": parse_variables(safe_get_value(result, 'bien_khac'))
        }

        # Add a row for each variable type and each variable within that type
        has_variables = False
        for var_type, variables_list in variable_fields.items():
            if variables_list:
                has_variables = True
                for var_name in variables_list:
                    row = original_row_values.copy()
                    row["STT"] = str(row_counter)
                    row["Variable_Type"] = var_type.replace("Biến ", "").replace(" nếu có", "").strip()
                    row["Variable_Name"] = var_name
                    row[var_type] = var_name  # Set the specific variable column
                    
                    normalized_data.append([row[col] for col in columns] + [row["Variable_Type"], row["Variable_Name"]])
                    row_counter += 1
        
        # If no variables found, still add the original row to keep track of the paper
        if not has_variables:
            row = original_row_values.copy()
            row["STT"] = str(row_counter)
            row["Variable_Type"] = "No Variables Identified"
            row["Variable_Name"] = "No Variables Identified"
            normalized_data.append([row[col] for col in columns] + [row["Variable_Type"], row["Variable_Name"]])
            row_counter += 1

    normalized_columns = columns + ["Variable_Type", "Variable_Name"]
    df = pd.DataFrame(normalized_data, columns=normalized_columns)
    
    # Ensure all columns are string type to avoid Arrow conversion issues
    for col in df.columns:
        if col != "STT" and col != "STT_Original":  # Keep STT as integer
            df[col] = df[col].astype(str)

    return df

def parse_authors(author_string):
    """Parse author string into individual authors"""
    if not author_string:
        return []
    
    try:
        text = str(author_string).strip()
        if text.lower() in ['no information', 'không có thông tin', 'no authors', 'n/a', 'na', 'none', 'null', 'unknown authors']:
            return []
        
        # Split by common author delimiters
        delimiters = [';', '&', ' and ', ' và ', ',', '\n']
        authors = [text]
        
        for delimiter in delimiters:
            new_authors = []
            for author in authors:
                new_authors.extend([a.strip() for a in author.split(delimiter)])
            authors = new_authors
        
        # Clean up authors
        cleaned_authors = []
        for author in authors:
            author = author.strip()
            if author and len(author) > 2 and author.lower() not in ['et al', 'et.al', 'and others']:
                cleaned_authors.append(author)
        
        return cleaned_authors
        
    except Exception as e:
        st.warning(f"Error in parse_authors with input {type(author_string)}: {e}")
        return [str(author_string)] if author_string else []

def parse_variables(var_string):
    """Parse variable string into individual variables"""
    # Handle different input types safely
    if not var_string:
        return []
    
    try:
        # Convert to string first, then check
        text = str(var_string).strip()
        if text.lower() in ['no information', 'không có thông tin', 'no variables', 'n/a', 'na', 'none', 'null']:
            return []
        
        import json, re
    except Exception as e:
        st.warning(f"Error in parse_variables with input {type(var_string)}: {e}")
        return []

    # If looks like a JSON list, parse it first
    candidates = []
    if (text.startswith('[') and text.endswith(']')) or (text.startswith('{') and text.endswith('}')):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                candidates = [str(x) for x in parsed]
        except Exception:
            candidates = []

    # Fallback: split by common delimiters
    if not candidates:
        delimiters = [',', ';', '&', ' and ', ' và ', '+', '|', '\n']
        candidates = [text]
        for delimiter in delimiters:
            new_candidates = []
            for v in candidates:
                new_candidates.extend([p.strip() for p in v.split(delimiter)])
            candidates = new_candidates

    try:
        # Normalize each candidate to lowercase and strip special characters
        normalized = []
        seen = set()
        for raw in candidates:
            v = str(raw).strip()
            # Remove surrounding and internal brackets/quotes commonly leaked from list stringification
            v = re.sub(r"^[\[\(\{\'\"\s]+", "", v)
            v = re.sub(r"[\]\)\}\'\"\s]+$", "", v)
            # Remove special chars like [](){}'"
            v = re.sub(r"[\[\]\(\)\{\}\'\"]", "", v)
            # Collapse multiple spaces
            v = re.sub(r"\s+", " ", v).strip()
            v_lower = v.lower()
            if (not v_lower or len(v_lower) <= 2 or v_lower in [
                'no information', 'không có thông tin', 'no variables', 'n/a', 'na', 'none', 'not available', 'not specified']):
                continue
            if v_lower not in seen:
                seen.add(v_lower)
                normalized.append(v_lower)

        return normalized
    except Exception as e:
        st.warning(f"Error in parse_variables processing: {e}")
        return []

def validate_variable_classification(result):
    """Validate and fix variable classification to prevent duplicates"""
    if not result:
        return result
    
    # Get all variable lists
    var_lists = {
        'bien_doc_lap': parse_variables(result.get('bien_doc_lap', '')),
        'bien_phu_thuoc': parse_variables(result.get('bien_phu_thuoc', '')),
        'bien_trung_gian': parse_variables(result.get('bien_trung_gian', '')),
        'bien_dieu_tiet': parse_variables(result.get('bien_dieu_tiet', '')),
        'bien_kiem_soat': parse_variables(result.get('bien_kiem_soat', '')),
        'bien_khac': parse_variables(result.get('bien_khac', ''))
    }
    
    # Track variables by priority (higher priority = keep, lower priority = remove)
    # Priority: Independent > Mediating > Dependent > Moderating > Control > Other
    # This ensures mediators are not misclassified as dependent variables
    priority_order = ['bien_doc_lap', 'bien_trung_gian', 'bien_phu_thuoc', 'bien_dieu_tiet', 'bien_kiem_soat', 'bien_khac']
    seen_vars = set()
    
    # Process each category in priority order
    for category in priority_order:
        original_vars = var_lists[category][:]
        filtered_vars = []
        
        for var in original_vars:
            if var not in seen_vars:
                seen_vars.add(var)
                filtered_vars.append(var)
            # If duplicate found, skip it (keep the higher priority one)
        
        # Update the result with filtered variables
        if filtered_vars:
            result[category] = ', '.join(filtered_vars)
        else:
            result[category] = ''
    
    return result

def normalize_token(value: str) -> str:
    """Normalize a single token to lowercase and remove special characters.

    - Trim whitespace
    - Remove brackets/quotes: [](){}'"
    - Collapse multiple spaces
    - Lowercase
    """
    import re
    if value is None:
        return ""
    v = str(value).strip()
    v = re.sub(r"^[\[\(\{\'\"\s]+", "", v)
    v = re.sub(r"[\]\)\}\'\"\s]+$", "", v)
    v = re.sub(r"[\[\]\(\)\{\}\'\"]", "", v)
    v = re.sub(r"\s+", " ", v).strip()
    return v.lower()


def normalize_analysis_data(results):
    """Transform analysis results into a normalized DataFrame (1 variable = 1 row).

    Returns a DataFrame containing the original columns duplicated per variable plus
    'STT_Original', 'Variable_Type', 'Variable_Name'.
    """
    # Build the original DataFrame first
    try:
        df = create_analysis_dataframe(results)
    except Exception:
        # Fallback: create minimal DataFrame from results if needed
        rows = []
        for i, r in enumerate(results, 1):
            if not r:
                continue
            row = {k: v for k, v in r.items()}
            row['STT'] = str(i)
            rows.append(row)
        df = pd.DataFrame(rows)

    normalized_rows = []

    for _, row in df.iterrows():
        stt_original = row.get('STT', '')
        # gather variable strings from common columns if present
        var_sources = {
            'Independent': row.get('Biến độc lập') or row.get('bien_doc_lap') or '',
            'Dependent': row.get('Biến phụ thuộc') or row.get('bien_phu_thuoc') or '',
            'Mediating': row.get('Biến trung gian') or row.get('bien_trung_gian') or '',
            'Moderating': row.get('Biến điều tiết') or row.get('bien_dieu_tiet') or '',
            'Control': row.get('Biến kiểm soát') or row.get('bien_kiem_soat') or '',
            'Other': row.get('Biến khác nếu có') or row.get('bien_khac') or ''
        }

        any_vars = False
        for var_type, var_text in var_sources.items():
            parsed = parse_variables(str(var_text)) if var_text else []
            if parsed:
                any_vars = True
                for var_name in parsed:
                    out = row.to_dict()
                    out['STT_Original'] = stt_original
                    out['Variable_Type'] = var_type
                    out['Variable_Name'] = var_name
                    normalized_rows.append(out)

        if not any_vars:
            out = row.to_dict()
            out['STT_Original'] = stt_original
            out['Variable_Type'] = 'No Variables Identified'
            out['Variable_Name'] = 'No Variables Identified'
            normalized_rows.append(out)

    if not normalized_rows:
        # Return empty DataFrame with expected columns
        cols = list(df.columns) + ['STT_Original', 'Variable_Type', 'Variable_Name']
        return pd.DataFrame([], columns=cols)

    norm_df = pd.DataFrame(normalized_rows)

    # Ensure expected columns exist
    if 'STT_Original' not in norm_df.columns:
        norm_df['STT_Original'] = norm_df.get('STT', '')

    # Normalize dtypes to strings where appropriate
    for c in norm_df.columns:
        if c not in ['STT', 'STT_Original']:
            try:
                norm_df[c] = norm_df[c].astype(str)
            except Exception:
                pass

    return norm_df

def normalize_analysis_data_authors(results):
    """Transform analysis results into a normalized DataFrame (1 author = 1 row).
    
    Returns a DataFrame containing the original columns duplicated per author plus
    'STT_Original', 'Author_Name'.
    """
    # Remove duplicates first
    if results:
        seen_filenames = set()
        unique_results = []
        
        for result in results:
            filename = result.get('filename', '')
            if filename not in seen_filenames:
                seen_filenames.add(filename)
                unique_results.append(result)
        
        results = unique_results  # Use unique results
    
    # Build the original DataFrame first
    try:
        df = create_analysis_dataframe(results)
    except Exception:
        # Fallback: create minimal DataFrame from results if needed
        rows = []
        for i, r in enumerate(results, 1):
            if not r:
                continue
            row = {k: v for k, v in r.items()}
            row['STT'] = str(i)
            rows.append(row)
        df = pd.DataFrame(rows)

    normalized_rows = []

    for _, row in df.iterrows():
        stt_original = row.get('STT', '')
        author_text = row.get('Tên tác giả', '') or row.get('tac_gia', '')
        
        # Parse authors
        authors = parse_authors(str(author_text)) if author_text else []
        
        if authors:
            # Create one row per author
            for author_name in authors:
                out = row.to_dict()
                out['STT_Original'] = stt_original
                out['Author_Name'] = author_name
                # Update the author column to show only this author
                out['Tên tác giả'] = author_name
                normalized_rows.append(out)
        else:
            # No authors found, keep original row
            out = row.to_dict()
            out['STT_Original'] = stt_original
            out['Author_Name'] = 'No Authors Identified'
            normalized_rows.append(out)

    if not normalized_rows:
        # Return empty DataFrame with expected columns
        cols = list(df.columns) + ['STT_Original', 'Author_Name']
        return pd.DataFrame([], columns=cols)

    norm_df = pd.DataFrame(normalized_rows)

    # Ensure expected columns exist
    if 'STT_Original' not in norm_df.columns:
        norm_df['STT_Original'] = norm_df.get('STT', '')

    # Normalize dtypes to strings where appropriate
    for c in norm_df.columns:
        if c not in ['STT', 'STT_Original']:
            try:
                norm_df[c] = norm_df[c].astype(str)
            except Exception:
                pass

    return norm_df

def safe_get_value(result, key, default='N/A'):
    """Safely get value and convert to string"""
    try:
        value = result.get(key, default)
        if value is None:
            return default
        elif isinstance(value, (list, dict)):
            return str(value)
        else:
            return str(value)
    except:
        return default

def debug_theory_extraction(result, filename):
    """Debug function to check theory extraction quality"""
    if not st.session_state.get('debug_mode', False):
        return
    
    theory_value = result.get('ly_thuyet_nen', 'Not found')
    with st.expander(f"🔍 Debug Theory Extraction for {filename}", expanded=False):
        st.write("**Raw AI Response for Theory:**")
        st.code(theory_value)
        
        # Check for common issues
        issues = []
        if theory_value.lower() in ['no information', 'không có thông tin', 'n/a']:
            issues.append("❌ Generic 'no information' response")
        if len(theory_value) < 10:
            issues.append("⚠️ Very short response - may be incomplete")
        if 'lý thuyết về' in theory_value.lower() and '(' not in theory_value:
            issues.append("⚠️ Generic description without specific theory name")
        if theory_value == theory_value.upper():
            issues.append("⚠️ All caps response - check formatting")
            
        if issues:
            st.warning("**Potential Issues:**")
            for issue in issues:
                st.write(issue)
        else:
            st.success("✅ Theory extraction looks good")

def format_complex_string_for_display(complex_string, type_of_data):
    """Formats complex strings like scales and model components for better display."""
    if not complex_string or complex_string.strip().lower() in ["n/a", "no information", "không có thông tin", "no explicit information", "none", "not available", "not specified"]:
        return "Không có thông tin"

    # Try to parse as JSON list/dict first
    try:
        parsed_data = json.loads(complex_string)
        if isinstance(parsed_data, list):
            return "; ".join([str(item) for item in parsed_data])
        elif isinstance(parsed_data, dict):
            return "; ".join([f"{k}: {v}" for k, v in parsed_data.items()])
    except json.JSONDecodeError:
        pass # Not a JSON, proceed with string parsing

    # Specific formatting for scales
    if type_of_data == "thang_do":
        lines = [line.strip() for line in complex_string.split(',') if line.strip()]
        formatted_lines = []
        scale_type = ""
        items = []
        for line in lines:
            if "scale" in line.lower() or "point" in line.lower() or "likert" in line.lower():
                scale_type = line
            else:
                items.append(line)
        
        if scale_type:
            formatted_lines.append(f"**Loại thang đo**: {scale_type}")
        if items:
            formatted_lines.append("**Các mục**: ")
            for item in items:
                formatted_lines.append(f"- {item}")
        return "\n".join(formatted_lines) if formatted_lines else complex_string

    # Specific formatting for model components
    elif type_of_data == "thanh_phan_mo_hinh":
        lines = [line.strip() for line in complex_string.split(';') if line.strip()]
        formatted_lines = [f"- {line}" for line in lines]
        return "\n".join(formatted_lines) if formatted_lines else complex_string

    # Default behavior for other complex strings
    return complex_string

def extract_response_text(response):
    """Robustly extract text from a model response object that may be multipart.

    Tries these in order:
    - response.text (if simple)
    - response.candidates[0].content.parts join
    - response.candidates[0].content.text
    - response.parts join
    - str(response)
    """
    def _find_text(obj, depth=0):
        """Recursively search for the first meaningful text in nested structures.

        This handles:
        - dicts with keys like 'text', 'content', 'body', 'message', 'parts'
        - lists/tuples of candidate parts or messages
        - SDK objects with attributes like .text, .content, .message, .parts
        """
        if depth > 6:
            return None
        if obj is None:
            return None
        # Direct string
        if isinstance(obj, str):
            return obj if obj.strip() else None
        # Primitive types
        if isinstance(obj, (int, float, bool)):
            return str(obj)
        # Mapping-like
        if isinstance(obj, dict):
            # Preferred keys order
            preferred_keys = ['text', 'content', 'body', 'message', 'output', 'response', 'parts']
            for key in preferred_keys:
                if key in obj:
                    val = obj[key]
                    if isinstance(val, str) and val.strip():
                        return val
                    found = _find_text(val, depth + 1)
                    if found:
                        return found
            # fallback to searching all values
            for v in obj.values():
                found = _find_text(v, depth + 1)
                if found:
                    return found
            return None
        # Iterable-like
        if isinstance(obj, (list, tuple)):
            for el in obj:
                found = _find_text(el, depth + 1)
                if found:
                    return found
            return None

        # Generic object: try common attributes
        try:
            if hasattr(obj, 'text'):
                t = getattr(obj, 'text')
                if isinstance(t, str) and t.strip():
                    return t
        except Exception:
            pass
        try:
            if hasattr(obj, 'content'):
                c = getattr(obj, 'content')
                found = _find_text(c, depth + 1)
                if found:
                    return found
        except Exception:
            pass
        try:
            if hasattr(obj, 'message'):
                m = getattr(obj, 'message')
                found = _find_text(m, depth + 1)
                if found:
                    return found
        except Exception:
            pass
        try:
            if hasattr(obj, 'parts'):
                parts = getattr(obj, 'parts')
                if isinstance(parts, (list, tuple)):
                    parts_text = []
                    for p in parts:
                        pt = _find_text(p, depth + 1)
                        if pt:
                            parts_text.append(pt)
                        else:
                            parts_text.append(str(p))
                    joined = ''.join(parts_text).strip()
                    if joined:
                        return joined
        except Exception:
            pass

        # No text found
        return None

    # 0) Prefer explicit candidates->content->parts text concatenation
    try:
        def _join_parts(parts_obj):
            if not parts_obj:
                return None
            collected = []
            try:
                for p in parts_obj:
                    t = None
                    try:
                        t = getattr(p, 'text', None)
                    except Exception:
                        t = None
                    if isinstance(t, str) and t.strip():
                        collected.append(t)
                    else:
                        # Some SDKs store text-like payloads under to_dict()/dict keys
                        try:
                            d = p.to_dict() if hasattr(p, 'to_dict') else None
                        except Exception:
                            d = None
                        if isinstance(d, dict):
                            for key in ['text', 'content', 'body']:
                                if key in d and isinstance(d[key], str) and d[key].strip():
                                    collected.append(d[key])
                                    break
                        else:
                            s = str(p)
                            if s and not s.startswith('<'):
                                collected.append(s)
                joined = ''.join(collected).strip()
                return joined if joined else None
            except Exception:
                return None

        # First check response.candidates
        cands = getattr(response, 'candidates', None)
        if cands:
            try:
                first_cand = cands[0] if isinstance(cands, (list, tuple)) and cands else None
                if first_cand is not None:
                    content = getattr(first_cand, 'content', None)
                    parts = getattr(content, 'parts', None) if content is not None else None
                    joined = _join_parts(parts)
                    if joined:
                        return joined
            except Exception:
                pass

        # Then check response.result.candidates
        result_obj = getattr(response, 'result', None)
        if result_obj is not None:
            try:
                res_cands = getattr(result_obj, 'candidates', None)
                if res_cands:
                    first_cand = res_cands[0] if isinstance(res_cands, (list, tuple)) and res_cands else None
                    if first_cand is not None:
                        content = getattr(first_cand, 'content', None)
                        parts = getattr(content, 'parts', None) if content is not None else None
                        joined = _join_parts(parts)
                        if joined:
                            return joined
            except Exception:
                pass
    except Exception:
        pass

    # 1) Try quick .text if available (guarded)
    try:
        txt_attr = getattr(response, 'text', None)
        if isinstance(txt_attr, str) and txt_attr.strip():
            return txt_attr.strip()
    except Exception:
        pass

    # 2) Candidates / results
    try:
        candidates = getattr(response, 'candidates', None) or getattr(response, 'results', None)
        found = _find_text(candidates)
        if found:
            return found.strip()
    except Exception:
        pass

    # 3) Try to find text anywhere in the response object
    try:
        found = _find_text(response)
        if found:
            return found.strip()
    except Exception:
        pass

    # 4) Try parts attribute directly
    try:
        parts = getattr(response, 'parts', None)
        if parts and isinstance(parts, (list, tuple)):
            parts_text = []
            for p in parts:
                pt = _find_text(p)
                parts_text.append(pt or str(p))
            joined = ''.join(parts_text).strip()
            if joined:
                return joined
    except Exception:
        pass

    # 5) Fallback to string conversion
    try:
        return str(response)
    except Exception:
        return ""

def main():
    # Import các thư viện cần thiết
    import time
    
    # Khởi tạo session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    
    # Temporary results storage (before save and analyze)
    if 'temp_analysis_results' not in st.session_state:
        st.session_state.temp_analysis_results = []
    
    # Flag to track if analysis is ready to be saved
    if 'analysis_ready_to_save' not in st.session_state:
        st.session_state.analysis_ready_to_save = False
    
    # Flag to track if user has reset the system
    if 'has_reset' not in st.session_state:
        st.session_state.has_reset = False
    
    # File uploader key to force refresh after reset
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0

    if 'learned_library' not in st.session_state:
        # Only load existing learned library if user hasn't reset
        if not st.session_state.has_reset:
            st.session_state.learned_library = load_learned_library()
        else:
            st.session_state.learned_library = []
    
    # Header
    st.title("📚 Systematic Literature Review Tool")
    st.subheader("🤖 Powered by Gemini AI - Tạo bảng phân tích tài liệu tự động")
    
    # Model Selection Sidebar
    st.sidebar.title("⚙️ Cài đặt AI Model")
    
    model_options = {
        "gemini-2.5-flash-lite-preview-06-17": {
            "name": "Gemini 2.5 Flash Lite",
            "speed": "Fastest ⚡",
            "quality": "Good",
            "description": "Tốc độ nhanh nhất, phù hợp batch lớn",
            "batch_rec": "1-1000 files"
        },
        "gemini-2.5-flash": {
            "name": "Gemini 2.5 Flash", 
            "speed": "Fast ⚡",
            "quality": "Very Good",
            "description": "Cân bằng tốc độ và chất lượng",
            "batch_rec": "1-500 files"
        },
        "gemini-1.5-flash": {
            "name": "Gemini 1.5 Flash",
            "speed": "Medium 🔄",
            "quality": "Good",
            "description": "Ổn định, độ tin cậy cao",
            "batch_rec": "1-200 files"
        },
        "gemini-1.5-pro": {
            "name": "Gemini 1.5 Pro",
            "speed": "Slower 🐌",
            "quality": "Excellent",
            "description": "Chất lượng cao nhất, phân tích sâu",
            "batch_rec": "1-50 files"
        }
    }
    
    selected_model = st.sidebar.selectbox(
        "🤖 Chọn AI Model:",
        options=list(model_options.keys()),
        index=0,  # Default to lite version
        format_func=lambda x: model_options[x]["name"]
    )
    
    model_info = model_options[selected_model]
    
    st.sidebar.info(f"""
    **🤖 {model_info['name']}**
    - ⚡ Tốc độ: {model_info['speed']}
    - 🎯 Chất lượng: {model_info['quality']}
    - 📝 {model_info['description']}
    - 📊 Khuyến nghị: {model_info['batch_rec']}
    """)
    
    # Initialize model
    model = init_gemini(selected_model)
    if not model:
        st.stop()
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📤 Upload File Mới", "📊 Kết Quả Phân Tích", "📈 Thống Kê Phân Tích", "🔍 Research Gaps & Opportunities", "🎯 Research Gap Analysis", "🚀 Complete Research Planning"])
    
    # Hide Tab 5 visually (keep code intact)
    st.markdown(
        """
        <style>
        /* Hide the 5th tab button only */
        div[data-baseweb="tab-list"] button:nth-child(5) { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    with tab1:
        st.header("📤 Upload và Phân Tích Documents")
        
        # Info về supported formats based on availability - không expose technical details
        if MARKITDOWN_AVAILABLE:
            st.info("🎯 **Advanced Document Processing** - Support multiple formats with enhanced analysis!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("📄 **PDF**\n- Research papers\n- Academic articles")
            with col2:
                st.markdown("📝 **Word**\n- .docx, .doc\n- Research reports")
            with col3:
                st.markdown("📊 **PowerPoint**\n- .pptx, .ppt\n- Presentations")
            with col4:
                st.markdown("📈 **Excel**\n- .xlsx, .xls\n- Data tables")
            
            supported_types = ['pdf', 'docx', 'doc', 'pptx', 'ppt', 'xlsx', 'xls']
            help_text = "Upload multiple files for batch analysis. Supported: PDF, Word, PowerPoint, Excel"
        else:
            st.info("📄 **Document Analysis** - PDF processing available")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("📄 **PDF Support**\n- Research papers\n- Academic articles")
            with col2:
                st.markdown("🔄 **Reliable Processing**\n- High-quality extraction\n- Batch analysis")
            
            supported_types = ['pdf']
            help_text = "Upload PDF files for systematic literature review analysis."
        
        # Thêm nút Reset và Debug mode ở đầu tab
        col_reset, col_debug = st.columns([2, 1])
        
        with col_reset:
            st.subheader("🔄 Reset Hệ Thống")
            st.warning("**Sử dụng nút này để xóa tất cả dữ liệu và bắt đầu lại từ đầu.**")
            reset_button = st.button("🔄 RESET TẤT CẢ DỮ LIỆU", type="primary", help="Xóa tất cả dữ liệu và bắt đầu lại từ đầu")
        
        with col_debug:
            st.subheader("🔍 Debug Mode")
            debug_mode = st.checkbox("Enable Debug", 
                                   value=st.session_state.get('debug_mode', False),
                                   help="Show detailed extraction analysis")
            st.session_state.debug_mode = debug_mode
        
        st.divider()
        
        # Handle Reset button
        if reset_button:
            # Clear all session state data
            st.session_state.analysis_results = []
            st.session_state.temp_analysis_results = []
            st.session_state.learned_library = []
            st.session_state.analysis_ready_to_save = False
            st.session_state.has_reset = True  # Mark that user has reset
            
            # Force file uploader to refresh by changing its key
            st.session_state.uploader_key += 1
            
            # Clear learned library file
            save_learned_library([])
            st.success("✅ Đã reset tất cả dữ liệu! Vui lòng upload files mới để bắt đầu.")
            st.rerun()
        
        uploaded_files = st.file_uploader(
            "Chọn documents để phân tích:",
            type=supported_types,
            accept_multiple_files=True,
            help=help_text,
            key=f"file_uploader_{st.session_state.uploader_key}"
        )
        
        if uploaded_files:
            st.success(f"✅ Đã upload **{len(uploaded_files)} files** thành công!")
            
            # Phần chọn số lượng phân tích
            st.subheader("⚙️ Cài đặt phân tích")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                analyze_all_upload = st.checkbox(
                    "🎯 Phân tích TẤT CẢ files", 
                    value=False, 
                    help="Tick để phân tích tất cả files đã upload"
                )
            
            with col2:
                if analyze_all_upload:
                    files_to_process = len(uploaded_files)
                    st.info(f"✅ Sẽ phân tích **TẤT CẢ {files_to_process} files**")
                else:
                    if len(uploaded_files) == 1:
                        # Special case: only 1 file - no need for slider
                        files_to_process = 1
                        st.info(f"📊 Sẽ phân tích **{files_to_process}/{len(uploaded_files)} file** duy nhất")
                    else:
                        # Multiple files - show slider
                        files_to_process = st.slider(
                            "📊 Chọn số lượng file để phân tích:", 
                            min_value=1, 
                            max_value=len(uploaded_files), 
                            value=min(5, len(uploaded_files)),
                            help=f"Chọn từ 1 đến {len(uploaded_files)} files"
                        )
                        st.info(f"📊 Sẽ phân tích **{files_to_process}/{len(uploaded_files)} files** đã chọn")
            
            # Hiển thị danh sách files sẽ được phân tích (preview)
            with st.expander("👀 Preview files sẽ được phân tích"):
                files_to_show = uploaded_files[:files_to_process]
                for i, file in enumerate(files_to_show, 1):
                    st.text(f"{i}. {file.name}")
                if files_to_process < len(uploaded_files):
                    st.text(f"... và bỏ qua {len(uploaded_files) - files_to_process} files khác")
            
            # Analyze button (stores to temporary results)
            analyze_button = st.button("🔍 Phân Tích Files", type="primary", help="Phân tích files và hiển thị preview kết quả")
            
            if analyze_button:
                files_to_analyze = uploaded_files[:files_to_process]
                
                # Thông báo cho batch lớn upload với time estimate theo model
                if files_to_process > 50:
                    # Điều chỉnh time estimate theo model
                    time_multiplier = 1.0
                    if "lite" in selected_model.lower():
                        time_multiplier = 0.6  # Nhanh hơn 40%
                    elif "pro" in selected_model.lower():
                        time_multiplier = 1.5  # Chậm hơn 50%
                    
                    min_time = int(files_to_process * 0.5 * time_multiplier)
                    max_time = int(files_to_process * 1.0 * time_multiplier)
                    
                    st.warning(f"⚠️ Đang phân tích {files_to_process} files với **{model_info['name']}**. Quá trình có thể mất {min_time}-{max_time} phút.")
                    st.info(f"🤖 Model hiện tại: **{model_info['name']}** ({model_info['speed']})")
                
                # Bắt đầu phân tích ngay lập tức (use placeholder so we can update it)
                start_msg = st.empty()
                start_msg.success(f"🚀 Bắt đầu phân tích {len(files_to_analyze)} files...")

                progress_bar = st.progress(0)
                status_text = st.empty()
                success_count = 0
                error_count = 0
                processed_count = 0
                
                # Container stats cho upload
                stats_container = st.empty()
                
                # Chỉ xử lý số files theo setting
                for i, uploaded_file in enumerate(files_to_analyze):
                    # Show status while processing current file
                    status_text.text(f"📄 Upload - Đang phân tích ({i+1}/{len(files_to_analyze)}): {uploaded_file.name[:50]}...")

                    # Hiển thị stats real-time (completed uses i+1)
                    with stats_container.container():
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Đã hoàn thành", f"{i+1}/{len(files_to_analyze)}")
                        with col2:
                            st.metric("Thành công", success_count)
                        with col3:
                            st.metric("Lỗi", error_count)
                    
#...
                    try:
                        # Trích xuất text
                        text_content = extract_text_from_document(uploaded_file)
                        
                        if text_content:
                            # Phân tích với AI
                            safe_filename = getattr(uploaded_file, 'name', 'unknown_file') or 'unknown_file'
                            result = analyze_document_with_gemini(model, text_content, safe_filename)
                            
                            if result:
                                # Debug theory extraction
                                debug_theory_extraction(result, safe_filename)
                                
                                # Validate and fix variable classification to prevent duplicates
                                result = validate_variable_classification(result)
                                
                                # Ensure result is a dict and add filename
                                if isinstance(result, dict):
                                    result['filename'] = safe_filename
                                    # Store in TEMPORARY storage instead of main results
                                    st.session_state.temp_analysis_results.append(result)
                                    success_count += 1
                                    
                                    # Do NOT add to learned library yet - only after Save and Analyze
                                else:
                                    st.error(f"Analysis result is not a dictionary: {type(result)}")
                                    error_count += 1
                            else:
                                error_count += 1
                        else:
                            error_count += 1
                            
                    except Exception as e:
                        error_count += 1
                        st.error(f"Lỗi với file {uploaded_file.name}: {str(e)}")
                    
                    # Pause nhỏ để tránh rate limit với batch lớn
                    if len(files_to_analyze) > 100 and i % 10 == 0:
                        import time
                        time.sleep(1)

                    # Update progress after processing this file
                    processed_count += 1
                    progress_percent = processed_count / len(files_to_analyze)
                    progress_bar.progress(progress_percent)
                
                progress_bar.progress(1.0)
                status_text.text("✅ Hoàn thành phân tích!")
                try:
                    start_msg.success(f"🎉 Đã phân tích xong {processed_count} file(s)! Thành công: {success_count}, Lỗi: {error_count}")
                except Exception:
                    # If placeholder was removed or not available, ignore
                    pass

                # Summary cuối cùng cho upload (use processed_count to reflect actual processed files)
                st.success(f"🎉 Đã phân tích xong {processed_count} file(s)! Thành công: {success_count}, Lỗi: {error_count}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("✅ Thành công", success_count, delta=f"{success_count/len(files_to_analyze)*100:.1f}%")
                with col2:
                    st.metric("❌ Lỗi", error_count, delta=f"{error_count/len(files_to_analyze)*100:.1f}%")
                
                # Set flag that analysis is ready to be saved
                if success_count > 0:
                    st.session_state.analysis_ready_to_save = True
                    st.info("📋 **Kết quả phân tích đã sẵn sàng!** Nhấn nút 'Save and Analyze' để lưu vào hệ thống và sử dụng ở các tab khác.")
        
        # Show Save and Analyze button if there are temporary results
        if st.session_state.analysis_ready_to_save and st.session_state.temp_analysis_results:
            st.divider()
            st.subheader("💾 Lưu và Áp Dụng Kết Quả")
            
            # Preview temporary results
            st.write(f"📋 **Preview kết quả phân tích:** {len(st.session_state.temp_analysis_results)} files")
            
            with st.expander("👀 Xem danh sách files đã phân tích"):
                for i, result in enumerate(st.session_state.temp_analysis_results, 1):
                    filename = result.get('filename', 'Unknown')
                    title = result.get('tieu_de', 'No title')[:50] + "..." if len(result.get('tieu_de', '')) > 50 else result.get('tieu_de', 'No title')
                    st.write(f"{i}. **{filename}** - {title}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                save_and_analyze_button = st.button(
                    "💾 Save and Analyze", 
                    type="primary", 
                    help="Lưu kết quả vào hệ thống và cho phép sử dụng ở các tab khác"
                )
            
            with col2:
                discard_button = st.button(
                    "🗑️ Discard", 
                    help="Bỏ kết quả phân tích này"
                )
            
            if save_and_analyze_button:
                # Hiển thị thanh progress và checklist
                st.subheader("⏳ Đang lưu và phân tích dữ liệu...")
                progress_bar = st.progress(0)
                status_area = st.empty()
                checklist_area = st.empty()
                
                # Danh sách các bước cần thực hiện
                total_steps = 4
                steps = [
                    "⏳ Chuẩn bị dữ liệu phân tích",
                    "⏳ Chuyển dữ liệu vào hệ thống chính",
                    "⏳ Cập nhật thư viện học tập",
                    "⏳ Lưu kết quả vào bộ nhớ"
                ]
                
                # Hiển thị checklist ban đầu
                with checklist_area.container():
                    for step in steps:
                        st.markdown(step)
                
                # Bước 1: Chuẩn bị dữ liệu
                progress_bar.progress(1/total_steps)
                with checklist_area.container():
                    st.markdown("✅ Chuẩn bị dữ liệu phân tích")
                    for step in steps[1:]:
                        st.markdown(step)
                status_area.info("🔄 Đang chuẩn bị dữ liệu phân tích...")
                time.sleep(0.5)  # Thêm độ trễ để người dùng thấy tiến trình
                
                # Bước 2: Chuyển dữ liệu vào hệ thống chính
                progress_bar.progress(2/total_steps)
                with checklist_area.container():
                    st.markdown("✅ Chuẩn bị dữ liệu phân tích")
                    st.markdown("✅ Chuyển dữ liệu vào hệ thống chính")
                    for step in steps[2:]:
                        st.markdown(step)
                status_area.info("🔄 Đang chuyển dữ liệu vào hệ thống chính...")
                
                # Move temporary results to main results
                for temp_result in st.session_state.temp_analysis_results:
                    st.session_state.analysis_results.append(temp_result)
                
                # Clear temporary results to prevent duplicates
                st.session_state.temp_analysis_results = []
                st.session_state.analysis_ready_to_save = False
                
                # Remove duplicates by filename to prevent same file being added multiple times
                seen_filenames = set()
                unique_results = []
                for result in st.session_state.analysis_results:
                    filename = result.get('filename', '')
                    if filename not in seen_filenames:
                        seen_filenames.add(filename)
                        unique_results.append(result)
                
                st.session_state.analysis_results = unique_results
                
                # Debug info
                original_count = len(st.session_state.analysis_results) + len(seen_filenames) - len(unique_results)
                st.info(f"🔄 **Đã loại bỏ {original_count - len(unique_results)} duplicates** - Giữ lại {len(unique_results)} files unique")
                time.sleep(0.5)  # Thêm độ trễ để người dùng thấy tiến trình
                
                # Bước 3: Cập nhật thư viện học tập
                progress_bar.progress(3/total_steps)
                with checklist_area.container():
                    st.markdown("✅ Chuẩn bị dữ liệu phân tích")
                    st.markdown("✅ Chuyển dữ liệu vào hệ thống chính")
                    st.markdown("✅ Cập nhật thư viện học tập")
                    for step in steps[3:]:
                        st.markdown(step)
                status_area.info("🔄 Đang cập nhật thư viện học tập...")
                
                # Add to learned library
                success_count = 0
                error_count = 0
                for temp_result in st.session_state.temp_analysis_results:
                    try:
                        safe_result = dict(temp_result)
                        learned_entry = add_to_learned_library(safe_result, {
                            'source': 'upload_analysis',
                            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                        success_count += 1
                    except Exception as library_error:
                        error_count += 1
                        st.error(f"Lỗi khi thêm vào thư viện cho {safe_result.get('filename', 'Unknown')}: {library_error}")
                
                time.sleep(0.5)  # Thêm độ trễ để người dùng thấy tiến trình
                
                # Bước 4: Lưu kết quả vào bộ nhớ
                progress_bar.progress(4/total_steps)
                with checklist_area.container():
                    st.markdown("✅ Chuẩn bị dữ liệu phân tích")
                    st.markdown("✅ Chuyển dữ liệu vào hệ thống chính")
                    st.markdown("✅ Cập nhật thư viện học tập")
                    st.markdown("✅ Lưu kết quả vào bộ nhớ")
                status_area.info("🔄 Đang lưu kết quả vào bộ nhớ...")
                
                # Save learned library
                saved_count = len(st.session_state.temp_analysis_results)
                save_success = save_learned_library(st.session_state.learned_library)
                time.sleep(0.5)  # Thêm độ trễ để người dùng thấy tiến trình
                
                # Hiển thị kết quả cuối cùng
                progress_bar.progress(1.0)
                if save_success:
                    status_area.success(f"✅ **Đã lưu thành công {saved_count} kết quả phân tích!**")
                    st.info("🔍 Chuyển sang các tab khác để xem kết quả:\n"
                           "- **Kết Quả Phân Tích**: Xem bảng chi tiết\n"
                           "- **Thống Kê Phân Tích**: Xem thống kê frequency\n"
                           "- **Research Gap Analysis**: Phân tích gaps và roadmap")
                else:
                    status_area.warning("⚠️ Đã lưu kết quả nhưng không thể lưu learned library")
                
                # Hiển thị thống kê
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("✅ Thành công", success_count)
                with col2:
                    st.metric("❌ Lỗi", error_count)
                
                # Clear temporary storage and reset flag
                st.session_state.temp_analysis_results = []
                st.session_state.analysis_ready_to_save = False
                st.session_state.has_reset = False  # Allow future loading of learned library
                
                # Thêm nút để tiếp tục
                if st.button("Tiếp tục", type="primary"):
                    st.rerun()
            
            elif discard_button:
                # Clear temporary results
                st.session_state.temp_analysis_results = []
                st.session_state.analysis_ready_to_save = False
                st.success("🗑️ Đã bỏ kết quả phân tích tạm thời.")
                st.rerun()
    
    with tab2:
        st.header("📊 Kết Quả Phân Tích")
        
        # Check if there are temporary results waiting to be saved
        if st.session_state.temp_analysis_results and not st.session_state.analysis_results:
            st.warning("⚠️ **Có kết quả phân tích chưa được lưu!**\n\n"
                      f"Có {len(st.session_state.temp_analysis_results)} file(s) đã được phân tích nhưng chưa được lưu vào hệ thống.\n"
                      "Vui lòng quay lại tab **'Upload File Mới'** và nhấn **'Save and Analyze'** để lưu kết quả.")
        elif st.session_state.temp_analysis_results and st.session_state.analysis_results:
            st.info(f"ℹ️ Hiển thị {len(st.session_state.analysis_results)} kết quả đã lưu. "
                   f"Có thêm {len(st.session_state.temp_analysis_results)} kết quả chưa lưu ở tab Upload.")
        
        if st.session_state.analysis_results:
            st.success(f"📈 Có **{len(st.session_state.analysis_results)}** kết quả phân tích")
            
            # View Mode Toggle
            st.subheader("🔄 **Chế độ hiển thị**")
            
            col_toggle1, col_toggle2, col_toggle3 = st.columns([2, 2, 2])
            
            # Removed view mode toggle - only Original View
            view_mode = "📋 Original View"
            
            with col_toggle1:
                st.info("**📋 Authors Separated View:**\n1 author = 1 row\nEach author gets separate row")
            
            # Check for and remove duplicates before creating DataFrame
            if st.session_state.analysis_results:
                original_count = len(st.session_state.analysis_results)
                seen_filenames = set()
                unique_results = []
                
                for result in st.session_state.analysis_results:
                    filename = result.get('filename', '')
                    if filename not in seen_filenames:
                        seen_filenames.add(filename)
                        unique_results.append(result)
                
                # Update session state if duplicates found
                if len(unique_results) != original_count:
                    removed_count = original_count - len(unique_results)
                    st.session_state.analysis_results = unique_results
                    st.warning(f"🔄 **Tự động xóa {removed_count} duplicates!** Còn lại {len(unique_results)} papers unique.")
            
            # Create DataFrame - Authors separated (1 author = 1 row)
            df = normalize_analysis_data_authors(st.session_state.analysis_results)
            view_type = "authors_separated"
            st.subheader("📋 **Bảng Systematic Literature Review**")
            st.info(f"📊 **{len(df)} author-paper combinations** trong bảng")
            
            # Save DataFrame to session state for Tab 3 to use
            st.session_state.tab2_dataframe = df
            st.session_state.tab2_view_type = view_type
            
            # Display DataFrame
            st.dataframe(df, use_container_width=True)
            
            # Export options
            st.subheader("💾 **Tải xuống kết quả**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Export as XLSX
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Literature Review', index=False)
                excel_buffer.seek(0)
                
                st.download_button(
                    label="📥 Tải Excel",
                    data=excel_buffer.getvalue(),
                    file_name=f"literature_review_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Export literature review as Excel file"
                )
            
            with col2:
                # CSV export
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Tải CSV",
                    data=csv,
                    file_name=f"literature_review_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Export literature review as CSV file"
                )
            
            with col3:
                col3a, col3b = st.columns(2)
                with col3a:
                    if st.button("🗑️ Xóa tất cả", help="Xóa toàn bộ analysis results"):
                        st.session_state.analysis_results = []
                        st.experimental_rerun()
                with col3b:
                    if st.button("🔄 Xóa duplicates", help="Xóa các papers trùng lặp"):
                        # Remove duplicates by filename
                        seen_filenames = set()
                        unique_results = []
                        for result in st.session_state.analysis_results:
                            filename = result.get('filename', '')
                            if filename not in seen_filenames:
                                seen_filenames.add(filename)
                                unique_results.append(result)
                        
                        removed_count = len(st.session_state.analysis_results) - len(unique_results)
                        st.session_state.analysis_results = unique_results
                        
                        if removed_count > 0:
                            st.success(f"✅ Đã xóa {removed_count} duplicates! Còn lại {len(unique_results)} papers unique.")
                        else:
                            st.info("ℹ️ Không có duplicates nào để xóa.")
                        st.experimental_rerun()
            
            # Thống kê tổng quan
            st.subheader("📈 Thống kê tổng quan")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📄 Tổng số tài liệu", len(df))
            
            with col2:
                # Năm xuất bản trung bình
                years = df["Năm xuất bản"].str.extract(r'(\d{4})').dropna()
                if not years.empty:
                    avg_year = years[0].astype(int).mean()
                    st.metric("📅 Năm TB", f"{avg_year:.0f}")
                else:
                    st.metric("📅 Năm TB", "N/A")
            
            with col3:
                # Số tài liệu có mô hình
                has_model = df["Mô hình có hay không"].str.contains("Yes|yes|Có|có", na=False).sum()
                st.metric("🏗️ Có mô hình", f"{has_model}/{len(df)}")
            
            with col4:
                # Tỷ lệ hoàn thành thông tin
                complete_info = df.apply(lambda row: (row != "N/A").sum() / len(row) * 100, axis=1).mean()
                st.metric("✅ Tỷ lệ thông tin", f"{complete_info:.1f}%")
        
        else:
            st.info("📝 Chưa có kết quả phân tích nào. Hãy upload file để bắt đầu!")
    
    with tab3:
        st.header("📈 Thống Kê Phân Tích")
        
        # Check if there are temporary results waiting to be saved
        if st.session_state.temp_analysis_results and not st.session_state.analysis_results:
            st.warning("⚠️ **Có kết quả phân tích chưa được lưu!**\n\n"
                      f"Có {len(st.session_state.temp_analysis_results)} file(s) đã được phân tích nhưng chưa được lưu vào hệ thống.\n"
                      "Vui lòng quay lại tab **'Upload File Mới'** và nhấn **'Save and Analyze'** để lưu kết quả.")
        elif st.session_state.temp_analysis_results and st.session_state.analysis_results:
            st.info(f"ℹ️ Hiển thị thống kê từ {len(st.session_state.analysis_results)} kết quả đã lưu. "
                   f"Có thêm {len(st.session_state.temp_analysis_results)} kết quả chưa lưu ở tab Upload.")
        
        if st.session_state.analysis_results:
            # Check if Tab 2 has DataFrame ready
            if hasattr(st.session_state, 'tab2_dataframe') and st.session_state.tab2_dataframe is not None:
                # Use DataFrame from Tab 2 (exactly what user sees)
                df = st.session_state.tab2_dataframe
                view_type = st.session_state.tab2_view_type
                
                st.success(f"📊 Phân tích thống kê trên **{len(st.session_state.analysis_results)}** tài liệu")
                st.info("ℹ️ **Sử dụng dữ liệu từ Tab 2:** Thống kê dựa trên bảng literature review")
            else:
                # Fallback: create original DataFrame if Tab 2 not visited yet
                df = create_analysis_dataframe(st.session_state.analysis_results)
                view_type = "original"
                st.success(f"📊 Phân tích thống kê trên **{len(st.session_state.analysis_results)}** tài liệu")
                st.warning("⚠️ **Chưa xem Tab 2.** Đang sử dụng dữ liệu gốc. Hãy xem Tab 2 để có thống kê chính xác.")
            
            # DEBUG INFO
            st.info(f"🐛 **DEBUG INFO:**\n"
                   f"- Real papers (session_state): {len(st.session_state.analysis_results)}\n"
                   f"- Current DataFrame rows: {len(df)}\n"
                   f"- View type: {view_type}")
            if view_type == "authors_separated":
                st.info(f"📊 **Sử dụng DataFrame từ Tab 2** - View Type: **Authors Separated**\n"
                       f"📋 **{len(df)}** author-paper combinations với đầy đủ thông tin")
            else:
                st.info(f"📊 **Sử dụng DataFrame từ Tab 2** - View Type: **{view_type}**\n"
                       f"📋 **{len(df)}** rows với đầy đủ thông tin")
            original_total_docs = len(st.session_state.analysis_results)  # FIXED: Real paper count
            total_variables = len(df)
            
            # ===== LITERATURE REVIEW ANALYSIS =====
            st.subheader("🅰️ **A. Tổng Quan Literature Review**")
            
            # Show basic statistics
            st.write(f"**📊 Tổng số papers:** {len(df)}")
            st.write(f"**📊 Tổng số columns:** {len(df.columns)}")
            
            st.markdown("---")
            
            # ===== TRADITIONAL ANALYSIS (UPDATED FOR CURRENT VIEW) =====
            
            def analyze_frequency_normalized(variable_column, title):
                """Analyze frequency for normalized data structure"""
                st.subheader(f"📊 {title}")
                st.caption(f"🔍 **Chỉ phân tích biến trong vai trò: {variable_column}**")
                
                if variable_column in df.columns:
                    all_values = []
                    papers_with_variables = {}  # Track which papers contain each variable
                    
                    # Iterate through normalized data to track variables per paper
                    for idx, row in df.iterrows():
                        value = row[variable_column]
                        paper_title = row.get("Tiêu đề bài nghiên cứu (Title)", f"Paper_{idx}")
                        
                        if value and value != "N/A" and str(value).strip():
                            value_str = normalize_token(value)
                            if (value_str != "no information" and 
                                value_str != "không có thông tin" and 
                                value_str != "no explicit information" and
                                value_str != "n/a" and 
                                value_str != "na" and
                                value_str != "" and
                                len(value_str) > 2):
                                clean_value = value_str
                                all_values.append(clean_value)
                                
                                # Track which papers contain this variable
                                if clean_value not in papers_with_variables:
                                    papers_with_variables[clean_value] = set()
                                papers_with_variables[clean_value].add(paper_title)
                    
                    if all_values:
                        frequency = pd.Series(all_values).value_counts()
                        total_papers = len(st.session_state.analysis_results)  # FIXED: Số papers thực tế
                        total_variable_instances = len(all_values)  # Tổng số lần biến xuất hiện
                        
                        # FIXED: Calculate Papers Using (%) correctly based on unique papers
                        papers_using_percent = []
                        for var_name in frequency.index:
                            unique_papers_count = len(papers_with_variables.get(var_name, set()))
                            papers_using_percent.append(round(unique_papers_count / total_papers * 100, 1))
                        
                        frequency_df = pd.DataFrame({
                            'Variable Name': frequency.index,
                            'Variable Share (%)': (frequency.values / total_variable_instances * 100).round(1),  # % trong tổng số lần biến xuất hiện
                            'Papers Using (%)': papers_using_percent,  # FIXED: % papers có chứa biến này
                            'Papers Count': [len(papers_with_variables.get(var_name, set())) for var_name in frequency.index],  # Số papers chứa biến
                            'Frequency': frequency.values  # Số lần biến xuất hiện
                        })
                        
                        # Sắp xếp lại thứ tự cột
                        frequency_df = frequency_df[['Variable Name', 'Papers Count', 'Papers Using (%)', 'Variable Share (%)', 'Frequency']]
                        
                        st.dataframe(frequency_df.head(10), use_container_width=True)
                        
                        # Debug: Show specific examples
                        if len(frequency_df) > 0:
                            top_var = frequency_df.iloc[0]
                            st.info(f"📝 **Ví dụ**: Biến '{top_var['Variable Name']}' trong vai trò **{variable_column}** có trong {top_var['Papers Count']} papers = {top_var['Papers Using (%)']}%")
                        
                        # Giải thích công thức tính toán
                        st.caption(f"""
                        📊 **Công thức tính:**
                        - **Papers Count**: Số papers có chứa biến này trong vai trò **{variable_column}** (unique papers)
                        - **Papers Using (%)**: Papers Count / {total_papers} papers total × 100%
                        - **Variable Share (%)**: % của biến này trong tổng số {total_variable_instances} lần xuất hiện của tất cả biến trong vai trò **{variable_column}**
                        - **Frequency**: Tổng số lần biến xuất hiện trong vai trò **{variable_column}**
                        """)
                        
                        # Removed bar chart
                        
                        return frequency_df
                    else:
                        st.info(f"Không có {title} có ý nghĩa để phân tích")
                        return pd.DataFrame()
                else:
                    st.warning(f"Column {variable_column} không tồn tại trong normalized data")
                    return pd.DataFrame()
            
            def ai_clean_extracted_data(raw_values, column_name, model):
                """AI-powered data cleaning instead of manual rules"""
                if not model or not raw_values:
                    return raw_values  # Fallback to original if AI unavailable
                
                # Determine what type of data we're cleaning
                data_type = "general"
                if "lý thuyết" in column_name.lower() or "theory" in column_name.lower():
                    data_type = "theories"
                elif "phân tích" in column_name.lower() or "method" in column_name.lower():
                    data_type = "methods"
                elif "thang đo" in column_name.lower() or "measurement" in column_name.lower():
                    data_type = "scales"
                elif "lĩnh vực" in column_name.lower() or "field" in column_name.lower():
                    data_type = "fields"
                
                # Create AI prompt for intelligent cleaning
                prompt = f"""
                Nhiệm vụ: Làm sạch và validate danh sách {data_type} từ systematic literature review.
                
                Dữ liệu đầu vào ({len(raw_values)} items):
                {chr(10).join([f"- {item}" for item in raw_values[:50]])}  
                
                YÊU CẦU CLEANING:
                
                1️⃣ **LOẠI BỎ (Remove completely):**
                - Chỉ có năm (VD: "1985", "1988)", "(2003)")
                - Tên biến nghiên cứu thay vì {data_type} (VD: "Customer Satisfaction", "Trust", "Quality")
                - Câu không hoàn chỉnh/đoạn văn bị cắt (VD: "One to 10) for performance", "However it")
                - Placeholder text (VD: "No specific", "Not specified", "No information")
                - Text quá ngắn không có ý nghĩa (< 5 ký tự)
                
                2️⃣ **GIỮ LẠI (Keep and clean):**
                - Tên {data_type} hợp lệ và có ý nghĩa
                - Theory names, model names, framework names (nếu data_type = theories)
                - Statistical method names (nếu data_type = methods)  
                - Scale names và measurement instruments (nếu data_type = scales)
                - Research field names (nếu data_type = fields)
                
                3️⃣ **CHUẨN HÓA (Standardize):**
                - Viết hoa chữ cái đầu
                - Loại bỏ dấu câu thừa
                - Gộp các tên gần giống nhau
                
                OUTPUT FORMAT: Trả về JSON array với chỉ các items đã cleaned:
                ["Item 1", "Item 2", "Item 3"]
                
                ⚠️ CHỈ trả về JSON array, không có text nào khác.
                """
                
                try:
                    response = model.generate_content(prompt)
                    response_text = extract_response_text(response)
                    
                    # Extract JSON from response
                    start_idx = response_text.find('[')
                    end_idx = response_text.rfind(']') + 1
                    
                    if start_idx != -1 and end_idx > start_idx:
                        json_str = response_text[start_idx:end_idx]
                        cleaned_list = json.loads(json_str)
                        
                        # Validate that it's a list of strings
                        if isinstance(cleaned_list, list):
                            return [str(item).strip() for item in cleaned_list if str(item).strip()]
                        else:
                            return raw_values  # Fallback
                    else:
                        return raw_values  # Fallback if no JSON found
                        
                except Exception as e:
                    st.warning(f"⚠️ AI cleaning failed: {e}. Using original data.")
                    return raw_values  # Fallback on error
            
            def analyze_frequency_original(column_name, title):
                """Original analyze_frequency for theories with correct Papers Using % calculation"""
                st.subheader(f"📊 {title}")
                all_values = []
                papers_with_items = {}  # Track which papers contain each item
                
                # Use current DataFrame for analysis
                if column_name in df.columns:
                    for idx, row in df.iterrows():
                        value = row[column_name]
                        paper_title = row.get("Tiêu đề bài nghiên cứu (Title)", f"Paper_{idx}")
                        
                        if value and value != "N/A":
                            # Filter out placeholder values that have no meaningful content
                            value_str = normalize_token(value)
                            if (value_str != "no information" and 
                                value_str != "không có thông tin" and 
                                value_str != "no explicit information" and
                                value_str != "n/a" and 
                                value_str != "na" and
                                value_str != "" and
                                len(value_str) > 2):
                                
                                # Enhanced splitting for better extraction
                                items = str(value).replace(";", ",").replace("&", ",").replace(" and ", ",").replace("+", ",").split(",")
                                for item in items:
                                    clean_item = normalize_token(item)
                                    clean_item_lower = clean_item
                                    
                                    # Additional filtering per item
                                    if (clean_item and 
                                        len(clean_item) > 2 and
                                        clean_item_lower != "no information" and
                                        clean_item_lower != "không có thông tin" and
                                        clean_item_lower != "no explicit information" and
                                        clean_item_lower != "n/a" and
                                        clean_item_lower != "na" and
                                        clean_item_lower != "none" and
                                        clean_item_lower != "not available" and
                                        clean_item_lower != "not specified"):
                                        
                                        # AI-powered validation instead of manual rules
                                        if len(clean_item.strip()) > 3:  # Basic length check only
                                            all_values.append(clean_item)
                                            
                                            # Track which papers contain this item
                                            if clean_item not in papers_with_items:
                                                papers_with_items[clean_item] = set()
                                            papers_with_items[clean_item].add(paper_title)
                
                if all_values:
                    # AI-powered cleaning before frequency analysis
                    with st.spinner("🤖 AI đang làm sạch dữ liệu..."):
                        cleaned_values = ai_clean_extracted_data(all_values, column_name, model)
                    
                    if cleaned_values:
                        frequency = pd.Series(cleaned_values).value_counts()
                        total_papers = len(st.session_state.analysis_results)  # Real paper count
                        
                        # FIXED: Calculate Papers Using (%) correctly
                        papers_using_percent = []
                        papers_count = []
                        for item_name in frequency.index:
                            unique_papers_count = len(papers_with_items.get(item_name, set()))
                            papers_count.append(unique_papers_count)
                            papers_using_percent.append(round(unique_papers_count / total_papers * 100, 1))
                        
                        frequency_df = pd.DataFrame({
                            'Tên': frequency.index,
                            'Papers Count': papers_count,  # Số papers chứa item này
                            'Papers Using (%)': papers_using_percent,  # FIXED: % papers có chứa item này
                            'Frequency': frequency.values  # Số lần xuất hiện
                        })
                        
                        st.dataframe(frequency_df.head(10), use_container_width=True)
                        
                        # Giải thích công thức tính toán
                        st.caption(f"""
                        📊 **Công thức tính:**
                        - **Papers Count**: Số papers có chứa {title.lower()} này (unique papers)
                        - **Papers Using (%)**: Papers Count / {total_papers} papers total × 100%
                        - **Frequency**: Tổng số lần {title.lower()} xuất hiện (có thể > Papers Count nếu 1 paper có nhiều lần)
                        """)
                        
                        # Removed bar chart
                        
                        return frequency_df
                    else:
                        st.info("🤖 AI đã filter hết dữ liệu không phù hợp")
                        return pd.DataFrame()
                else:
                    st.info("Không có dữ liệu có ý nghĩa để phân tích")
                    return pd.DataFrame()
            
            # Main Statistical Analysis Categories
            st.markdown("---")
            st.markdown("## 🅱️ **B. Phân Tích Biến Nghiên Cứu**")
            
            col1, col2 = st.columns(2)
            with col1:
                dependent_stats = analyze_frequency_original("Biến phụ thuộc", "Biến Phụ Thuộc")
            with col2:
                independent_stats = analyze_frequency_original("Biến độc lập", "Biến Độc Lập")
            
            col3, col4 = st.columns(2)
            with col3:
                mediating_stats = analyze_frequency_original("Biến trung gian", "Biến Trung Gian")
            with col4:
                moderating_stats = analyze_frequency_original("Biến điều tiết", "Biến Điều Tiết")
            
            col5, col6 = st.columns(2)
            with col5:
                control_stats = analyze_frequency_original("Biến kiểm soát", "Biến Kiểm Soát")
            with col6:
                other_vars_stats = analyze_frequency_original("Biến khác nếu có", "Biến Khác")
            
            st.markdown("---")
            st.markdown("## 🆎 **C. Lý Thuyết & Phương Pháp**")
            
            col7, col8 = st.columns(2)
            with col7:
                theory_stats = analyze_frequency_original("Lý Thuyết nền", "Lý Thuyết Nền")

                # ===== Thêm bảng mô tả lý thuyết =====
                st.markdown("### 📚 **Chi Tiết Lý Thuyết Nền**")
                if "ai_theory_desc" not in st.session_state:
                    st.session_state["ai_theory_desc"] = {}

                def split_theories(text: str):
                    if not text:
                        return []
                    delimiters = [",", ";", "&", " và ", " and ", "|", "\n", "+"]
                    parts = [text]
                    for d in delimiters:
                        newp = []
                        for p in parts:
                            newp.extend(p.split(d))
                        parts = newp
                    return [p.strip().title() for p in parts if p.strip()]

                # Thu thập lý thuyết và bài báo liên quan
                theory_papers = {}
                for _idx, _row in df.iterrows():
                    theories_list = split_theories(str(_row.get("Lý Thuyết nền", "")))
                    for th in theories_list:
                        theory_papers.setdefault(th, []).append(_row["Tiêu đề bài nghiên cứu (Title)"])

                # Lấy mô tả bằng Gemini nếu chưa cache
                import google.generativeai as genai
                # Sử dụng API key từ danh sách
                api_key_secret = None
                if 'api_keys' in st.session_state and st.session_state.api_keys:
                    api_key_secret = st.session_state.api_keys[st.session_state.current_key_index]
                else:
                    # Fallback to secrets/env if session state not initialized
                    api_key_secret = globals().get("api_key_secret", st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY")))
                
                model_name = globals().get("model_name", st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash"))
                if api_key_secret:
                    genai.configure(api_key=api_key_secret)
                    model_desc = genai.GenerativeModel(model_name)
                
                # Tạo context text từ các bài đã upload (giới hạn 20.000 ký tự để tránh quá dài)
                context_chunks = []
                if "analysis_results" in st.session_state and st.session_state.analysis_results:
                    st.write("📂 analysis_results keys:", st.session_state.analysis_results[0].keys())
                    for res in st.session_state.analysis_results:
                        # Ưu tiên raw_text vì nó chứa nội dung gốc
                        content = res.get("raw_text") or res.get("content", "")
                        if content and isinstance(content, str):
                            context_chunks.append(content[:2000])
                context_text = "\n\n".join(context_chunks)[:20000]  # Thêm dấu cách giữa các bài
                
                # DEBUG: Kiểm tra dữ liệu
                with st.expander("🔍 DEBUG: Kiểm tra dữ liệu"):
                    st.write("Số bài đã upload:", len(context_chunks))
                    st.write("Tổng ký tự context_text:", len(context_text))
                    st.write("10 dòng đầu context_text:")
                    st.code("\n".join(context_text.split("\n")[:10]))
                    
                    # Test Gemini response
                    if api_key_secret:
                        try:
                            test_resp = model_desc.generate_content("Hello, respond with 'TEST_OK' if you can read this.")
                            st.write("Gemini test response:", test_resp.text if hasattr(test_resp, "text") else str(test_resp)[:200])
                        except Exception as e:
                            st.error(f"Gemini error: {str(e)}")

                description_rows = []
                for th, papers in theory_papers.items():
                    cache = st.session_state["ai_theory_desc"]
                    if th not in cache:
                        desc = "-"; source = "-"
                        if api_key_secret and context_text.strip():
                            # Step 1: Try to find representative sentence inside context
                            try:
                                prompt1 = (
                                    f"Find ONE representative sentence (<=35 words) in the text below that explains the theory '{th}'. "
                                    "If no such sentence exists, reply with ONLY the word NONE.\n\nTEXT:\n" + context_text)
                                resp1 = model_desc.generate_content(prompt1, safety_settings={"HARASSMENT":"block_none"})
                                sent = resp1.text.strip()
                                if sent.upper() != "NONE" and len(sent.split()) <= 40:
                                    desc = sent
                                    source = "Gemini-match"
                                else:
                                    # Step 2: general concise explanation
                                    prompt2 = f"Explain concisely the {th} in max 35 words."
                                    resp2 = model_desc.generate_content(prompt2, safety_settings={"HARASSMENT":"block_none"})
                                    desc = resp2.text.strip()
                                    source = "Gemini-summary"
                            except Exception:
                                desc = "-"; source = "Error"
                        cache[th] = {"desc": desc, "source": source}
                    else:
                        desc = cache[th]["desc"]
                        source = cache[th]["source"]

                    description_rows.append({
                        "Theory": th,
                        "Description": desc,
                        "Source": source,
                        "Papers": ", ".join(list(set(papers)))
                    })

                theory_df = pd.DataFrame(description_rows)
                st.dataframe(theory_df, use_container_width=True)
            with col8:
                method_stats = analyze_frequency_original("PP phân tích dữ liệu (Statistical method)", "Phương Pháp Phân Tích")
            
            col9, col10 = st.columns(2)
            with col9:
                # Remove frequency table for 'Thang đo' and display per paper
                st.markdown("## 📏 **PHÂN TÍCH THANG ĐO CHI TIẾT**")
                st.info("Hiển thị các thang đo và các mục đo lường được trích xuất từ mỗi bài nghiên cứu.")
                
                if "Thang đo (Measurement)" in df.columns:
                    measurement_data = df[df["Thang đo (Measurement)"].apply(lambda x: x.strip().lower() not in ["n/a", "no information", "không có thông tin", "no explicit information", "none", "not available", "not specified", ""])].copy()
                    
                    if not measurement_data.empty:
                        for index, row in measurement_data.iterrows():
                            scale_str = row["Thang đo (Measurement)"]
                            filename = row["Tiêu đề bài nghiên cứu (Title)"] # Using title as identifier
                            
                            with st.expander(f"**Bài báo: {filename}**"): # Use filename or title for expander
                                # Use the existing formatting function
                                formatted_output = format_complex_string_for_display(scale_str, "thang_do")
                                st.markdown(formatted_output)
                    else:
                        st.info("Không có bài báo nào có thông tin thang đo chi tiết để phân tích.")
                else:
                    st.warning("Cột 'Thang đo (Measurement)' không tồn tại.")

            with col10:
                field_stats = analyze_frequency_original("Lĩnh vực NC", "Lĩnh Vực Nghiên Cứu")

            # ====== METHOD SUMMARY ======
            st.markdown("### ⚙️ **TỔNG HỢP PHƯƠNG PHÁP PHÂN TÍCH**")
            st.caption("Thống kê tất cả phương pháp phân tích dữ liệu được ghi trong các bài, tần suất và danh sách bài sử dụng.")

            # Sử dụng API key từ danh sách
            api_key_secret = None
            if 'api_keys' in st.session_state and st.session_state.api_keys:
                api_key_secret = st.session_state.api_keys[st.session_state.current_key_index]
            else:
                # Fallback to secrets/env if session state not initialized
                api_key_secret = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
            
            model_name = st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash")

            if api_key_secret:
                use_ai_extract = True  # mặc định bật
                st.markdown("✅ Đã phát hiện GOOGLE_API_KEY trong secrets — AI extraction ENABLED")
            else:
                use_ai_extract = st.checkbox("🔮 Sử dụng Gemini AI để trích xuất phương pháp (tốn credit)")
            if use_ai_extract and "ai_method_cache" not in st.session_state:
                st.session_state["ai_method_cache"] = {}

            def extract_methods_ai(text: str):
                """Gọi Gemini flash để trích xuất danh sách phương pháp từ chuỗi đầu vào."""
                import google.generativeai as genai
                if not text:
                    return []
                try:
                    # Sử dụng hàm init_gemini để tận dụng cơ chế luân chuyển API key
                    model = init_gemini(model_name)
                    prompt = (
                        "Extract a concise, comma-separated list of statistical data analysis methods mentioned "
                        "in the following sentence. Only return the list.\nSentence: " + text
                    )
                    response = model.generate_content(prompt, safety_settings={"HARASSMENT": "block_none"})
                    methods_raw = extract_response_text(response)
                    return [m.strip().title() for m in methods_raw.split(",") if m.strip()]
                except Exception as e:
                    # Fallback to rule-based splitting if API fails
                    return split_methods(text)


            # Tách phương pháp bằng các delimiter phổ biến
            def split_methods(method_str):
                if not method_str or method_str.lower() in ["n/a", "no information", "không có thông tin", "none", "not specified"]:
                    return []
                delimiters = [",", ";", "&", " and ", " và ", "|", "\n", "+"]
                parts = [method_str]
                for d in delimiters:
                    new_parts = []
                    for p in parts:
                        new_parts.extend(p.split(d))
                    parts = new_parts
                cleaned = [p.strip().title() for p in parts if p.strip()]
                return cleaned

            # Thu thập tất cả phương pháp và papers
            method_freq = {}
            method_papers = {}
            for _idx, _row in df.iterrows():
                raw_text = str(_row["PP phân tích dữ liệu (Statistical method)"])
                if use_ai_extract:
                    cache = st.session_state["ai_method_cache"]
                    if raw_text not in cache:
                        cache[raw_text] = extract_methods_ai(raw_text)
                    methods_list = cache[raw_text]
                else:
                    methods_list = split_methods(raw_text)
                for m in methods_list:
                    method_freq[m] = method_freq.get(m, 0) + 1
                    method_papers.setdefault(m, []).append(_row["Tiêu đề bài nghiên cứu (Title)"])

            # ---- Công thức/Ghi chú cho một số phương pháp phổ biến ----
            formula_dict = {
                "Linear Regression": "Y = β₀ + ΣβᵢXᵢ + ε",
                "Multiple Linear Regression": "Y = β₀ + ΣβᵢXᵢ + ε",
                "Logistic Regression": "log(p / (1-p)) = β₀ + ΣβᵢXᵢ",
                "Probit Regression": "Φ⁻¹(p) = β₀ + ΣβᵢXᵢ",
                "Poisson Regression": "log(λ) = β₀ + ΣβᵢXᵢ",
                "Sem": "Structural Equation Model",
                "Pls-Sem": "Partial Least Squares SEM",
                "Anova": "F = MS_between / MS_within",
                "Manova": "Multivariate ANOVA",
                "T-Test": "t = (x̄₁ - x̄₂) / SE",
                "Chi-Square": "χ² = Σ (O-E)² / E",
                "Factor Analysis": "X = ΛF + ε"
            }

            # ---- Tạo bảng tổng hợp ----
            summary_rows = [
                {
                    "Method": m,
                    "Frequency": f,
                    "Papers": ", ".join(method_papers[m]),
                    "Formula/Note": formula_dict.get(m, "-")
                }
                for m, f in sorted(method_freq.items(), key=lambda x: x[1], reverse=True)
            ]
            method_df = pd.DataFrame(summary_rows)
            st.dataframe(method_df, use_container_width=True)

            # ---- Kết thúc bảng phương pháp ----

            
            st.markdown("---")
            st.subheader("📊 Mô Hình Có Hay Không")
            st.markdown("## 🆑 **D. Ngữ Cảnh & Mô Hình Nghiên Cứu**")
            
            col11, col12 = st.columns(2)
            with col11:
                location_stats = analyze_frequency_original("Địa điểm NC (Area)", "Địa Điểm Nghiên Cứu")
            with col12:
                # Model existence analysis
                st.subheader("📊 Mô Hình Có Hay Không")
                
                # Filter out meaningless model data
                meaningful_model_data = []
                for value in df["Mô hình có hay không"]:
                    if value and value != "N/A":
                        value_str = str(value).strip().lower()
                        if (value_str != "no information" and 
                            value_str != "không có thông tin" and 
                            value_str != "no explicit information" and
                            value_str != "n/a" and 
                            value_str != "na" and
                            value_str != "" and
                            len(value_str) > 2):
                            meaningful_model_data.append(value)
                
                if len(meaningful_model_data) > 0:
                    model_counts = pd.Series(meaningful_model_data).value_counts()
                    total_meaningful_models = len(meaningful_model_data)
                    
                    model_df = pd.DataFrame({
                        'Trạng thái': model_counts.index,
                        'Số lượng': model_counts.values,
                        'Tỷ lệ (%)': (model_counts.values / total_meaningful_models * 100).round(1)
                    })
                    st.dataframe(model_df, use_container_width=True)
                else:
                    st.info("Không có dữ liệu mô hình có ý nghĩa")
            
            # Model components analysis - Display as list per paper
            st.markdown("---")
            st.markdown("## 🏗️ **PHÂN TÍCH CẤU TRÚC MÔ HÌNH**")
            st.info("Hiển thị các thành phần và cấu trúc mô hình được trích xuất từ mỗi bài nghiên cứu.")
            
            if "Mô hình gồm các thành phần gì (nếu có)" in df.columns:
                model_components_data = df[df["Mô hình có hay không"].str.contains("Yes|yes|Có|có", na=False)]
                
                if not model_components_data.empty:
                    for index, row in model_components_data.iterrows():
                        model_components_str = row["Mô hình gồm các thành phần gì (nếu có)"]
                        filename = row["Tiêu đề bài nghiên cứu (Title)"] # Using title as identifier
                        
                        with st.expander(f"**Bài báo: {filename}**"): # Use filename or title for expander
                            if model_components_str and model_components_str.strip().lower() not in ["n/a", "no information", "không có thông tin", "no explicit information", "none", "not available", "not specified", ""]: # Ensure not empty after cleaning
                                # Use the existing formatting function
                                formatted_output = format_complex_string_for_display(model_components_str, "thanh_phan_mo_hinh")
                                st.markdown(formatted_output)
                            else:
                                st.info("Không có thông tin thành phần mô hình chi tiết cho bài báo này.")
                else:
                    st.info("Không có bài báo nào có mô hình để phân tích cấu trúc.")
            else:
                st.warning("Cột 'Mô hình gồm các thành phần gì (nếu có)' không tồn tại.")
            
            st.markdown("---")
            st.markdown("## 📅 **THỜI GIAN & XUẤT BẢN ANALYSIS**")
            
            col13, col14 = st.columns(2)
            with col13:
                # Year distribution
                st.subheader("📊 Phân Bố Năm Xuất Bản")
                years = df["Năm xuất bản"].str.extract(r'(\d{4})')[0].dropna()
                if len(years) > 0:
                    year_counts = years.value_counts().sort_index()
                    year_df = pd.DataFrame({
                        'Năm': year_counts.index,
                        'Số bài': year_counts.values
                    })
                    st.dataframe(year_df.tail(10), use_container_width=True)
                    # Removed line chart
                else:
                    st.info("Không có dữ liệu năm xuất bản")
            
            with col14:
                # Sample size analysis - only meaningful data
                st.subheader("📊 Thông Tin Mẫu")
                
                # Filter meaningful sample data
                meaningful_samples = []
                for value in df["Mẫu (sample)"]:
                    if value and value != "N/A":
                        value_str = str(value).strip().lower()
                        if (value_str != "no information" and 
                            value_str != "không có thông tin" and 
                            value_str != "no explicit information" and
                            value_str != "n/a" and 
                            value_str != "na" and
                            value_str != "" and
                            len(value_str) > 2):
                            meaningful_samples.append(value)
                
                if len(meaningful_samples) > 0:
                    # Extract numbers from meaningful sample descriptions only
                    import re
                    sample_numbers = []
                    for sample in meaningful_samples:
                        numbers = re.findall(r'\d+', str(sample))
                        if numbers:
                            try:
                                sample_numbers.append(int(numbers[0]))
                            except:
                                pass
                
                    if sample_numbers:
                        sample_stats = pd.DataFrame({
                            'Thống kê': ['Trung bình', 'Trung vị', 'Min', 'Max'],
                            'Giá trị': [
                                round(sum(sample_numbers)/len(sample_numbers)),
                                sorted(sample_numbers)[len(sample_numbers)//2],
                                min(sample_numbers),
                                max(sample_numbers)
                            ]
                        })
                        st.dataframe(sample_stats, use_container_width=True)
                    else:
                        st.info("Không trích xuất được số liệu mẫu có ý nghĩa")
                else:
                    st.info("Không có dữ liệu mẫu có ý nghĩa")
            
            st.markdown("---")
            st.subheader("📈 Tổng Hợp Thống Kê")
            
            # Define total documents count
            total_docs = len(st.session_state.analysis_results)  # FIXED: Real paper count 
            original_total_docs = total_docs  # For backward compatibility
            
            # Summary metrics
            col_summary1, col_summary2, col_summary3, col_summary4 = st.columns(4)
            
            def count_meaningful_values(column_name):
                """Count only meaningful values, excluding placeholders"""
                meaningful_values = []
                for value in df[column_name].unique():
                    if value and value != "N/A":
                        value_str = str(value).strip().lower()
                        if (value_str != "no information" and 
                            value_str != "không có thông tin" and 
                            value_str != "no explicit information" and
                            value_str != "n/a" and 
                            value_str != "na" and
                            value_str != "" and
                            value_str != "none" and
                            value_str != "not available" and
                            value_str != "not specified" and
                            len(value_str) > 2):
                            meaningful_values.append(value)
                return len(meaningful_values)
            
            with col_summary1:
                st.metric("Tổng tài liệu", total_docs)
                unique_theories = count_meaningful_values("Lý Thuyết nền")
                st.metric("Lý thuyết unique", unique_theories)
            
            with col_summary2:
                unique_methods = count_meaningful_values("PP phân tích dữ liệu (Statistical method)")
                st.metric("Phương pháp unique", unique_methods)
                unique_fields = count_meaningful_values("Lĩnh vực NC")
                st.metric("Lĩnh vực unique", unique_fields)
            
            with col_summary3:
                has_model = len(df[df["Mô hình có hay không"].str.contains("Yes|Có", case=False, na=False)])
                st.metric("Có mô hình", f"{has_model}/{total_docs}")
                unique_locations = count_meaningful_values("Địa điểm NC (Area)")
                st.metric("Địa điểm unique", unique_locations)
            
            with col_summary4:
                unique_measurements = count_meaningful_values("Thang đo (Measurement)")
                st.metric("Thang đo unique", unique_measurements)
                with_doi = len(df[df["Số DOI"] != "N/A"])
                st.metric("Có DOI", f"{with_doi}/{total_docs}")
            
            st.subheader("💾 Tải Xuống Thống Kê")
            if st.button("📥 Tải Xuống Báo Cáo Thống Kê Đầy Đủ"):
                # Create comprehensive statistics report
                stats_report = f"""
=== BÁO CÁO THỐNG KÊ SYSTEMATIC LITERATURE REVIEW ===
Tổng số tài liệu phân tích: {total_docs}
Ngày tạo báo cáo: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

LƯU Ý: Thống kê chỉ bao gồm dữ liệu có ý nghĩa, loại bỏ:
- "No information" / "Không có thông tin"
- "No Explicit Information"
- "N/A" / "Not available" / "Not specified"
- Các giá trị trống hoặc không rõ ràng

=== BIẾN NGHIÊN CỨU (CHỈ DỮ LIỆU CÓ Ý NGHĨA) ===
- Lý thuyết nền unique: {unique_theories}
- Phương pháp phân tích unique: {unique_methods}  
- Lĩnh vực nghiên cứu unique: {unique_fields}
- Địa điểm nghiên cứu unique: {unique_locations}
- Thang đo unique: {unique_measurements}

=== MÔ HÌNH NGHIÊN CỨU ===
- Có mô hình: {has_model}/{total_docs} ({round(has_model/total_docs*100,1)}%)
- Không mô hình: {total_docs-has_model}/{total_docs} ({round((total_docs-has_model)/total_docs*100,1)}%)

=== THÔNG TIN XUẤT BẢN ===
- Có DOI: {with_doi}/{total_docs} ({round(with_doi/total_docs*100,1)}%)

=== TOP 10 BIẾN PHỤ THUỘC (CÓ Ý NGHĨA) ===
{dependent_stats.head(10).to_string() if len(dependent_stats) > 0 else "Không có dữ liệu có ý nghĩa"}

=== TOP 10 BIẾN ĐỘC LẬP (CÓ Ý NGHĨA) ===
{independent_stats.head(10).to_string() if len(independent_stats) > 0 else "Không có dữ liệu có ý nghĩa"}

=== TOP 10 LÝ THUYẾT NỀN (CÓ Ý NGHĨA) ===
{theory_stats.head(10).to_string() if len(theory_stats) > 0 else "Không có dữ liệu có ý nghĩa"}

=== TOP 10 PHƯƠNG PHÁP PHÂN TÍCH (CÓ Ý NGHĨA) ===
{method_stats.head(10).to_string() if len(method_stats) > 0 else "Không có dữ liệu có ý nghĩa"}

=== CHẤT LƯỢNG DỮ LIỆU ===
Báo cáo này chỉ tính những thông tin có giá trị thực tế,
loại bỏ các placeholder values để đảm bảo độ chính xác
của phân tích thống kê systematic literature review.
                """
                
                st.download_button(
                    label="📄 Download Statistics Report (TXT)",
                    data=stats_report,
                    file_name=f"literature_statistics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        else:
            st.info("📝 Chưa có dữ liệu để thống kê. Hãy upload và phân tích file trước!")


    with tab4:
        st.header("🔍 Research Gaps & Opportunities")
        st.subheader("🤖 AI-Powered Research Proposal Generator")
        
        # Check if there are temporary results waiting to be saved
        if st.session_state.temp_analysis_results and not st.session_state.analysis_results:
            st.warning("⚠️ **Có kết quả phân tích chưa được lưu!**\n\n"
                      f"Có {len(st.session_state.temp_analysis_results)} file(s) đã được phân tích nhưng chưa được lưu vào hệ thống.\n"
                      "Vui lòng quay lại tab **'Upload File Mới'** và nhấn **'Save and Analyze'** để lưu kết quả trước khi sử dụng tính năng này.")
        elif st.session_state.temp_analysis_results and st.session_state.analysis_results:
            st.info(f"ℹ️ Sử dụng {len(st.session_state.analysis_results)} kết quả đã lưu để tạo research gaps. "
                   f"Có thêm {len(st.session_state.temp_analysis_results)} kết quả chưa lưu ở tab Upload.")
        
        # Kiểm tra cache trước
        if 'research_plan_cache' not in st.session_state:
            st.session_state.research_plan_cache = {}
        
        # Only use data from current session analysis (saved results)
        # Do NOT use learned_library from previous sessions
        all_research_data = []
        
        # Add ONLY from current session analysis (saved results)
        if st.session_state.analysis_results:
            all_research_data.extend(st.session_state.analysis_results)
        
        # Do NOT add from learned library to prevent showing old session data
        # This ensures Reset functionality works properly
        
        if not all_research_data:
            if st.session_state.has_reset:
                st.info("🔄 **Đã Reset - Cần Upload Files Mới**\n\nVui lòng:\n1. Upload files vào tab **'Upload File Mới'**\n2. Nhấn **'Phân Tích Files'**\n3. Nhấn **'Save and Analyze'**\n\nSau đó bạn có thể sử dụng tính năng Research Gap Analysis.")
            else:
                st.info("🔍 **Cần dữ liệu để phân tích research gaps**\n\nHãy upload và phân tích ít nhất 3-5 file nghiên cứu để hệ thống có thể:\n- Phát hiện khoảng trống nghiên cứu\n- Đề xuất topics mới\n- Generate research proposals")
        else:
            # Research Gap Analysis Functions
            def analyze_research_patterns(data):
                """Analyze patterns from research data"""
                patterns = {
                    'theories': {},
                    'methods': {},
                    'fields': {},
                    'variables_independent': {},
                    'variables_dependent': {},
                    'variable_combinations': {},
                    'theory_method_pairs': {},
                    'underexplored_areas': []
                }
                
                for item in data:
                    # Count theories
                    theories = str(item.get('ly_thuyet_nen', '')).strip()
                    if theories and theories.lower() not in ['no information', 'không có thông tin']:
                        theory_list = [t.strip() for t in theories.replace(',', ';').split(';')]
                        for theory in theory_list:
                            if theory:
                                patterns['theories'][theory] = patterns['theories'].get(theory, 0) + 1
                    
                    # Count methods
                    methods = str(item.get('pp_phan_tich', '')).strip()
                    if methods and methods.lower() not in ['no information', 'không có thông tin']:
                        patterns['methods'][methods] = patterns['methods'].get(methods, 0) + 1
                    
                    # Count fields
                    field = str(item.get('linh_vuc_nc', '')).strip()
                    if field and field.lower() not in ['no information', 'không có thông tin']:
                        patterns['fields'][field] = patterns['fields'].get(field, 0) + 1
                    
                    # Count variables
                    indep_vars = str(item.get('bien_doc_lap', '')).strip()
                    if indep_vars and indep_vars.lower() not in ['no information', 'không có thông tin']:
                        var_list = [v.strip() for v in indep_vars.replace(',', ';').split(';')]
                        for var in var_list:
                            if var:
                                patterns['variables_independent'][var] = patterns['variables_independent'].get(var, 0) + 1
                    
                    dep_vars = str(item.get('bien_phu_thuoc', '')).strip()
                    if dep_vars and dep_vars.lower() not in ['no information', 'không có thông tin']:
                        var_list = [v.strip() for v in dep_vars.replace(',', ';').split(';')]
                        for var in var_list:
                            if var:
                                patterns['variables_dependent'][var] = patterns['variables_dependent'].get(var, 0) + 1
                
                return patterns
            
            def generate_research_proposals(model, patterns, num_proposals=5):
                """Generate research proposals using AI"""
                if not model:
                    return []
                
                # Prepare data summary for AI
                top_theories = dict(sorted(patterns['theories'].items(), key=lambda x: x[1], reverse=True)[:10])
                top_methods = dict(sorted(patterns['methods'].items(), key=lambda x: x[1], reverse=True)[:8])
                top_fields = dict(sorted(patterns['fields'].items(), key=lambda x: x[1], reverse=True)[:8])
                top_indep_vars = dict(sorted(patterns['variables_independent'].items(), key=lambda x: x[1], reverse=True)[:15])
                top_dep_vars = dict(sorted(patterns['variables_dependent'].items(), key=lambda x: x[1], reverse=True)[:15])
                
                prompt = f"""
                Dựa trên phân tích systematic literature review từ {len(all_research_data)} bài nghiên cứu, hãy đề xuất {num_proposals} đề tài nghiên cứu mới để lấp đầy khoảng trống nghiên cứu.

                **DỮ LIỆU PHÂN TÍCH:**

                **🏗️ Lý thuyết được sử dụng nhiều nhất:**
                {chr(10).join([f"- {theory}: {count} lần" for theory, count in top_theories.items()])}

                **📊 Phương pháp phân tích phổ biến:**
                {chr(10).join([f"- {method}: {count} lần" for method, count in top_methods.items()])}

                **🎯 Lĩnh vực nghiên cứu:**
                {chr(10).join([f"- {field}: {count} lần" for field, count in top_fields.items()])}

                **📈 Biến độc lập thường dùng:**
                {chr(10).join([f"- {var}: {count} lần" for var, count in top_indep_vars.items()])}

                **📊 Biến phụ thuộc thường dùng:**
                {chr(10).join([f"- {var}: {count} lần" for var, count in top_dep_vars.items()])}

                **YÊU CẦU TẠO ĐỀ XUẤT:**

                Hãy tạo {num_proposals} đề xuất nghiên cứu SÁNG TẠO và THỰC TIỄN với:

                1️⃣ **ƯU TIÊN KHOẢNG TRỐNG:**
                - Kết hợp lý thuyết chưa được kết hợp
                - Ứng dụng lý thuyết vào lĩnh vực mới
                - Biến nghiên cứu chưa được kết hợp
                - Phương pháp mới cho các mô hình cũ

                2️⃣ **ĐẢM BẢO TÍNH KHOA HỌC:**
                - Có căn cứ lý thuyết vững chắc
                - Biến nghiên cứu rõ ràng và đo lường được
                - Phương pháp phù hợp với mô hình
                - Có ý nghĩa thực tiễn

                3️⃣ **FORMAT ĐẦU RA (JSON):**
                Trả về JSON array với {num_proposals} đề xuất, mỗi đề xuất có:

                ```json
                {{
                  "proposal_id": "P1",
                  "title": "Tên đề tài nghiên cứu cụ thể",
                  "research_gap": "Khoảng trống nghiên cứu được giải quyết",
                  "theoretical_foundation": "Lý thuyết nền đề xuất (có thể kết hợp nhiều lý thuyết)",
                  "research_model": "Mô tả mô hình nghiên cứu đề xuất",
                  "independent_variables": ["Biến độc lập 1", "Biến độc lập 2", "..."],
                  "dependent_variables": ["Biến phụ thuộc 1", "Biến phụ thuộc 2", "..."],
                  "mediating_variables": ["Biến trung gian 1", "Biến trung gian 2", "..."],
                  "moderating_variables": ["Biến điều tiết 1", "Biến điều tiết 2", "..."],
                  "methodology": "Phương pháp nghiên cứu đề xuất",
                  "research_field": "Lĩnh vực nghiên cứu",
                  "target_population": "Đối tượng nghiên cứu",
                  "practical_contribution": "Đóng góp thực tiễn",
                  "theoretical_contribution": "Đóng góp lý thuyết",
                  "rationale": "Lý do tại sao đề xuất này quan trọng và khả thi"
                }}
                ```

                **LƯU Ý QUAN TRỌNG:**
                - Tên đề tài phải CỤ THỂ, SÁNG TẠO và THỰC TIỄN
                - Biến nghiên cứu phải RÕ RÀNG và ĐÚNG CHUYÊN MÔN
                - Lý thuyết nền phải PHÙ HỢP với mô hình
                - Phương pháp phải PHẢN ÁNH xu hướng hiện đại
                - ĐẢM BẢO output là JSON hợp lệ
                - TẤT CẢ nội dung bằng TIẾNG VIỆT
                """
                
                # Use retry logic for transient API errors (500s) and provide a fallback model
                max_retries = 3
                backoff = 1.0
                response = None
                last_exc = None
                for attempt in range(1, max_retries + 1):
                    try:
                        # Request JSON directly when possible
                        try:
                            response = model.generate_content(
                                prompt,
                                generation_config={"response_mime_type": "application/json"}
                            )
                        except Exception:
                            # Fallback without config if SDK/version doesn't support it
                            response = model.generate_content(prompt)
                        break
                    except Exception as e:
                        last_exc = e
                        msg = str(e)
                        # If it's an internal 500-like transient error, retry with exponential backoff
                        if '500' in msg or 'Internal' in msg or 'internal error' in msg.lower():
                            st.warning(f"API internal error (attempt {attempt}/{max_retries}) — đang thử lại sau {backoff}s...")
                            time.sleep(backoff)
                            backoff *= 2
                            # try a fallback model on the second attempt
                            if attempt == 2:
                                try:
                                    model = init_gemini('gemini-2.5-flash')
                                    st.info("Đã chuyển sang model thay thế để thử lại.")
                                except Exception:
                                    pass
                            continue
                        else:
                            # Non-retryable error
                            st.error(f"Lỗi khi gọi API: {e}")
                            return []

                if response is None:
                    st.error("API trả về lỗi nội bộ nhiều lần. Vui lòng thử lại sau vài phút hoặc kiểm tra https://developers.generativeai.google/guide/troubleshooting")
                    if last_exc:
                        st.error(str(last_exc))
                    return []

                # Debug: safely preview response without accessing properties that may raise
                st.write("🔍 Debug - Response type:", type(response))
                try:
                    preview_text = extract_response_text(response)
                    if preview_text:
                        st.write("🔍 Debug - Response preview:")
                        st.code(preview_text[:1000])
                except Exception as _e:
                    st.write("🔍 Debug - Unable to preview response text:", str(_e))

                # Extract text safely
                response_text = extract_response_text(response)

                # Extract JSON array from the text response
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1

                if start_idx != -1 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    try:
                        proposals = json.loads(json_str)
                        return proposals
                    except json.JSONDecodeError:
                        # Try a more permissive fix: replace single quotes and trailing commas
                        cleaned = json_str.replace("'", '"')
                        cleaned = cleaned.replace(',]', ']')
                        cleaned = cleaned.replace(',}', '}')
                        try:
                            proposals = json.loads(cleaned)
                            return proposals
                        except Exception as e:
                            st.error(f"Lỗi parse JSON sau khi làm sạch: {e}")
                            return []
                else:
                    # Attempt 1: extract JSON from code fences or JSON-like blocks
                    try:
                        import re
                        fenced = re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", response_text, flags=re.DOTALL)
                        if fenced:
                            try:
                                return json.loads(fenced[0])
                            except Exception:
                                cleaned = fenced[0].replace("'", '"').replace(',]', ']').replace(',}', '}')
                                return json.loads(cleaned)
                    except Exception:
                        pass

                    # Attempt 2: ask model to convert previous text to strict JSON
                    strict_prompt = (
                        "Hãy CHỈ trả về JSON array hợp lệ (không có giải thích). "
                        "Chuyển đổi nội dung sau thành JSON theo đúng schema đã mô tả trước đó:\n\n" + (response_text or "")
                    )
                    try:
                        try:
                            resp2 = model.generate_content(
                                strict_prompt,
                                generation_config={"response_mime_type": "application/json"}
                            )
                        except Exception:
                            resp2 = model.generate_content(strict_prompt)
                        resp2_text = extract_response_text(resp2)
                        s2_start = resp2_text.find('[')
                        s2_end = resp2_text.rfind(']') + 1
                        if s2_start != -1 and s2_end > s2_start:
                            s2_json = resp2_text[s2_start:s2_end]
                            try:
                                return json.loads(s2_json)
                            except Exception:
                                cleaned = s2_json.replace("'", '"').replace(',]', ']').replace(',}', '}')
                                try:
                                    return json.loads(cleaned)
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    st.error("Không tìm thấy JSON hợp lệ trong response")
                    if response_text:
                        st.info(response_text[:1000])
                    st.write("🔍 Full response object:", response)
                    return []
            
            # Main content
            st.info(f"📊 **Đang phân tích {len(all_research_data)} nghiên cứu** để tìm khoảng trống và tạo đề xuất...")
            
            # Analyze patterns
            with st.spinner("🔍 Đang phân tích patterns từ dữ liệu nghiên cứu..."):
                patterns = analyze_research_patterns(all_research_data)
            
            # Display current research landscape
            st.subheader("📊 **Current Research Landscape**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**🏗️ Top Theories:**")
                top_theories = dict(sorted(patterns['theories'].items(), key=lambda x: x[1], reverse=True)[:5])
                for theory, count in top_theories.items():
                    st.write(f"• {theory}: {count} lần")
            
            with col2:
                st.write("**📊 Top Methods:**")
                top_methods = dict(sorted(patterns['methods'].items(), key=lambda x: x[1], reverse=True)[:5])
                for method, count in top_methods.items():
                    st.write(f"• {method}: {count} lần")
            
            with col3:
                st.write("**🎯 Top Fields:**")
                top_fields = dict(sorted(patterns['fields'].items(), key=lambda x: x[1], reverse=True)[:5])
                for field, count in top_fields.items():
                    st.write(f"• {field}: {count} lần")
            
            # Generate proposals button
            st.markdown("---")
            st.subheader("🤖 **AI Research Proposal Generator**")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                num_proposals = st.slider("📝 Số lượng đề xuất nghiên cứu:", min_value=3, max_value=10, value=5)
            with col2:
                generate_btn = st.button("🚀 **Generate Proposals**", type="primary")
            
            if generate_btn:
                # Initialize a model for gap analysis proposals
                research_gap_model = init_gemini('gemini-2.5-pro')
                with st.spinner("🤖 AI đang phân tích gaps và tạo research proposals..."):
                    proposals = generate_research_proposals(research_gap_model, patterns, num_proposals)
                
                if proposals:
                    st.success(f"✅ Đã tạo thành công {len(proposals)} research proposals!")
                    
                    # Display proposals
                    for i, proposal in enumerate(proposals, 1):
                        with st.expander(f"📋 **Proposal {i}: {proposal.get('title', 'N/A')}**", expanded=True):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**🎯 Research Gap:**")
                                st.write(proposal.get('research_gap', 'N/A'))
                                
                                st.write("**🏗️ Theoretical Foundation:**")
                                st.write(proposal.get('theoretical_foundation', 'N/A'))
                                
                                st.write("**📊 Research Model:**")
                                st.write(proposal.get('research_model', 'N/A'))
                                
                                st.write("**📈 Methodology:**")
                                st.write(proposal.get('methodology', 'N/A'))
                            
                            with col2:
                                st.write("**📊 Variables:**")
                                if proposal.get('independent_variables'):
                                    st.write("*Biến độc lập:*", ', '.join(proposal['independent_variables']))
                                if proposal.get('dependent_variables'):
                                    st.write("*Biến phụ thuộc:*", ', '.join(proposal['dependent_variables']))
                                if proposal.get('mediating_variables'):
                                    st.write("*Biến trung gian:*", ', '.join(proposal['mediating_variables']))
                                if proposal.get('moderating_variables'):
                                    st.write("*Biến điều tiết:*", ', '.join(proposal['moderating_variables']))
                                
                                st.write("**🎯 Research Field:**")
                                st.write(proposal.get('research_field', 'N/A'))
                                
                                st.write("**👥 Target Population:**")
                                st.write(proposal.get('target_population', 'N/A'))
                            
                            st.write("**💡 Rationale & Importance:**")
                            st.write(proposal.get('rationale', 'N/A'))
                            
                            col_contrib1, col_contrib2 = st.columns(2)
                            with col_contrib1:
                                st.write("**🏭 Practical Contribution:**")
                                st.write(proposal.get('practical_contribution', 'N/A'))
                            with col_contrib2:
                                st.write("**📚 Theoretical Contribution:**")
                                st.write(proposal.get('theoretical_contribution', 'N/A'))
                    
                    # Export proposals
                    st.markdown("---")
                    st.subheader("💾 **Export Research Proposals**")
                    
                    # Prepare export data
                    export_data = []
                    for i, proposal in enumerate(proposals, 1):
                        export_data.append({
                            'Proposal_ID': f"P{i}",
                            'Title': proposal.get('title', ''),
                            'Research_Gap': proposal.get('research_gap', ''),
                            'Theoretical_Foundation': proposal.get('theoretical_foundation', ''),
                            'Research_Model': proposal.get('research_model', ''),
                            'Independent_Variables': ', '.join(proposal.get('independent_variables', [])),
                            'Dependent_Variables': ', '.join(proposal.get('dependent_variables', [])),
                            'Mediating_Variables': ', '.join(proposal.get('mediating_variables', [])),
                            'Moderating_Variables': ', '.join(proposal.get('moderating_variables', [])),
                            'Methodology': proposal.get('methodology', ''),
                            'Research_Field': proposal.get('research_field', ''),
                            'Target_Population': proposal.get('target_population', ''),
                            'Practical_Contribution': proposal.get('practical_contribution', ''),
                            'Theoretical_Contribution': proposal.get('theoretical_contribution', ''),
                            'Rationale': proposal.get('rationale', ''),
                            'Generated_Date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                    
                    export_df = pd.DataFrame(export_data)
                    csv_export = export_df.to_csv(index=False, encoding='utf-8-sig')
                    
                    st.download_button(
                        label="📥 **Download Research Proposals (CSV)**",
                        data=csv_export,
                        file_name=f"research_proposals_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Download các đề xuất nghiên cứu thành file CSV"
                    )

    with tab5:
        st.header("🎯 Research Gap Analysis")
        st.subheader("📊 Kumar DM Framework Implementation")
        
        # Check if there are temporary results waiting to be saved
        if st.session_state.temp_analysis_results and not st.session_state.analysis_results:
            st.warning("⚠️ **Có kết quả phân tích chưa được lưu!**\n\n"
                      f"Có {len(st.session_state.temp_analysis_results)} file(s) đã được phân tích nhưng chưa được lưu vào hệ thống.\n"
                      "Vui lòng quay lại tab **'Upload File Mới'** và nhấn **'Save and Analyze'** để lưu kết quả trước khi sử dụng tính năng này.")
        elif st.session_state.temp_analysis_results and st.session_state.analysis_results:
            st.info(f"ℹ️ Sử dụng {len(st.session_state.analysis_results)} kết quả đã lưu để phân tích gap. "
                   f"Có thêm {len(st.session_state.temp_analysis_results)} kết quả chưa lưu ở tab Upload.")
        
        # Display Kumar framework image reference
        st.info("🔬 **Framework Reference:** Kumar DM (2020). Journey through research gaps: White Paper.")
        
        # Fallback function for when AI fails
        def create_simplified_gap_analysis(documents):
            """Create basic gap analysis when AI fails"""
            
            # Extract basic patterns
            theories = []
            methods = []
            fields = []
            
            for doc in documents:
                if doc.get('ly_thuyet_nen'):
                    theories.append(doc.get('ly_thuyet_nen'))
                if doc.get('pp_phan_tich'):
                    methods.append(doc.get('pp_phan_tich'))
                if doc.get('linh_vuc_nc'):
                    fields.append(doc.get('linh_vuc_nc'))
            
            # Basic gap analysis
            return {
                "kumar_gap_analysis": {
                    "gap_in_population": {
                        "gaps": ["Cần nghiên cứu thêm các nhóm đối tượng khác", "Mở rộng phạm vi địa lý"],
                        "opportunities": ["Nghiên cứu cross-cultural", "Nghiên cứu đa ngành"]
                    },
                    "gap_in_methods": {
                        "gaps": ["Áp dụng phương pháp nghiên cứu mới", "Kết hợp định tính và định lượng"],
                        "opportunities": ["Mixed-methods research", "Longitudinal studies"]
                    },
                    "gap_in_theory": {
                        "gaps": ["Phát triển framework lý thuyết mới", "Kiểm định lý thuyết trong bối cảnh mới"],
                        "opportunities": ["Theory integration", "Model extension"]
                    },
                    "known_vs_unknown": {
                        "known": [f"Đã có {len(set(theories))} lý thuyết được áp dụng", f"Đã sử dụng {len(set(methods))} phương pháp"],
                        "unknown": ["Hiệu ứng long-term", "Tác động cross-industry", "Ảnh hưởng cultural factors"]
                    },
                    "summary": {
                        "total_gaps": 6,
                        "critical_gap": "Thiếu nghiên cứu longitudinal và cross-cultural",
                        "direction": "Mở rộng nghiên cứu đa lĩnh vực và đa phương pháp"
                }
            }
        }

        # Generate comprehensive research roadmap
        def generate_research_roadmap(kumar_data, all_research_data):
            """Generate comprehensive research roadmap and topic suggestions"""
            
            # Extract patterns from current research
            theories_used = []
            methods_used = []
            fields_studied = []
            populations_studied = []
            
            for doc in all_research_data:
                if doc.get('ly_thuyet_nen'):
                    theories_used.extend([t.strip() for t in str(doc.get('ly_thuyet_nen')).split(',') if t.strip()])
                if doc.get('pp_phan_tich'):
                    methods_used.extend([m.strip() for m in str(doc.get('pp_phan_tich')).split(',') if m.strip()])
                if doc.get('linh_vuc_nc'):
                    fields_studied.append(doc.get('linh_vuc_nc'))
                if doc.get('mau'):
                    populations_studied.append(doc.get('mau'))
            
            # Create comprehensive roadmap
            roadmap = {
                'critical_gaps': [
                    "Thiếu nghiên cứu longitudinal theo dõi tác động dài hạn",
                    "Chưa có nghiên cứu cross-cultural trong bối cảnh đa văn hóa",
                    "Thiếu tích hợp công nghệ mới (AI, blockchain) vào mô hình nghiên cứu",
                    "Chưa có framework tổng hợp kết hợp multiple theories",
                    "Thiếu nghiên cứu về behavioral mechanisms và psychological processes"
                ],
                'immediate_opportunities': [
                    "Replication studies với sample size lớn hơn",
                    "Meta-analysis của existing studies trong lĩnh vực",
                    "Mixed-methods approach kết hợp quan sát thực tế",
                    "Technology-mediated research với digital platforms",
                    "Cross-industry comparative studies"
                ],
                'suggested_topics': [
                    {
                        'title': 'Tác Động Dài Hạn của Digital Transformation lên Customer Experience',
                        'description': 'Nghiên cứu longitudinal 3-5 năm về sự thay đổi hành vi khách hàng trong môi trường số',
                        'gap_addressed': 'Gap in longitudinal studies và technology integration',
                        'theoretical_framework': 'Technology Acceptance Model + Customer Experience Theory + Digital Transformation Framework',
                        'methodology': 'Mixed-methods: Survey panel + In-depth interviews + Digital analytics',
                        'target_population': 'Khách hàng các ngành dịch vụ (banking, retail, hospitality)',
                        'priority': 'High'
                    },
                    {
                        'title': 'Cross-Cultural Framework cho Customer Satisfaction trong ASEAN',
                        'description': 'Phát triển unified model đo lường satisfaction across different cultures',
                        'gap_addressed': 'Gap in cross-cultural perspective và regional context',
                        'theoretical_framework': 'Cultural Dimensions Theory + SERVQUAL + National Culture Framework',
                        'methodology': 'Multi-country survey + Cultural adaptation của measurement scales',
                        'target_population': 'Consumers từ 6 quốc gia ASEAN (1000+ each)',
                        'priority': 'High'
                    },
                    {
                        'title': 'AI-Powered Personalization và Customer Loyalty Mechanisms',
                        'description': 'Khám phá psychological mechanisms của AI personalization effect',
                        'gap_addressed': 'Gap in AI integration và behavioral mechanisms',
                        'theoretical_framework': 'Personalization Theory + Psychological Reactance + Trust Theory',
                        'methodology': 'Experimental design + Neurological measures + Behavioral tracking',
                        'target_population': 'Digital natives (Gen Z, Millennials) trong e-commerce',
                        'priority': 'High'
                    },
                    {
                        'title': 'Sustainable Business Practices và Brand Authenticity Perception',
                        'description': 'Nghiên cứu tác động của sustainability initiatives lên brand perception',
                        'gap_addressed': 'Gap in sustainability context và authenticity measures',
                        'theoretical_framework': 'Brand Authenticity Theory + Sustainability Framework + Consumer Values Theory',
                        'methodology': 'Scenario-based experiments + Content analysis + Consumer surveys',
                        'target_population': 'Conscious consumers across age groups',
                        'priority': 'Medium'
                    },
                    {
                        'title': 'Multi-Channel Customer Journey Mapping với Real-Time Analytics',
                        'description': 'Develop comprehensive framework cho omnichannel customer experience',
                        'gap_addressed': 'Gap in multi-touchpoint analysis và real-time insights',
                        'theoretical_framework': 'Customer Journey Theory + Omnichannel Framework + Experience Economy',
                        'methodology': 'Journey mapping + Digital tracking + Moment-based surveys',
                        'target_population': 'Omnichannel users trong retail và banking',
                        'priority': 'Medium'
                    },
                    {
                        'title': 'Social Media Influence và Word-of-Mouth trong Post-Pandemic Era',
                        'description': 'Nghiên cứu sự thay đổi của influence patterns sau COVID-19',
                        'gap_addressed': 'Gap in post-pandemic context và digital influence',
                        'theoretical_framework': 'Social Influence Theory + Digital WOM + Crisis Communication Theory',
                        'methodology': 'Social network analysis + Content mining + Survey research',
                        'target_population': 'Social media users với experience mua sắm post-pandemic',
                        'priority': 'Medium'
                    },
                    {
                        'title': 'Employee Experience và Customer Experience Correlation',
                        'description': 'Khám phá mối liên hệ giữa EX và CX trong service industries',
                        'gap_addressed': 'Gap in employee-customer experience linkage',
                        'theoretical_framework': 'Service-Profit Chain + Employee Engagement Theory + Customer Experience Management',
                        'methodology': 'Dual-sided surveys + Organizational analysis + Performance metrics',
                        'target_population': 'Service employees và customers trong hospitality, healthcare',
                        'priority': 'Low'
                    },
                    {
                        'title': 'Blockchain Trust và Financial Service Adoption',
                        'description': 'Nghiên cứu trust mechanisms trong blockchain-based financial services',
                        'gap_addressed': 'Gap in emerging technology adoption và trust formation',
                        'theoretical_framework': 'Technology Trust Theory + Blockchain Framework + Financial Behavior Theory',
                        'methodology': 'Technology acceptance surveys + Trust experiments + Usage analytics',
                        'target_population': 'Fintech users và potential adopters',
                        'priority': 'Low'
                    }
                ],
                'research_roadmap': {
                    'short_term': [
                        "Pilot studies cho AI personalization research",
                        "Survey development cho cross-cultural framework",
                        "Literature review comprehensive cho sustainable business practices",
                        "Stakeholder interviews cho omnichannel journey mapping",
                        "Technology infrastructure setup cho digital tracking"
                    ],
                    'medium_term': [
                        "Main data collection cho longitudinal digital transformation study",
                        "Cross-country implementation của cultural satisfaction framework",
                        "Experimental phases cho AI personalization mechanisms",
                        "Multi-industry rollout của sustainability authenticity research",
                        "Integration testing cho real-time analytics platforms"
                    ],
                    'long_term': [
                        "Longitudinal follow-up và trend analysis",
                        "Theory development từ integrated findings",
                        "Policy recommendations cho industry applications",
                        "Global scaling của successful frameworks",
                        "Next-generation research platform development"
                    ]
                },
                'strategic_recommendations': {
                    'theory_building': [
                        "Integrate multiple theories thành unified frameworks",
                        "Develop context-specific theories cho emerging markets",
                        "Create digital-native theoretical models",
                        "Build cross-disciplinary theoretical bridges"
                    ],
                    'methodology': [
                        "Adopt mixed-methods approaches systematically",
                        "Implement real-time data collection methods",
                        "Use AI/ML cho pattern discovery",
                        "Develop mobile-first research designs"
                    ],
                    'context_expansion': [
                        "Expand sang emerging markets (Vietnam, Philippines, etc.)",
                        "Include post-pandemic behavioral shifts",
                        "Study digital-first business models",
                        "Research sustainability-conscious segments"
                    ]
                },
                'success_metrics': [
                    "Publication trong top-tier journals (3+ papers/year)",
                    "Industry adoption của research frameworks (2+ implementations)",
                    "Policy influence và regulatory citations",
                    "Conference presentations và academic recognition",
                    "Practical impact measures trong business applications"
                ],
                'collaboration_opportunities': [
                    "Partnership với tech companies cho data access",
                    "Collaboration với government agencies cho policy research",
                    "Joint research với international universities",
                    "Industry consortiums cho funding và implementation",
                    "Cross-functional teams với practitioners"
                ]
            }
            
            return roadmap

        # Enhanced AI prompt for gap analysis
        def analyze_with_kumar_framework(model, documents):
            """Analyze documents using Kumar's research gap framework with retry mechanism"""
            
            # Limit documents to prevent overload
            max_docs = min(5, len(documents))  # Reduce to 5 docs max
            selected_docs = documents[:max_docs]
            
            # Create concise data summary
            data_summary = ""
            for i, doc in enumerate(selected_docs):
                data_summary += f"\n--- Doc {i+1} ---\n"
                data_summary += f"Title: {str(doc.get('tieu_de', 'Unknown'))[:100]}...\n"
                data_summary += f"Field: {doc.get('linh_vuc_nc', 'Unknown')}\n"
                data_summary += f"Theory: {str(doc.get('ly_thuyet_nen', 'Unknown'))[:100]}...\n"
                data_summary += f"Method: {doc.get('pp_phan_tich', 'Unknown')}\n"
                data_summary += f"Variables: {str(doc.get('bien_doc_lap', ''))[:50]} → {str(doc.get('bien_phu_thuoc', ''))[:50]}\n"
                data_summary += f"Key Findings: {str(doc.get('ket_qua_nc', 'Unknown'))[:100]}...\n"
            
            prompt = f"""
            Phân tích KUMAR FRAMEWORK (2020) cho {len(selected_docs)} nghiên cứu:

            **GAPS CẦN XÁC ĐỊNH:**
            1. Gap in Population, Analysis, Methods, Theory
            2. Known vs Unknown areas
            3. Research opportunities

            **DATA:**
            {data_summary[:8000]}

            **JSON FORMAT:**
            {{
              "kumar_gap_analysis": {{
                "gap_in_population": {{"gaps": ["Gap 1", "Gap 2"], "opportunities": ["Opp 1"]}},
                "gap_in_methods": {{"gaps": ["Gap 1"], "opportunities": ["Opp 1"]}},
                "gap_in_theory": {{"gaps": ["Gap 1"], "opportunities": ["Opp 1"]}},
                "known_vs_unknown": {{"known": ["Fact 1"], "unknown": ["Area 1"]}},
                "summary": {{"total_gaps": 3, "critical_gap": "Gap description", "direction": "Research direction"}}
            }}

            Trả về JSON ngắn gọn bằng tiếng Việt.
            """
            
            # Retry mechanism with different models
            models_to_try = [
                ('gemini-2.5-flash', "Fast model"),
                ('gemini-1.5-flash', "Stable model"),
                ('gemini-1.5-pro', "Pro model")
            ]
            
            for model_name, model_desc in models_to_try:
                try:
                    st.info(f"🔄 Trying {model_desc}: {model_name}")
                    
                    # Initialize model for this attempt
                    retry_model = init_gemini(model_name)
                    if not retry_model:
                        continue
                    
                    response = retry_model.generate_content(prompt)
                    if hasattr(response, 'text') and isinstance(getattr(response, 'text'), str):
                        response_text = extract_response_text(response)
                    else:
                        candidates = getattr(response, 'candidates', None) or getattr(response, 'results', None)
                        if candidates and len(candidates) > 0:
                            first = candidates[0]
                            content = getattr(first, 'content', None) or getattr(first, 'message', None) or first
                            parts = getattr(content, 'parts', None)
                            if parts:
                                response_text = ''.join([str(p) for p in parts]).strip()
                            else:
                                response_text = str(getattr(content, 'text', content)).strip()
                        else:
                            response_text = str(response)
                    
                    # Extract JSON from response
                    start_idx = response_text.find('{')
                    end_idx = response_text.rfind('}') + 1
                    
                    if start_idx != -1 and end_idx > start_idx:
                        json_str = response_text[start_idx:end_idx]
                        gap_analysis = json.loads(json_str)
                        st.success(f"✅ Success with {model_desc}")
                        return gap_analysis
                    else:
                        st.warning(f"⚠️ No valid JSON from {model_desc}")
                        continue
                        
                except json.JSONDecodeError as e:
                    st.warning(f"⚠️ JSON parse error with {model_desc}: {e}")
                    continue
                except Exception as e:
                    error_msg = str(e)
                    st.warning(f"⚠️ Error with {model_desc}: {error_msg}")
                    
                    # If 500 error, try next model immediately
                    if "500" in error_msg or "internal error" in error_msg.lower():
                        st.info("🔄 API overload detected, trying fallback model...")
                        continue
                    else:
                        continue
            
            # If all models fail, return simplified analysis
            st.error("❌ All models failed. Generating simplified analysis...")
            return create_simplified_gap_analysis(selected_docs)
        
        # Get data for analysis
        all_research_data = []
        
        # Add from current session analysis
        if st.session_state.analysis_results:
            all_research_data.extend(st.session_state.analysis_results)
        
        # Do NOT add from learned library to prevent showing old session data
        # This ensures Reset functionality works properly
        
        if not all_research_data:
            if st.session_state.has_reset:
                st.info("🔄 **Đã Reset - Cần Upload Files Mới**\n\nVui lòng:\n1. Upload files vào tab **'Upload File Mới'**\n2. Nhấn **'Phân Tích Files'**\n3. Nhấn **'Save and Analyze'**\n\nSau đó bạn có thể sử dụng Kumar Framework Analysis.")
            else:
                st.info("🔍 **Cần dữ liệu để phân tích research gaps**\n\nHãy upload và phân tích ít nhất 3-5 file nghiên cứu để áp dụng Kumar Framework")
        else:
            st.success(f"📊 **Đã có {len(all_research_data)} nghiên cứu** sẵn sàng để phân tích gaps")
            
            # Kumar Gap Analysis Button
            if st.button("🔍 **Phân Tích Research Gaps (Kumar Framework)**", type="primary"):
                # Use premium model for gap analysis
                gap_model = init_gemini('gemini-2.5-pro')
                
                with st.spinner("🔬 Đang áp dụng Kumar Framework để phân tích research gaps..."):
                    gap_results = analyze_with_kumar_framework(gap_model, all_research_data)
                
                if gap_results:
                    st.success("✅ **Phân tích Kumar Framework hoàn thành!**")
                    
                    # Display results in organized sections
                    kumar_data = gap_results.get('kumar_gap_analysis', {})
                    
                    # Summary Section
                    st.markdown("---")
                    st.subheader("📋 **Tổng Quan Research Gaps**")
                    
                    summary = kumar_data.get('summary', {})
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("🔍 Total Gaps", summary.get('total_gaps_identified', 0))
                    with col2:
                        st.metric("📊 Documents", len(all_research_data))
                    with col3:
                        st.metric("🎯 Priority", "High")
                    with col4:
                        st.metric("🔬 Framework", "Kumar 2020")
                    
                    # Critical Gap Highlight
                    if summary.get('most_critical_gap'):
                        st.error(f"🚨 **Most Critical Gap:** {summary.get('most_critical_gap')}")
                    
                    # NEW: Comprehensive Research Gap Summary & Future Roadmap (MOVED UP)
                    st.markdown("---")
                    st.subheader("📊 **Research Gap Summary & Future Roadmap**")
                    
                    # Generate comprehensive summary and recommendations
                    comprehensive_summary = generate_research_roadmap(kumar_data, all_research_data)
                    
                    # Display comprehensive analysis
                    st.write("### 🔍 **Tổng Kết Research Gaps Hiện Tại**")
                    
                    # Critical gaps summary
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🚨 Gaps Quan Trọng Nhất:**")
                        critical_gaps = comprehensive_summary.get('critical_gaps', [])
                        for i, gap in enumerate(critical_gaps[:5], 1):
                            st.write(f"{i}. {gap}")
                    
                    with col2:
                        st.markdown("**🎯 Cơ Hội Nghiên Cứu Ngay:**")
                        immediate_opportunities = comprehensive_summary.get('immediate_opportunities', [])
                        for i, opp in enumerate(immediate_opportunities[:5], 1):
                            st.write(f"{i}. {opp}")
                    
                    # Future research topics
                    st.write("### 💡 **Đề Xuất Đề Tài Nghiên Cứu Cụ Thể**")
                    
                    suggested_topics = comprehensive_summary.get('suggested_topics', [])
                    
                    # Display in expandable sections
                    for i, topic in enumerate(suggested_topics[:8], 1):
                        with st.expander(f"🔬 **Đề Tài {i}: {topic.get('title', 'Unknown')}**"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**📝 Mô Tả:**")
                                st.write(topic.get('description', 'N/A'))
                                
                                st.write("**🎯 Gap Addressed:**")
                                st.write(topic.get('gap_addressed', 'N/A'))
                                
                                st.write("**📚 Lý Thuyết Đề Xuất:**")
                                st.write(topic.get('theoretical_framework', 'N/A'))
                            
                            with col2:
                                st.write("**🔧 Phương Pháp:**")
                                st.write(topic.get('methodology', 'N/A'))
                                
                                st.write("**👥 Đối Tượng:**")
                                st.write(topic.get('target_population', 'N/A'))
                                
                                st.write("**⚡ Độ Ưu Tiên:**")
                                priority = topic.get('priority', 'Medium')
                                if priority == 'High':
                                    st.error(f"🔴 {priority}")
                                elif priority == 'Medium':
                                    st.warning(f"🟡 {priority}")
                                else:
                                    st.info(f"🔵 {priority}")
                    
                    # Research roadmap timeline
                    st.write("### 🗺️ **Research Roadmap (Timeline)**")
                    
                    roadmap = comprehensive_summary.get('research_roadmap', {})
                    
                    # Short-term (6-12 months)
                    st.write("#### 🚀 **Phase 1: Short-term (6-12 tháng)**")
                    short_term = roadmap.get('short_term', [])
                    for item in short_term:
                        st.write(f"• {item}")
                    
                    # Medium-term (1-2 years)
                    st.write("#### 📈 **Phase 2: Medium-term (1-2 năm)**")
                    medium_term = roadmap.get('medium_term', [])
                    for item in medium_term:
                        st.write(f"• {item}")
                    
                    # Long-term (3+ years)
                    st.write("#### 🎯 **Phase 3: Long-term (3+ năm)**")
                    long_term = roadmap.get('long_term', [])
                    for item in long_term:
                        st.write(f"• {item}")
                    
                    # Strategic recommendations
                    st.write("### 🎯 **Strategic Recommendations**")
                    
                    strategic_recs = comprehensive_summary.get('strategic_recommendations', {})
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**🏗️ Theory Building:**")
                        theory_recs = strategic_recs.get('theory_building', [])
                        for rec in theory_recs:
                            st.write(f"• {rec}")
                    
                    with col2:
                        st.write("**🔧 Methodology:**")
                        method_recs = strategic_recs.get('methodology', [])
                        for rec in method_recs:
                            st.write(f"• {rec}")
                    
                    with col3:
                        st.write("**🌍 Context Expansion:**")
                        context_recs = strategic_recs.get('context_expansion', [])
                        for rec in context_recs:
                            st.write(f"• {rec}")
                    
                    # Success metrics
                    st.write("### 📊 **Success Metrics & KPIs**")
                    
                    success_metrics = comprehensive_summary.get('success_metrics', [])
                    for metric in success_metrics:
                        st.write(f"• {metric}")
                    
                    # Collaboration opportunities
                    st.write("### 🤝 **Collaboration Opportunities**")
                    
                    collaborations = comprehensive_summary.get('collaboration_opportunities', [])
                    for collab in collaborations:
                        st.write(f"• {collab}")
                    
                    # Gap Categories Analysis
                    st.markdown("---")
                    st.subheader("🎯 **Research Gaps by Category**")
                    
                    gap_categories = [
                        ("gap_in_population", "👥 Population Gaps", "🔵"),
                        ("gap_in_analysis", "📊 Analysis Gaps", "🟢"), 
                        ("gap_in_implication", "💡 Implication Gaps", "🟡"),
                        ("gap_in_methods", "🔧 Method Gaps", "🟠"),
                        ("gap_in_concepts", "💭 Concept Gaps", "🟣"),
                        ("gap_in_perspectives", "👀 Perspective Gaps", "🔴"),
                        ("gap_in_theory", "📚 Theory Gaps", "🔶"),
                        ("gap_in_methodology", "⚙️ Methodology Gaps", "⚫")
                    ]
                    
                    # Display gaps in tabs
                    gap_tabs = st.tabs([f"{emoji} {name.split(' ')[1]}" for _, name, emoji in gap_categories])
                    
                    for i, (gap_key, gap_name, emoji) in enumerate(gap_categories):
                        with gap_tabs[i]:
                            gap_data = kumar_data.get(gap_key, {})
                            
                            st.write(f"### {emoji} **{gap_name}**")
                            
                            # Identified Gaps (handle both old and new format)
                            gaps = gap_data.get('identified_gaps', gap_data.get('gaps', []))
                            if gaps:
                                st.write("**🔍 Identified Gaps:**")
                                for gap in gaps:
                                    st.write(f"• {gap}")
                            
                            # Opportunities
                            opportunities = gap_data.get('opportunities', [])
                            if opportunities:
                                st.write("**🎯 Research Opportunities:**")
                                for opp in opportunities:
                                    st.write(f"• {opp}")
                    
                    # Known vs Unknown Analysis
                    st.markdown("---")
                    st.subheader("🔄 **Known vs Unknown Analysis**")
                    
                    known_unknown = kumar_data.get('known_vs_unknown', {})
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("### ✅ **Past Knowledge (Known)**")
                        past_knowledge = known_unknown.get('past_knowledge', known_unknown.get('known', []))
                        for knowledge in past_knowledge:
                            st.write(f"• {knowledge}")
                    
                    with col2:
                        st.write("### ❓ **Expected Knowledge (Unknown)**")
                        expected_knowledge = known_unknown.get('expected_knowledge', known_unknown.get('unknown', []))
                        for knowledge in expected_knowledge:
                            st.write(f"• {knowledge}")
                    
                    # Inconsistencies and Conflicts
                    st.write("### ⚡ **Inconsistencies & Conflicts**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**🔄 Inconsistencies:**")
                        inconsistencies = known_unknown.get('inconsistencies', [])
                        for item in inconsistencies:
                            st.write(f"• {item}")
                    
                    with col2:
                        st.write("**💭 Controversies:**")
                        controversies = known_unknown.get('controversies', [])
                        for item in controversies:
                            st.write(f"• {item}")
                    
                    with col3:
                        st.write("**⚔️ Conflicts:**")
                        conflicts = known_unknown.get('conflicts', [])
                        for item in conflicts:
                            st.write(f"• {item}")
                    
                    # Framework Opportunities
                    st.markdown("---")
                    st.subheader("🚀 **Framework & Opportunities**")
                    
                    framework_ops = kumar_data.get('framework_opportunities', {})
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**📚 Current Conceptual Frame:**")
                        st.write(framework_ops.get('current_conceptual_frame', 'N/A'))
                    
                    with col2:
                        st.write("**🆕 New Theoretical Frame:**")
                        st.write(framework_ops.get('new_theoretical_frame', 'N/A'))
                    
                    st.write("**🎯 Research Opportunities:**")
                    research_ops = framework_ops.get('research_opportunities', [])
                    for opp in research_ops:
                        st.write(f"• {opp}")
                    
                    # Priority Matrix
                    st.markdown("---")
                    st.subheader("📊 **Priority Matrix**")
                    
                    priority_gaps = kumar_data.get('priority_gaps', {})
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.error("**🔴 High Priority**")
                        high_priority = priority_gaps.get('high_priority', [])
                        for item in high_priority:
                            st.write(f"• {item}")
                    
                    with col2:
                        st.warning("**🟡 Medium Priority**")
                        medium_priority = priority_gaps.get('medium_priority', [])
                        for item in medium_priority:
                            st.write(f"• {item}")
                    
                    with col3:
                        st.info("**🔵 Low Priority**")
                        low_priority = priority_gaps.get('low_priority', [])
                        for item in low_priority:
                            st.write(f"• {item}")
                    
                    # Research Direction
                    st.markdown("---")
                    st.subheader("🎯 **Recommended Research Direction**")
                    
                    if summary.get('research_direction', summary.get('direction')):
                        direction = summary.get('research_direction', summary.get('direction'))
                        st.success(f"**🚀 Hướng nghiên cứu:** {direction}")
                    
                    if summary.get('theoretical_contribution'):
                        st.info(f"**📚 Đóng góp lý thuyết:** {summary.get('theoretical_contribution')}")
                    
                    # Export Kumar Analysis
                    st.markdown("---")
                    st.subheader("💾 **Export Kumar Gap Analysis**")
                    
                    # Prepare comprehensive export data
                    export_data = {
                        'Kumar_Gap_Analysis': gap_results,
                        'Comprehensive_Research_Roadmap': comprehensive_summary,
                        'Analysis_Date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'Documents_Analyzed': len(all_research_data),
                        'Framework_Reference': 'Kumar DM (2020). Journey through research gaps: White Paper.',
                        'Export_Version': '2.0_Comprehensive'
                    }
                    
                    # Convert to JSON for download
                    json_export = json.dumps(export_data, ensure_ascii=False, indent=2)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            label="📥 **Download Complete Analysis (JSON)**",
                            data=json_export,
                            file_name=f"comprehensive_gap_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            help="Download phân tích Kumar Framework + Research Roadmap đầy đủ"
                        )
                    
                    with col2:
                        # Create simplified CSV for research topics
                        topics_df = pd.DataFrame(comprehensive_summary.get('suggested_topics', []))
                        if not topics_df.empty:
                            csv_export = topics_df.to_csv(index=False, encoding='utf-8-sig')
                            st.download_button(
                                label="📊 **Download Research Topics (CSV)**",
                                data=csv_export,
                                file_name=f"research_topics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                help="Download danh sách đề tài nghiên cứu đề xuất"
                            )

    with tab6:
        st.header("🚀 Complete Research Planning")
        st.subheader("📋 From Gap Analysis to Full Implementation Guide")
        
        # Check if there are temporary results waiting to be saved
        if st.session_state.temp_analysis_results and not st.session_state.analysis_results:
            st.warning("⚠️ **Có kết quả phân tích chưa được lưu!**\n\n"
                      f"Có {len(st.session_state.temp_analysis_results)} file(s) đã được phân tích nhưng chưa được lưu vào hệ thống.\n"
                      "Vui lòng quay lại tab **'Upload File Mới'** và nhấn **'Save and Analyze'** để lưu kết quả trước khi sử dụng tính năng này.")
        elif st.session_state.temp_analysis_results and st.session_state.analysis_results:
            st.info(f"ℹ️ Sử dụng {len(st.session_state.analysis_results)} kết quả đã lưu để tạo research plan. "
                   f"Có thêm {len(st.session_state.temp_analysis_results)} kết quả chưa lưu ở tab Upload.")
        
        # Enhanced comprehensive research planning function
        def generate_complete_research_plan(model, kumar_data, all_research_data, selected_topic=None):
            """Generate complete research plan including survey, methodology, timeline"""
            # Kiểm tra cache trước khi gọi API
            cache_key = f"plan_{len(all_research_data)}"
            cached_result = check_and_use_cache(cache_key, len(all_research_data))
            if cached_result:
                return cached_result
            
            # Prepare comprehensive prompt for complete research planning
            data_summary = ""
            for i, doc in enumerate(all_research_data[:8]):  # Limit for prompt size
                data_summary += f"\n--- Document {i+1} ---\n"
                data_summary += f"Title: {str(doc.get('tieu_de', 'Unknown'))[:100]}...\n"
                data_summary += f"Theory: {str(doc.get('ly_thuyet_nen', 'Unknown'))[:100]}...\n"
                data_summary += f"Method: {doc.get('pp_phan_tich', 'Unknown')}\n"
                data_summary += f"Variables: {str(doc.get('bien_doc_lap', ''))[:50]} → {str(doc.get('bien_phu_thuoc', ''))[:50]}\n"
                data_summary += f"Scales: {str(doc.get('thang_do', 'Unknown'))[:100]}...\n"
            
            # Đơn giản hóa prompt để tránh lỗi timeout và JSON không hợp lệ
            prompt = f"""
            Dựa trên phân tích systematic literature review, hãy tạo 1 đề tài nghiên cứu ngắn gọn.

            **DỮ LIỆU NGHIÊN CỨU:**
            {data_summary[:5000]}

            **YÊU CẦU TẠO RESEARCH PLAN:**

            1️⃣ **ĐỀ TÀI NGHIÊN CỨU:**
            - Tên đề tài ngắn gọn
            - Vấn đề nghiên cứu 
            - 2-3 câu hỏi nghiên cứu
            - 2-3 mục tiêu nghiên cứu
            - Ý nghĩa nghiên cứu

            2️⃣ **PHƯƠNG PHÁP NGHIÊN CỨU:**
            - Research design
            - Population và sampling method
            - Sample size
            - 2-3 biến độc lập
            - 1-2 biến phụ thuộc

            3️⃣ **THANG ĐO:**
            - 1-2 thang đo chính
            - 3-5 câu hỏi mẫu

            4️⃣ **TIMELINE:**
            - 2-3 giai đoạn chính
            - Thời gian dự kiến

            **FORMAT ĐẦU RA (JSON):**
            {{
              "complete_research_plan": {{
                "research_topic": {{
                  "title": "Tên đề tài cụ thể",
                  "research_problem": "Vấn đề nghiên cứu",
                  "research_questions": ["Main question", "Sub question 1", "Sub question 2"],
                  "objectives": ["Objective 1", "Objective 2", "Objective 3"],
                  "significance": "Ý nghĩa nghiên cứu"
                }},
                "theoretical_framework": {{
                  "main_theories": ["Theory 1", "Theory 2"],
                  "research_model": "Mô tả mô hình",
                  "hypotheses": ["H1: ...", "H2: ...", "H3: ..."],
                  "gaps_addressed": "Literature gaps được giải quyết"
                }},
                "methodology": {{
                  "research_design": "Design type",
                  "population": "Target population",
                  "sampling_method": "Sampling approach",
                  "sample_size": "Size và justification",
                  "data_collection": "Collection method",
                  "analysis_plan": "Statistical analysis"
                }},
                "variables_and_scales": {{
                  "independent_variables": [
                    {{"name": "Variable name", "definition": "Definition", "scale": "Scale details"}}
                  ],
                  "dependent_variables": [
                    {{"name": "Variable name", "definition": "Definition", "scale": "Scale details"}}
                  ],
                  "mediating_variables": [
                    {{"name": "Variable name", "definition": "Definition", "scale": "Scale details"}}
                  ],
                  "control_variables": [
                    {{"name": "Variable name", "definition": "Definition", "scale": "Scale details"}}
                  ]
                }},
                "survey_questionnaire": {{
                  "demographics": [
                    {{"question": "Question text", "type": "multiple_choice", "options": ["Option1", "Option2"]}},
                    {{"question": "Question text", "type": "likert", "scale": "1-5", "anchors": ["Strongly disagree", "Strongly agree"]}}
                  ],
                  "main_constructs": [
                    {{
                      "construct_name": "Construct name",
                      "items": [
                        {{"code": "CONST1", "question": "Question text", "scale": "1-7 Likert"}},
                        {{"code": "CONST2", "question": "Question text", "scale": "1-7 Likert"}}
                      ]
                    }}
                  ]
                }},
                "timeline": {{
                  "phase_1": {{"duration": "Months 1-3", "activities": ["Activity 1", "Activity 2"]}},
                  "phase_2": {{"duration": "Months 4-5", "activities": ["Activity 1", "Activity 2"]}},
                  "phase_3": {{"duration": "Months 6-8", "activities": ["Activity 1", "Activity 2"]}},
                  "phase_4": {{"duration": "Months 9-12", "activities": ["Activity 1", "Activity 2"]}}
                }},
                "implementation": {{
                  "ethical_considerations": ["Ethics point 1", "Ethics point 2"],
                  "tools_platforms": ["Tool 1", "Tool 2"],
                  "budget_estimation": "Budget details",
                  "resources_needed": ["Resource 1", "Resource 2"],
                  "risks_mitigation": ["Risk 1: Mitigation", "Risk 2: Mitigation"]
                }},
                "expected_outputs": {{
                  "academic_papers": ["Paper 1 target", "Paper 2 target"],
                  "conferences": ["Conference 1", "Conference 2"],
                  "practical_applications": ["Application 1", "Application 2"],
                  "policy_recommendations": ["Policy 1", "Policy 2"]
              }}
            }}

            **LƯU Ý:**
            - TẤT CẢ nội dung bằng TIẾNG VIỆT
            - Chi tiết và cụ thể, có thể implement ngay
            - Scales phải realistic và validated
            - Timeline feasible cho 1 năm research
            - Survey questions professional và clear
            """
            
            # Retry mechanism for comprehensive planning
            models_to_try = [
                ('gemini-2.5-pro', "Premium comprehensive model"),
                ('gemini-2.5-flash', "Fast comprehensive model"),
                ('gemini-1.5-pro', "Fallback pro model")
            ]
            
            for model_name, model_desc in models_to_try:
                try:
                    st.info(f"🔄 Generating complete plan with {model_desc}...")
                    
                    retry_model = init_gemini(model_name)
                    if not retry_model:
                        continue
                    
                    # Gọi API với cấu hình đơn giản và timeout
                    try:
                        # Thêm timeout cho API call để tránh lỗi Deadline Exceeded
                        import threading
                        import queue
                        
                        response_queue = queue.Queue()
                        
                        def call_api():
                            try:
                                api_response = retry_model.generate_content(prompt)
                                response_queue.put(("success", api_response))
                            except Exception as api_error:
                                response_queue.put(("error", api_error))
                        
                        # Tạo thread để gọi API
                        api_thread = threading.Thread(target=call_api)
                        api_thread.daemon = True
                        api_thread.start()
                        
                        # Đợi tối đa 30 giây
                        api_thread.join(timeout=30)
                        
                        if api_thread.is_alive():
                            # Thread vẫn đang chạy sau 30 giây -> timeout
                            raise TimeoutError("API call timed out after 30 seconds")
                        
                        # Lấy kết quả từ queue
                        status, result = response_queue.get(block=False)
                        
                        if status == "error":
                            raise result
                        
                        response = result
                    except TimeoutError as timeout_err:
                        st.warning(f"⚠️ API timeout: {str(timeout_err)}")
                        continue
                    except Exception as api_call_error:
                        st.warning(f"⚠️ Lỗi khi gọi API: {str(api_call_error)}")
                        continue
                    
                    # Xử lý phản hồi an toàn hơn
                    response_text = ""
                    try:
                        # Kiểm tra nếu phản hồi có nhiều phần
                        if hasattr(response, 'parts'):
                            for part in response.parts:
                                if hasattr(part, 'text'):
                                    response_text += part.text
                                elif hasattr(part, 'function_call'):
                                    response_text += str(part.function_call)
                        # Cách cũ nếu là phản hồi đơn giản
                        elif hasattr(response, 'text'):
                            response_text = extract_response_text(response)
                        # Phương pháp dự phòng
                        else:
                            response_text = str(response)
                        
                        response_text = response_text.strip()
                    except Exception as text_error:
                        st.warning(f"⚠️ Không thể trích xuất văn bản từ phản hồi: {text_error}")
                        continue
                    
                    # Cải thiện xử lý JSON từ phản hồi
                    try:
                        # Phương pháp 1: Tìm JSON từ chuỗi phản hồi
                        start_idx = response_text.find('{')
                        end_idx = response_text.rfind('}') + 1

                        if start_idx != -1 and end_idx > start_idx:
                            json_str = response_text[start_idx:end_idx]
                            try:
                                research_plan = json.loads(json_str)
                                st.success(f"✅ Complete plan generated with {model_desc}")

                                # Lưu kết quả vào cache để sử dụng lần sau
                                if 'research_plan_cache' not in st.session_state:
                                    st.session_state.research_plan_cache = {}
                                cache_key = f"plan_{len(all_research_data)}"
                                st.session_state.research_plan_cache[cache_key] = research_plan

                                return research_plan
                            except json.JSONDecodeError as json_err:
                                # Thử phương pháp sửa JSON
                                st.info(f"🔄 Đang thử sửa JSON từ {model_desc}...")
                                fixed_json = self_repair_json(json_str)
                                if fixed_json:
                                    st.success(f"✅ Đã sửa và phân tích JSON thành công!")
                                    return fixed_json

                        # Phương pháp 2: Tìm JSON bằng regex (nếu phương pháp 1 thất bại)
                        import re
                        json_pattern = r'\{(?:[^{}]|(?R))*\}'
                        matches = re.findall(json_pattern, response_text, re.DOTALL)
                        if matches:
                            for potential_json in matches:
                                try:
                                    research_plan = json.loads(potential_json)
                                    st.success(f"✅ Tìm thấy JSON hợp lệ với regex từ {model_desc}")
                                    return research_plan
                                except Exception:
                                    continue

                        # Nếu không tìm thấy JSON hợp lệ
                        st.warning(f"⚠️ Không tìm thấy JSON hợp lệ từ {model_desc}")
                    except Exception as json_extract_error:
                        st.warning(f"⚠️ Lỗi khi xử lý JSON từ {model_desc}: {json_extract_error}")

                    # Nếu tất cả các phương pháp đều thất bại, tiếp tục với model tiếp theo
                    continue
                        
                except Exception as e:
                    st.warning(f"⚠️ Lỗi với {model_desc}: {str(e)}")
                    # Đợi 2 giây trước khi thử model tiếp theo để tránh rate limit
                    time.sleep(2)
                    continue
            
            # Fallback simplified plan if all models fail
            st.error("❌ All models failed. Generating simplified template...")
            return create_fallback_research_plan()

        def self_repair_json(json_str):
            """Cố gắng sửa JSON không hợp lệ"""
            try:
                # Thử phương pháp 1: Sửa lỗi dấu phẩy
                fixed_json = json_str
                # Sửa dấu phẩy sau phần tử cuối cùng trong mảng hoặc đối tượng
                fixed_json = re.sub(r',\s*}', '}', fixed_json)
                fixed_json = re.sub(r',\s*]', ']', fixed_json)
                
                # Thử phương pháp 2: Thêm dấu ngoặc kép cho các key không có
                fixed_json = re.sub(r'(\s*)([a-zA-Z0-9_]+)(\s*):(\s*)', r'\1"\2"\3:\4', fixed_json)
                
                # Thử phương pháp 3: Sửa các chuỗi không có dấu ngoặc kép
                # Quá phức tạp để làm trong regex đơn giản, nên chúng ta sẽ thử parse trực tiếp
                
                try:
                    return json.loads(fixed_json)
                except:
                    pass
                
                # Thử phương pháp 4: Sử dụng thư viện bên ngoài nếu có
                try:
                    # Thử import demjson3 (có thể không được cài đặt)
                    try:
                        import demjson3  # type: ignore
                        return demjson3.decode(json_str)
                    except ImportError:
                        pass
                except:
                    pass
                    
                # Thử phương pháp 5: Sử dụng ast.literal_eval cho các cấu trúc Python đơn giản
                try:
                    import ast
                    # Thay thế các key không có dấu ngoặc kép
                    python_dict_str = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)(\s*:)', r'\1"\2"\3', json_str)
                    return ast.literal_eval(python_dict_str)
                except:
                    pass
                
                return None
            except Exception as repair_error:
                st.warning(f"⚠️ Không thể sửa JSON: {repair_error}")
                return None
                
        def check_and_use_cache(cache_key, data_length):
            """Kiểm tra và sử dụng cache nếu có"""
            if 'research_plan_cache' not in st.session_state:
                st.session_state.research_plan_cache = {}
                
            if cache_key in st.session_state.research_plan_cache:
                st.success(f"✅ Sử dụng kết quả từ cache - tiết kiệm API! ({data_length} tài liệu)")
                cached_plan = st.session_state.research_plan_cache[cache_key]
                
                # Hiển thị nút để xóa cache nếu muốn tạo mới
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("🔄 Tạo mới", help="Xóa cache và tạo kế hoạch nghiên cứu mới"):
                        del st.session_state.research_plan_cache[cache_key]
                        st.rerun()
                
                # Nếu người dùng không muốn tạo mới, sử dụng cache
                if cache_key in st.session_state.research_plan_cache:
                    return cached_plan
            
            return None

        def create_fallback_research_plan():
            """Create a basic research plan template when AI fails"""
            # Tạo mẫu nghiên cứu dựa trên dữ liệu đã phân tích (nếu có)
            research_title = "Nghiên cứu tác động của công nghệ số đến hành vi người tiêu dùng"
            research_problem = "Thiếu hiểu biết về cách công nghệ số thay đổi hành vi mua sắm"
            
            # Thử trích xuất thông tin từ dữ liệu đã phân tích
            if all_research_data:
                try:
                    # Lấy một số chủ đề từ dữ liệu hiện có
                    topics = []
                    fields = []
                    methods = []
                    
                    for item in all_research_data[:5]:  # Chỉ xem 5 mục đầu tiên
                        if isinstance(item, dict):
                            # Lấy lĩnh vực nghiên cứu
                            field = item.get('linh_vuc_nc', '')
                            if field and field not in fields and field != 'No information':
                                fields.append(field)
                                
                            # Lấy phương pháp phân tích
                            method = item.get('pp_phan_tich', '')
                            if method and method not in methods and method != 'No information':
                                methods.append(method)
                                
                            # Lấy tiêu đề làm chủ đề tiềm năng
                            title = item.get('tieu_de', '')
                            if title and title not in topics and title != 'No information':
                                topics.append(title)
                    
                    # Sử dụng thông tin thu thập được để cải thiện mẫu
                    if fields:
                        research_title = f"Nghiên cứu về {fields[0]}"
                    
                    st.info(f"📊 Đã tạo mẫu nghiên cứu dựa trên {len(all_research_data)} tài liệu đã phân tích")
                except Exception as e:
                    st.warning(f"⚠️ Không thể trích xuất thông tin từ dữ liệu: {e}")
            
            return {
                "complete_research_plan": {
                    "research_topic": {
                        "title": research_title,
                        "research_problem": research_problem,
                        "research_questions": [
                            "Các yếu tố nào ảnh hưởng đến hành vi người tiêu dùng trong lĩnh vực này?",
                            "Có mối quan hệ giữa các biến độc lập và biến phụ thuộc không?",
                            "Các yếu tố nhân khẩu học có tác động điều tiết đến mối quan hệ này không?"
                        ],
                        "objectives": [
                            "Xác định các yếu tố ảnh hưởng chính",
                            "Phân tích mối quan hệ giữa các biến",
                            "Đánh giá tác động điều tiết của các yếu tố nhân khẩu học"
                        ],
                        "significance": "Cung cấp hiểu biết sâu sắc và định hướng chiến lược cho các nhà nghiên cứu và doanh nghiệp"
                    },
                    "methodology": {
                        "research_design": "Quantitative cross-sectional survey",
                        "population": "Người tiêu dùng trực tuyến 18-65 tuổi",
                        "sampling_method": "Stratified random sampling",
                        "sample_size": "385 respondents (95% confidence, 5% margin of error)",
                        "data_collection": "Online survey via Google Forms/Qualtrics",
                        "analysis_plan": "Descriptive statistics, SEM using SmartPLS"
                    },
                    "timeline": {
                        "phase_1": {"duration": "Tháng 1-3", "activities": ["Literature review", "Questionnaire design"]},
                        "phase_2": {"duration": "Tháng 4-5", "activities": ["Pilot test", "Survey refinement"]},
                        "phase_3": {"duration": "Tháng 6-8", "activities": ["Data collection", "Data cleaning"]},
                        "phase_4": {"duration": "Tháng 9-12", "activities": ["Data analysis", "Report writing"]}
                    },
                    "implementation": {
                        "ethical_considerations": ["Ensure informed consent", "Data privacy by design"],
                        "tools_platforms": ["Qualtrics", "SmartPLS", "Python"],
                        "budget_estimation": "TBD",
                        "resources_needed": ["Survey platform", "Data analysts"],
                        "risks_mitigation": ["Non-response: follow-ups", "Data quality checks"]
                    }
                }
            }

        # Main content for Complete Research Planning
        st.info("🎯 **All-in-One Research Planning Tool** - Từ gap analysis đến implementation guide hoàn chỉnh!")
        
        # Get data for comprehensive planning
        all_research_data = []
        
        # Add from current session analysis
        if st.session_state.analysis_results:
            all_research_data.extend(st.session_state.analysis_results)
        
        # Do NOT add from learned library to prevent showing old session data
        # This ensures Reset functionality works properly
        
        if not all_research_data:
            if st.session_state.has_reset:
                st.info("🔄 **Đã Reset - Cần Upload Files Mới**\n\nVui lòng:\n1. Upload files vào tab **'Upload File Mới'**\n2. Nhấn **'Phân Tích Files'**\n3. Nhấn **'Save and Analyze'**\n\nSau đó bạn có thể sử dụng Complete Research Planning.")
            else:
                st.info("🔍 **Cần dữ liệu để tạo complete research plan**\n\nHãy upload và phân tích ít nhất 3-5 file nghiên cứu để hệ thống có thể tạo:\n- Complete research proposal\n- Methodology design\n- Survey questionnaire\n- Implementation timeline")
        else:
            st.success(f"📊 **Đã có {len(all_research_data)} nghiên cứu** để tạo complete research plan")
            
            # Generate Complete Research Plan Button
            if st.button("🚀 **Tạo Complete Research Plan**", type="primary"):
                # Use premium model for comprehensive planning
                planning_model = init_gemini('gemini-2.5-pro')
                
                with st.spinner("🔬 Đang tạo complete research plan với methodology, survey, timeline..."):
                    # First run Kumar analysis to get gap data
                    kumar_results = {"kumar_gap_analysis": {"summary": {}}}  # Simplified for this context
                    
                    research_plan = generate_complete_research_plan(
                        planning_model, 
                        kumar_results, 
                        all_research_data
                    )
                
                if research_plan and research_plan.get('complete_research_plan'):
                    st.success("✅ **Complete Research Plan đã được tạo thành công!**")
                    
                    plan_data = research_plan['complete_research_plan']
                    
                    # Display comprehensive research plan
                    st.markdown("---")
                    st.subheader("📋 **1. ĐỀ TÀI NGHIÊN CỨU**")
                    
                    topic_data = plan_data.get('research_topic', {})
                    st.success(f"**🎯 Tên đề tài:** {topic_data.get('title', 'N/A')}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**❓ Vấn đề nghiên cứu:**")
                        st.write(topic_data.get('research_problem', 'N/A'))
                        
                        st.write("**🎯 Mục tiêu nghiên cứu:**")
                        objectives = topic_data.get('objectives', [])
                        for i, obj in enumerate(objectives, 1):
                            st.write(f"{i}. {obj}")
                    
                    with col2:
                        st.write("**❓ Câu hỏi nghiên cứu:**")
                        questions = topic_data.get('research_questions', [])
                        for i, q in enumerate(questions, 1):
                            st.write(f"RQ{i}: {q}")
                        
                        st.write("**💡 Ý nghĩa nghiên cứu:**")
                        st.write(topic_data.get('significance', 'N/A'))
                    
                    # Theoretical Framework
                    st.markdown("---")
                    st.subheader("📚 **2. KHUNG LÝ THUYẾT**")
                    
                    theory_data = plan_data.get('theoretical_framework', {})
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**🏗️ Lý thuyết nền:**")
                        theories = theory_data.get('main_theories', [])
                        for theory in theories:
                            st.write(f"• {theory}")
                        
                        st.write("**🔍 Gaps được giải quyết:**")
                        st.write(theory_data.get('gaps_addressed', 'N/A'))
                    
                    with col2:
                        st.write("**📊 Mô hình nghiên cứu:**")
                        st.write(theory_data.get('research_model', 'N/A'))
                        
                        st.write("**📝 Giả thuyết nghiên cứu:**")
                        hypotheses = theory_data.get('hypotheses', [])
                        for hyp in hypotheses:
                            st.write(f"• {hyp}")
                    
                    # Methodology
                    st.markdown("---")
                    st.subheader("🔬 **3. PHƯƠNG PHÁP NGHIÊN CỨU**")
                    
                    method_data = plan_data.get('methodology', {})
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write("**📋 Research Design:**")
                        st.write(method_data.get('research_design', 'N/A'))
                        
                        st.write("**👥 Population:**")
                        st.write(method_data.get('population', 'N/A'))
                    
                    with col2:
                        st.write("**🎯 Sampling Method:**")
                        st.write(method_data.get('sampling_method', 'N/A'))
                        
                        st.write("**📊 Sample Size:**")
                        st.write(method_data.get('sample_size', 'N/A'))
                    
                    with col3:
                        st.write("**📥 Data Collection:**")
                        st.write(method_data.get('data_collection', 'N/A'))
                        
                        st.write("**📈 Analysis Plan:**")
                        st.write(method_data.get('analysis_plan', 'N/A'))
                    
                    # Variables and Scales
                    st.markdown("---")
                    st.subheader("📊 **4. BIẾN NGHIÊN CỨU & THANG ĐO**")
                    
                    variables_data = plan_data.get('variables_and_scales', {})
                    
                    # Display variables in tabs
                    var_tabs = st.tabs(["📈 Independent", "📉 Dependent", "🔄 Mediating", "⚙️ Control"])
                    
                    var_types = [
                        ('independent_variables', 0),
                        ('dependent_variables', 1), 
                        ('mediating_variables', 2),
                        ('control_variables', 3)
                    ]
                    
                    for var_type, tab_idx in var_types:
                        with var_tabs[tab_idx]:
                            variables = variables_data.get(var_type, [])
                            if variables:
                                for var in variables:
                                    with st.expander(f"📋 {var.get('name', 'Variable')}"):
                                        st.write(f"**📝 Definition:** {var.get('definition', 'N/A')}")
                                        st.write(f"**📏 Scale:** {var.get('scale', 'N/A')}")
                            else:
                                st.info("Không có biến trong category này")
                    
                    # Survey Questionnaire
                    st.markdown("---")
                    st.subheader("📋 **5. BẢNG KHẢO SÁT CHI TIẾT**")
                    
                    survey_data = plan_data.get('survey_questionnaire', {})
                    
                    # Demographics section
                    st.write("### 👥 **Demographics Questions**")
                    demographics = survey_data.get('demographics', [])
                    if demographics:
                        for i, demo in enumerate(demographics, 1):
                            st.write(f"**Q{i}.** {demo.get('question', 'N/A')}")
                            st.write(f"*Type:* {demo.get('type', 'N/A')}")
                            if demo.get('options'):
                                st.write(f"*Options:* {', '.join(demo.get('options', []))}")
                            if demo.get('anchors'):
                                st.write(f"*Scale:* {demo.get('scale', '')} - {' | '.join(demo.get('anchors', []))}")
                            st.write("")
                    
                    # Main constructs
                    st.write("### 📊 **Main Construct Questions**")
                    constructs = survey_data.get('main_constructs', [])
                    if constructs:
                        for construct in constructs:
                            st.write(f"#### 📋 **{construct.get('construct_name', 'Construct')}**")
                            items = construct.get('items', [])
                            for item in items:
                                st.write(f"**{item.get('code', 'CODE')}.** {item.get('question', 'Question')}")
                                st.write(f"*Scale:* {item.get('scale', 'N/A')}")
                                st.write("")
                    
                    # Timeline & Milestones
                    st.markdown("---")
                    st.subheader("🗓️ **6. TIMELINE & MILESTONES**")
                    
                    timeline_data = plan_data.get('timeline', {})
                    
                    phases = [
                        ('phase_1', '🚀 Phase 1', 'primary'),
                        ('phase_2', '📋 Phase 2', 'secondary'),
                        ('phase_3', '📊 Phase 3', 'success'),
                        ('phase_4', '📈 Phase 4', 'info')
                    ]
                    
                    for phase_key, phase_name, color in phases:
                        phase_data = timeline_data.get(phase_key, {})
                        if phase_data:
                            with st.expander(f"{phase_name}: {phase_data.get('duration', 'N/A')}", expanded=True):
                                activities = phase_data.get('activities', [])
                                for activity in activities:
                                    st.write(f"• {activity}")
                    
                    # Implementation Details
                    st.markdown("---")
                    st.subheader("⚙️ **7. IMPLEMENTATION GUIDE**")
                    
                    impl_data = plan_data.get('implementation', {})
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**🛡️ Ethical Considerations:**")
                        ethics = impl_data.get('ethical_considerations', [])
                        for eth in ethics:
                            st.write(f"• {eth}")
                        
                        st.write("**🔧 Tools & Platforms:**")
                        tools = impl_data.get('tools_platforms', [])
                        for tool in tools:
                            st.write(f"• {tool}")
                    
                    with col2:
                        st.write("**💰 Budget Estimation:**")
                        st.write(impl_data.get('budget_estimation', 'N/A'))
                        
                        st.write("**📦 Resources Needed:**")
                        resources = impl_data.get('resources_needed', [])
                        for resource in resources:
                            st.write(f"• {resource}")
                    
                    st.write("**⚠️ Risks & Mitigation:**")
                    risks = impl_data.get('risks_mitigation', [])
                    for risk in risks:
                        st.write(f"• {risk}")
                    
                    # Expected Outputs
                    st.markdown("---")
                    st.subheader("🎯 **8. EXPECTED OUTPUTS**")
                    
                    output_data = plan_data.get('expected_outputs', {})
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**📄 Academic Papers:**")
                        papers = output_data.get('academic_papers', [])
                        for paper in papers:
                            st.write(f"• {paper}")
                        
                        st.write("**🎤 Conferences:**")
                        conferences = output_data.get('conferences', [])
                        for conf in conferences:
                            st.write(f"• {conf}")
                    
                    with col2:
                        st.write("**🏭 Practical Applications:**")
                        applications = output_data.get('practical_applications', [])
                        for app in applications:
                            st.write(f"• {app}")
                        
                        st.write("**🏛️ Policy Recommendations:**")
                        policies = output_data.get('policy_recommendations', [])
                        for policy in policies:
                            st.write(f"• {policy}")
                    
                    # Export Complete Research Plan
                    st.markdown("---")
                    st.subheader("💾 **Export Complete Research Plan**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Export full plan as JSON
                        full_json = json.dumps(research_plan, ensure_ascii=False, indent=2)
                        st.download_button(
                            label="📥 **Download Full Plan (JSON)**",
                            data=full_json,
                            file_name=f"complete_research_plan_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            help="Download complete research plan với tất cả details"
                        )
                    
                    with col2:
                        # Export survey questionnaire as CSV
                        if survey_data:
                            survey_items = []
                            
                            # Add demographics
                            for i, demo in enumerate(survey_data.get('demographics', []), 1):
                                survey_items.append({
                                    'Section': 'Demographics',
                                    'Question_Code': f'DEMO{i}',
                                    'Question_Text': demo.get('question', ''),
                                    'Question_Type': demo.get('type', ''),
                                    'Scale': demo.get('scale', ''),
                                    'Options': ', '.join(demo.get('options', [])) if demo.get('options') else ''
                                })
                            
                            # Add main constructs
                            for construct in survey_data.get('main_constructs', []):
                                for item in construct.get('items', []):
                                    survey_items.append({
                                        'Section': construct.get('construct_name', ''),
                                        'Question_Code': item.get('code', ''),
                                        'Question_Text': item.get('question', ''),
                                        'Question_Type': 'Likert',
                                        'Scale': item.get('scale', ''),
                                        'Options': ''
                                    })
                            
                            if survey_items:
                                survey_df = pd.DataFrame(survey_items)
                                survey_csv = survey_df.to_csv(index=False, encoding='utf-8-sig')
                                
                                st.download_button(
                                    label="📋 **Download Survey (CSV)**",
                                    data=survey_csv,
                                    file_name=f"research_survey_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    help="Download survey questionnaire để import vào platforms"
                                )
                    
                    with col3:
                        # Export timeline as CSV
                        if timeline_data:
                            timeline_items = []
                            for phase_key, phase_data in timeline_data.items():
                                for activity in phase_data.get('activities', []):
                                    timeline_items.append({
                                        'Phase': phase_key.replace('_', ' ').title(),
                                        'Duration': phase_data.get('duration', ''),
                                        'Activity': activity,
                                        'Status': 'Planned'
                                    })
                            
                            if timeline_items:
                                timeline_df = pd.DataFrame(timeline_items)
                                timeline_csv = timeline_df.to_csv(index=False, encoding='utf-8-sig')
                                
                                st.download_button(
                                    label="🗓️ **Download Timeline (CSV)**",
                                    data=timeline_csv,
                                    file_name=f"research_timeline_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    help="Download timeline để project management"
                    )

        # Đoạn code liên quan đến deep-research đã được loại bỏ hoàn toàn

if __name__ == "__main__":
    main() 