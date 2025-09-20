# 📚 Systematic Literature Review Tool

**🚀 Powered by Microsoft MarkItDown + Gemini AI**

Công cụ phân tích tài liệu nghiên cứu tự động với khả năng tạo bảng systematic literature review từ **multiple file formats** với **structure preservation**.

## ✨ **Tính Năng Chính**

### 📄 **Multi-Format Support** ⭐ **NEW** ⭐
- **PDF** - Research papers, academic articles
- **Word** - .docx, .doc research reports
- **PowerPoint** - .pptx, .ppt presentations  
- **Excel** - .xlsx, .xls data tables

### 🤖 **AI-Powered Analysis**
- **Gemini AI** models (2.5 Flash, Lite, Pro)
- **Smart extraction** với 22 columns systematic review
- **Language preservation** - giữ nguyên ngôn ngữ gốc
- **Structure-aware** - hiểu được headings, tables, lists

### 📊 **Advanced Analytics**
- **Frequency analysis** - lý thuyết, biến, phương pháp
- **Statistical insights** - visualizations & charts  
- **Batch processing** - 1-1000 files
- **Real-time monitoring** - progress & success rates

### 🧠 **Learning System** ⭐ **NEW** ⭐
- **Automatic knowledge capture** - learns from every analyzed document
- **Persistent knowledge base** - builds cumulative research insights
- **Smart search & retrieval** - find patterns across learned papers
- **Knowledge export** - CSV/JSON export of learned data
- **Research analytics** - top theories, methods, and trends

### 💾 **Export & Visualization**
- **CSV/Excel export** - compatible với Excel
- **Interactive tables** - search, sort, filter
- **Statistical reports** - frequency summaries

## 🛠️ **Technology Stack**

- **Frontend**: Streamlit with Arrow-optimized DataFrames
- **Document Processing**: Microsoft MarkItDown  
- **AI Engine**: Google Gemini AI
- **Data Analysis**: Pandas with type-safe operations
- **Fallback Extractors**: PyMuPDF, PyPDF2
- **Learning System**: JSON-based persistent storage

## 🚀 **Installation & Setup**

### **Prerequisites**
- Python 3.10+
- Gemini AI API key

### **Install Dependencies**
```bash
pip install -r requirements_new.txt
```

### **Run Application**
```bash
streamlit run app_literature_analyzer.py --server.port 8502
```

### **Access Application**
```
http://localhost:8502
```

## 📋 **Systematic Review Columns**

1. STT (Serial Number)
2. Năm xuất bản (Publication Year)
3. Tên tác giả (Authors)
4. Tiêu đề bài nghiên cứu (Title)
5. Mục tiêu nghiên cứu (Research Objectives)
6. Lĩnh vực NC (Research Field)
7. Địa điểm NC (Research Area)
8. Lý Thuyết nền (Theoretical Foundation)
9. Mẫu (Sample)
10. PP phân tích dữ liệu (Statistical Method)
11. Biến độc lập (Independent Variables)
12. Biến phụ thuộc (Dependent Variables)
13. Biến trung gian (Mediating Variables)
14. Biến điều tiết (Moderating Variables)
15. Biến kiểm soát (Control Variables)
16. Biến khác nếu có (Other Variables)
17. Findings (Research Results)
18. Hạn chế và hướng NC tiếp theo (Limitations)
19. Thang đo (Measurement)
20. Mô hình có hay không (Model Existence)
21. Mô hình gồm các thành phần gì (Model Components)
22. Số DOI (DOI Number)

## 🎯 **Usage Workflow**

### **Step 1: Upload Documents**
- Drag & drop multiple files
- Support: PDF, Word, PowerPoint, Excel
- Choose quantity (1-1000) or analyze all

### **Step 2: AI Analysis**
- Real-time progress monitoring
- Structure-preserved extraction
- Smart content analysis

### **Step 3: Review Results**
- Interactive 22-column table
- Search, sort, filter capabilities
- Download CSV/Excel

### **Step 4: Statistical Analysis**
- Frequency analysis by category
- Visual charts & insights
- Export statistical reports

### **Step 5: Learning System** ⭐ **NEW** ⭐
- Automatic knowledge capture from analysis
- Search learned documents by theory/method
- Export knowledge base for team sharing
- Analytics on accumulated research insights

## ⚙️ **AI Model Options**

- **Gemini 2.5 Flash Lite** - Fastest (default)
- **Gemini 2.5 Flash** - Balanced speed/quality
- **Gemini 1.5 Flash** - Stable & reliable
- **Gemini 1.5 Pro** - Highest quality

## 🔧 **Advanced Features**

### **Document Processing Priority**
1. **Microsoft MarkItDown** - Primary (structure preservation)
2. **PyMuPDF** - Fallback 1 (fast extraction)
3. **PyPDF2** - Fallback 2 (compatibility)

### **Batch Processing**
- **Smart chunking** - optimal for AI limits
- **Error handling** - continue on failures
- **Rate limiting** - API-friendly pauses
- **Progress tracking** - real-time updates

### **Multi-language Support**
- **Language detection** - automatic
- **Preservation mode** - keep original language
- **Unicode handling** - international characters

## 📊 **Statistical Analysis Features**

### **Frequency Analysis**
- **Theoretical Foundations** - count & percentage
- **Dependent Variables** - popularity ranking
- **Independent Variables** - usage patterns
- **Statistical Methods** - methodology trends

### **Visualizations**
- **Bar charts** - top 10 items per category
- **Summary metrics** - unique counts
- **Percentage breakdowns** - relative frequencies

## 💡 **Best Practices**

### **File Preparation**
- **Quality documents** - clear, readable text
- **Structured content** - proper headings/sections
- **Research papers** - academic format preferred

### **Batch Optimization**
- **5-20 files** - optimal for testing
- **50+ files** - use Lite model for speed
- **100+ files** - allow extra processing time

### **Result Quality**
- **Review outputs** - verify accuracy
- **Check extractions** - validate key fields
- **Export regularly** - backup results

## 🚨 **Troubleshooting**

### **Common Issues**
- **File format errors** - check supported types
- **Extraction failures** - try different file quality
- **API limits** - use smaller batches
- **Memory issues** - restart for large batches

### **Performance Tips**
- **Lite model** - for speed over quality
- **Smaller batches** - for consistent results
- **Good internet** - for API reliability

## 📈 **Version History**

### **v2.0** ⭐ **Current** ⭐
- **Microsoft MarkItDown** integration
- **Multi-format support** (PDF, Word, PPT, Excel)
- **Enhanced structure preservation**
- **Improved AI analysis accuracy**

### **v1.0**
- **Basic PDF support**
- **22-column systematic review**
- **Gemini AI integration**
- **Statistical analysis**

## 🤝 **Contributing**

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 **License**

This project is for academic and research purposes.

---

**🎯 Perfect tool for systematic literature review and academic research analysis!**

*Powered by Microsoft MarkItDown & Google Gemini AI* 