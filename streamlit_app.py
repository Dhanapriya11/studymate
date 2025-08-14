import os
import pandas as pd
import streamlit as st
import re
from dotenv import load_dotenv

from src.pdf_ingest import PDFProcessor
from src.vectorstore import VectorStore
from src.rag import RAGPipeline
from src.llm_hf import HFLLM

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(page_title='StudyMate — PDF Q&A', page_icon='📚', layout='wide')
load_dotenv()

# Load custom CSS
with open('assets/styles.css', 'r') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize session states
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
    
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.chunks_df = None

if 'question_history' not in st.session_state:
    st.session_state.question_history = []

# Apply theme attribute to body
theme = "dark" if st.session_state.dark_mode else "light"
st.markdown(f"<body data-theme='{theme}'></body>", unsafe_allow_html=True)

# =====================
# HEADER SECTION
# =====================
st.markdown('<div class="styled-card">', unsafe_allow_html=True)
st.title('📚 StudyMate — AI-Powered PDF Q&A')
st.caption('Upload PDFs, ask questions, get grounded answers with sources.')
st.markdown('</div>', unsafe_allow_html=True)

# Dark mode toggle
col1, col2, col3 = st.columns([1, 5, 1])
with col2:
    if st.button("🌓 Toggle Dark Mode" if not st.session_state.dark_mode else "☀️ Toggle Light Mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# Apply theme dynamically
theme = "dark" if st.session_state.dark_mode else "light"
st.markdown(f"""
<style>
    body {{
        background-color: var(--background-{theme});
        color: var(--text-{theme});
    }}
    [data-testid="stAppViewContainer"] {{
        background-color: var(--background-{theme});
    }}
</style>
""", unsafe_allow_html=True)

# =====================
# MAIN LAYOUT
# =====================
tab1, tab2, tab3 = st.tabs(["📄 Upload & Process", "💬 Ask Questions", "🗂️ History"])

# =====================
# TAB 1: UPLOAD & PROCESS
# =====================
with tab1:
    st.markdown('<div class="styled-card">', unsafe_allow_html=True)
    st.header("Upload PDF Documents")
    
    # Settings panel
    with st.expander('⚙️ Advanced Settings', expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            max_tokens = st.slider('Chunk size (words)', 200, 1000, 500, step=50)
            overlap = st.slider('Chunk overlap', 0, 200, 50, step=10)
        with col2:
            top_k = st.slider('Top-K chunks to retrieve', 1, 10, 4)
            embed_model = st.text_input('Embedding model', 'all-MiniLM-L6-v2')
        
        model_id = st.text_input('Generation model', 'google/flan-t5-small')
        st.caption("Recommended models: google/flan-t5-small (fast), google/flan-t5-base (balanced), ibm-granite/granite-3.3-2b-instruct (powerful but slow)")
        hf_token = st.text_input(
            'HuggingFace token (required for private models)',
            value=os.getenv('HF_TOKEN'),
            type='password'
        )
    
    # PDF Upload Section
    st.subheader("Upload PDFs")
    uploaded = st.file_uploader(
        'Choose one or more PDF files',
        type=['pdf'],
        accept_multiple_files=True,
        key="pdf_uploader"
    )
    
    if uploaded:
        st.success(f"✅ {len(uploaded)} file(s) uploaded successfully!")
    
    # Build Knowledge Base
    if uploaded and st.button('🔧 Build Knowledge Base', type='primary', use_container_width=True):
        with st.status('Processing documents...', expanded=True) as status:
            processor = PDFProcessor()
            status.update(label='🔍 Extracting text and chunking...')
            chunks = processor.extract_chunks(uploaded, max_tokens=max_tokens, overlap=overlap)

            status.update(label='🧠 Building embeddings & FAISS index...')
            vs = VectorStore(model_name=embed_model)
            vs.build(chunks)

            st.session_state.vector_store = vs
            df = pd.DataFrame([{
                'chunk_id': c.chunk_id,
                'pdf_name': c.pdf_name,
                'page': c.page_num,
                'text': c.text[:500] + ('…' if len(c.text) > 500 else '')
            } for c in chunks])
            st.session_state.chunks_df = df

            status.update(label='✅ Knowledge base ready!', state='complete')
    
    # Preview indexed chunks
    if st.session_state.chunks_df is not None:
        st.subheader("Indexed Chunks Preview")
        st.dataframe(st.session_state.chunks_df, use_container_width=True, height=250)
        
        # Document Summarization
        if st.button('📝 Summarize Documents', use_container_width=True):
            with st.spinner('Generating summary...'):
                try:
                    @st.cache_resource
                    def get_llm(model_id):
                        return HFLLM(model_id)

                    llm = get_llm(model_id)

                    # Safely build context from all chunks
                    all_context = ""
                    chunks_list = getattr(st.session_state.vector_store, "meta", [])

                    if not chunks_list:
                        st.error("No chunks found in the vector store. Please rebuild the knowledge base.")
                        st.stop()

                    for chunk in chunks_list:
                        if hasattr(chunk, "text"):  # Object with .text attribute
                            all_context += f"{chunk.text}\n\n"
                        elif isinstance(chunk, dict) and "text" in chunk:  # Dictionary
                            all_context += f"{chunk['text']}\n\n"
                        else:
                            st.warning(f"Unknown chunk format: {chunk}")

                    # Limit context size for model
                    max_context_chars = 2000 if "small" in model_id.lower() else 4000
                    all_context = all_context[:max_context_chars]

                    summary_prompt = f"""
You are StudyMate, an academic assistant. Please provide a concise summary of the following documents.
Focus on the main topics, key points, and important information.

<documents>
{all_context}
</documents>

Summary:
"""

                    summary = llm.generate(summary_prompt, max_new_tokens=300)

                    if "Error generating response" in summary or "Unable to generate response" in summary:
                        st.error(summary)
                        st.info("""
                        **Troubleshooting tips:**
                        - Try a different model in the Settings panel
                        - Reduce the number of uploaded documents
                        - Use a smaller model like `google/flan-t5-small`
                        - Clear Hugging Face cache with `huggingface-cli delete-cache`
                        """)
                    else:
                        st.markdown('### 📋 Document Summary')
                        st.write(summary)

                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# =====================
# TAB 2: ASK QUESTIONS
# =====================
with tab2:
    st.markdown('<div class="styled-card">', unsafe_allow_html=True)
    st.header("Ask Questions About Your Documents")
    
    # Check if knowledge base exists
    if st.session_state.vector_store is None:
        st.info("💡 Please upload PDFs and build a knowledge base in the 'Upload & Process' tab first.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()
    
    # Question input
    q = st.text_input('Enter your question:', placeholder='e.g., What are the main findings of the research?')
    
    if q:
        try:
            # Use cached LLM initialization for better performance
            @st.cache_resource
            def get_llm(model_id):
                return HFLLM(model_id)
            
            llm = get_llm(model_id)
        except Exception as e:
            st.error(f'LLM setup error: {e}')
            st.info("""
            **Troubleshooting tips:**
            1. Check your disk space - models require several hundred MBs to download
            2. Try a smaller model like `google/flan-t5-small` or `sshleifer/tiny-gpt2`
            3. If you see a 'from_tf=True' error, the model has TensorFlow weights only
            4. Check your internet connection for downloading models
            """)
            st.stop()

        rag = RAGPipeline(st.session_state.vector_store, llm)
        with st.spinner('Thinking...'):
            answer, sources = rag.answer(q, top_k=top_k)

        st.markdown('### ✅ Answer')
        # Handle LLM errors gracefully
        if "Error generating response" in answer or "Unable to generate response" in answer:
            st.error(answer)
            st.info("""
            **Troubleshooting tips:**
            - Try a different model in the Settings panel
            - Check your disk space and internet connection
            - Use a smaller model like `google/flan-t5-small` or `sshleifer/tiny-gpt2`
            - Clear Hugging Face cache by running `huggingface-cli delete-cache` in your terminal/command prompt (make sure to type the command exactly without extra characters)
            - Change download location to a drive with more space by setting the `HF_HOME` environment variable
            """)
        else:
            # Format answer with proper markdown for bullet points
            st.markdown(answer)
        
        # Export functionality
        export_content = f"Question: {q}\n\nAnswer: {answer}\n\nSources:\n"
        for s in sources:
            export_content += f"- {s.pdf_name}, Page {s.page_num}, ID {s.chunk_id}\n"
        
        st.download_button(
            label="📥 Export Answer",
            data=export_content,
            file_name=f"studymate_answer_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
        
        # Save to history
        st.session_state.question_history.append({
            'question': q,
            'answer': answer,
            'sources': sources,
            'timestamp': pd.Timestamp.now()
        })

        st.markdown('### 📚 Sources Used')
        for i, s in enumerate(sources):
            with st.expander(f"Source {i+1}: {s.pdf_name} (Page {s.page_num})"):
                st.markdown(f"**Chunk ID:** {s.chunk_id}")
                st.write(s.text)
        
        with st.expander('ℹ️ Debug / Context Sent'):
            st.code(rag.build_context(sources))
    
    st.markdown('</div>', unsafe_allow_html=True)

# =====================
# TAB 3: HISTORY
# =====================
with tab3:
    st.markdown('<div class="styled-card">', unsafe_allow_html=True)
    st.header("Question History")
    
    if not st.session_state.question_history:
        st.info("No questions asked yet. Start asking questions in the 'Ask Questions' tab!")
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()
    
    # Search functionality
    search_term = st.text_input('🔍 Search in history:', placeholder='Search questions and answers...')
    
    # Clear History Button
    if st.button('🗑️ Clear History'):
        st.session_state.question_history.clear()
        st.success("History cleared!")
        st.rerun()
    
    # Filter history based on search term
    if search_term:
        filtered_history = []
        for item in st.session_state.question_history:
            found_in_question = search_term.lower() in item['question'].lower()
            found_in_answer = search_term.lower() in item['answer'].lower()
            if found_in_question or found_in_answer:
                # Add location info to the item
                item_copy = item.copy()
                item_copy['found_in'] = []
                if found_in_question:
                    item_copy['found_in'].append('question')
                if found_in_answer:
                    item_copy['found_in'].append('answer')
                filtered_history.append(item_copy)
        
        st.markdown(f"Found {len(filtered_history)} matching entries")
    else:
        # Add found_in info to all items when not searching
        filtered_history = []
        for item in st.session_state.question_history:
            item_copy = item.copy()
            item_copy['found_in'] = ['all']  # Not searching, so show all
            filtered_history.append(item_copy)
    
    # Show last 10 questions (or filtered results)
    display_history = filtered_history[-10:] if not search_term else filtered_history
    
    # Function to highlight search terms
    def highlight_text(text, search_term):
        if not search_term:
            return text
        # Case insensitive replacement with highlighting
        pattern = re.compile(f'({re.escape(search_term)})', re.IGNORECASE)
        highlighted = pattern.sub(r'**\1**', text)  # Bold the search term
        return highlighted
    
    for i, item in enumerate(reversed(display_history)):
        # Show where the term was found
        if search_term and 'found_in' in item:
            found_locations = ', '.join(item['found_in'])
            st.markdown(f"**Found in: {found_locations}**")
        
        # Highlight search terms in question and answer
        highlighted_question = highlight_text(item['question'], search_term)
        highlighted_answer = highlight_text(item['answer'], search_term)
        
        st.markdown(f"**Q:** {highlighted_question}")
        st.markdown(f"**Answer:** {highlighted_answer}")
        st.markdown(f"**Asked at:** {item['timestamp']}")
        st.markdown("---")  # Separator line
    
    st.markdown('</div>', unsafe_allow_html=True)
