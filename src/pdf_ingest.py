from dataclasses import dataclass
from typing import List, IO
import fitz  # PyMuPDF
from .text_utils import clean_text, chunk_text

@dataclass
class Chunk:
    text: str
    pdf_name: str
    page_num: int
    chunk_id: str

class PDFProcessor:
    def extract_chunks(self, uploaded_files: List[IO], max_tokens=500, overlap=50) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        for f in uploaded_files:
            pdf_bytes = f.read()
            doc = fitz.open(stream=pdf_bytes, filetype='pdf')
            pdf_name = getattr(f, 'name', 'uploaded.pdf')
            for page_index in range(len(doc)):
                page = doc[page_index]
                text = page.get_text('text')
                text = clean_text(text)
                if not text:
                    continue
                pieces = chunk_text(text, max_tokens=max_tokens, overlap=overlap)
                for j, piece in enumerate(pieces):
                    cid = f"{pdf_name}::p{page_index+1}::c{j+1}"
                    all_chunks.append(Chunk(text=piece, pdf_name=pdf_name, page_num=page_index+1, chunk_id=cid))
        return all_chunks
