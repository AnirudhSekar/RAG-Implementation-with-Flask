from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama.llms import OllamaLLM
from langchain.document_loaders import TextLoader
from langchain.schema import Document
from langchain.chains import RetrievalQA  # Make sure this is imported

from PyPDF2 import PdfReader

from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Initialize the Ollama LLM
llm = OllamaLLM(model="llama3")  # Replace "llama3" with the desired Ollama model

def extract_text_from_pdf(file):
    try:
        # Assuming you are using PyMuPDF or PyPDF2
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting PDF content: {e}")
        return ""

def create_qa_chain(document):
    """Creates a RetrievalQA chain."""
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents([document])  # Pass the list containing the single document

    # Preprocess text (replace newlines with spaces)
    preprocessed_texts = [text.page_content.replace("\n", " ") for text in texts]

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(preprocessed_texts, embeddings)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return qa_chain

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Log the entire files object to check what is being sent
            print("Request Files:", request.files)

            # Check if the file exists in the request
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400
            
            file = request.files['file']
            print(f"Received file: {file.filename}")  # Debug log to confirm file is received

            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400

            file_contents = ""
            
            # Process the file based on its extension
            if file.filename.endswith('.pdf'):
                file_contents = extract_text_from_pdf(file)
            else:
                file_contents = file.read().decode('utf-8-sig', errors='replace')

            document = Document(page_content=file_contents)
            qa_chain = create_qa_chain(document)
            query = request.form['query']
            result = qa_chain.run(query)
            return render_template('index.html', result=result)

        except Exception as e:
            return jsonify({'error': f'An error occurred: {e}'}), 500

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
