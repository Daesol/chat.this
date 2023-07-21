from flask import Flask, request, jsonify
from pdf_processing import process_pdf
from vectorization import vectorize_chunks
from question_answering import answer_question

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_pdf():
    pdf_file = request.files['pdf']
    if pdf_file and pdf_file.filename.endswith('.pdf'):
        text = process_pdf(pdf_file)
        chunks = vectorize_chunks(text)
        return jsonify({'message': 'PDF processed successfully', 'chunks': chunks}), 200
    else:
        return jsonify({'error': 'Invalid PDF file'}), 400

@app.route('/answer', methods=['POST'])
def answer_query():
    data = request.get_json()
    query = data.get('query')
    chunks = data.get('chunks')
    response = answer_question(query, chunks)
    return jsonify({'response': response}), 200

if __name__ == '__main__':
    app.run()
