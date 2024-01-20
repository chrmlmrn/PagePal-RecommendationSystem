from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

book = pd.read_csv('Books.csv')
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('neighbors_model.pkl', 'rb') as neighbors_file:
    neighbors_model = pickle.load(neighbors_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    query_title = request.args.get('title')  
    query_vector = vectorizer.transform([query_title])
    distances, indices = neighbors_model.kneighbors(query_vector)

    similar_books = []
    for idx in indices[0]:
        title = book.iloc[idx]['Book-Title']
        image_url = book.iloc[idx]['Image-URL-L']
        similar_books.append((title, image_url))

    return render_template('index.html', recommendations=similar_books, query_title=query_title)

if __name__ == '__main__':
    app.run(debug=True)