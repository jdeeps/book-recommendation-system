
from flask import Flask, url_for, request, render_template, jsonify, json
import csv
from flask_pymongo import PyMongo
from control_logic import *


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/user')
def user():
    return render_template('user.html')


#Item or book title based recommendation
@app.route('/recommendations', methods=['POST'])
def recommendations():
    
    recommendations = []

    if request.method == 'POST':
        title = request.form['title']
        recommendations = recommend(title)

    if recommendations is None:
        return render_template('error.html')
    else:
        return render_template('result.html', recommendations=recommendations, n=len(recommendations['isbn']))
   
#Item based recommendation call to backend      
def recommend(title):
    
    rec_books = title_rec(title)

    if rec_books is not None:

        result = {'isbn':[],'title':[],'author':[],'rating':[],'image_url':[],'users_rated':[],'publisher':[]}
      
        url = 'https://www.google.com/search?q='
       
        for i in range(0,len(rec_books)):
            
            result['isbn'].append(url+str(rec_books[i]['ISBN']))           
            result['title'].append(rec_books[i]['Book-Title'])
            result['author'].append(rec_books[i]['Book-Author'])
            result['rating'].append(round(rec_books[i]['Avg_Rating'],2))
            result['image_url'].append(rec_books[i]['Image-URL-L'])
            result['users_rated'].append(rec_books[i]['Total_No_Of_Users_Rated'])
            result['publisher'].append(rec_books[i]['Publisher'])
                
        return result
    
    else:
        return None

# User based recommendation
@app.route('/userrecommendations', methods=['POST'])
def userrecommendations():

    if request.method == 'POST':
        
        user = request.form['userid']

        recommendations = user_recommend(user)
        
    if recommendations is None:
        return render_template('error.html')

    else:
        return render_template('result.html',recommendations=recommendations, n = len(recommendations['isbn']))

#User based recommendation call to backend        
def user_recommend(user):

    user_rec_book = user_rec(user)

    if user_rec_book is not None:

        result = {'isbn':[],'title':[],'author':[],'rating':[],'image_url':[],'users_rated':[],'publisher':[]}
        url = 'https://www.google.com/search?q='
        
        for i in user_rec_book.index:

            result['isbn'].append(url+str(user_rec_book.loc[i]['ISBN']))
            result['title'].append(user_rec_book.loc[i]['Book-Title'])
            result['author'].append(user_rec_book.loc[i]['Book-Author'])
            result['rating'].append(round(user_rec_book.loc[i]['Avg_Rating'],2))
            result['image_url'].append(user_rec_book.loc[i]['Image-URL-L'])
            result['users_rated'].append(user_rec_book.loc[i]['Total_No_Of_Users_Rated'])
            result['publisher'].append(user_rec_book.loc[i]['Publisher'])
      
        return result
    else:
        return None


if __name__ == '__main__':
    app.run(debug=True)
