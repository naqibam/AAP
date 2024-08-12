from flask import Flask, jsonify, request,render_template,redirect, url_for,session, flash
from flask_sqlalchemy import SQLAlchemy

import urllib.parse
import pypyodbc as odbc

from credentials import username, password


app = Flask(__name__)

app.secret_key = 'supersecretkey'


params = urllib.parse.quote_plus("Driver={ODBC Driver 18 for SQL Server};Server=tcp:aap-221734q.database.windows.net,1433;Database=aap;Uid=aap-user;Pwd=pM3sEHAZ+_2hu5-;Encrypt=yes;TrustServerCertificate=yes;Connection Timeout=30;")
app.config['SQLALCHEMY_DATABASE_URI'] = "mssql+pyodbc:///?odbc_connect=" + params
#app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
#app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://aap:mysql@localhost:3306/aap'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
 
    app.run(debug=True)