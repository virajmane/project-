from demo import solve 
from flask import Flask

app = Flask(__name__) 
@app.route("/")
def index():
    ans = solve() 
    return ans
app.run(host="0.0.0.0", port=5000, debug=True)