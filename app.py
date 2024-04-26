import pandas as pd
from flask import Flask, render_template, request

# Load the dataset
df = pd.read_csv("C:\Users\THOTA CHANDRASEKHAR\Downloads\Search Engine Project\Search-Engine")

# Ensure consistent column names
df.columns = df.columns.str.strip().str.replace(' ', '_')

# Fill NaN values in 'Movies&WebSeries' column with an empty string
df['Movies&WebSeries'].fillna('', inplace=True)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def search():
    if request.method == "POST":
        search_text = request.form.get("search_text")
        if search_text:
            # Filter the DataFrame and handle NaN values in 'Movies&WebSeries' column
            results = df[df['Subtitles'].str.contains(search_text, case=False, na=False)]['Movies&WebSeries'].tolist()
            return render_template("results.html", search_text=search_text, results=results)
        else:
            return render_template("results.html", search_text="Nothing", results=None)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
