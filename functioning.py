from flask import Flask, redirect, url_for, render_template, request, session
from datetime import timedelta

app = Flask(__name__)

# Session makes it easy to pass values
# When the browser is closed session data is deleted since sessions are temporary
# a secret key is always needed to start a session
app.secret_key = "hello"

# For permanent sessions we can specify the time to which they should last
app.permanent_session_lifetime = timedelta(minutes=5)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/login", methods=["POST", "GET"])
def login():
    if request.method == "POST":

        # for temporary sessions we can have it as false
        session.permanent = True
        name = request.form["nm"]
        session["name"] = name
        return redirect(url_for("user"))
    else:
        # This means if a person has logged in already then redirect him to user page
        if "name" in session:
            return redirect(url_for("user"))
        else:
            return render_template("login.html")


@app.route("/user")
def user():
    if "name" in session:
        # Currently we are not using the name
        name_customer = session["name"]
        return render_template("user.html")
    else:
        return redirect(url_for("login"))


@app.route("/logout")
def logout():
    # Deleting session values
    session.pop("name", None)
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)
