from flask import Flask, redirect, render_template, request, url_for
from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__)

# This tells Flask-SQLAlchemy where the SQLite database file should live.
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"

# This turns off a feature we do not need and avoids warning messages.
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Create the SQLAlchemy object. We use it to define models and talk to the DB.
db = SQLAlchemy(app)


class User(db.Model):
    # A model is a Python class that represents a database table.
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False, unique=True)

    def __repr__(self):
        return f"<User {self.name}>"


with app.app_context():
    # Create the database and table automatically if they do not exist yet.
    db.create_all()


@app.route("/")
def index():
    # query.all() gets every row from the User table.
    users = User.query.order_by(User.id.desc()).all()
    return render_template("index.html", users=users)


@app.route("/add", methods=["GET", "POST"])
def add_user():
    if request.method == "POST":
        name = request.form["name"].strip()
        email = request.form["email"].strip()

        # A simple check so empty values are not saved.
        if not name or not email:
            return render_template(
                "add.html",
                error="Name and email are required.",
                form_data={"name": name, "email": email},
            )

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return render_template(
                "add.html",
                error="This email is already in use. Try a different one.",
                form_data={"name": name, "email": email},
            )

        # Create a new User object and add it to the session.
        user = User(name=name, email=email)
        db.session.add(user)

        # commit() saves the changes permanently to the database.
        db.session.commit()

        return redirect(url_for("index"))

    return render_template("add.html", error=None, form_data={"name": "", "email": ""})


@app.route("/edit/<int:user_id>", methods=["GET", "POST"])
def edit_user(user_id):
    # get_or_404() finds the row by ID or shows a 404 page if missing.
    user = User.query.get_or_404(user_id)

    if request.method == "POST":
        name = request.form["name"].strip()
        email = request.form["email"].strip()

        if not name or not email:
            return render_template(
                "edit.html",
                user=user,
                error="Name and email are required.",
            )

        duplicate_user = User.query.filter(
            User.email == email, User.id != user.id
        ).first()
        if duplicate_user:
            return render_template(
                "edit.html",
                user=user,
                error="Another user already uses this email.",
            )

        # Update the existing object with new form values.
        user.name = name
        user.email = email

        # commit() saves the updated values.
        db.session.commit()

        return redirect(url_for("index"))

    return render_template("edit.html", user=user, error=None)


@app.route("/delete/<int:user_id>", methods=["POST"])
def delete_user(user_id):
    user = User.query.get_or_404(user_id)

    # Delete the object from the session, then commit the change.
    db.session.delete(user)
    db.session.commit()

    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
