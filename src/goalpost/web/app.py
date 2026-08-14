"""GoalPost Flask application."""

from flask import Flask, render_template


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    @app.route("/")
    def home():
        return render_template("home.html")

    @app.route("/nfl")
    def nfl():
        return render_template("nfl.html")

    @app.route("/ncaab")
    def ncaab():
        return render_template("ncaab.html")

    @app.route("/mlb")
    def mlb():
        return render_template("mlb.html")

    @app.route("/nba")
    def nba():
        return render_template("nba.html")

    @app.route("/nhl")
    def nhl():
        return render_template("nhl.html")

    @app.route("/soccer")
    def soccer():
        return render_template("soccer.html")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
