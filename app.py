"""
LeafTrack-SG – Flask Application Entry Point
AI-Based Crop Disease Detection, Segmentation & Severity Assessment
"""
import os
import sys
from functools import wraps

# Ensure project root is in Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, send_from_directory, redirect, url_for, session, request, jsonify
from flask_cors import CORS

import config
from database.db import init_db, populate_disease_library
from routes.prediction_routes import prediction_bp
from routes.history_routes import history_bp
from routes.progression_routes import progression_bp
from routes.auth_routes import auth_bp


# ── Login-required decorator ──────────────────────────────────────────────────
def login_required(f):
    """
    Every page and API route is protected by this decorator.
    Browser requests  → redirect to /login?next=<requested path>
    API/XHR requests  → return 401 JSON so the frontend can handle it
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            # Distinguish AJAX calls (Accept: application/json) from page loads
            if request.headers.get("Accept", "").startswith("application/json") or \
               request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({
                    "success": False,
                    "error": "Please log in to continue.",
                    "login_required": True,
                }), 401
            return redirect(url_for("login_page", next=request.path))
        return f(*args, **kwargs)
    return decorated


# ── No-cache decorator — prevents back-button access after logout ─────────────
def no_cache(f):
    """Add Cache-Control headers so browsers don't cache protected pages."""
    @wraps(f)
    def decorated(*args, **kwargs):
        resp = f(*args, **kwargs)
        # Flask returns different response types; handle both
        from flask import make_response
        response = make_response(resp)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"]         = "no-cache"
        response.headers["Expires"]        = "0"
        return response
    return decorated


def create_app() -> Flask:
    app = Flask(__name__,
                template_folder="templates",
                static_folder="static")

    app.config["SECRET_KEY"]          = config.SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"]  = config.MAX_IMAGE_SIZE_MB * 1024 * 1024
    app.config["UPLOAD_FOLDER"]       = config.UPLOADS_DIR

    # Enable CORS for all API routes
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register blueprints
    app.register_blueprint(prediction_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(progression_bp)
    app.register_blueprint(auth_bp)

    # ── Auth pages (the ONLY public routes) ───────────────────────────────────
    @app.route("/login")
    def login_page():
        # Already logged in → go straight to dashboard
        if "user_id" in session:
            return redirect(url_for("dashboard"))
        msg = request.args.get("msg", "")
        return render_template("auth.html", mode="login",
                               next_url=request.args.get("next", "/dashboard"),
                               flash_msg=msg)

    @app.route("/signup")
    def signup_page():
        if "user_id" in session:
            return redirect(url_for("dashboard"))
        return render_template("auth.html", mode="signup",
                               next_url="/dashboard",
                               flash_msg="")

    # ── All remaining pages: login required + no-cache ────────────────────────

    @app.route("/")
    @login_required
    @no_cache
    def home():
        # Root always goes to dashboard for logged-in users
        return redirect(url_for("dashboard"))

    @app.route("/dashboard")
    @login_required
    @no_cache
    def dashboard():
        return render_template("dashboard.html",
                               user_name=session.get("user_name", ""))

    @app.route("/diagnosis")
    @login_required
    @no_cache
    def diagnosis():
        return render_template("diagnosis.html",
                               user_name=session.get("user_name", ""))

    @app.route("/results")
    @login_required
    @no_cache
    def results():
        return render_template("results.html",
                               user_name=session.get("user_name", ""))

    @app.route("/disease-library")
    @login_required
    @no_cache
    def disease_library():
        return render_template("disease_library.html",
                               user_name=session.get("user_name", ""))

    @app.route("/progression")
    @login_required
    @no_cache
    def progression():
        return render_template("progression.html",
                               user_name=session.get("user_name", ""))

    @app.route("/progression/<session_id>")
    @login_required
    @no_cache
    def progression_detail(session_id):
        return render_template("progression.html", session_id=session_id,
                               user_name=session.get("user_name", ""))

    @app.route("/report")
    @login_required
    @no_cache
    def report_page():
        return render_template("report.html",
                               user_name=session.get("user_name", ""))

    # ── Logout page (GET /logout for convenience) ─────────────────────────────
    @app.route("/logout")
    def logout_page():
        session.clear()
        resp = redirect(url_for("login_page", msg="You have been logged out successfully."))
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"]         = "no-cache"
        resp.headers["Expires"]        = "0"
        return resp

    # ── Static file serving for uploads ───────────────────────────────────────
    @app.route("/uploads/<path:filename>")
    @login_required
    def uploaded_file(filename):
        return send_from_directory(config.UPLOADS_DIR, filename)

    # ── Health Check (public — used by monitoring) ────────────────────────────
    @app.route("/api/health")
    def health_check():
        classifier_ready   = os.path.exists(config.CLASSIFIER_MODEL_PATH)
        segmentation_ready = os.path.exists(config.SEGMENTATION_MODEL_PATH)
        return jsonify({
            "status": "online",
            "classifier_model":   "ready" if classifier_ready   else "not trained",
            "segmentation_model": "ready" if segmentation_ready else "not trained",
            "version": "1.0.0",
        })

    return app


# ─── Database Init ────────────────────────────────────────────────────────────
init_db()
try:
    populate_disease_library()
except Exception:
    pass   # Already populated

# ─── Create app instance ──────────────────────────────────────────────────────
app = create_app()

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                    LeafTrack-SG                                  ║
║        AI Crop Disease Detection System                          ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Server:  http://localhost:5000                                  ║
║  Login:   http://localhost:5000/login                            ║
║                                                                  ║
║  All pages require login.                                        ║
║  New users → /signup                                             ║
╚══════════════════════════════════════════════════════════════════╝
""")
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
    )
