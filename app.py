"""Enterprise Flask application Controller and HTTP route definitions.

This module instantiates the Flask server utilizing an Application Factory pattern,
connects route endpoints to the Hugging Face BART domain summarizer engine, handles
runtime validation, and integrates robust HTTP exception handling.
"""

import logging
from typing import Any, Tuple
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
from werkzeug.exceptions import HTTPException

from config import config_by_name, Config
from summarizer import SummarizationEngine, SummaryResult, MODEL_REGISTRY, DEFAULT_MODEL_ID, count_words

# Configure Flask Controller Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_app(config_name: str = "default") -> Flask:
    """Application Factory function responsible for configuring and creating the Flask instance.

    Args:
        config_name (str): Selector keyword matching environment ('development', 'production', or 'default').

    Returns:
        Flask: Fully initialized Flask application ready to serve requests.
    """
    app = Flask(__name__)
    app.config.from_object(config_by_name.get(config_name, Config))

    # Initialize application directories and workspace storage
    Config.initialize_app(app)
    
    # Initialize summarizer singleton in background or first invocation
    logger.info("Application factory completed setup successfully.")
    
    register_routes(app)
    register_error_handlers(app)
    
    return app


LENGTH_PROFILES = {
    "short": {"max_length": 80, "min_length": 25},
    "medium": {"max_length": 150, "min_length": 45},
    "detailed": {"max_length": 250, "min_length": 80}
}

def get_hyperparams_from_profile(profile: str) -> tuple[int, int]:
    """Retrieves validated model generation parameters mapped from a user-selected profile.

    Args:
        profile (str): Length profile requested by the user ('short', 'medium', 'detailed').

    Returns:
        tuple[int, int]: (max_length, min_length) values.
    """
    safe_profile = str(profile).strip().lower()
    if safe_profile not in LENGTH_PROFILES:
        safe_profile = "medium"
    
    settings = LENGTH_PROFILES[safe_profile]
    return settings["max_length"], settings["min_length"]


def register_routes(app: Flask) -> None:
    """Registers standard HTML presentation and REST API routing endpoints onto the application.

    Args:
        app (Flask): Target Flask web application instance.
    """

    @app.route("/", methods=["GET"])
    def index() -> str:
        """Renders the interactive web dashboard for user text submission.

        Returns:
            str: Compiled HTML rendering of the index page.
        """
        return render_template(
            "index.html",
            max_words=Config.MAX_INPUT_WORDS,
            models=MODEL_REGISTRY.values(),
            default_model=DEFAULT_MODEL_ID
        )

    @app.route("/summarize", methods=["POST"])
    def summarize() -> Any:
        """Processes form text payloads, executes Seq2Seq AI inference, and presents comparative metrics.

        Returns:
            Any: Compiled HTML rendering of result page upon success, or redirect upon validation failure.
        """
        input_text = request.form.get("input_text", "").strip()
        
        # Convert user-friendly length profile into exact neural hyperparameters
        max_len, min_len = get_hyperparams_from_profile(
            request.form.get("length_profile", "medium")
        )

        # Input Validation Checkpoints (consistent with client-side Javascript rules)
        if not input_text:
            flash("Please paste or type text into the input terminal before generating a summary.", "warning")
            return redirect(url_for("index"))

        word_count = count_words(input_text)
        if word_count > Config.MAX_INPUT_WORDS:
            flash(
                f"Input length ({word_count:,} words) surpasses the maximum threshold of {Config.MAX_INPUT_WORDS:,} words. Please shorten your text.",
                "danger"
            )
            return redirect(url_for("index"))
        
        if word_count < 5:
            flash("Input text is too abbreviated (< 5 words). Please provide a fuller sentence or paragraph.", "warning")
            return redirect(url_for("index"))

        try:
            logger.info(f"Received web summarization request for text comprising {word_count} words.")
            
            model_id = request.form.get("model_id", DEFAULT_MODEL_ID)
            
            engine = SummarizationEngine()
            result: SummaryResult = engine.summarize(
                raw_text=input_text,
                max_length=max_len,
                min_length=min_len,
                model_id=model_id
            )
            
            return render_template("result.html", result=result)

        except Exception as err:
            logger.exception("Unexpected AI neural engine inference exception occurred.")
            flash(f"Model processing error occurred: {str(err)}", "danger")
            return redirect(url_for("index"))

    @app.route("/api/summarize", methods=["POST"])
    def api_summarize() -> Tuple[Response, int]:
        """RESTful JSON endpoint allowing external consumer apps to invoke transformer inference.

        Expected JSON body:
            {"text": "long english string...", "max_length": 130, "min_length": 30}

        Returns:
            Tuple[Response, int]: JSON response payload accompanied by HTTP status code.
        """
        payload = request.get_json(silent=True)
        if not payload or "text" not in payload:
            return jsonify({"error": "Missing required JSON field 'text'"}), 400

        text_data = str(payload.get("text", "")).strip()
        word_count = count_words(text_data)
        
        if word_count < 5 or word_count > Config.MAX_INPUT_WORDS:
            return jsonify({"error": f"Word frequency ({word_count}) violates allowable boundaries (5 - {Config.MAX_INPUT_WORDS} words)"}), 422

        try:
            # Safely clamp API hyperparameter parameters against DoS computational attacks
            max_len, min_len = get_hyperparams_from_profile(
                payload.get("length_profile", "medium")
            )

            model_id = str(payload.get("model_id", DEFAULT_MODEL_ID)).strip()

            engine = SummarizationEngine()
            output: SummaryResult = engine.summarize(
                raw_text=text_data,
                max_length=max_len,
                min_length=min_len,
                model_id=model_id
            )

            return jsonify({
                "status": "success",
                "metrics": {
                    "original_word_count": output.original_word_count,
                    "summary_word_count": output.summary_word_count,
                    "compression_percentage": output.compression_percentage,
                    "processing_time_seconds": output.processing_time_seconds,
                    "execution_device": output.device_used
                },
                "data": {
                    "original_text": output.original_text,
                    "summary_text": output.summary_text
                }
            }), 200

        except Exception as exc:
            logger.exception("REST API execution failure.")
            return jsonify({"error": f"Server AI processing breakdown: {str(exc)}"}), 500


def register_error_handlers(app: Flask) -> None:
    """Assigns graceful custom error presentation page renderers for HTTP fault codes.

    Args:
        app (Flask): Target Flask web application instance.
    """

    @app.errorhandler(404)
    def page_not_found(error: Any) -> Tuple[Any, int]:
        """Handles HTTP 404 resource not found errors."""
        flash("The requested web page resource could not be located. Redirecting to primary terminal.", "info")
        return redirect(url_for("index")), 404

    @app.errorhandler(500)
    def internal_server_error(error: Any) -> Tuple[Any, int]:
        """Handles HTTP 500 unhandled internal application crashes."""
        logger.error(f"Internal HTTP 500 error triggered: {str(error)}")
        flash("An internal server anomaly occurred during processing. Please retry your request.", "danger")
        return redirect(url_for("index")), 500


# Application entrypoint execution instance
app = create_app("development")

if __name__ == "__main__":
    logger.info("Starting local development HTTP web server on port 5000...")
    # Run server listening on standard loopback interface and accessible port
    app.run(host="0.0.0.0", port=5000, debug=True)
