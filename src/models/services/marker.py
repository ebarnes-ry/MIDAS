import logging
from pathlib import Path

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser

logger = logging.getLogger(__name__)

class MarkerService:
    def __init__(self, **settings):
        """
        Initializes the MarkerService using the robust PdfConverter class
        and pulls configuration, including API keys, from the provided settings.
        """
        self.settings = settings
        self.models = create_model_dict()

        # Build the configuration dictionary for Marker's ConfigParser
        # This is how we pass settings like the API key to the underlying library
        config = self._build_cli_config()
        config_parser = ConfigParser(config)
        
        self.converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=self.models,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service()
        )
        logger.info("MarkerService initialized, forcing JSON output and using config for LLM settings.")

    def _get_math_block_correction_prompt(self) -> str:
        return """Pay special attention to mathematical content that may be incorrectly classified as Figure, Picture, or Table blocks.

CRITICAL MATH DETECTION RULES:
1. If a block contains mathematical equations, formulas, expressions, or mathematical notation, reclassify it as:
   - "Equation" for standalone mathematical expressions
   - "TextInlineMath" for text containing embedded math
   - "Text" for text blocks that should contain inline math

2. Look for these mathematical indicators:
   - LaTeX notation (\\frac, \\int, \\sum, \\sqrt, etc.)
   - Mathematical symbols (=, +, -, ×, ÷, ±, ≠, ≤, ≥, ∞, π, etc.)
   - Matrices and vectors
   - Fractions, integrals, summations, square roots, limits, derivatives, etc.
   - Equation-like structures with variables (x, y, z, a, b, c, etc.)
   - Mathematical expressions with parentheses and operators
   - Coordinate systems, graphs with mathematical content
   - Function notation f(x), g(x), etc.

3. When reclassifying math content:
   - Preserve any existing <math> tags
   - Convert mathematical content to proper LaTeX format within <math></math> tags
   - Ensure mathematical expressions are properly formatted as LaTeX
   - For standalone equations, use block-level math formatting
   - For inline math, embed within text using inline math formatting
   - Handle matrices, vectors, and other mathematical structures with proper LaTeX formatting

4. Reformat the HTML content to properly represent the mathematical notation using LaTeX within <math> tags.

Your goal is to ensure mathematical content is correctly identified and properly formatted, without changing the underlying mathematical meaning."""

    def _build_cli_config(self):
        """Build the CLI configuration for Marker's PdfConverter."""
        cli_config = self._build_basic_config()
        
        # Configure LLM if enabled
        if self.settings.get("use_llm", False):
            self._configure_llm(cli_config)
        
        return cli_config
    
    def _build_basic_config(self):
        """Build basic configuration settings that don't depend on LLM."""
        config = {
            "output_format": self.settings.get("output_format", "json"),
            "force_ocr": self.settings.get("force_ocr", False),
            "strip_existing_ocr": self.settings.get("strip_existing_ocr", False),
            "use_llm": self.settings.get("use_llm", False),
        }
        
        # Handle image extraction settings
        if self.settings.get("disable_image_extraction", False):
            config["disable_image_extraction"] = True
            logger.info("Image extraction disabled - LLM will generate descriptions instead")
        elif "extract_images" in self.settings:
            config["extract_images"] = self.settings["extract_images"]
        else:
            config["extract_images"] = True  # Default
        
        if self.settings.get("redo_inline_math", False):
            config["redo_inline_math"] = True
        
        return config
    
    def _configure_llm(self, cli_config):
        """Configure LLM-specific settings."""
        llm_service = self.settings.get("llm_service", "gemini")
        service_config = self.settings.get(llm_service, {})
        
        # Set LLM service class
        service_classes = {
            "gemini": "marker.services.gemini.GoogleGeminiService",
            "openai": "src.models.services.marker_openai.TruststoreOpenAIService",
        }

        if llm_service not in service_classes:
            raise ValueError(
                f"Unsupported Marker LLM service '{llm_service}'. "
                f"Supported services: {', '.join(sorted(service_classes))}"
            )

        cli_config["use_llm"] = True
        cli_config["llm_service"] = service_classes[llm_service]
        
        # Set block correction prompt
        if not self.settings.get("block_correction_prompt"):
            cli_config["block_correction_prompt"] = self._get_math_block_correction_prompt()
            logger.info("Using math-focused block correction prompt")
        else:
            cli_config["block_correction_prompt"] = self.settings["block_correction_prompt"]
        
        # Configure service-specific settings
        if llm_service == "gemini":
            self._configure_gemini(cli_config, service_config)
        elif llm_service == "openai":
            self._configure_openai(cli_config, service_config)
    
    def _configure_gemini(self, cli_config, gemini_config):
        """Configure Gemini-specific settings."""
        import os
        api_key = (
            gemini_config.get("api_key")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        
        if not api_key:
            logger.error("Marker is configured to use Gemini, but GEMINI_API_KEY is not set.")
            return
        
        # Set API key
        cli_config["gemini_api_key"] = api_key
        logger.info("Gemini API key loaded successfully")
        
        # Set optional Gemini parameters
        if gemini_config.get("model"):
            cli_config["gemini_model_name"] = gemini_config["model"]
        if gemini_config.get("max_tokens"):
            cli_config["gemini_max_tokens"] = gemini_config["max_tokens"]
        if gemini_config.get("temperature") is not None:
            cli_config["gemini_temperature"] = gemini_config["temperature"]

    def _configure_openai(self, cli_config, openai_config):
        """Configure Marker OpenAI service settings."""
        import os
        api_key = (
            openai_config.get("api_key")
            or os.environ.get("OPENAI_API_KEY")
        )

        if not api_key:
            logger.error("Marker is configured to use OpenAI, but OPENAI_API_KEY is not set.")
            return

        cli_config["openai_api_key"] = api_key
        logger.info("OpenAI API key loaded successfully for Marker")

        if openai_config.get("model"):
            cli_config["openai_model"] = openai_config["model"]
        if openai_config.get("base_url"):
            cli_config["openai_base_url"] = openai_config["base_url"]
        if openai_config.get("image_format"):
            cli_config["openai_image_format"] = openai_config["image_format"]
        if openai_config.get("max_output_tokens"):
            cli_config["max_output_tokens"] = openai_config["max_output_tokens"]
        if openai_config.get("timeout"):
            cli_config["timeout"] = openai_config["timeout"]
        if openai_config.get("max_retries") is not None:
            cli_config["max_retries"] = openai_config["max_retries"]
        if openai_config.get("retry_wait_time") is not None:
            cli_config["retry_wait_time"] = openai_config["retry_wait_time"]
    
    def convert_document(self, file_path: str):
        """
        Converts a single document using the configured PdfConverter.
        """
        logger.info(f"Converting document: {file_path}")
        
        rendered_output = self.converter(file_path)
        
        if not hasattr(rendered_output, 'children'):
             raise TypeError(f"Marker returned a {type(rendered_output).__name__} object instead of JsonOutput. The library's API may have changed or the config is wrong.")
        
        return rendered_output
