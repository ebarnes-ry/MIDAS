import pytest
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open

from src.models.services.marker import MarkerService
from src.models.providers.base import ModelError


class TestMarkerService:
    """Test suite for MarkerService functionality"""
    
    @pytest.fixture
    def mock_marker_imports(self):
        """Mock all Marker library imports"""
        with patch('src.models.services.marker.create_model_dict') as mock_create_dict, \
             patch('src.models.services.marker.ConfigParser') as mock_config_parser, \
             patch('src.models.services.marker.PdfConverter') as mock_converter_class:
            
            # Setup mocks
            mock_artifact_dict = {"model1": "artifact1", "model2": "artifact2"}
            mock_create_dict.return_value = mock_artifact_dict
            
            mock_parser = Mock()
            mock_config_parser.return_value = mock_parser
            mock_parser.generate_config_dict.return_value = {"config": "dict"}
            mock_parser.get_processors.return_value = ["processor1", "processor2"]
            mock_parser.get_renderer.return_value = "renderer"
            mock_parser.get_llm_service.return_value = "llm_service"
            
            mock_converter = Mock()
            mock_converter_class.return_value = mock_converter
            
            yield {
                'create_dict': mock_create_dict,
                'config_parser': mock_config_parser,
                'converter_class': mock_converter_class,
                'converter': mock_converter,
                'parser': mock_parser,
                'artifact_dict': mock_artifact_dict
            }
    
    def test_initialization_basic(self, mock_marker_imports):
        """
        Test: Basic MarkerService initialization without LLM
        How: Create service with minimal settings
        Ensures: Service can be initialized for basic document conversion
        """
        settings = {"use_llm": False, "output_format": "json"}
        
        service = MarkerService(**settings)
        
        # Verify settings stored
        assert service.settings == settings
        
        # Verify Marker components initialized
        mock_marker_imports['create_dict'].assert_called_once()
        mock_marker_imports['config_parser'].assert_called_once()
        mock_marker_imports['converter_class'].assert_called_once()
        
        # Verify converter was initialized with correct parameters
        converter_call = mock_marker_imports['converter_class'].call_args
        assert converter_call[1]['config'] == {"config": "dict"}
        assert converter_call[1]['artifact_dict'] == mock_marker_imports['artifact_dict']
        assert converter_call[1]['processor_list'] == ["processor1", "processor2"]
        assert converter_call[1]['renderer'] == "renderer"
        assert converter_call[1]['llm_service'] == "llm_service"
    
    def test_initialization_with_gemini_llm(self, mock_marker_imports):
        """
        Test: MarkerService initialization with Gemini LLM
        How: Create service with Gemini configuration
        Ensures: Service can be configured to use Gemini for enhanced processing
        """
        settings = {
            "use_llm": True,
            "llm_service": "gemini",
            "output_format": "json",
            "gemini": {"api_key": "test-gemini-key"}
        }
        
        service = MarkerService(**settings)
        
        # Verify CLI config was built correctly
        cli_config = service._build_cli_config()
        assert cli_config["llm_service"] == "marker.services.gemini.GoogleGeminiService"
        assert cli_config["gemini_api_key"] == "test-gemini-key"
        assert cli_config["output_format"] == "json"
    
    def test_initialization_with_ollama_llm(self, mock_marker_imports):
        """
        Test: MarkerService initialization with Ollama LLM
        How: Create service with Ollama configuration
        Ensures: Service can be configured to use local Ollama for processing
        """
        settings = {
            "use_llm": True,
            "llm_service": "ollama",
            "output_format": "json",
            "ollama": {
                "base_url": "http://localhost:11434",
                "model": "llama3.2-vision"
            }
        }
        
        service = MarkerService(**settings)
        
        cli_config = service._build_cli_config()
        assert cli_config["llm_service"] == "marker.services.ollama.OllamaService"
        assert cli_config["ollama_base_url"] == "http://localhost:11434"
        assert cli_config["ollama_model"] == "llama3.2-vision"
    
    def test_initialization_with_openai_llm(self, mock_marker_imports):
        """
        Test: MarkerService initialization with OpenAI LLM
        How: Create service with OpenAI configuration
        Ensures: Service can be configured to use OpenAI for processing
        """
        settings = {
            "use_llm": True,
            "llm_service": "openai",
            "output_format": "json",
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
                "api_key": "test-openai-key"
            }
        }
        
        service = MarkerService(**settings)
        
        cli_config = service._build_cli_config()
        assert cli_config["llm_service"] == "src.models.services.marker_openai.TruststoreOpenAIService"
        assert cli_config["openai_base_url"] == "https://api.openai.com/v1"
        assert cli_config["openai_model"] == "gpt-4o-mini"
        assert cli_config["openai_api_key"] == "test-openai-key"
    
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "env-gemini-key"})
    def test_environment_variable_fallback(self, mock_marker_imports):
        """
        Test: Environment variable fallback for API keys
        How: Create service without explicit API key, verify env var is used
        Ensures: Service can use environment variables as fallback for API keys
        """
        settings = {
            "use_llm": True,
            "llm_service": "gemini",
            "gemini": {}  # No explicit API key
        }
        
        service = MarkerService(**settings)
        
        cli_config = service._build_cli_config()
        assert cli_config["gemini_api_key"] == "env-gemini-key"
    
    def test_default_gemini_service(self, mock_marker_imports):
        """
        Test: Default LLM service is Gemini when use_llm=True
        How: Create service with use_llm=True but no explicit service
        Ensures: Service defaults to Gemini as recommended in Marker docs
        """
        settings = {
            "use_llm": True,
            "gemini": {"api_key": "test-key"}
        }
        
        service = MarkerService(**settings)
        
        cli_config = service._build_cli_config()
        assert cli_config["llm_service"] == "marker.services.gemini.GoogleGeminiService"
    
    def test_no_llm_configuration(self, mock_marker_imports):
        """
        Test: Configuration without LLM service
        How: Create service with use_llm=False
        Ensures: Service can operate without LLM for basic document processing
        """
        settings = {"use_llm": False, "output_format": "markdown"}
        
        service = MarkerService(**settings)
        
        cli_config = service._build_cli_config()
        assert "llm_service" not in cli_config
        assert cli_config["output_format"] == "markdown"
    
    def test_convert_document_success(self, mock_marker_imports):
        """
        Test: Successful document conversion
        How: Mock successful converter call
        Ensures: Service can convert documents and return results
        """
        mock_converter = mock_marker_imports['converter']
        expected_result = {"pages": [{"blocks": []}], "metadata": {}}
        mock_converter.return_value = expected_result
        
        settings = {"use_llm": False}
        service = MarkerService(**settings)
        
        result = service.convert_document("test.pdf")
        
        # Verify converter was called with correct file path
        mock_converter.assert_called_once_with("test.pdf")
        assert result == expected_result
    
    def test_convert_document_failure(self, mock_marker_imports):
        """
        Test: Document conversion failure handling
        How: Mock converter failure and verify error handling
        Ensures: Service handles conversion failures gracefully
        """
        mock_converter = mock_marker_imports['converter']
        mock_converter.side_effect = Exception("File not found")
        
        settings = {"use_llm": False}
        service = MarkerService(**settings)
        
        with pytest.raises(Exception) as exc_info:
            service.convert_document("nonexistent.pdf")
        
        assert "File not found" in str(exc_info.value)
    
    def test_convert_document_with_different_formats(self, mock_marker_imports):
        """
        Test: Document conversion with different output formats
        How: Test various output format configurations
        Ensures: Service supports different output formats (JSON, markdown, etc.)
        """
        mock_converter = mock_marker_imports['converter']
        mock_converter.return_value = "converted_content"
        
        # Test JSON format
        settings_json = {"use_llm": False, "output_format": "json"}
        service_json = MarkerService(**settings_json)
        result_json = service_json.convert_document("test.pdf")
        assert result_json == "converted_content"
        
        # Test markdown format
        settings_md = {"use_llm": False, "output_format": "markdown"}
        service_md = MarkerService(**settings_md)
        result_md = service_md.convert_document("test.pdf")
        assert result_md == "converted_content"
    
    def test_cli_config_building_comprehensive(self, mock_marker_imports):
        """
        Test: Comprehensive CLI config building with all options
        How: Test config building with all possible LLM services and settings
        Ensures: Service can build correct CLI configurations for all supported LLM services
        """
        # Test Gemini with all options
        settings_gemini = {
            "use_llm": True,
            "llm_service": "gemini",
            "output_format": "json",
            "gemini": {"api_key": "gemini-key"}
        }
        service_gemini = MarkerService(**settings_gemini)
        config_gemini = service_gemini._build_cli_config()
        
        assert config_gemini["llm_service"] == "marker.services.gemini.GoogleGeminiService"
        assert config_gemini["gemini_api_key"] == "gemini-key"
        assert config_gemini["output_format"] == "json"
        
        # Test Ollama with custom settings
        settings_ollama = {
            "use_llm": True,
            "llm_service": "ollama",
            "output_format": "markdown",
            "ollama": {
                "base_url": "http://custom:11434",
                "model": "custom-model"
            }
        }
        service_ollama = MarkerService(**settings_ollama)
        config_ollama = service_ollama._build_cli_config()
        
        assert config_ollama["llm_service"] == "marker.services.ollama.OllamaService"
        assert config_ollama["ollama_base_url"] == "http://custom:11434"
        assert config_ollama["ollama_model"] == "custom-model"
        assert config_ollama["output_format"] == "markdown"
        
        # Test OpenAI with custom base URL
        settings_openai = {
            "use_llm": True,
            "llm_service": "openai",
            "output_format": "json",
            "openai": {
                "base_url": "https://custom.openai.com/v1",
                "model": "gpt-4",
                "api_key": "openai-key"
            }
        }
        service_openai = MarkerService(**settings_openai)
        config_openai = service_openai._build_cli_config()
        
        assert config_openai["llm_service"] == "src.models.services.marker_openai.TruststoreOpenAIService"
        assert config_openai["openai_base_url"] == "https://custom.openai.com/v1"
        assert config_openai["openai_model"] == "gpt-4"
        assert config_openai["openai_api_key"] == "openai-key"
    
    def test_initialization_error_handling(self):
        """
        Test: Initialization error handling when Marker imports fail
        How: Mock import failures and verify proper error handling
        Ensures: Service provides clear error messages when Marker library is unavailable
        """
        with patch('src.models.services.marker.create_model_dict', side_effect=ImportError("Marker not installed")):
            with pytest.raises(ImportError) as exc_info:
                MarkerService(use_llm=False)
            
            assert "Marker not installed" in str(exc_info.value)
    
    def test_converter_initialization_error(self, mock_marker_imports):
        """
        Test: Converter initialization error handling
        How: Mock converter initialization failure
        Ensures: Service handles converter setup failures gracefully
        """
        mock_marker_imports['converter_class'].side_effect = Exception("Converter initialization failed")
        
        with pytest.raises(Exception) as exc_info:
            MarkerService(use_llm=False)
        
        assert "Converter initialization failed" in str(exc_info.value)
    
    def test_settings_validation(self, mock_marker_imports):
        """
        Test: Settings validation and default values
        How: Test service with minimal and missing settings
        Ensures: Service uses sensible defaults and validates required settings
        """
        # Test with minimal settings
        service_minimal = MarkerService()
        assert service_minimal.settings == {}
        
        # Test with partial settings
        settings_partial = {"output_format": "json"}
        service_partial = MarkerService(**settings_partial)
        assert service_partial.settings == settings_partial
        
        # Test default output format
        config = service_partial._build_cli_config()
        assert config["output_format"] == "json"
    
    def test_multiple_document_conversion(self, mock_marker_imports):
        """
        Test: Multiple document conversions with same service instance
        How: Convert multiple documents using the same service instance
        Ensures: Service can handle multiple conversions efficiently
        """
        mock_converter = mock_marker_imports['converter']
        mock_converter.return_value = "converted_content"
        
        settings = {"use_llm": False}
        service = MarkerService(**settings)
        
        # Convert multiple documents
        result1 = service.convert_document("doc1.pdf")
        result2 = service.convert_document("doc2.pdf")
        result3 = service.convert_document("doc3.pdf")
        
        # Verify all conversions succeeded
        assert result1 == "converted_content"
        assert result2 == "converted_content"
        assert result3 == "converted_content"
        
        # Verify converter was called for each document
        assert mock_converter.call_count == 3
        assert mock_converter.call_args_list[0][0][0] == "doc1.pdf"
        assert mock_converter.call_args_list[1][0][0] == "doc2.pdf"
        assert mock_converter.call_args_list[2][0][0] == "doc3.pdf"


class TestMarkerServiceIntegration:
    """Integration tests for MarkerService with ModelManager"""
    
    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create a mock config file for testing"""
        config_content = """
providers:
  ollama_local:
    type: ollama
    settings:
      host: "http://localhost:11434"

services:
  marker:
    type: marker
    settings:
      use_llm: true
      llm_service: "gemini"
      output_format: "json"

tasks:
  vision:
    provider: ollama_local
    model: "test-model"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        return config_file
    
    @patch('src.models.services.marker.create_model_dict')
    @patch('src.models.services.marker.ConfigParser')
    @patch('src.models.services.marker.PdfConverter')
    def test_model_manager_integration(self, mock_converter_class, mock_config_parser, mock_create_dict, mock_config):
        """
        Test: MarkerService integration with ModelManager
        How: Create ModelManager with Marker service and verify access
        Ensures: ModelManager can properly initialize and provide access to MarkerService
        """
        from src.models.manager import ModelManager
        
        # Setup mocks
        mock_create_dict.return_value = {}
        mock_parser = Mock()
        mock_config_parser.return_value = mock_parser
        mock_parser.generate_config_dict.return_value = {}
        mock_parser.get_processors.return_value = []
        mock_parser.get_renderer.return_value = None
        mock_parser.get_llm_service.return_value = None
        mock_converter_class.return_value = Mock()
        
        # Create ModelManager
        manager = ModelManager(mock_config)
        
        # Access Marker service through property
        marker_service = manager.marker
        assert isinstance(marker_service, MarkerService)
        
        # Verify service was initialized with correct settings
        assert marker_service.settings["use_llm"] is True
        assert marker_service.settings["llm_service"] == "gemini"
        assert marker_service.settings["output_format"] == "json"
