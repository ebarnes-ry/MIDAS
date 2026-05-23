import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.models.services.marker import MarkerService
from src.models.manager import ModelManager


class TestMarkerService:
    
    def test_init_no_llm(self):
        settings = {"use_llm": False, "output_format": "json"}
        provider = MarkerService(**settings)
        assert provider.settings == settings
        assert provider.converter is not None
    
    def test_cli_config_gemini_default(self):
        settings = {
            "use_llm": True,
            "gemini": {"api_key": "test-key"}
        }
        provider = MarkerService(**settings)
        cli_config = provider._build_cli_config()
        
        assert cli_config["llm_service"] == "marker.services.gemini.GoogleGeminiService"
        assert cli_config["gemini_api_key"] == "test-key"
    
    def test_cli_config_ollama(self):
        settings = {
            "use_llm": True,
            "llm_service": "ollama",
            "ollama": {
                "base_url": "http://localhost:11434",
                "model": "test-model"
            }
        }
        provider = MarkerService(**settings)
        cli_config = provider._build_cli_config()
        
        assert cli_config["llm_service"] == "marker.services.ollama.OllamaService"
        assert cli_config["ollama_base_url"] == "http://localhost:11434"
        assert cli_config["ollama_model"] == "test-model"
    
    def test_cli_config_openai(self):
        settings = {
            "use_llm": True,
            "llm_service": "openai",
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
                "api_key": "test-openai-key"
            }
        }
        provider = MarkerService(**settings)
        cli_config = provider._build_cli_config()
        
        assert cli_config["llm_service"] == "src.models.services.marker_openai.TruststoreOpenAIService"
        assert cli_config["openai_base_url"] == "https://api.openai.com/v1"
        assert cli_config["openai_model"] == "gpt-4o-mini"
        assert cli_config["openai_api_key"] == "test-openai-key"
    
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "env-gemini-key"})
    def test_env_var_fallback(self):
        settings = {"use_llm": True, "gemini": {}}
        provider = MarkerService(**settings)
        cli_config = provider._build_cli_config()
        
        assert cli_config["gemini_api_key"] == "env-gemini-key"
    
    @patch('src.models.providers.marker.PdfConverter')
    def test_convert_document(self, mock_converter_class):
        mock_converter = MagicMock()
        mock_converter_class.return_value = mock_converter
        mock_converter.return_value = "converted_result"
        
        settings = {"use_llm": False}
        provider = MarkerService(**settings)
        result = provider.convert_document("test.pdf")
        
        mock_converter.assert_called_once_with("test.pdf")
        assert result == "converted_result"


class TestMarkerManagerIntegration:
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"})
    def test_marker_provider_registration(self):
        config_path = Path("src/config/config.yaml")
        if not config_path.exists():
            pytest.skip("Config file not found")
        
        manager = ModelManager(config_path)
        
        marker_provider = manager.marker
        assert isinstance(marker_provider, MarkerService)
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"})
    def test_get_provider_for_task(self):
        config_path = Path("src/config/config.yaml")
        if not config_path.exists():
            pytest.skip("Config file not found")
        
        manager = ModelManager(config_path)
        
        provider = manager.get_provider_for_task("document_conversion")
        assert isinstance(provider, MarkerService)


@pytest.mark.integration
class TestMarkerRealCall:
    
    @pytest.fixture
    def sample_image_path(self):
        return "./benchmarks/data/samples/input_cases/one_problem/multi_choice_diagram.png"
    
    @pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")
    def test_gemini_real_call(self, sample_image_path):
        settings = {
            "use_llm": True,
            "llm_service": "gemini",
            "output_format": "json",
            "gemini": {"api_key": os.getenv("GOOGLE_API_KEY")}
        }
        
        provider = MarkerService(**settings)
        result = provider.convert_document(sample_image_path)
        
        print(f"Result type: {type(result)}")
        
        # Handle Marker's JSONOutput object
        if hasattr(result, 'children'):
            print(f"Number of children blocks: {len(result.children)}")
            print("Block structure:")
            for i, child in enumerate(result.children[:1]):  # Show first page
                print(f"Page {i}: {child.block_type}")
                if hasattr(child, 'children') and child.children:
                    for j, block in enumerate(child.children[:5]):  # Show first 5 blocks
                        print(f"  Block {j}: {block.block_type} - {block.html[:100]}")
            
            if hasattr(result, 'metadata'):
                print(f"Metadata keys: {list(result.metadata.keys())}")
                if 'page_stats' in result.metadata:
                    stats = result.metadata['page_stats'][0]
                    print(f"Text extraction method: {stats['text_extraction_method']}")
                    print(f"Block counts: {stats['block_counts']}")
        
        elif isinstance(result, str):
            print(f"Result length: {len(result)}")
            print("First 1000 characters:")
            print(result[:1000])
        elif isinstance(result, dict):
            print(f"Result keys: {list(result.keys())}")
        
        assert result is not None
    
    @pytest.mark.skipif(not os.getenv("TEST_OLLAMA"), reason="TEST_OLLAMA not set")
    def test_ollama_real_call(self, sample_image_path):
        settings = {
            "use_llm": True,
            "llm_service": "ollama",
            "output_format": "json",
            "ollama": {
                "base_url": "http://localhost:11434",
                "model": "llama3.2-vision"
            }
        }
        
        provider = MarkerService(**settings)
        result = provider.convert_document(sample_image_path)
        
        print(f"Result type: {type(result)}")
        if isinstance(result, str):
            print(f"Result length: {len(result)}")
            print("First 1000 characters:")
            print(result[:1000])
        elif isinstance(result, dict):
            print(f"Result keys: {list(result.keys())}")
            if 'children' in result:
                print(f"Number of children: {len(result['children'])}")
        
        assert result is not None
        assert isinstance(result, (str, dict))
