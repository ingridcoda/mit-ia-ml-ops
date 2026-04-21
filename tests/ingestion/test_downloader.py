"""
test_downloader.py — Testes unitários aprimorados para o KaggleDownloader.
"""
import os
from unittest.mock import patch

import pytest

from src.ingestion.downloader import KaggleDownloader


@pytest.fixture
def downloader_fake(null_logger):
    """Instancia o downloader com credenciais mockadas para teste."""
    # Adicionamos o patch do load_dotenv aqui também para evitar leitura do disco real
    with patch("src.ingestion.downloader.load_dotenv"), \
            patch.dict(os.environ, {"KAGGLE_USERNAME": "test_user", "KAGGLE_KEY": "test_key"}):
        return KaggleDownloader(
            dataset="uciml/german-credit",
            logger=null_logger,
            skip_if_exists=True
        )


class TestKaggleDownloader:
    # AJUSTE: Mockamos o load_dotenv para ele não ler o seu .env real e "sujar" o teste
    @patch("src.ingestion.downloader.load_dotenv")
    def test_autenticacao_erro_variavel_faltante(self, mock_load_dotenv, null_logger):
        """Verifica se levanta erro caso falte o usuário ou a chave no ambiente."""
        # clear=True garante que o ambiente está limpo, e o mock impede o load_dotenv de restaurá-lo
        with patch.dict(os.environ, {"KAGGLE_KEY": "some_key"}, clear=True):
            with pytest.raises(EnvironmentError, match="Configure as credenciais"):
                KaggleDownloader("any/dataset", null_logger)

    @patch("kaggle.api.kaggle_api_extended.KaggleApi")
    def test_load_executa_download_se_pasta_vazia(self, mock_api_class, downloader_fake, tmp_path):
        """Garante que a API do Kaggle é acionada quando não há dados locais."""
        mock_api = mock_api_class.return_value
        dest_dir = tmp_path / "raw"
        dest_dir.mkdir()

        def mock_download_effect(*args, **kwargs):
            (dest_dir / "credit.csv").touch()

        mock_api.dataset_download_files.side_effect = mock_download_effect

        result = downloader_fake.load(dest_dir)

        mock_api.authenticate.assert_called_once()
        mock_api.dataset_download_files.assert_called_once()
        assert len(result) == 1

    def test_skip_download_se_arquivos_existem(self, downloader_fake, tmp_path):
        """Valida a idempotência: se o arquivo existe, pula a chamada de API."""
        dest_dir = tmp_path / "raw"
        dest_dir.mkdir()
        (dest_dir / "credit.csv").touch()

        with patch("kaggle.api.kaggle_api_extended.KaggleApi") as mock_api:
            downloader_fake.load(dest_dir)
            mock_api.return_value.authenticate.assert_not_called()

    @patch("kaggle.api.kaggle_api_extended.KaggleApi")
    def test_force_download_ignora_arquivos(self, mock_api_class, null_logger, tmp_path):
        """Garante que force=True obriga o download mesmo com arquivos presentes."""
        dest_dir = tmp_path / "raw"
        dest_dir.mkdir()
        (dest_dir / "credit.csv").touch()

        with patch("src.ingestion.downloader.load_dotenv"), \
                patch.dict(os.environ, {"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"}):
            downloader = KaggleDownloader("any/dataset", null_logger, force=True)
            downloader.load(dest_dir)

            mock_api_class.return_value.authenticate.assert_called_once()
