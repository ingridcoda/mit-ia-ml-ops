"""
quality/report_writer.py — Escritor de Relatórios Técnicos de Qualidade.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.quality.base import QualityReportWriterBase


class QualityReportWriter(QualityReportWriterBase):
    def write(self, summary: dict[str, Any], output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"credit_quality_report_{timestamp}.json"

        # Adaptado para o retorno do Great Expectations V1.0+
        report = {
            "timestamp": timestamp,
            "success": summary.get("success", False),
            "message": "Auditoria concluída. Detalhes completos disponíveis no objeto de resultados."
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        self._logger.info("Relatório de conformidade salvo: %s", report_path)
        return report_path
