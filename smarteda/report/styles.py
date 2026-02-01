"""Estilos e formatação para relatórios."""

from typing import Dict, Any
from enum import Enum


class AlertLevel(Enum):
    """Níveis de alerta."""
    INFO = "info"
    WARNING = "warning"
    SUCCESS = "success"
    ERROR = "error"


class Styles:
    """Estilos para relatórios Markdown."""

    # Badges (HTML)
    BADGES = {
        "numeric": '<span style="background-color:#3498db;color:white;padding:2px 8px;border-radius:4px;font-size:12px;">Numérico</span>',
        "categorical": '<span style="background-color:#9b59b6;color:white;padding:2px 8px;border-radius:4px;font-size:12px;">Categórico</span>',
        "temporal": '<span style="background-color:#e67e22;color:white;padding:2px 8px;border-radius:4px;font-size:12px;">Temporal</span>',
        "binary": '<span style="background-color:#1abc9c;color:white;padding:2px 8px;border-radius:4px;font-size:12px;">Binário</span>',
        "id": '<span style="background-color:#95a5a6;color:white;padding:2px 8px;border-radius:4px;font-size:12px;">ID</span>',
        "text": '<span style="background-color:#34495e;color:white;padding:2px 8px;border-radius:4px;font-size:12px;">Texto</span>',
    }

    # Ícones (Unicode)
    ICONS = {
        "check": "✓",
        "cross": "✗",
        "warning": "⚠",
        "info": "ℹ",
        "chart": "📊",
        "table": "📋",
        "calendar": "📅",
        "number": "🔢",
        "category": "🏷️",
        "target": "🎯",
        "correlation": "🔗",
        "importance": "⭐",
        "missing": "❓",
        "outlier": "⚡",
        "trend": "📈",
    }

    # Alertas
    ALERT_STYLES = {
        AlertLevel.INFO: {
            "prefix": "> ℹ️ **Info:**",
            "html_class": "alert-info",
            "color": "#3498db"
        },
        AlertLevel.WARNING: {
            "prefix": "> ⚠️ **Atenção:**",
            "html_class": "alert-warning",
            "color": "#f39c12"
        },
        AlertLevel.SUCCESS: {
            "prefix": "> ✅ **Sucesso:**",
            "html_class": "alert-success",
            "color": "#27ae60"
        },
        AlertLevel.ERROR: {
            "prefix": "> ❌ **Erro:**",
            "html_class": "alert-error",
            "color": "#e74c3c"
        }
    }

    @staticmethod
    def badge(badge_type: str) -> str:
        """Retorna badge HTML."""
        return Styles.BADGES.get(badge_type, "")

    @staticmethod
    def badge_plain(badge_type: str) -> str:
        """Retorna badge em texto puro."""
        return f"[{badge_type.upper()}]"

    @staticmethod
    def icon(icon_type: str) -> str:
        """Retorna ícone."""
        return Styles.ICONS.get(icon_type, "")

    @staticmethod
    def alert(message: str, level: AlertLevel = AlertLevel.INFO) -> str:
        """Formata mensagem de alerta."""
        style = Styles.ALERT_STYLES[level]
        return f"{style['prefix']} {message}"

    @staticmethod
    def alert_html(message: str, level: AlertLevel = AlertLevel.INFO) -> str:
        """Formata alerta com HTML."""
        style = Styles.ALERT_STYLES[level]
        return f'<div style="background-color:{style["color"]}22;border-left:4px solid {style["color"]};padding:10px;margin:10px 0;">{message}</div>'

    @staticmethod
    def progress_bar(value: float, max_value: float = 1.0, width: int = 20) -> str:
        """Cria barra de progresso em texto."""
        if max_value == 0:
            percentage = 0
        else:
            percentage = value / max_value
        filled = int(percentage * width)
        empty = width - filled
        return f"[{'█' * filled}{'░' * empty}] {percentage * 100:.1f}%"

    @staticmethod
    def format_table(df, max_rows: int = 20) -> str:
        """Formata DataFrame como tabela Markdown."""
        if len(df) == 0:
            return "*Sem dados*"

        df_display = df.head(max_rows)

        # Cabeçalho
        headers = " | ".join(str(col) for col in df_display.columns)
        separator = " | ".join("---" for _ in df_display.columns)

        # Linhas
        rows = []
        for _, row in df_display.iterrows():
            cells = []
            for val in row:
                if isinstance(val, float):
                    if abs(val) < 0.01 and val != 0:
                        cells.append(f"{val:.2e}")
                    else:
                        cells.append(f"{val:.4f}")
                else:
                    cells.append(str(val))
            rows.append(" | ".join(cells))

        table = f"| {headers} |\n| {separator} |\n"
        table += "\n".join(f"| {row} |" for row in rows)

        if len(df) > max_rows:
            table += f"\n\n*... e mais {len(df) - max_rows} linhas*"

        return table

    @staticmethod
    def collapsible(title: str, content: str) -> str:
        """Cria seção colapsável (HTML)."""
        return f"""
<details>
<summary><strong>{title}</strong></summary>

{content}

</details>
"""

    @staticmethod
    def metric_card(title: str, value: str, subtitle: str = "") -> str:
        """Cria card de métrica (HTML)."""
        subtitle_html = f'<div style="font-size:12px;color:#666;">{subtitle}</div>' if subtitle else ""
        return f"""
<div style="display:inline-block;background:#f8f9fa;border-radius:8px;padding:15px;margin:5px;min-width:150px;text-align:center;">
    <div style="font-size:24px;font-weight:bold;color:#2c3e50;">{value}</div>
    <div style="font-size:14px;color:#7f8c8d;">{title}</div>
    {subtitle_html}
</div>
"""
