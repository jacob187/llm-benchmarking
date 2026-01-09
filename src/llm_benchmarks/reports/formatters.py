"""Report formatters for different output formats."""

import json
from abc import ABC, abstractmethod
from pathlib import Path

from .models import Report


class ReportFormatter(ABC):
    """Base class for report formatters (Strategy pattern)."""

    @abstractmethod
    def format(self, report: Report) -> str:
        """
        Format a report to a string.

        Args:
            report: Report to format

        Returns:
            Formatted string
        """
        pass

    @abstractmethod
    def get_extension(self) -> str:
        """Get file extension for this format."""
        pass

    def save(self, report: Report, filename: str | None = None, directory: str | None = None) -> Path:
        """
        Save formatted report to file.

        Args:
            report: Report to save
            filename: Filename (auto-generated if None)
            directory: Directory to save in (default: data/processed/)

        Returns:
            Path to saved file
        """
        # Default directory
        if directory is None:
            directory = "data/processed"

        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename if not provided
        if filename is None:
            timestamp = report.generated_at.strftime("%Y%m%d_%H%M%S")
            ext = self.get_extension()
            filename = f"benchmark_report_{timestamp}{ext}"

        output_path = output_dir / filename

        # Format and write
        formatted = self.format(report)
        with open(output_path, "w") as f:
            f.write(formatted)

        return output_path


class MarkdownFormatter(ReportFormatter):
    """Formats reports as Markdown."""

    def format(self, report: Report) -> str:
        """Format report as Markdown."""
        lines = []

        # Title
        lines.append(f"# {report.title}\n")

        # Metadata
        if report.metadata:
            # Filter out detailed_models from display metadata
            display_metadata = {
                k: v for k, v in report.metadata.items()
                if k != "detailed_models"
            }

            if display_metadata:
                lines.append("---\n")
                for key, value in display_metadata.items():
                    # Format key nicely
                    formatted_key = key.replace("_", " ").title()
                    if isinstance(value, list):
                        value = ", ".join(value)
                    lines.append(f"**{formatted_key}:** {value}")
                lines.append("\n---\n")

        # Generation timestamp
        timestamp_str = report.generated_at.strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"*Generated: {timestamp_str}*\n")

        # Sections and tables
        section_idx = 0
        table_idx = 0

        for section in report.sections:
            # Add section heading and content
            heading = "#" * section.level
            lines.append(f"{heading} {section.title}\n")
            if section.content:
                lines.append(f"{section.content}\n")

            # If this is the "Detailed Benchmark Data" section, add expandable details
            if section.title == "Detailed Benchmark Data" and "detailed_models" in report.metadata:
                detailed_html = self._create_detailed_models_html(report.metadata["detailed_models"])
                lines.append(detailed_html)

            # Check if there's a table that follows this section
            if table_idx < len(report.tables):
                lines.append(report.tables[table_idx].to_markdown())
                table_idx += 1

            section_idx += 1

        # Footer
        lines.append(
            "\n---\n\n"
            "*This report was generated automatically by LLM Benchmark Tracker.*\n"
        )

        return "\n".join(lines)

    def _create_detailed_models_html(self, models_dict: dict) -> str:
        """Create expandable HTML details section for models."""
        lines = [
            "\n<details>",
            "<summary>Click to expand detailed benchmark data for all models</summary>\n",
        ]

        # Sort models alphabetically
        for model_name in sorted(models_dict.keys()):
            model_data = models_dict[model_name]

            lines.append(f"### {model_data['name']}\n")

            if model_data.get('average_score'):
                lines.append(f"**Average Score:** {model_data['average_score']:.2f}\n")

            lines.append("**Benchmarks:**")
            for bench_name, score in sorted(model_data['benchmarks'].items()):
                lines.append(f"- {bench_name}: {score:.2f}")

            lines.append(f"\n**Sources:** {', '.join(sorted(model_data['sources']))}\n")
            lines.append("---\n")

        lines.append("</details>\n")

        return "\n".join(lines)

    def get_extension(self) -> str:
        """Get file extension."""
        return ".md"


class JSONFormatter(ReportFormatter):
    """Formats reports as JSON."""

    def __init__(self, indent: int = 2):
        """
        Initialize JSON formatter.

        Args:
            indent: JSON indentation level
        """
        self.indent = indent

    def format(self, report: Report) -> str:
        """Format report as JSON."""
        return report.to_json(indent=self.indent)

    def get_extension(self) -> str:
        """Get file extension."""
        return ".json"


class HTMLFormatter(ReportFormatter):
    """Formats reports as HTML."""

    def format(self, report: Report) -> str:
        """Format report as HTML."""
        lines = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{report.title}</title>",
            "<style>",
            self._get_css(),
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{report.title}</h1>",
        ]

        # Metadata
        if report.metadata:
            display_metadata = {
                k: v for k, v in report.metadata.items()
                if k != "detailed_models"
            }

            if display_metadata:
                lines.append('<div class="metadata">')
                for key, value in display_metadata.items():
                    formatted_key = key.replace("_", " ").title()
                    if isinstance(value, list):
                        value = ", ".join(value)
                    lines.append(f"<p><strong>{formatted_key}:</strong> {value}</p>")
                lines.append("</div>")

        # Timestamp
        timestamp_str = report.generated_at.strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f'<p class="timestamp">Generated: {timestamp_str}</p>')

        # Sections and tables
        table_idx = 0
        for section in report.sections:
            lines.append(f"<h{section.level}>{section.title}</h{section.level}>")
            if section.content:
                # Convert markdown-style content to HTML paragraphs
                paragraphs = section.content.split("\n\n")
                for para in paragraphs:
                    if para.strip():
                        lines.append(f"<p>{para}</p>")

            # Add table if one follows
            if table_idx < len(report.tables):
                lines.append(self._table_to_html(report.tables[table_idx]))
                table_idx += 1

        lines.append("</body>")
        lines.append("</html>")

        return "\n".join(lines)

    def _table_to_html(self, table) -> str:
        """Convert TableData to HTML table."""
        lines = ['<table>']

        if table.caption:
            lines.append(f'<caption>{table.caption}</caption>')

        # Header
        lines.append('<thead><tr>')
        for header in table.headers:
            lines.append(f'<th>{header}</th>')
        lines.append('</tr></thead>')

        # Body
        lines.append('<tbody>')
        for row in table.rows:
            lines.append('<tr>')
            for cell in row:
                lines.append(f'<td>{cell}</td>')
            lines.append('</tr>')
        lines.append('</tbody>')

        lines.append('</table>')
        return "\n".join(lines)

    def _get_css(self) -> str:
        """Get CSS styles for HTML report."""
        return """
            body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
            h1 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .metadata { background-color: #f9f9f9; padding: 15px; margin: 20px 0; border-left: 4px solid #4CAF50; }
            .timestamp { color: #666; font-style: italic; }
            caption { caption-side: top; font-weight: bold; padding: 10px; }
        """

    def get_extension(self) -> str:
        """Get file extension."""
        return ".html"
