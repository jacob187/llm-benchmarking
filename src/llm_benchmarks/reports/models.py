"""Data models for report generation."""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any


@dataclass
class ReportSection:
    """A section in a report."""

    title: str
    content: str
    level: int = 2  # Heading level (1-6)

    def to_markdown(self) -> str:
        """
        Convert section to markdown.

        Returns:
            Markdown string
        """
        heading = "#" * self.level
        return f"{heading} {self.title}\n\n{self.content}\n"


@dataclass
class TableData:
    """Table data for reports."""

    headers: list[str]
    rows: list[list[str]]
    caption: str | None = None

    def to_markdown(self) -> str:
        """
        Convert table to markdown format.

        Returns:
            Markdown table string
        """
        if not self.headers or not self.rows:
            return ""

        lines = []

        # Add caption if provided
        if self.caption:
            lines.append(f"**{self.caption}**\n")

        # Header row
        lines.append("| " + " | ".join(self.headers) + " |")

        # Separator row
        lines.append("| " + " | ".join(["---"] * len(self.headers)) + " |")

        # Data rows
        for row in self.rows:
            # Ensure row has same number of columns as headers
            padded_row = row + [""] * (len(self.headers) - len(row))
            lines.append("| " + " | ".join(padded_row[:len(self.headers)]) + " |")

        return "\n".join(lines) + "\n"


@dataclass
class Report:
    """Complete benchmark report."""

    title: str
    sections: list[ReportSection] = field(default_factory=list)
    tables: list[TableData] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)

    def add_section(self, title: str, content: str, level: int = 2) -> "Report":
        """
        Add a section to the report.

        Args:
            title: Section title
            content: Section content
            level: Heading level (1-6)

        Returns:
            Self for chaining
        """
        self.sections.append(ReportSection(title, content, level))
        return self

    def add_table(self, headers: list[str], rows: list[list[str]], caption: str | None = None) -> "Report":
        """
        Add a table to the report.

        Args:
            headers: Table headers
            rows: Table rows
            caption: Optional table caption

        Returns:
            Self for chaining
        """
        self.tables.append(TableData(headers, rows, caption))
        return self

    def to_markdown(self) -> str:
        """
        Convert entire report to markdown.

        Returns:
            Complete markdown report
        """
        lines = []

        # Title
        lines.append(f"# {self.title}\n")

        # Metadata
        if self.metadata:
            lines.append("---\n")
            for key, value in self.metadata.items():
                lines.append(f"**{key}:** {value}")
            lines.append("\n---\n")

        # Generation timestamp
        timestamp_str = self.generated_at.strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"*Generated: {timestamp_str}*\n")

        # Sections
        for section in self.sections:
            lines.append(section.to_markdown())

        # Tables
        for table in self.tables:
            lines.append(table.to_markdown())

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert report to dictionary for JSON serialization.

        Returns:
            Dictionary representation
        """
        return {
            "title": self.title,
            "generated_at": self.generated_at.isoformat(),
            "metadata": self.metadata,
            "sections": [
                {
                    "title": s.title,
                    "content": s.content,
                    "level": s.level
                }
                for s in self.sections
            ],
            "tables": [
                {
                    "headers": t.headers,
                    "rows": t.rows,
                    "caption": t.caption
                }
                for t in self.tables
            ]
        }

    def to_json(self, indent: int = 2) -> str:
        """
        Convert report to JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Report":
        """
        Create report from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Report instance
        """
        report = cls(
            title=data["title"],
            generated_at=datetime.fromisoformat(data["generated_at"])
        )
        report.metadata = data.get("metadata", {})

        for section_data in data.get("sections", []):
            report.add_section(
                section_data["title"],
                section_data["content"],
                section_data.get("level", 2)
            )

        for table_data in data.get("tables", []):
            report.add_table(
                table_data["headers"],
                table_data["rows"],
                table_data.get("caption")
            )

        return report

    def __str__(self) -> str:
        """String representation returns markdown."""
        return self.to_markdown()
