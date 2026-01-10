#!/usr/bin/env python3
"""Integration test for report generation with multiple formats."""

from llm_benchmarks.pipeline import BenchmarkPipeline
from llm_benchmarks.reports import (
    ReportGenerator,
    MarkdownFormatter,
    JSONFormatter,
    HTMLFormatter
)


def main():
    print("=" * 60)
    print("REPORT GENERATION INTEGRATION TEST")
    print("=" * 60)

    # Run pipeline to get data
    print("\n1. Running pipeline...")
    pipeline = BenchmarkPipeline()
    result = pipeline.run(skip_scrape=True, analyses=["summary"], save_cache=False)

    print(f"   ✓ Loaded {len(result.models)} models")
    print(f"   ✓ Generated {len(result.analyses)} analyses")

    # Generate structured report
    print("\n2. Generating structured report...")
    generator = ReportGenerator()
    report = generator.generate(
        models=result.models,
        analyses=result.analyses,
        blog_insights=None
    )

    print(f"   ✓ Report created: {report.title}")
    print(f"   ✓ Sections: {len(report.sections)}")
    print(f"   ✓ Tables: {len(report.tables)}")

    # Test JSON export
    print("\n3. Testing JSON serialization...")
    json_data = report.to_json()
    print(f"   ✓ JSON size: {len(json_data)} chars")
    print(f"   Preview: {json_data[:150]}...")

    # Test formatters
    print("\n4. Testing formatters...")

    # Markdown
    print("\n   a) Markdown Formatter")
    md_formatter = MarkdownFormatter()
    md_output = md_formatter.format(report)
    md_path = md_formatter.save(report, "test_report.md")
    print(f"      ✓ Markdown generated ({len(md_output)} chars)")
    print(f"      ✓ Saved to: {md_path}")

    # JSON
    print("\n   b) JSON Formatter")
    json_formatter = JSONFormatter(indent=2)
    json_output = json_formatter.format(report)
    json_path = json_formatter.save(report, "test_report.json")
    print(f"      ✓ JSON generated ({len(json_output)} chars)")
    print(f"      ✓ Saved to: {json_path}")

    # HTML
    print("\n   c) HTML Formatter")
    html_formatter = HTMLFormatter()
    html_output = html_formatter.format(report)
    html_path = html_formatter.save(report, "test_report.html")
    print(f"      ✓ HTML generated ({len(html_output)} chars)")
    print(f"      ✓ Saved to: {html_path}")

    # Test Report.from_dict (round-trip)
    print("\n5. Testing round-trip serialization...")
    from llm_benchmarks.reports import Report

    report_dict = report.to_dict()
    restored_report = Report.from_dict(report_dict)

    print(f"   ✓ Original title: {report.title}")
    print(f"   ✓ Restored title: {restored_report.title}")
    print(f"   ✓ Sections match: {len(report.sections) == len(restored_report.sections)}")
    print(f"   ✓ Tables match: {len(report.tables) == len(restored_report.tables)}")

    # Show preview of each format
    print("\n6. Format previews:")

    print("\n   Markdown preview:")
    print("   " + "-" * 57)
    for line in md_output.split("\n")[:15]:
        print(f"   {line}")
    print("   ...")

    print("\n   JSON preview:")
    print("   " + "-" * 57)
    for line in json_output.split("\n")[:10]:
        print(f"   {line}")
    print("   ...")

    print("\n   HTML preview:")
    print("   " + "-" * 57)
    for line in html_output.split("\n")[:10]:
        print(f"   {line}")
    print("   ...")

    print("\n" + "=" * 60)
    print("✓ ALL TESTS COMPLETE")
    print("=" * 60)
    print(f"\nGenerated reports:")
    print(f"  - {md_path}")
    print(f"  - {json_path}")
    print(f"  - {html_path}")
    print()


if __name__ == "__main__":
    main()
