"""Tests for response-parsing logic in 03_summarize_readmes.py."""

import importlib
import json
import re
import sys

# Import module
spec = importlib.util.spec_from_file_location("summarize_readmes", "03_summarize_readmes.py")
summarize_readmes = importlib.util.module_from_spec(spec)
sys.modules["summarize_readmes"] = summarize_readmes
spec.loader.exec_module(summarize_readmes)

PROJECT_TYPES = summarize_readmes.PROJECT_TYPES
TARGET_AUDIENCES = summarize_readmes.TARGET_AUDIENCES


def _parse_response(raw: str, full_name: str = "owner/repo") -> tuple[str, str, str, str, str]:
    """Replicate the parsing logic from summarize_readme (lines 126-155)."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    repo_name = full_name.split("/")[1]

    try:
        obj = json.loads(raw)
        title = obj.get("title") or repo_name
        summary = obj.get("summary") or raw
        project_type = obj.get("project_type", "Other")
        if project_type not in PROJECT_TYPES:
            project_type = "Other"
        tagline = obj.get("tagline", "")
        tagline = tagline.strip().strip("'\"")
        target_audience = obj.get("target_audience", "Developers")
        if target_audience not in TARGET_AUDIENCES:
            target_audience = "Developers"
    except (json.JSONDecodeError, AttributeError):
        title = repo_name
        summary = raw
        project_type = "Other"
        tagline = ""
        target_audience = "Developers"

    summary = re.sub(r"^#+\s+.*?\n+", "", summary).strip()

    return title, summary, project_type, tagline, target_audience


class TestValidJsonResponse:
    def test_full_valid_response(self):
        raw = json.dumps(
            {
                "title": "My Project",
                "summary": "A great project that does things.",
                "project_type": "Library",
                "tagline": "Fast data processing library",
                "target_audience": "Data & ML Engineers",
            }
        )
        title, summary, project_type, tagline, audience = _parse_response(raw)
        assert title == "My Project"
        assert summary == "A great project that does things."
        assert project_type == "Library"
        assert tagline == "Fast data processing library"
        assert audience == "Data & ML Engineers"

    def test_null_title_falls_back_to_repo_name(self):
        raw = json.dumps({"title": None, "summary": "desc", "project_type": "CLI Tool"})
        title, _, _, _, _ = _parse_response(raw, "myorg/awesome-tool")
        assert title == "awesome-tool"


class TestInvalidProjectType:
    def test_unknown_project_type_falls_back_to_other(self):
        raw = json.dumps({"title": "X", "summary": "Y", "project_type": "SomethingInvalid"})
        _, _, project_type, _, _ = _parse_response(raw)
        assert project_type == "Other"


class TestInvalidTargetAudience:
    def test_unknown_audience_falls_back_to_developers(self):
        raw = json.dumps({"title": "X", "summary": "Y", "target_audience": "Aliens"})
        _, _, _, _, audience = _parse_response(raw)
        assert audience == "Developers"


class TestMalformedJson:
    def test_invalid_json_falls_back_to_raw(self):
        raw = "This is not JSON at all"
        title, summary, project_type, tagline, audience = _parse_response(raw, "org/myrepo")
        assert title == "myrepo"
        assert summary == "This is not JSON at all"
        assert project_type == "Other"
        assert tagline == ""
        assert audience == "Developers"


class TestMarkdownFencedJson:
    def test_fenced_json_is_parsed(self):
        inner = json.dumps(
            {
                "title": "Fenced",
                "summary": "Works fine.",
                "project_type": "Framework",
                "tagline": "Web framework",
                "target_audience": "Developers",
            }
        )
        raw = f"```json\n{inner}\n```"
        title, summary, project_type, tagline, audience = _parse_response(raw)
        assert title == "Fenced"
        assert project_type == "Framework"

    def test_fenced_without_lang_tag(self):
        inner = json.dumps({"title": "NoLang", "summary": "ok"})
        raw = f"```\n{inner}\n```"
        title, _, _, _, _ = _parse_response(raw)
        assert title == "NoLang"


class TestTaglineQuoteStripping:
    def test_strips_double_quotes(self):
        raw = json.dumps({"title": "T", "summary": "S", "tagline": '"Quoted tagline"'})
        _, _, _, tagline, _ = _parse_response(raw)
        assert tagline == "Quoted tagline"

    def test_strips_single_quotes(self):
        raw = json.dumps({"title": "T", "summary": "S", "tagline": "'Single quoted'"})
        _, _, _, tagline, _ = _parse_response(raw)
        assert tagline == "Single quoted"

    def test_no_quotes_unchanged(self):
        raw = json.dumps({"title": "T", "summary": "S", "tagline": "Plain tagline"})
        _, _, _, tagline, _ = _parse_response(raw)
        assert tagline == "Plain tagline"
