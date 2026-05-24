from pathlib import Path
import json
import re


ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "docs" / "wiki" / "experiment-forms.html"
FORMS_MD = ROOT / "docs" / "wiki" / "experiment-forms.md"


def _html() -> str:
    return HTML.read_text(encoding="utf-8")


def _function_body(html: str, name: str) -> str:
    match = re.search(rf"function {name}\([^)]*\) \{{(?P<body>.*?)\n    \}}", html, re.S)
    assert match is not None
    return match.group("body")


def _first_ps_command_lines(function_body: str) -> list[str]:
    start = function_body.index("psCommand([") + len("psCommand([")
    depth = 1
    pos = start
    while depth:
        ch = function_body[pos]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        pos += 1
    array_source = function_body[start : pos - 1]
    tokens = re.findall(r'(`(?:\\.|[^`])*`|"(?:\\.|[^"])*")', array_source)
    lines = []
    for token in tokens:
        if token.startswith('"'):
            lines.append(json.loads(token))
        else:
            lines.append(token[1:-1])
    return lines


def _render_lines(lines: list[str], data: dict[str, str]) -> str:
    rendered = []
    for line in lines:
        for key, value in data.items():
            line = line.replace(f"${{data.{key}}}", value)
            line = line.replace(f"${{psQuote(data.{key})}}", f'"{value}"')
        rendered.append(line)
    return "\n".join(
        line if idx == len(rendered) - 1 else f"{line} `"
        for idx, line in enumerate(rendered)
    )


def test_experiment_form_exposes_generated_command_blocks() -> None:
    html = _html()

    assert 'href="#generated-commands"' in html
    assert 'id="generated-commands"' in html
    assert 'id="copyAllCommands"' in html

    for command_id in (
        "commandWithTutor",
        "commandWithoutTutor",
        "commandExport",
        "commandAnalyze",
    ):
        assert f'id="{command_id}"' in html
        assert "readonly" in html


def test_experiment_form_generates_condition_specific_command_args() -> None:
    html = _html()

    for expected in (
        "--model-provider openai_compat",
        "--model-base-url",
        "--model-name",
        "--vision-model-name",
        "--model-enable-multimodal",
        "--global-help-hotkey",
        "--max-overlay-targets 4",
        "--model-provider stub",
        "--stub-mode correct",
        "--no-model-enable-multimodal",
        "--help-udp-port 0",
        "No LLM/VLM/overlay/help is expected",
        'logs\\\\${participant}_${trial}_${condition}.jsonl',
        'questionnaires\\\\${participant}_${trial}.json',
        'recordings\\\\${participant}_${trial}.mp4',
        'artifacts\\\\experiments\\\\${study}',
        'artifacts\\\\analysis\\\\${study}',
    ):
        assert expected in html


def test_experiment_form_copy_buttons_have_clipboard_fallback() -> None:
    html = _html()

    assert "navigator.clipboard.writeText" in html
    assert ".select()" in html
    assert "document.execCommand(\"copy\")" in html
    assert "copyCommandText" in html
    assert 'id="commandAllFallback"' in html


def test_experiment_form_keeps_passive_command_free_of_tutor_setup() -> None:
    html = _html()
    body = _function_body(html, "buildWithoutTutorCommand")

    assert "--help-udp-port 0" in body
    assert "--model-provider stub" in body
    assert "--stub-mode correct" in body
    assert "--no-model-enable-multimodal" in body
    assert "--global-help-hotkey" not in body
    assert "--model-base-url" not in body
    assert "vision-saved-games-dir" not in body


def test_experiment_form_renders_p02_without_tutor_acceptance_commands() -> None:
    html = _html()
    data = {
        "study": "fa18c_quest3_simitutor_2026",
        "participant": "P02",
        "trial": "T01",
        "condition": "without_tutor",
        "sessionId": "sess-P02-T01",
        "rawLogPath": r"logs\P02_T01_without_tutor.jsonl",
        "questionnairePath": r"questionnaires\P02_T01.json",
        "recordingPath": r"recordings\P02_T01.mp4",
        "exportOutputDir": r"artifacts\experiments\fa18c_quest3_simitutor_2026",
        "analysisOutputDir": r"artifacts\analysis\fa18c_quest3_simitutor_2026",
        "group": "novice",
        "experimenter": "E01",
        "commandLanguage": "zh",
        "dcsAircraft": "FA-18C_hornet",
        "dcsMission": "fa18c_cold_start_fixed.miz",
        "vrSetup": "Quest 3 Link",
        "monitorSetup": "fa18c_composite_panel_v2",
    }

    passive = _render_lines(
        _first_ps_command_lines(_function_body(html, "buildWithoutTutorCommand")),
        data,
    )
    assert '--session-id "sess-P02-T01"' in passive
    assert "--help-udp-port 0" in passive
    assert "--model-provider stub" in passive
    assert "--stub-mode correct" in passive
    assert "--no-model-enable-multimodal" in passive
    assert '--output "logs\\P02_T01_without_tutor.jsonl"' in passive

    export = _render_lines(
        _first_ps_command_lines(_function_body(html, "buildExportCommand")),
        data | {"provider": "stub", "textModel": "not_used", "visionModel": "not_used"},
    )
    assert 'experiment-export "logs\\P02_T01_without_tutor.jsonl"' in export
    assert '--participant-id "P02"' in export
    assert '--condition "without_tutor"' in export
    assert '--questionnaire "questionnaires\\P02_T01.json"' in export
    assert '--recording-ref "recordings\\P02_T01.mp4"' in export
    assert '--output-dir "artifacts\\experiments\\fa18c_quest3_simitutor_2026"' in export
    assert "--strict" in export

    analyze = _render_lines(
        _first_ps_command_lines(_function_body(html, "buildAnalyzeCommand")),
        data,
    )
    assert 'experiment-analyze "artifacts\\experiments\\fa18c_quest3_simitutor_2026"' in analyze
    assert '--output-dir "artifacts\\analysis\\fa18c_quest3_simitutor_2026"' in analyze


def test_experiment_form_maps_mixed_language_to_cli_supported_zh() -> None:
    html = _html()

    assert 'const commandLanguage = formLanguage === "en" ? "en" : "zh";' in html
    assert "Tutor message language=mixed maps to --lang zh" in html
    assert "--lang ${data.commandLanguage}" in html


def test_experiment_form_load_regenerates_commands() -> None:
    html = _html()

    assert "function updateGeneratedCommands()" in html
    assert "updateGeneratedCommands();" in html
    assert "form.addEventListener(\"input\", updateGeneratedCommands)" in html
    assert "form.addEventListener(\"change\", updateGeneratedCommands)" in html


def test_experiment_forms_markdown_mentions_generated_commands() -> None:
    text = FORMS_MD.read_text(encoding="utf-8")

    assert "Generated commands" in text
    assert "live-dcs" in text
    assert "experiment-export" in text
    assert "experiment-analyze" in text
