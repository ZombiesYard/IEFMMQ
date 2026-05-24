from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "docs" / "wiki" / "experiment-forms.html"
FORMS_MD = ROOT / "docs" / "wiki" / "experiment-forms.md"


def _html() -> str:
    return HTML.read_text(encoding="utf-8")


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
