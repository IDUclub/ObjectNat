import pathlib
import tomllib


def main() -> None:
    pyproject_path = pathlib.Path("pyproject.toml")
    version_file_path = pathlib.Path("src/objectnat/_version.py")

    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)
    version = data["tool"]["poetry"]["version"]

    text = version_file_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    new_lines: list[str] = []

    replaced = False
    for line in lines:
        if line.lstrip().startswith("VERSION"):
            new_lines.append(f'VERSION = "{version}"')
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        raise SystemExit(f"VERSION line not found in {version_file_path}. " "Expected a line starting with `VERSION`.")

    end_newline = "\n" if text.endswith("\n") else ""
    version_file_path.write_text("\n".join(new_lines) + end_newline, encoding="utf-8")

    print(f"Synced VERSION in {version_file_path} to {version}")


if __name__ == "__main__":
    main()
