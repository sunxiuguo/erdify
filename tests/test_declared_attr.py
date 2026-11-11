"""Static SQLAlchemy relationship factories (#298); their code must never run."""

import ast
from pathlib import Path

import pytest

from erdify.parser import parse_models_directory
from erdify.parser.dialects.base import ParseContext
from erdify.parser.dialects.sqlalchemy import SQLAlchemyDialect
from tests.ground_truth import sqlalchemy_ref
from tests.ground_truth.schema import from_erdify

HEADER = """
from sqlalchemy import ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, declared_attr, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class Owner(Base):
    __tablename__ = "owner"
    id: Mapped[int] = mapped_column(primary_key=True)
"""

FACTORY = """    @declared_attr
    def owner(cls) -> Mapped["Owner"]:
        return relationship("Owner")
"""


@pytest.mark.parametrize("assignment_first", [False, True])
@pytest.mark.parametrize(
    ("name", "annotation", "value"),
    [
        ("title", "str", "None"),
        ("title", "str", 'mapped_column(default="untitled")'),
        ("owner", '"Owner"', 'relationship("Owner")'),
    ],
)
def test_split_declarations_are_preserved(
    tmp_path: Path, assignment_first: bool, name: str, annotation: str, value: str
):
    annotation_line = f"    {name}: Mapped[{annotation}]\n"
    assignment = f"    {name} = {value}\n"
    body = assignment + annotation_line if assignment_first else annotation_line + assignment
    source = (
        HEADER
        + f"""
class Document(Base):
    __tablename__ = "document"
    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("owner.id"))
{body}
"""
    )
    reference = sqlalchemy_ref.load(source)
    assert reference.checks == ()
    (tmp_path / "models.py").write_text(source, encoding="utf-8")
    assert from_erdify(tmp_path, ("sqlalchemy",), relationships=True) == reference


@pytest.mark.parametrize("bases", ["Annotations, OwnedMixin", "OwnedMixin, Annotations"])
def test_annotation_only_sibling_mixin_keeps_factory(tmp_path: Path, bases: str):
    source = (
        HEADER
        + f"""
class Annotations:
    owner: Mapped["Owner"]

class OwnedMixin:
    owner_id: Mapped[int] = mapped_column(ForeignKey("owner.id"))
{FACTORY}
class Document({bases}, Base):
    __tablename__ = "document"
    id: Mapped[int] = mapped_column(primary_key=True)
"""
    )
    reference = sqlalchemy_ref.load(source)
    assert reference.checks == ()
    (tmp_path / "models.py").write_text(source, encoding="utf-8")
    assert from_erdify(tmp_path, ("sqlalchemy",), relationships=True) == reference


@pytest.mark.parametrize(
    ("decorator", "same_class", "subclass"),
    [
        ("@declared_attr", "", ""),
        ("@declared_attr\n    @classmethod", "", ""),
        ("@classmethod\n    @declared_attr", "", ""),
        ("@property\n    @declared_attr", "", ""),
        ("@staticmethod\n    @declared_attr", "", ""),
        ("@declared_attr", "    owner = None\n", ""),
        ("@declared_attr", "", "    owner = None\n"),
        ("@declared_attr", "    def owner(self):\n        return None\n", ""),
        ("@declared_attr", "", "    @property\n    def owner(self):\n        return None\n"),
        ("@declared_attr", '    owner: Mapped["Owner"]\n', ""),
        ("@declared_attr", "", '    owner: Mapped["Owner"]\n'),
    ],
)
def test_binding_and_decorator_order_match_sqlalchemy(
    tmp_path: Path, decorator: str, same_class: str, subclass: str
):
    factory = FACTORY.replace("@declared_attr", decorator)
    source = (
        HEADER
        + f"""
class OwnedMixin:
    owner_id: Mapped[int] = mapped_column(ForeignKey("owner.id"))
{factory}{same_class}
class Document(OwnedMixin, Base):
    __tablename__ = "document"
    id: Mapped[int] = mapped_column(primary_key=True)
{subclass}
"""
    )
    reference = sqlalchemy_ref.load(source)
    assert reference.checks == ()
    (tmp_path / "models.py").write_text(source, encoding="utf-8")
    assert from_erdify(tmp_path, ("sqlalchemy",), relationships=True) == reference


def test_declared_relationships_follow_annotations_and_mro(tmp_path: Path):
    fixture = (
        Path(__file__).parent / "fixtures/sqlalchemy_edge_cases/declared_attr_relationships.py"
    )
    (tmp_path / "models.py").write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    entities, _ = parse_models_directory(tmp_path)
    assert entities["Document"].relationships == [("Owner", "one", "owner")]
    assert entities["Document"].nullable_relationships == ["owner"]
    assert entities["Document"].inverse_relationships == {"owner": "documents"}
    assert ("Document", "many", "documents") in entities["Owner"].relationships
    assert entities["Archive"].inverse_relationships == {"owner": "archives"}
    assert entities["Draft"].inverse_relationships == {}
    assert entities["document_tags"].is_link_table
    assert not any(name.endswith("Mixin") for name in entities)


@pytest.mark.parametrize(
    "body",
    [
        'def owner(cls) -> Mapped["Owner"]:\n    return relationship()',
        '@property\ndef owner(cls) -> Mapped["Owner"]:\n    return relationship()',
        '@declared_attr\ndef owner(cls):\n    return relationship("Owner")',
        '@declared_attr\ndef owner(cls) -> Mapped["Owner"]:\n    pass',
        '@declared_attr\ndef owner(cls) -> Mapped["Owner"]:\n    return',
        '@declared_attr\ndef owner(cls) -> Mapped["Owner"]:\n    return make_relationship()',
        '@declared_attr\ndef owner(cls) -> Mapped["Owner"]:\n    return mapped_column()',
        '@declared_attr\ndef owner(cls) -> Mapped["Owner"]:\n'
        "    if cls.enabled:\n        return relationship()",
        '@declared_attr\ndef owner(cls) -> Mapped["Owner"]:\n'
        "    def nested():\n        return relationship()\n    return nested()",
        '@declared_attr\nasync def owner(cls) -> Mapped["Owner"]:\n    return relationship()',
        '@declared_attr\ndef _owner(cls) -> Mapped["Owner"]:\n    return relationship()',
        '@declared_attr\ndef owner(cls) -> Mapped["Owner | Other"]:\n    return relationship()',
        '@declared_attr\ndef owner(cls) -> Mapped[list["Owner"]]:\n'
        '    return relationship(secondary="link")',
    ],
)
def test_unsupported_bodies_are_not_guessed(body: str):
    node = ast.parse(
        "class Model:\n" + "\n".join("    " + line for line in body.splitlines())
    ).body[0]
    assert isinstance(node, ast.ClassDef)
    fields, relationships = SQLAlchemyDialect(ParseContext()).class_fields(node, "Model")
    assert not fields
    assert not relationships


def test_unsupported_declaration_still_shadows_a_base():
    node = ast.parse(
        "class Model:\n"
        "    @declared_attr\n"
        '    def owner(cls) -> Mapped["Owner"]:\n'
        "        return choose_relationship(cls)\n"
    ).body[0]
    assert isinstance(node, ast.ClassDef)
    assert SQLAlchemyDialect(ParseContext()).declared_names(node) == ["owner"]


@pytest.mark.parametrize("method_last", [False, True])
def test_last_declaration_wins_within_class(method_last: bool):
    assignment = '    owner: Mapped["Other"] = relationship()\n'
    method = (
        "    @declared_attr\n"
        '    def owner(cls) -> Mapped["Owner"]:\n'
        "        return relationship()\n"
    )
    body = assignment + method if method_last else method + assignment
    node = ast.parse("class Model:\n" + body).body[0]
    assert isinstance(node, ast.ClassDef)
    _, relationships = SQLAlchemyDialect(ParseContext()).class_fields(node, "Model")
    assert relationships["owner"].target == ("Owner" if method_last else "Other")


def test_factory_relationships_keep_declaration_order():
    body = FACTORY + FACTORY.replace("owner", "editor")
    node = ast.parse("class Model:\n" + body).body[0]
    assert isinstance(node, ast.ClassDef)
    _, relationships = SQLAlchemyDialect(ParseContext()).class_fields(node, "Model")
    assert list(relationships) == ["owner", "editor"]
