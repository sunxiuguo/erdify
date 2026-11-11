"""SQLModel and SQLAlchemy 2.0 (ORM classes and Core association tables)."""

import ast
from typing import Dict, Iterable, List, Tuple

from ...config import EntityInfo, FieldInfo, is_structural_link_table
from ..annotations import expr_name, is_call_to, is_mapped_annotation
from ..common import _ClassMembers, _Rel
from .annotated import AnnotatedDialect, extract_foreign_key_arg


class SQLModelDialect(AnnotatedDialect):
    """SQLModel: a class declared with ``table=True``."""

    kind = "sqlmodel"

    def detect(self, class_node: ast.ClassDef) -> bool:
        for keyword in class_node.keywords:
            if keyword.arg == "table" and isinstance(keyword.value, ast.Constant):
                if keyword.value.value:
                    return True
        return False


class SQLAlchemyDialect(AnnotatedDialect):
    """SQLAlchemy 2.0: a concrete mapped class has ``__tablename__`` and ``Mapped[...]``.

    Mixins / abstract bases (``Mapped`` fields but no ``__tablename__``) are not
    detected, so they are not emitted as entities, but their fields are still
    inherited.
    """

    kind = "sqlalchemy"
    reads_annotated_columns = True

    def detect(self, class_node: ast.ClassDef) -> bool:
        return has_tablename(class_node) and has_mapped_field(class_node)

    def body_members(self, class_node: ast.ClassDef) -> Iterable[ast.stmt]:
        """Read static relationship factories as annotated attributes."""
        return self._attribute_members(class_node).values()

    def declared_names(self, class_node: ast.ClassDef) -> List[str]:
        """Value bindings shadow inherited members; an annotation alone does not."""
        return [name for name in self._attribute_members(class_node) if not name.startswith("_")]

    def _attribute_members(self, class_node: ast.ClassDef) -> Dict[str, ast.stmt]:
        members = attribute_bindings(class_node)
        for name, node in list(members.items()):
            if isinstance(node, ast.FunctionDef):
                declaration = declared_relationship(node)
                if declaration is not None:
                    members[name] = declaration
        return members

    def merge(
        self,
        class_node: ast.ClassDef,
        merge_order: List[str],
        members: Dict[str, _ClassMembers],
    ) -> Tuple[Dict[str, FieldInfo], Dict[str, _Rel]]:
        """Annotations alone do not replace a factory anywhere in the concrete MRO."""
        bindings: Dict[str, ast.stmt] = {}
        for name in merge_order:
            for attr, node in attribute_bindings(self.classes.all[name]).items():
                if not (isinstance(node, ast.AnnAssign) and node.value is None):
                    bindings[attr] = node
        factories = {
            attr
            for attr, node in bindings.items()
            if isinstance(node, ast.FunctionDef) and is_declared_attr(node)
        }
        adjusted: Dict[str, _ClassMembers] = {}
        for name, (fields, rels, declared) in members.items():
            annotation_only = {
                attr
                for attr, node in attribute_bindings(self.classes.all[name]).items()
                if attr in factories and isinstance(node, ast.AnnAssign) and node.value is None
            }
            adjusted[name] = (
                {attr: value for attr, value in fields.items() if attr not in annotation_only},
                rels,
                [attr for attr in declared if attr not in annotation_only],
            )
        return super().merge(class_node, merge_order, adjusted)

    def extra_entities(self, trees: Iterable[ast.Module]) -> List[EntityInfo]:
        """Core association tables (:func:`core_association_tables`, #34)."""
        return core_association_tables(trees)


def attribute_bindings(class_node: ast.ClassDef) -> Dict[str, ast.stmt]:
    """Final class-body bindings; annotation-only statements never replace a value."""
    members: Dict[str, ast.stmt] = {}
    annotations: Dict[str, ast.expr] = {}
    for node in class_node.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            annotations[node.target.id] = node.annotation
            if node.value is not None or node.target.id not in members:
                members[node.target.id] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for part in ast.walk(target):
                    if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Store):
                        members[part.id] = node
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            members[node.name] = node
    # Python stores annotations separately from values. A split declaration
    # (`title: Mapped[str]` and `title = mapped_column()`) is still a column,
    # in either order; only a method/property replaces it with a descriptor.
    for name, node in members.items():
        if name in annotations and isinstance(node, (ast.AnnAssign, ast.Assign)):
            members[name] = ast.AnnAssign(
                target=ast.Name(id=name, ctx=ast.Store()),
                annotation=annotations[name],
                value=node.value,
                simple=1,
            )
    return members


def is_declared_attr(node: ast.FunctionDef) -> bool:
    """Only the outermost decorator determines whether this is a declared attribute."""
    decorators = [expr_name(decorator) for decorator in node.decorator_list]
    return decorators in (["declared_attr"], ["declared_attr", "classmethod"])


def declared_relationship(node: ast.FunctionDef) -> ast.AnnAssign | None:
    """A direct, annotated ``@declared_attr`` relationship, without executing code."""
    if not is_declared_attr(node):
        return None
    # Do not search nested functions or choose a branch of a computed factory.
    body = node.body[1:] if ast.get_docstring(node) is not None else node.body
    if (
        node.returns is not None
        and len(body) == 1
        and isinstance(body[0], ast.Return)
        and body[0].value is not None
        and is_call_to(body[0].value, "relationship")
    ):
        return ast.AnnAssign(
            target=ast.Name(id=node.name, ctx=ast.Store()),
            annotation=node.returns,
            value=body[0].value,
            simple=1,
        )
    return None


def core_association_tables(trees: Iterable[ast.Module]) -> List[EntityInfo]:
    """Link entities from module-level Core ``Table(...)`` assignments.

    ``relationship(secondary=Table(...))`` points at a SQLAlchemy Core table
    rather than a mapped class. Only class definitions become entities in the
    normal passes, so such association tables - and their M:N - would be lost.
    Parse module-level ``name = Table("t", metadata, Column(...), ...)``
    assignments that are structurally association tables (a composite PK
    made up entirely of FK columns) and return them as link entities (#34).
    """
    entities: List[EntityInfo] = []
    for tree in trees:
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            if not is_call_to(node.value, "Table"):
                continue

            entity = build_core_table_entity(node.targets[0].id, node.value)
            if entity is not None:
                entities.append(entity)
    return entities


def has_tablename(class_node: ast.ClassDef) -> bool:
    """Check if a class assigns __tablename__ (annotated or plain)."""
    for node in class_node.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "__tablename__":
                return True
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__tablename__":
                    return True
    return False


def has_mapped_field(class_node: ast.ClassDef) -> bool:
    """Check if a class has at least one Mapped[...] annotated field."""
    for node in class_node.body:
        if isinstance(node, ast.AnnAssign) and is_mapped_annotation(node.annotation):
            return True
    return False


def build_core_table_entity(var_name: str, call: ast.Call) -> EntityInfo | None:
    """Build a link EntityInfo from a Core ``Table(...)`` call, or None if it
    is not structurally an association table."""
    # Table name is the first positional string arg; fall back to variable name.
    table_name = var_name
    if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
        table_name = call.args[0].value

    fields: List[FieldInfo] = []
    for arg in call.args:
        if is_call_to(arg, "Column"):
            field = parse_core_column(arg)
            if field is not None:
                fields.append(field)

    if not is_structural_link_table(fields):
        return None

    entity = EntityInfo(
        name=var_name,
        table_name=table_name,
        base_classes=[],
        source="sqlalchemy",
    )
    entity.fields = fields
    entity.is_link_table = True
    return entity


def parse_core_column(call: ast.Call) -> FieldInfo | None:
    """Parse a Core ``Column("name", <Type>, ForeignKey(...), primary_key=...)``."""
    if not call.args or not isinstance(call.args[0], ast.Constant):
        return None
    col_name = str(call.args[0].value)

    is_primary_key = False
    for keyword in call.keywords:
        if keyword.arg == "primary_key" and isinstance(keyword.value, ast.Constant):
            is_primary_key = bool(keyword.value.value)

    fk_target = extract_foreign_key_arg(call)

    # An optional positional column type (e.g. Column("id", Integer, ...)).
    type_str = ""
    for arg in call.args[1:]:
        if isinstance(arg, ast.Name):
            type_str = arg.id
            break
        if isinstance(arg, ast.Attribute):
            type_str = arg.attr
            break

    return FieldInfo(
        name=col_name,
        type_str=type_str,
        is_primary_key=is_primary_key,
        is_foreign_key=fk_target is not None,
        foreign_table=fk_target,
    )
