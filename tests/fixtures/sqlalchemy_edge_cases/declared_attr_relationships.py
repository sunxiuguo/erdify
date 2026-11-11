"""Declared relationships: qualified calls, annotations, MRO and secondary tables."""

from sqlalchemy import Column, ForeignKey, Table, orm


class Base(orm.DeclarativeBase):
    pass


document_tags = Table(
    "document_tags",
    Base.metadata,
    Column("document_id", ForeignKey("document.id"), primary_key=True),
    Column("tag_id", ForeignKey("tag.id"), primary_key=True),
)


class OwnedMixin:
    owner_id: orm.Mapped[int | None] = orm.mapped_column(ForeignKey("owner.id"))

    @orm.declared_attr
    @classmethod
    def owner(cls) -> orm.Mapped["Owner | None"]:
        """The owner is optional, including on inherited relationships."""
        return orm.relationship("Owner", back_populates="documents")


class TransitiveOwnedMixin(OwnedMixin):
    pass


class ArchivedMixin:
    @orm.declared_attr
    def owner(cls) -> orm.Mapped["Owner | None"]:
        return orm.relationship("Owner", back_populates="archives")


class TaggedMixin:
    @orm.declared_attr
    def tags(cls) -> orm.Mapped[list["Tag"]]:
        return orm.relationship("Tag", secondary=document_tags)


class Owner(Base):
    __tablename__ = "owner"
    id: orm.Mapped[int] = orm.mapped_column(primary_key=True)

    @orm.declared_attr
    def documents(cls) -> orm.Mapped[list["Document"]]:
        return orm.relationship("Document", back_populates="owner")

    archives: orm.Mapped[list["Archive"]] = orm.relationship(back_populates="owner")


class Document(TransitiveOwnedMixin, TaggedMixin, Base):
    __tablename__ = "document"
    id: orm.Mapped[int] = orm.mapped_column(primary_key=True)


class Archive(ArchivedMixin, OwnedMixin, Base):
    __tablename__ = "archive"
    id: orm.Mapped[int] = orm.mapped_column(primary_key=True)


class Draft(OwnedMixin, Base):
    __tablename__ = "draft"
    id: orm.Mapped[int] = orm.mapped_column(primary_key=True)
    owner: orm.Mapped["Owner | None"] = orm.relationship()


class Tag(Base):
    __tablename__ = "tag"
    id: orm.Mapped[int] = orm.mapped_column(primary_key=True)
