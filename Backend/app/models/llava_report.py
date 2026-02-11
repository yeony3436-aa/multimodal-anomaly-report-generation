from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Text, BigInteger, String, Integer, Float, TIMESTAMP
from sqlalchemy.sql import func

class Base(DeclarativeBase):
    pass

class LlavaReport(Base):
    __tablename__ = "llava_reports"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    filename: Mapped[str] = mapped_column(Text)
    image_path: Mapped[str] = mapped_column(Text)
    heatmap_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    overlay_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    dataset: Mapped[str] = mapped_column(String(50))
    category: Mapped[str] = mapped_column(String(50))
    ground_truth: Mapped[str | None] = mapped_column(Text, nullable=True)

    decision: Mapped[str] = mapped_column(String(20))
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    has_defect: Mapped[int] = mapped_column(Integer, default=0)
    defect_type: Mapped[str] = mapped_column(Text, default="")
    location: Mapped[str] = mapped_column(Text, default="")
    severity: Mapped[str] = mapped_column(Text, default="")

    defect_description: Mapped[str] = mapped_column(Text, default="")
    possible_cause: Mapped[str] = mapped_column(Text, default="")
    product_description: Mapped[str] = mapped_column(Text, default="")

    summary: Mapped[str] = mapped_column(Text, default="")
    impact: Mapped[str] = mapped_column(Text, default="")
    recommendation: Mapped[str] = mapped_column(Text, default="")

    inference_time: Mapped[float | None] = mapped_column(Float, nullable=True)  # seconds
    datetime: Mapped[object] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
