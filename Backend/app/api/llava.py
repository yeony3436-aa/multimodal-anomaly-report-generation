from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
from typing import Optional

from app.db.session import get_db
from app.models.llava_report import LlavaReport
from app.schemas.llava_report import ReportListDTO, ReportDTO

router = APIRouter(prefix="/llava", tags=["llava"])


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


@router.get("/reports", response_model=ReportListDTO)
async def list_reports(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    dataset: Optional[str] = None,
    category: Optional[str] = None,
    decision: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(LlavaReport)
    cnt_stmt = select(func.count()).select_from(LlavaReport)

    def apply_filters(q):
        if dataset:
            q = q.where(LlavaReport.dataset == dataset)
        if category:
            q = q.where(LlavaReport.category == category)
        if decision:
            q = q.where(func.lower(LlavaReport.decision) == decision.strip().lower())
        df = _parse_dt(date_from)
        dt = _parse_dt(date_to)
        if df:
            q = q.where(LlavaReport.datetime >= df)
        if dt:
            q = q.where(LlavaReport.datetime <= dt)
        return q

    stmt = apply_filters(stmt).order_by(LlavaReport.datetime.desc()).limit(limit).offset(offset)
    cnt_stmt = apply_filters(cnt_stmt)

    total = (await db.execute(cnt_stmt)).scalar_one()
    rows = (await db.execute(stmt)).scalars().all()

    items = [
        ReportDTO(
            id=r.id,
            filename=r.filename,
            image_path=r.image_path,
            heatmap_path=getattr(r, "heatmap_path", None),
            overlay_path=getattr(r, "overlay_path", None),
            dataset=r.dataset,
            category=r.category,
            ground_truth=r.ground_truth,
            decision=r.decision,
            confidence=r.confidence,
            has_defect=r.has_defect,
            defect_type=r.defect_type,
            location=r.location,
            severity=r.severity,
            defect_description=r.defect_description,
            possible_cause=r.possible_cause,
            product_description=r.product_description,
            summary=r.summary,
            impact=r.impact,
            recommendation=r.recommendation,
            inference_time=r.inference_time,
            datetime=r.datetime.isoformat(),
        )
        for r in rows
    ]

    return ReportListDTO(items=items, total=int(total))
